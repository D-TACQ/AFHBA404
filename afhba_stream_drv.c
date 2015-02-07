/* ------------------------------------------------------------------------- */
/* afhba_stream_drv.c D-TACQ ACQ400 FMC  DRIVER
 * afhba_stream_drv.c
 *
 *  Created on: 19 Jan 2015
 *      Author: pgm
 */

/* ------------------------------------------------------------------------- */
/*   Copyright (C) 2015 Peter Milne, D-TACQ Solutions Ltd                    *
 *                      <peter dot milne at D hyphen TACQ dot com>           *
 *                                                                           *
 *  This program is free software; you can redistribute it and/or modify     *
 *  it under the terms of Version 2 of the GNU General Public License        *
 *  as published by the Free Software Foundation;                            *
 *                                                                           *
 *  This program is distributed in the hope that it will be useful,          *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of           *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            *
 *  GNU General Public License for more details.                             *
 *                                                                           *
 *  You should have received a copy of the GNU General Public License        *
 *  along with this program; if not, write to the Free Software              *
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.                */
/* ------------------------------------------------------------------------- */

/*
 * prefix afs : acq fiber stream
 */


#ifndef EXPORT_SYMTAB
#define EXPORT_SYMTAB
#include <linux/module.h>
#endif

#include "acq-fiber-hba.h"
#include "afhba_stream_drv.h"



int RX_TO = 10*HZ;
module_param(RX_TO, int, 0644);
MODULE_PARM_DESC(RX_TO, "RX timeout (jiffies) [0.1Hz]");

int WORK_TO = HZ/10;
module_param(WORK_TO, int, 0644);
MODULE_PARM_DESC(WORK_TO,
	"WORK timeout (jiffies) [10Hz] - decrease for hi fifo stat poll rate");

int SMOO = 7;
module_param(SMOO, int, 0644);
MODULE_PARM_DESC(SMOO, "rate smoothing factor 0..9 none..smooth");

int stalls = 0;
module_param(stalls, int, 0644);
MODULE_PARM_DESC(stalls, "number of times ISR ran with no buffers to queue");

int buffer_debug = 0;
module_param(buffer_debug, int, 0644);


int nbuffers = NBUFFERS;
module_param(nbuffers, int, 0444);
MODULE_PARM_DESC(nbuffers, "number of host-side buffers");

int buffer_len = BUFFER_LEN;
module_param(buffer_len, int, 0644);
MODULE_PARM_DESC(buffer_len, "length of each buffer in bytes");


int stop_on_skipped_buffer = 0;
module_param(stop_on_skipped_buffer, int, 0644);

int transfer_buffers = 0x7fffffff;
module_param(transfer_buffers, int, 0664);
MODULE_PARM_DESC(transfer_buffers, "number of buffers to transfer");

int aurora_to_ms = 1000;
module_param(aurora_to_ms, int, 0644);
MODULE_PARM_DESC(aurora_to_ms, "timeout on aurora connect");

int aurora_monitor = 0;
module_param(aurora_monitor, int, 0644);
MODULE_PARM_DESC(aurora_monitor, "enable to check cable state in run loop, disable for debug");

int eot_interrupt = 1;
module_param(eot_interrupt, int, 0644);
MODULE_PARM_DESC(eot_interrupt, "1: interrupt every, 0: interrupt none, N: interrupt interval");

static struct file_operations afs_fops_dma;
static struct file_operations afs_fops_dma_poll;

static int getOrder(int len)
{
	int order;
	len /= PAGE_SIZE;

	for (order = 0; 1 << order < len; ++order){
		;
	}
	return order;
}

static int getAFDMAC_Order(int len)
{
	int order;
	len /= AFDMAC_PAGE;

	for (order = 0; 1 << order < len; ++order){
		;
	}
	return order;
}

void init_descriptors_ht(struct AFHBA_STREAM_DEV *sdev)
{
	int ii;

	for (ii = 0; ii < sdev->nbuffers; ++ii){
		u32 descr = sdev->hbx[ii].descr;

		descr &= ~AFDMAC_DESC_LEN_MASK;
		descr |= getAFDMAC_Order(sdev->buffer_len)<< AFDMAC_DESC_LEN_SHL;
		switch(eot_interrupt){
		case 0:
			descr &= ~AFDMAC_DESC_EOT;
			break;
		case 1:
			descr |= AFDMAC_DESC_EOT;
			break;
		default:
			if (ii%eot_interrupt == 0){
				descr |= AFDMAC_DESC_EOT;
			}else{
				descr &= ~AFDMAC_DESC_EOT;
			}
			break;
		}

		sdev->hbx[ii].descr = descr;
	}
}





#define COPY_FROM_USER(src, dest, len) \
	if (copy_from_user(src, dest, len)) { return -EFAULT; }


static void write_descr(struct AFHBA_DEV *adev, unsigned offset, int idesc)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	u32 descr = sdev->hbx[idesc].descr;

	if (sdev->job.buffers_queued < 5){
		dev_info(pdev(adev), "write_descr(%d) [%d] offset:%04x = %08x",
				sdev->job.buffers_queued, idesc, offset, descr);
	}
	dev_dbg(pdev(adev), "ibuf %d offset:%04x = %08x", idesc, offset, descr);
	writel(descr, adev->mappings[REMOTE_BAR].va+offset);
}

void _afs_write_dmareg(struct AFHBA_DEV *adev, int regoff, u32 value)

{
	u32* dma_regs = (u32*)(adev->mappings[REMOTE_BAR].va + DMA_BASE);
	void* va = &dma_regs[regoff];
	dev_dbg(pdev(adev), "_afs_write_dmareg %04lx = %08x",
			va-adev->mappings[REMOTE_BAR].va, value);
	writel(value, va);
}

u32 _afs_read_dmareg(struct AFHBA_DEV *adev, int regoff)
{
	u32* dma_regs = (u32*)(adev->mappings[REMOTE_BAR].va + DMA_BASE);
	void* va = &dma_regs[regoff];
	u32 value = readl(va);
	dev_dbg(pdev(adev), "_afs_read_dmareg %04lx = %08x",
			va-adev->mappings[REMOTE_BAR].va, value);
	return adev->stream_dev->dma_regs[regoff] = value;
}

void _afs_write_pcireg(struct AFHBA_DEV *adev, int regoff, u32 value)

{
	u32* dma_regs = (u32*)(adev->mappings[REMOTE_BAR].va + PCIE_BASE);
	void* va = &dma_regs[regoff];
	dev_dbg(pdev(adev), "_afs_write_pcireg %04lx = %08x",
				va-adev->mappings[REMOTE_BAR].va, value);
	dev_dbg(pdev(adev), "%p = %08x", va, value);
	writel(value, va);
}

u32 _afs_read_pcireg(struct AFHBA_DEV *adev, int regoff)
{
	u32* dma_regs = (u32*)(adev->mappings[REMOTE_BAR].va + PCIE_BASE);
	void* va = &dma_regs[regoff];
	u32 value = readl(va);
	dev_dbg(pdev(adev), "_afs_read_pcireg %04lx = %08x",
			va-adev->mappings[REMOTE_BAR].va, value);
	return adev->stream_dev->dma_regs[regoff] = value;
}
static void afs_load_push_descriptor(struct AFHBA_DEV *adev, int idesc)
{
/* change descr status .. */
	write_descr(adev, DMA_PUSH_DESC_FIFO, idesc);
}


static int _afs_dma_started(struct AFHBA_DEV *adev, int shl)
{
	u32 ctrl = DMA_CTRL_RD(adev);
	ctrl >>= shl;
	return (ctrl&DMA_CTRL_EN) != 0;
}


static inline int afs_push_dma_started(struct AFHBA_DEV *adev)
{
	return _afs_dma_started(adev, DMA_CTRL_PUSH_SHL);
}
static inline int afs_pull_dma_started(struct AFHBA_DEV *adev)
{
	return _afs_dma_started(adev, DMA_CTRL_PULL_SHL);
}

static int afs_aurora_lane_up(struct AFHBA_DEV *adev)
{
	u32 stat = afhba_read_reg(adev, AURORA_STATUS_REG);
	return (stat & AFHBA_AURORA_STAT_LANE_UP) != 0;
}

static void _afs_pcie_mirror_init(struct AFHBA_DEV *adev)
{
	int ireg;

	for (ireg = PCIE_CNTRL; ireg <= PCIE_BUFFER_CTRL; ++ireg){
		PCI_REG_WRITE(adev, ireg, afhba_read_reg(adev, ireg*sizeof(u32)));
	}
}
static int _afs_comms_init(struct AFHBA_DEV *adev)
{
	struct AFHBA_STREAM_DEV* sdev = adev->stream_dev;
	int to = 0;

	afhba_write_reg(adev, AURORA_CONTROL_REG, AFHBA_AURORA_CTRL_ENA);

	while(!afs_aurora_lane_up(adev)){
		msleep(1);
		if (++to > aurora_to_ms){
			return 0;
		}
	}

	_afs_pcie_mirror_init(adev);
	return sdev->comms_init_done = true;
}

int afs_comms_init(struct AFHBA_DEV *adev)
{
	struct AFHBA_STREAM_DEV* sdev = adev->stream_dev;

	if (afs_aurora_lane_up(adev)){
		if (!sdev->comms_init_done){
			_afs_comms_init(adev);
		}
		return sdev->comms_init_done;
	}else{
		dev_dbg(pdev(adev), "aurora lane down");
		return sdev->comms_init_done = false;
	}
}


/* @@todo : dma implement PUSH only */

static void afs_dma_reset(struct AFHBA_DEV *adev)
{
	DMA_CTRL_CLR(adev, DMA_CTRL_EN);
	DMA_CTRL_SET(adev, DMA_CTRL_FIFO_RST);
	DMA_CTRL_CLR(adev, DMA_CTRL_FIFO_RST);
	//DMA_CTRL_SET(adev, DMA_CTRL_EN);
}

static void afs_start_dma(struct AFHBA_DEV *adev)
{
	DMA_CTRL_SET(adev, DMA_CTRL_EN);
}

static void afs_stop_dma(struct AFHBA_DEV *adev)
{
	DMA_CTRL_CLR(adev, DMA_CTRL_EN);
}

#define RTDMAC_DATA_FIFO_CNT	0x1000
#define RTDMAC_DESC_FIFO_CNT	0x1000

#define DATA_FIFO_SZ	(RTDMAC_DATA_FIFO_CNT*sizeof(unsigned))
#define DESC_FIFO_SZ	(RTDMAC_DESC_FIFO_CNT*sizeof(unsigned))

static void mark_empty(struct device *dev, struct HostBuffer *hb){
	u32 mark_len = 2 * sizeof(u32);
	u32 offset = hb->req_len - mark_len;
	u32 *pmark = (u32*)(hb->va + offset);

	pmark[0] = EMPTY1;
	pmark[1] = EMPTY2;

	/* direction may be wrong - we're trying to flush */
	dma_sync_single_for_device(dev, hb->pa, hb->req_len, PCI_DMA_TODEVICE);
}


static int is_marked_empty(struct device *dev, struct HostBuffer *hb){
	u32 mark_len = 2 * sizeof(u32);
	u32 offset = hb->req_len - mark_len;
	u32 *pmark = (u32*)(hb->va + offset);
	int is_empty;

	dma_sync_single_for_cpu(dev, hb->pa, hb->req_len, PCI_DMA_FROMDEVICE);

	is_empty = pmark[0] == EMPTY1 && pmark[1] == EMPTY2;

	return is_empty;
}

static int queue_next_free_buffer(struct AFHBA_DEV *adev)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	int rc = 0;

	if (mutex_lock_interruptible(&sdev->list_mutex)){
		return -ERESTARTSYS;
	}
	if (!list_empty_careful(&sdev->bp_empties.list)){
		struct HostBuffer *hb = HB_ENTRY(sdev->bp_empties.list.next);

		mark_empty(&adev->pci_dev->dev, hb);

		afs_load_push_descriptor(adev, hb->ibuf);
		hb->bstate = BS_FILLING;
		list_move_tail(&hb->list, &sdev->bp_filling.list);
		rc = 1;
	}
	mutex_unlock(&sdev->list_mutex);
	return rc;
}

static void queue_free_buffers(struct AFHBA_DEV *adev)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	struct JOB *job = &sdev->job;
	int in_queue =  job->buffers_queued -
			(job->buffers_received+job->buffers_discarded);

	while (job->buffers_queued < job->buffers_demand){
		if (queue_next_free_buffer(adev)){
	                ++job->buffers_queued;
		        ++in_queue;
		}else{
			if (in_queue == 0){
				++stalls;
			}
			break;
		}
        }
}

struct HostBuffer* hb_from_descr(struct AFHBA_DEV *adev, u32 inflight_descr)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	int ii;

	for (ii = 0; ii < nbuffers; ++ii){
		if (sdev->hbx[ii].descr == inflight_descr){
			return &sdev->hbx[ii];
		}
	}
	return 0;
}

static void report_inflight(
	struct AFHBA_DEV *adev, int ibuf, int is_error, char *msg)
{
	struct AFHBA_STREAM_DEV* sdev = adev->stream_dev;
	u32 inflight_descr = DMA_PUSH_DESC_STA_RD(adev);
	struct HostBuffer*  inflight = hb_from_descr(adev, inflight_descr);

	if (sdev->job.buffers_demand == 0){
		return;
	}
	if (is_error){
		dev_err(pdev(adev),
			"%30s: buffer %02d  last descr:%08x [%02d] fifo:%08x",
			msg,
			ibuf,
			inflight_descr,
			inflight? inflight->ibuf: -1,
			DMA_DESC_FIFSTA_RD(adev));
	}else{
		dev_dbg(pdev(adev),
			"%30s: buffer %02d last descr:%08x [%02d] fifo:%08x",
			msg,
			ibuf,
			inflight_descr,
			inflight? inflight->ibuf: -1,
			DMA_DESC_FIFSTA_RD(adev));
	}
}
static void report_stuck_buffer(struct AFHBA_DEV *adev, int ibuf)
{
	report_inflight(adev, ibuf, 0, "buffer was skipped");
}

static void return_empty(struct AFHBA_DEV *adev, struct HostBuffer *hb)
/** caller MUST lock the list */
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	dev_dbg(pdev(adev), "ibuf %d", hb->ibuf);
	hb->bstate = BS_EMPTY;
	list_move_tail(&hb->list, &sdev->bp_empties.list);
}
static int queue_full_buffers(struct AFHBA_DEV *adev)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	struct HostBuffer* hb;
	struct HostBuffer* tmp;
	struct HostBuffer* first = 0;
	int nrx = 0;
	int ifilling = 0;

	if (mutex_lock_interruptible(&sdev->list_mutex)){
		return -ERESTARTSYS;
	}

	list_for_each_entry_safe(hb, tmp, &sdev->bp_filling.list, list){
		if (++ifilling == 1){
			first = hb;
		}
		if (is_marked_empty(&adev->pci_dev->dev, hb)){
			if (ifilling > 1){
				break; 	/* top 2 buffers empty: no action */
			}else{
				continue;  /* check skipped data. */
			}
		}else{
			if (ifilling > 1 && first && hb != first){
				sdev->job.buffers_discarded++;
				report_stuck_buffer(adev, first->ibuf);
				return_empty(adev, first);
				first = 0;
				if (stop_on_skipped_buffer){
					sdev->job.please_stop = PS_PLEASE_STOP;
				}
			}
			if (buffer_debug){
				report_inflight(adev, hb->ibuf, 0, "->FULL");
			}

			hb->bstate = BS_FULL;
			list_move_tail(&hb->list, &sdev->bp_full.list);
			sdev->job.buffers_received++;
			++nrx;
		}
	}

	mutex_unlock(&sdev->list_mutex);
	return nrx;
}



static void init_histo_buffers(struct AFHBA_STREAM_DEV* sdev)
{
	int ii;

	sdev->data_fifo_histo = kzalloc(DATA_FIFO_SZ, GFP_KERNEL);
	sdev->desc_fifo_histo =	kzalloc(DESC_FIFO_SZ, GFP_KERNEL);

	/* give it a test pattern .. */

	for (ii = 0; ii != RTDMAC_DATA_FIFO_CNT; ++ii){
		sdev->data_fifo_histo[ii] = 0x70000000 + ii;
	}
	for (ii = 0; ii != RTDMAC_DESC_FIFO_CNT; ++ii){
		sdev->desc_fifo_histo[ii] = 0x50000000 + ii;
	}
}

int afs_init_buffers(struct AFHBA_DEV* adev)
{
	struct AFHBA_STREAM_DEV* sdev = adev->stream_dev;
	struct HostBuffer *hb;
	int order = getOrder(BUFFER_LEN);
	int ii;

	dev_dbg(pdev(adev), "afs_init_buffers() 01 order=%d", order);

	sdev->hbx = kzalloc(sizeof(struct HostBuffer)*nbuffers, GFP_KERNEL);
        INIT_LIST_HEAD(&sdev->bp_empties.list);
	INIT_LIST_HEAD(&sdev->bp_filling.list);
	INIT_LIST_HEAD(&sdev->bp_full.list);
	spin_lock_init(&sdev->job_lock);

	mutex_init(&sdev->list_mutex);
	mutex_lock(&sdev->list_mutex);

	sdev->buffer_len = BUFFER_LEN;
	dev_dbg(pdev(adev), "allocating %d buffers size:%d order:%d dev.dma_mask:%08llx",
			nbuffers, BUFFER_LEN, order, *adev->pci_dev->dev.dma_mask);

	for (hb = sdev->hbx, ii = 0; ii < nbuffers; ++ii, ++sdev->nbuffers, ++hb){
		void *buf = (void*)__get_free_pages(GFP_KERNEL|GFP_DMA32, order);

		if (!buf){
			dev_err(pdev(adev), "failed to allocate buffer %d", ii);
			break;
		}

		dev_dbg(pdev(adev), "buffer %2d allocated at %p, map it", ii, buf);

		hb->ibuf = ii;
		hb->pa = dma_map_single(&adev->pci_dev->dev, buf,
				BUFFER_LEN, PCI_DMA_FROMDEVICE);
		hb->va = buf;
		hb->len = BUFFER_LEN;

		dev_dbg(pdev(adev), "buffer %2d allocated, map done", ii);

		if ((hb->pa & (AFDMAC_PAGE-1)) != 0){
			dev_err(pdev(adev), "HB NOT PAGE ALIGNED");
			WARN_ON(true);
			return -1;
		}

		hb->descr = hb->pa | 0 | AFDMAC_DESC_EOT | (ii&AFDMAC_DESC_ID_MASK);
		hb->bstate = BS_EMPTY;

		dev_dbg(pdev(adev), "[%d] %p %08x %d %08x",
		    ii, hb->va, hb->pa, hb->len, hb->descr);
		list_add_tail(&hb->list, &sdev->bp_empties.list);
	}
	sdev->nbuffers = nbuffers;
	sdev->init_descriptors = init_descriptors_ht;
	sdev->init_descriptors(sdev);
	init_waitqueue_head(&sdev->work.w_waitq);
	init_waitqueue_head(&sdev->return_waitq);

	mutex_unlock(&sdev->list_mutex);

	init_histo_buffers(sdev);
	dev_dbg(pdev(adev), "afs_init_buffers() 99");
	return 0;
}


static irqreturn_t afs_rx_isr(int irq, void *data)
{
	struct AFHBA_DEV* adev = (struct AFHBA_DEV*)data;
	struct AFHBA_STREAM_DEV* sdev = adev->stream_dev;

	dev_dbg(pdev(adev), "01 irq %d", irq);


       if (sdev->job.buffers_demand == 0 &&
		       sdev->job.please_stop != PS_PLEASE_STOP){
	       return !IRQ_HANDLED;
       }

       sdev->job.ints++;
       set_bit(WORK_REQUEST, &sdev->work.w_to_do);
       wake_up_interruptible(&sdev->work.w_waitq);

       dev_dbg(pdev(adev), "99");
       return IRQ_HANDLED;
}

static irqreturn_t afs_null_isr(int irq, void* data)
{
	struct AFHBA_DEV* adev = (struct AFHBA_DEV*)data;

	dev_info(pdev(adev), "afs_null_isr %d", irq);
	return IRQ_HANDLED;
}
static int hook_interrupts(struct AFHBA_DEV* adev)
{
	struct pci_dev *dev = adev->pci_dev;
	struct AFHBA_STREAM_DEV* sdev = adev->stream_dev;

	static const char* irq_names[4] = {
		"%s-dma", "%s-line", "%s-ppnf", "%s-spare"
	};
	int rc;
	int nvec;
	int iv;

	dev_dbg(pdev(adev), "%d IRQ %d", __LINE__, dev->irq);

	rc = pci_enable_msi_block(dev, nvec = 4);
	if (rc < 0){
		dev_warn(pdev(adev), "pci_enable_msi_block() returned %d", rc);
		rc = pci_enable_msi(dev);
		nvec = 1;
	}

	if (rc < 0){
		dev_err(pdev(adev), "pci_enable_msi FAILED");
		return rc;
	}

	for (iv = 0; iv < nvec; ++iv){
		snprintf(sdev->irq_names[iv], 32, irq_names[iv], adev->name);
	}

	rc = request_irq(dev->irq+0, afs_rx_isr,
			 	IRQF_SHARED, sdev->irq_names[0], adev);
	if (rc){
		dev_err(pdev(adev), "request_irq %d failed", dev->irq+0);
	}

	for (iv = 1; iv < nvec; ++iv){
		rc = request_irq(dev->irq+iv, afs_null_isr, IRQF_SHARED,
				sdev->irq_names[iv], adev);
		if (rc){
			dev_err(pdev(adev), "request_irq %d failed", dev->irq+iv);
		}else{
			dev_info(pdev(adev), "request_irq %s %d OK",
					sdev->irq_names[iv], dev->irq+iv);
		}
	}

	return rc;
}


static void smooth(unsigned *rate, unsigned *old, unsigned *new)
{
#define RATE	*rate
#define OLD	*old
#define NEW	*new

	if (likely(NEW > OLD)){
		RATE = (SMOO*RATE + (10-SMOO)*(NEW-OLD))/10;
	}else{
		RATE = 0;
	}
	OLD = NEW;
#undef NEW
#undef OLD
#undef RATE
}


static int as_mon(void *arg)
{
	struct AFHBA_DEV* adev = (struct AFHBA_DEV*)arg;
	wait_queue_head_t waitq;

	init_waitqueue_head(&waitq);

	while(!kthread_should_stop()){
		struct JOB *job = &adev->stream_dev->job;
		wait_event_interruptible_timeout(waitq, 0, HZ);

		smooth(&job->rx_rate,
			&job->rx_buffers_previous, &job->buffers_received);

		smooth(&job->int_rate, &job->int_previous, &job->ints);
	}

	return 0;
}


static void check_fifo_status(struct AFHBA_DEV* adev)
{
#ifdef TODOLATER  /** @@todo */
	u32 desc_sta = DMA_PUSH_DESC_STA_RD(adev);
	u32 desc_flags = check_fifo_xxxx(tdev->desc_fifo_histo, desc_sta);
	u32 data_sta = rtd_read_reg(tdev, RTMT_C_DATA_FIFSTA);
	u32 data_flags = check_fifo_xxxx(tdev->data_fifo_histo, data_sta);

	if ((data_flags & RTMT_H_XX_DMA_FIFSTA_FULL)   &&
					tdev->job.errors < 10){
		/** @@todo .. do something! */
		err("GAME OVER: %d FIFSTA_DATA_OVERFLOW: 0x%08x",
		    tdev->idx, data_sta);
		if (++tdev->job.errors == 10){
			err("too many errors, turning reporting off ..");
		}
	}
	if ((desc_flags & RTMT_H_XX_DMA_FIFSTA_FULL) != 0 &&
					tdev->job.errors < 10){
		err("GAME OVER: %d FIFSTA_DESC_OVERFLOW: 0x%08x",
		    tdev->idx, desc_sta);
		if (++tdev->job.errors == 10){
			err("too many errors, turning reporting off ..");
		}
	}
#endif
}

int job_is_go(struct JOB* job)
{
	return !job->please_stop && job->buffers_queued < job->buffers_demand;
}
static int afs_isr_work(void *arg)
{
	struct AFHBA_DEV* adev = (struct AFHBA_DEV*)arg;
	struct AFHBA_STREAM_DEV* sdev = adev->stream_dev;
	struct JOB* job = &sdev->job;

	int loop_count = 0;
/* this is going to be the top RT process */
	struct sched_param param = { .sched_priority = 10 };
	int please_check_fifo = 0;

	sched_setscheduler(current, SCHED_FIFO, &param);
	afs_comms_init(adev);

	for ( ; 1; ++loop_count){
		int timeout = wait_event_interruptible_timeout(
			sdev->work.w_waitq,
			test_and_clear_bit(WORK_REQUEST, &sdev->work.w_to_do) ||
			kthread_should_stop(),
			WORK_TO) == 0;

		if (!timeout || loop_count%10 == 0){
			dev_dbg(pdev(adev), "TIMEOUT? %d queue_free_buffers() ? %d",
			    timeout, job_is_go(job)  );
		}

		if (aurora_monitor && !afs_comms_init(adev)){
			if (job_is_go(job)){
				dev_warn(pdev(adev), "job is go but aurora is down");
			}
			continue;
		}

	        if (job_is_go(job)){
	        	queue_free_buffers(adev);
			if (!afs_push_dma_started(adev)){
				afs_start_dma(adev);
			}
		}


		if (job->buffers_demand > 0 ){
			if (queue_full_buffers(adev) > 0){
				wake_up_interruptible(&sdev->return_waitq);
			}
		}

		spin_lock(&sdev->job_lock);
		switch(job->please_stop){
		case PS_STOP_DONE:
			break;
		case PS_PLEASE_STOP:
			afs_stop_dma(adev);
			job->please_stop = PS_STOP_DONE;
			break;
		default:
			if (afs_push_dma_started(adev)){
				please_check_fifo = 1;
			}
		}
		spin_unlock(&sdev->job_lock);

		if (please_check_fifo){
			check_fifo_status(adev);
			please_check_fifo = 0;
		}
	}

	afs_stop_dma(adev);
	return 0;
}


static void startWork(struct AFHBA_DEV *adev)
{
	adev->stream_dev->work.w_task = kthread_run(afs_isr_work, adev, adev->name);
	adev->stream_dev->work.mon_task = kthread_run(as_mon, adev, adev->mon_name);
}

ssize_t afs_histo_read(
	struct file *file, char *buf, size_t count, loff_t *f_pos)
{
	unsigned *the_histo = PD(file)->private;
	int maxentries = PD(file)->minor == MINOR_DATA_FIFO ?
		RTDMAC_DATA_FIFO_CNT: RTDMAC_DESC_FIFO_CNT;
	unsigned cursor = *f_pos;	/* f_pos counts in entries */
	int rc;

	if (cursor >= maxentries){
		return 0;
	}else{
		int headroom = (maxentries - cursor) * sizeof(unsigned);
		if (count > headroom){
			count = headroom;
		}
	}
	rc = copy_to_user(buf, the_histo+cursor, count);
	if (rc){
		return -1;
	}

	*f_pos += count/sizeof(unsigned);
	return count;
}

static struct file_operations afs_fops_histo = {
	.read = afs_histo_read,
	.release = afhba_release
};


static int rtm_t_start_stream(struct AFHBA_DEV *adev, unsigned buffers_demand)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	struct JOB *job = &sdev->job;

	dev_dbg(pdev(adev), "01");
	afs_dma_reset(adev);
	memset(job, 0, sizeof(struct JOB));

	job->buffers_demand = buffers_demand;
	if (unlikely(list_empty_careful(&sdev->bp_empties.list))){
		dev_err(pdev(adev), "no free buffers");
		return -ERESTARTSYS;
	}

	spin_lock(&sdev->job_lock);
	job->please_stop = PS_OFF;
	spin_unlock(&sdev->job_lock);
	set_bit(WORK_REQUEST, &sdev->work.w_to_do);
	wake_up_interruptible(&sdev->work.w_waitq);
	dev_dbg(pdev(adev), "99");
	return 0;
}

int afs_histo_open(struct inode *inode, struct file *file, unsigned *histo)
{
	file->f_op = &afs_fops_histo;
	PD(file)->private = histo;
	return 0;
}

int afs_reset_buffers(struct AFHBA_DEV *adev)
/* handle with care! */
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	struct HostBuffer *hb = sdev->hbx;
	int ii;

	if (mutex_lock_interruptible(&sdev->list_mutex)){
		return -1;
	}
        INIT_LIST_HEAD(&sdev->bp_empties.list);
	INIT_LIST_HEAD(&sdev->bp_filling.list);
	INIT_LIST_HEAD(&sdev->bp_full.list);



	for (ii = 0; ii < nbuffers; ++ii, ++sdev->nbuffers, ++hb){
		hb->bstate = BS_EMPTY;
		list_add_tail(&hb->list, &sdev->bp_empties.list);
	}

	sdev->init_descriptors(sdev);
	memset(sdev->data_fifo_histo, 0, DATA_FIFO_SZ);
	memset(sdev->desc_fifo_histo, 0, DESC_FIFO_SZ);

	mutex_unlock(&sdev->list_mutex);
	return 0;
}

int afs_dma_open(struct inode *inode, struct file *file)
{
	struct AFHBA_DEV *adev = PD(file)->dev;
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;

	int ii;

	dev_dbg(pdev(adev), "45: DMA open");

	if (afs_reset_buffers(adev)){
		return -ERESTARTSYS;
	}
	/** @@todo protect with lock ? */
	if (sdev->pid == 0){
		sdev->pid = current->pid;
	}

	if (sdev->pid != current->pid){
		return -EBUSY;
	}

	if (sdev->buffer_len == 0) sdev->buffer_len = BUFFER_LEN;
	sdev->req_len = min(sdev->buffer_len, BUFFER_LEN);

	for (ii = 0; ii != nbuffers; ++ii){
		sdev->hbx[ii].req_len = sdev->req_len;
	}

	if ((file->f_flags & O_NONBLOCK) != 0){
		file->f_op = &afs_fops_dma_poll;
	}else{
		file->f_op = &afs_fops_dma;
	}

	dev_dbg(pdev(adev), "99");
	return 0;
}

int afs_dma_release(struct inode *inode, struct file *file)
{
	struct AFHBA_DEV *adev = PD(file)->dev;
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;

	struct HostBuffer *hb;
	struct HostBuffer *tmp;

	dev_dbg(pdev(adev), "01 %s %d %p<-%p->%p",
		adev->name, PD(file)->minor,
		PD(file)->my_buffers.prev,
		&PD(file)->my_buffers,
		PD(file)->my_buffers.next);

	if (mutex_lock_interruptible(&sdev->list_mutex)){
		return -ERESTARTSYS;
	}
	list_for_each_entry_safe(hb, tmp, &PD(file)->my_buffers, list){
		dev_dbg(pdev(adev), "returning %d", hb->ibuf);
		return_empty(adev, hb);
	}

	mutex_unlock(&sdev->list_mutex);

	dev_dbg(pdev(adev), "90");
	sdev->job.please_stop = PS_PLEASE_STOP;
	sdev->job.buffers_demand = 0;

	if (sdev->onStop){
		sdev->onStop(adev);
		sdev->onStop = 0;
	}
	sdev->pid = 0;
	return afhba_release(inode, file);
}

ssize_t afs_dma_read(
	struct file *file, char __user *buf, size_t count, loff_t *f_pos)
/* returns when buffer[s] available
 * data is buffer index as array of unsigned
 * return len is sizeof(array)
 */
{
	struct AFHBA_DEV *adev = PD(file)->dev;
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	struct JOB *job = &sdev->job;
	int rc;

	dev_dbg(pdev(adev), "01 ps %u count %ld demand %d received %d waiting %d",
	    (unsigned)*f_pos,	(long)count,
		job->buffers_demand, job->buffers_received,
		!list_empty(&sdev->bp_full.list));

	if (job->buffers_received >= job->buffers_demand &&
		list_empty(&sdev->bp_full.list)	){
		dev_dbg(pdev(adev), "job done");
		return 0;
	}

	if (*f_pos == 0){
		rc = wait_event_interruptible(
			sdev->return_waitq, !list_empty(&sdev->bp_full.list));
	}else{
		rc = wait_event_interruptible_timeout(
			sdev->return_waitq,
			!list_empty(&sdev->bp_full.list), RX_TO);
	}

	dev_dbg(pdev(adev), "done waiting, rc %d", rc);

	if (rc < 0){
		dev_dbg(pdev(adev), "RESTART");
		return -ERESTARTSYS;
	}else if (mutex_lock_interruptible(&sdev->list_mutex)){
		return -ERESTARTSYS;
	}else{
		struct HostBuffer *hb;
		struct HostBuffer *tmp;
		int nbytes = 0;

		list_for_each_entry_safe(hb, tmp, &sdev->bp_full.list, list){
			if (nbytes+sizeof(int) > count){
				dev_dbg(pdev(adev), "quit nbytes %d count %lu",
				    nbytes, (long)count);
				break;
			}

			if (copy_to_user(buf+nbytes, &hb->ibuf, sizeof(int))){
				rc = -EFAULT;
				goto read99;
			}
			dev_dbg(pdev(adev), "add my_buffers %d", hb->ibuf);

			list_move_tail(&hb->list, &PD(file)->my_buffers);
			hb->bstate = BS_FULL_APP;
			nbytes += sizeof(int);
		}

		if (rc == 0 && nbytes == 0){
			dev_dbg(pdev(adev), "TIMEOUT");
			rc = -ETIMEDOUT;
		}else{
			*f_pos += nbytes;
			dev_dbg(pdev(adev), "return %d", nbytes);
			rc = nbytes;
		}
	}
read99:
	mutex_unlock(&sdev->list_mutex);
	return rc;
}

ssize_t afs_dma_read_poll(
	struct file *file, char __user *buf, size_t count, loff_t *f_pos)
/* returns when buffer[s] available
 * data is buffer index as array of unsigned
 * return len is sizeof(array)
 */
{
	struct AFHBA_DEV *adev = PD(file)->dev;
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	struct JOB *job = &sdev->job;

	int rc = 0;
	struct HostBuffer *hb;
	struct HostBuffer *tmp;
	int nbytes = 0;

	dev_dbg(pdev(adev), "01 ps %u count %ld demand %d received %d waiting %d",
	    (unsigned)*f_pos,	(long)count,
	    job->buffers_demand, job->buffers_received,
	    !list_empty(&sdev->bp_full.list)	);

	if (job->buffers_received >= job->buffers_demand &&
	    list_empty(&sdev->bp_full.list)	){
		dev_dbg(pdev(adev), "job done");
		return 0;
	}

	if (!afs_push_dma_started(adev)){
		afs_start_dma(adev);
	}
	if (queue_full_buffers(adev)){
		list_for_each_entry_safe(hb, tmp, &sdev->bp_full.list, list){
			if (nbytes+sizeof(int) > count){
				dev_dbg(pdev(adev), "quit nbytes %d count %lu",
				    nbytes, (long)count);
				break;
			}

			if (copy_to_user(buf+nbytes, &hb->ibuf, sizeof(int))){
				rc = -EFAULT;
				goto read99;
			}
			dev_dbg(pdev(adev), "add my_buffers %d", hb->ibuf);

			list_move_tail(&hb->list, &PD(file)->my_buffers);
			hb->bstate = BS_FULL_APP;
			nbytes += sizeof(int);
		}

		if (rc == 0 && nbytes == 0){
			dev_dbg(pdev(adev), "TIMEOUT");
			rc = -ETIMEDOUT;
		}else{
			*f_pos += nbytes;
			dev_dbg(pdev(adev), "return %d", nbytes);
			rc = nbytes;
		}
	}
read99:
	return rc;
}


ssize_t afs_dma_write(
	struct file *file, const char *buf, size_t count, loff_t *f_pos)
/* write completed data.
 * data is array of full buffer id's
 * id's are removed from full and placed onto empty.
 */
{
	struct AFHBA_DEV *adev = PD(file)->dev;
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;

	int nbytes = 0;
	int rc = 0;

	dev_dbg(pdev(adev), "pos %u count %lu", (unsigned)*f_pos, (long)count);

	if (mutex_lock_interruptible(&sdev->list_mutex)){
		return -ERESTARTSYS;
	}
	while (nbytes+sizeof(int) <= count){
		int id;

		if (copy_from_user(&id, buf+nbytes, sizeof(int))){
			return -EFAULT;
		}
		dev_dbg(pdev(adev), "[%u] recycle buffer %d",
				(unsigned)(nbytes/sizeof(int)), id);

		if (id < 0){
			dev_err(pdev(adev), "ID < 0");
			rc = -100;
			goto write99;
		}else if (id >= nbuffers){
			dev_err(pdev(adev), "ID > NBUFFERS");
			rc = -101;
			goto write99;
		}else if (sdev->hbx[id].bstate != BS_FULL_APP){
			dev_err(pdev(adev), "STATE != BS_FULL_APP %d",
					sdev->hbx[id].bstate);
			rc = -102;
			goto write99;
		}else{
			struct HostBuffer *hb;

			rc = -1;

			list_for_each_entry(
					hb, &PD(file)->my_buffers, list){

				dev_dbg(pdev(adev), "listing %d", hb->ibuf);
				assert(hb != 0);
				assert(hb->ibuf >= 0 && hb->ibuf < nbuffers);
				if (hb->ibuf == id){
					return_empty(adev, hb);
					nbytes += sizeof(int);
					rc = 0;
					break;
				}
			}
			if (rc == -1){
				dev_err(pdev(adev), "ATTEMPT TO RET BUFFER NOT MINE");
				goto write99;
			}
		}
	}

	*f_pos += nbytes;
	rc = nbytes;

write99:
	mutex_unlock(&sdev->list_mutex);
	dev_dbg(pdev(adev), "99 return %d", rc);
	return rc;
}

long  afs_dma_ioctl(struct file *file,
                        unsigned int cmd, unsigned long arg)
{
	struct AFHBA_DEV *adev = PD(file)->dev;
	void* varg = (void*)arg;


	switch(cmd){
	case RTM_T_START_STREAM:
		return rtm_t_start_stream(adev, transfer_buffers);
	case RTM_T_START_STREAM_MAX: {
		u32 my_transfer_buffers;
		COPY_FROM_USER(&my_transfer_buffers, varg, sizeof(u32));
		return rtm_t_start_stream(adev, my_transfer_buffers);
	}
	/*
	case RTM_T_START_LLC: {
		struct LLC_DEF llc_def;
		COPY_FROM_USER(&llc_def, varg, sizeof(struct LLC_DEF));
		rtm_t_start_llc(adev, &llc_def);
		return 0;
	} case RTM_T_START_AOLLC: {
		struct AO_LLC_DEF ao_llc_def;
		COPY_FROM_USER(&ao_llc_def, varg, sizeof(struct AO_LLC_DEF));
		rtm_t_start_aollc(adev, &ao_llc_def);
		return 0;

	} */ default:
		return -ENOTTY;
	}

}

int afs_mmap_host(struct file* file, struct vm_area_struct* vma)
/**
 * mmap the host buffer.
 */
{
	struct AFHBA_DEV *adev = PD(file)->dev;
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;

	int ibuf = PD(vma->vm_file)->minor&NBUFFERS_MASK;
	struct HostBuffer *hb = &sdev->hbx[ibuf];
	unsigned long vsize = vma->vm_end - vma->vm_start;
	unsigned long psize = hb->len;
	unsigned pfn = hb->pa >> PAGE_SHIFT;

	dev_dbg(pdev(adev), "%c vsize %lu psize %lu %s",
		'D', vsize, psize, vsize>psize? "EINVAL": "OK");

	if (vsize > psize){
		return -EINVAL;                   /* request too big */
	}
	if (remap_pfn_range(
		vma, vma->vm_start, pfn, vsize, vma->vm_page_prot)){
		return -EAGAIN;
	}else{
		return 0;
	}
}

static struct file_operations afs_fops_dma = {
	.open = afs_dma_open,
	.release = afs_dma_release,
	.read = afs_dma_read,
	.write = afs_dma_write,
	.unlocked_ioctl = afs_dma_ioctl
};

static struct file_operations afs_fops_dma_poll = {
	.open = afs_dma_open,
	.release = afs_dma_release,
	.read = afs_dma_read_poll,
	.write = afs_dma_write,
	.unlocked_ioctl = afs_dma_ioctl
};


int afs_open(struct inode *inode, struct file *file)
{
	struct AFHBA_DEV *adev = DEV(file);

	dev_dbg(pdev(adev), "01");
	if (adev == 0){
		return -ENODEV;
	}
	dev_dbg(pdev(adev), "33: minor %d", PD(file)->minor);

	switch((PD(file)->minor)){
	case MINOR_DMAREAD:
		return afs_dma_open(inode, file);
	case MINOR_DATA_FIFO:
		return afs_histo_open(
				inode, file, adev->stream_dev->data_fifo_histo);
	case MINOR_DESC_FIFO:
		return afs_histo_open(
				inode, file, adev->stream_dev->desc_fifo_histo);
	default:
		if (PD(file)->minor <= NBUFFERS_MASK){
			return 0;
		}else{
			dev_err(pdev(adev),"99 adev %p name %s", adev, adev->name);
			return -ENODEV;
		}
	}

}

static struct file_operations afs_fops = {
	.open = afs_open,
	.mmap = afs_mmap_host,
	.release = afhba_release,
};

int afhba_stream_drv_init(struct AFHBA_DEV* adev)
{
	adev->stream_dev = kzalloc(sizeof(struct AFHBA_STREAM_DEV), GFP_KERNEL);

	afs_init_buffers(adev);
	hook_interrupts(adev);
	startWork(adev);
	adev->stream_fops = &afs_fops;
	afs_init_procfs(adev);
	return 0;
}
int afhba_stream_drv_del(struct AFHBA_DEV* adev)
{
	return 0;
}
