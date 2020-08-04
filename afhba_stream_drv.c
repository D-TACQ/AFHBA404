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

#include <linux/poll.h>
#include <linux/version.h>


#define REVID	"R1070"

#define DEF_BUFFER_LEN 0x100000

int RX_TO = 1*HZ;
module_param(RX_TO, int, 0644);
MODULE_PARM_DESC(RX_TO, "RX timeout (jiffies) [0.1Hz]");

int WORK_TO = HZ/100;
module_param(WORK_TO, int, 0644);
MODULE_PARM_DESC(WORK_TO,
	"WORK timeout (jiffies) [50Hz] - decrease for hi fifo stat poll rate");

int amon_jiffies = HZ/3;
module_param(amon_jiffies, int, 0644);
MODULE_PARM_DESC(amon_jiffies, "aurora monitor poll rate");

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

int buffer_len = DEF_BUFFER_LEN;
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

int eot_interrupt = 0;
module_param(eot_interrupt, int, 0644);
MODULE_PARM_DESC(eot_interrupt, "1: interrupt every, 0: interrupt none, N: interrupt interval");

int aurora_status_read_count = 0;
module_param(aurora_status_read_count, int, 0644);
MODULE_PARM_DESC(aurora_status_read_count, "number of amon polls");

int dma_descriptor_ram = 1;
module_param(dma_descriptor_ram, int, 0644);
MODULE_PARM_DESC(dma_descriptor_ram, "descriptors in RAM not FIFO >1 :: validate on load");

int assume_stuck_buffers_are_ok = 1;
module_param(assume_stuck_buffers_are_ok, int, 0644);
MODULE_PARM_DESC(assume_stuck_buffers_are_ok, "assume that stuck buffers are ok to release into the wild");

int max_dma_load_retry = 2;
module_param(max_dma_load_retry, int, 0644);
MODULE_PARM_DESC(max_dma_load_retry, "number of times to retry loading descriptors. HACK: once should be enough");

static struct file_operations afs_fops_dma;
static struct file_operations afs_fops_dma_poll;

int max_empty_backlog_check = 2;
module_param(max_empty_backlog_check, int , 0644);
MODULE_PARM_DESC(max_empty_backlog_check, "set to one to look only at top of deck, set to two to check skips");



int use_llc_multi = 0;
module_param(use_llc_multi, int, 0644);
MODULE_PARM_DESC(use_llc_multi, "use LLC for multi descriptor transfer");

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
		int len = sdev->hbx[ii].req_len;

		if (len == 0) len = sdev->buffer_len;
		descr &= ~AFDMAC_DESC_LEN_MASK;
		descr |= getAFDMAC_Order(len)<< AFDMAC_DESC_LEN_SHL;
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

	sdev->push_ram_cursor = sdev->pull_ram_cursor = 0;
}





#define COPY_FROM_USER(to, from, len) \
	if (copy_from_user(to, from, len)) { return -EFAULT; }

#define COPY_TO_USER(to, from, len) \
	if (copy_to_user(to, from, len)) { return -EFAULT; }

static void write_descr(struct AFHBA_DEV *adev, unsigned offset, int idesc)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	u32 descr = sdev->hbx[idesc].descr;

	if (sdev->job.buffers_queued < 5){
		dev_info(pdev(adev), "write_descr(%d) [%d] offset:%04x = %08x",
				sdev->job.buffers_queued, idesc, offset, descr);
	}
	DEV_DBG(pdev(adev), "ibuf %d offset:%04x = %08x", idesc, offset, descr);
	writel(descr, adev->remote+offset);
	/* force write posting through to avoid message backup */
	DMA_DESC_FIFSTA_RD(adev);
}

static int _write_ram_descr(struct AFHBA_DEV *adev, unsigned offset, int idesc, int *cursor)
/* returns "address" of entry or ERR */
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	u32 descr = sdev->hbx[idesc].descr;
	u32 descrr;
	int cr = *cursor;

	if (cr < sdev->nbuffers){
		int addr = cr;
		void* va = adev->remote + offset + cr*sizeof(unsigned);

		DEV_DBG(pdev(adev), "_write_ram_descr() ibuf %d offset:%04x = %08x cursor:%d",
					idesc, offset, descr, cr);
		writel(descr, va);
		descrr = readl(va);
		if (descr != descrr){
			dev_err(pdev(adev), "descriptor [%4d] wrote 0x%08x read 0x%08x",
					cr, descr, descrr);
		}
		*cursor = cr + 1;
		return addr;
	}else{
		return -1;
	}
}

static int validate_dma_descriptor_ram(
		struct AFHBA_DEV *adev, unsigned offset, unsigned max_id, int phase)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	u32 descr;
	u32 id;
	int errors = 0;

	for (id = 0; id < max_id; ++id){
		descr = readl(adev->remote+offset+id*sizeof(unsigned));
		if (descr != sdev->hbx[id].descr){
			dev_err(pdev(adev), "%s phase:%d descriptor mismatch at [%d] w:%08x r:%08x",
					"validate_dma_descriptor_ram", phase, id, sdev->hbx[id].descr, descr);
			errors += 1;
		}
		dev_dbg(pdev(adev), "%s phase:%d descriptor at [%d] w:%08x r:%08x",
				"validate_dma_descriptor_ram", phase, id, sdev->hbx[id].descr, descr);
	}
	if (errors){
		dev_err(pdev(adev), "%s descriptor errors %d out of %d",
				"validate_dma_descriptor_ram", errors, id);
		return errors;
	}else{
		dev_info(pdev(adev), "%s %d descriptors PASS",
				"validate_dma_descriptor_ram", id);
		return 0;
	}
}
static void write_ram_descr(struct AFHBA_DEV *adev, unsigned offset, int idesc)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	int addr;

	if (offset == DMA_PUSH_DESC_RAM){
		addr = _write_ram_descr(adev, offset, idesc, &sdev->push_ram_cursor);
		if (addr != -1){
			_afs_write_dmareg(adev, DMA_PUSH_DESC_LEN, addr);
		}
	}else{
		dev_warn(pdev(adev), "write_ram_descr  valid PUSH only");
		addr = _write_ram_descr(adev, offset, idesc, &sdev->pull_ram_cursor);
		if (addr != -1){
			_afs_write_dmareg(adev, DMA_PULL_DESC_LEN, addr);
		}
	}
}
u32 _afs_read_zynqreg(struct AFHBA_DEV *adev, int regoff)
{
	u32* dma_regs = (u32*)(adev->remote + ZYNQ_BASE);
	void* va = &dma_regs[regoff];
	u32 value = readl(va);
	DEV_DBG(pdev(adev), "%04lx = %08x", va-adev->remote, value);
	return adev->stream_dev->dma_regs[regoff] = value;
}

void _afs_write_comreg(struct AFHBA_DEV *adev, int regoff, u32 value)
{
	u32* dma_regs = (u32*)(adev->mappings[adev->remote_com_bar].va + COMMON_BASE);
	void* va = &dma_regs[regoff];
	DEV_DBG(pdev(adev), "%04lx = %08x", va-adev->remote, value);
	writel(value, va);
}

void _afs_write_dmareg(struct AFHBA_DEV *adev, int regoff, u32 value)

{
	u32* dma_regs = (u32*)(adev->remote + DMA_BASE);
	void* va = &dma_regs[regoff];
	DEV_DBG(pdev(adev), "%04lx = %08x", va-adev->remote, value);
	writel(value, va);
}

u32 _afs_read_dmareg(struct AFHBA_DEV *adev, int regoff)
{
	u32* dma_regs = (u32*)(adev->remote + DMA_BASE);
	void* va = &dma_regs[regoff];
	u32 value = readl(va);
	DEV_DBG(pdev(adev), "%04lx = %08x", va-adev->remote, value);
	return adev->stream_dev->dma_regs[regoff] = value;
}

void _afs_write_pcireg(struct AFHBA_DEV *adev, int regoff, u32 value)

{
	u32* dma_regs = (u32*)(adev->remote + PCIE_BASE);
	void* va = &dma_regs[regoff];
	DEV_DBG(pdev(adev), "%04lx = %08x", va-adev->remote, value);
	DEV_DBG(pdev(adev), "%p = %08x", va, value);
	writel(value, va);
}

u32 _afs_read_pcireg(struct AFHBA_DEV *adev, int regoff)
{
	u32* dma_regs = (u32*)(adev->remote + PCIE_BASE);
	void* va = &dma_regs[regoff];
	u32 value = readl(va);
	DEV_DBG(pdev(adev), "%04lx = %08x", va-adev->remote, value);
	return adev->stream_dev->dma_regs[regoff] = value;
}
static void afs_load_push_descriptor(struct AFHBA_DEV *adev, int idesc)
{
	if (dma_descriptor_ram){
		if  (!adev->stream_dev->job.dma_started){
			write_ram_descr(adev, DMA_PUSH_DESC_RAM, idesc);
		}
		/* else .. NO ACTION, descriptor already loaded in RAM */
	}else{
		write_descr(adev, DMA_PUSH_DESC_FIFO, idesc);
	}
}

void afs_load_pull_descriptor(struct AFHBA_DEV *adev, int idesc)
{
	if (dma_descriptor_ram){
		if  (!adev->stream_dev->job.dma_started){
			write_ram_descr(adev, DMA_PULL_DESC_RAM, idesc);
		}
		/* else .. NO ACTION, descriptor already loaded in RAM */
	}else{
		write_descr(adev, DMA_PUSH_DESC_FIFO, idesc);
	}
}

static void afs_init_dma_clr(struct AFHBA_DEV *adev)
{
	DMA_CTRL_RD(adev);
	afs_dma_reset(adev, DMA_BOTH_SEL);
}

static void afs_configure_streaming_dma(
		struct AFHBA_DEV *adev, enum DMA_SEL dma_sel)
{
	u32 dma_ctrl = DMA_CTRL_RD(adev);
	u32 check;
	dma_ctrl &= ~dma_pp(dma_sel, DMA_CTRL_LOW_LAT|DMA_CTRL_RECYCLE);
	if (dma_descriptor_ram){
		dma_ctrl |= DMA_CTRL_RAM;
	}else{
		dma_ctrl &= ~DMA_CTRL_RAM;
	}
	DMA_CTRL_WR(adev, dma_ctrl);
	check = DMA_CTRL_RD(adev);
	if (check != dma_ctrl){
		dev_err(pdev(adev), "%s setting DMA_CTRL w:%08x r:%08x",
				"afs_configure_streaming_dma", dma_ctrl, check);
	}
}

static void afs_dma_set_recycle(
		struct AFHBA_DEV *adev, enum DMA_SEL dma_sel, int enable)
{
	u32 dma_ctrl = DMA_CTRL_RD(adev);

	dma_ctrl &= ~dma_pp(dma_sel, DMA_CTRL_RECYCLE);
	DMA_CTRL_WR(adev, dma_ctrl);
}

static void afs_load_llc_single_dma(
	struct AFHBA_DEV *adev, enum DMA_SEL dma_sel, u32 pa, unsigned len)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	u32 dma_ctrl = DMA_CTRL_RD(adev);
	u32 len64 = ((len/64-1) + (len%64!=0));
	u32 offset = DMA_DIR_DESC_FIFO(dma_sel);
	u32 dma_desc;

	dev_dbg(pdev(adev), "afs_load_llc_single_dma %s 0x%08x %d",
			sDMA_SEL(dma_sel), pa, len);

	len64 <<= AFDMAC_DESC_LEN_SHL;
	len64 &= AFDMAC_DESC_LEN_MASK;

	dma_desc = pa&AFDMAC_DESC_ADDR_MASK;
	dma_desc |= len64;
	dma_desc |= sdev->shot&AFDMAC_DESC_ID_MASK;

	dma_ctrl &= ~dma_pp(dma_sel, DMA_CTRL_RAM);
	dma_ctrl |= dma_pp(dma_sel, DMA_CTRL_LOW_LAT|DMA_CTRL_RECYCLE);

	dev_dbg(pdev(adev),
		"afs_load_llc_single_dma len64:%08x dma_desc:%08x dma_ctrl:%08x",
		len64, dma_desc, dma_ctrl);

	_afs_write_dmareg(adev, DMA_DIR_DESC_LEN(dma_sel), 0);
	DMA_CTRL_WR(adev, dma_ctrl);
	writel(dma_desc, adev->remote+offset);
	afs_start_dma(adev, dma_sel);
}

static void afs_load_dram_descriptors(
	struct AFHBA_DEV *adev, enum DMA_SEL dma_sel,
	struct XLLC_DEF* buffers,
	int nbufs
)
/**< load nbufs buffers to DRAM in regular mode */
{
	u32 dma_ctrl = DMA_CTRL_RD(adev);
	u32 offset = DMA_DIR_DESC_RAM(dma_sel);
	int idesc;

	dma_ctrl &= ~dma_pp(dma_sel, DMA_CTRL_EN|DMA_CTRL_LOW_LAT|DMA_CTRL_RECYCLE);
	dma_ctrl |= dma_pp(dma_sel, DMA_CTRL_RAM);
	DMA_CTRL_WR(adev, dma_ctrl);

	for (idesc = 0; idesc < nbufs; ++idesc, offset += 4){
		struct XLLC_DEF* bd = &buffers[idesc];
		u32 dma_desc = bd->pa
		    | getAFDMAC_Order(bd->len)<< AFDMAC_DESC_LEN_SHL
		    | idesc;
		dev_dbg(pdev(adev),"%s() [%d] 0x%08x",
				"afs_load_dram_descriptors", idesc, dma_desc);
		writel(dma_desc, adev->remote+offset);
	}

	dma_ctrl |= dma_pp(dma_sel, DMA_CTRL_EN);
	_afs_write_dmareg(adev, DMA_DIR_DESC_LEN(dma_sel), nbufs-1);
	DMA_CTRL_WR(adev, dma_ctrl);
	afs_start_dma(adev, dma_sel);
}


#define LL_MAX_CNT	16
#define LL_BLOCK	64
#define LL_NB(cnt)	((cnt)-1)
#define LL_MAX_LEN	(LL_MAX_CNT*LL_BLOCK)

static void afs_load_dram_descriptors_ll(
	struct AFHBA_DEV *adev, enum DMA_SEL dma_sel,
	struct XLLC_DEF* buffers,
	int nbufs
)
/**< load nbufs buffers to DRAM in regular mode */
{
	u32 dma_ctrl = DMA_CTRL_RD(adev);
	int cursor = 0;
	int ib = 0;

	for ( ; ib < nbufs; ++ib){
		struct XLLC_DEF* bd = &buffers[ib];
		int idb = 0;
		for ( ; idb*LL_MAX_LEN < bd->len; ++idb, ++cursor){
			u32 dma_desc;
			int cnt;
			int residue = bd->len - idb*LL_MAX_LEN;
			unsigned reg_off = DMA_DIR_DESC_RAM(dma_sel) + cursor*4;

			residue = min(residue, LL_MAX_LEN);
			cnt = 1 + residue/LL_BLOCK;

			dma_desc = (bd->pa + idb*LL_MAX_LEN)
					| LL_NB(cnt)<< AFDMAC_DESC_LEN_SHL
					| (idb&0x0f);

			dev_dbg(pdev(adev), "%s() [%d] [%04x] %p := 0x%08x",
				__FUNCTION__, idb, reg_off, adev->remote+reg_off, dma_desc);

			writel(dma_desc, adev->remote+reg_off);
		}
	}

	_afs_write_dmareg(adev, DMA_DIR_DESC_LEN(dma_sel), cursor-1);

	dma_ctrl &= ~dma_pp(dma_sel, DMA_CTRL_RECYCLE);
	dma_ctrl |= dma_pp(dma_sel, DMA_CTRL_LOW_LAT|DMA_CTRL_RAM);

	DMA_CTRL_WR(adev, dma_ctrl);
	afs_start_dma(adev, dma_sel);
}

static int _afs_dma_started(struct AFHBA_DEV *adev, int shl)
{
	u32 ctrl = DMA_CTRL_RD(adev);
	ctrl >>= shl;
	return (ctrl&DMA_CTRL_EN) != 0;
}


static inline int afs_dma_started(struct AFHBA_DEV *adev, enum DMA_SEL dma_sel)
{
	return _afs_dma_started(adev, dma_pp(dma_sel, DMA_CTRL_PUSH_SHL));
}



static int afs_aurora_lane_up(struct AFHBA_DEV *adev)
{

	u32 stat;

	stat = afhba_read_reg(adev, ASR(adev->ACR));

	++aurora_status_read_count;
	++adev->aurora_status_read_count;
	if (adev->aurora_status_read_count == 1){
		dev_info(pdev(adev), "afs_aurora_lane_up %c status 0x%08x",
				aurora_id(adev), stat);
	}

	return (stat & AFHBA_AURORA_STAT_LANE_UP) != 0;
}

/* super-paranoid remote access */
void __afs_dma_reset(struct AFHBA_DEV *adev, u32 dma_sel)
{
	if (afs_comms_ready(adev)){
		DMA_CTRL_CLR(adev, dma_pp(dma_sel, DMA_CTRL_EN));
		DMA_CTRL_SET(adev, dma_pp(dma_sel, DMA_CTRL_FIFO_RST));
		DMA_CTRL_CLR(adev, dma_pp(dma_sel, DMA_CTRL_FIFO_RST));
	}
}
void __afs_stop_dma(struct AFHBA_DEV *adev, u32 dma_sel)
{
	if (afs_comms_ready(adev)){
		DMA_CTRL_CLR(adev, dma_pp(dma_sel, DMA_CTRL_EN));
	}
}

void __afs_start_dma(struct AFHBA_DEV *adev, u32 dma_sel)
{
	if (afs_comms_ready(adev)){
		u32 dmacr;
		DMA_CTRL_SET(adev, dma_pp(dma_sel, DMA_CTRL_EN));
		dmacr = DMA_CTRL_RD(adev);
		if ((dmacr&dma_pp(dma_sel, DMA_CTRL_EN)) == 0){
			dev_err(pdev(adev), "DMA_CTRL_EN NOT SET wanted:%08x got:%08x",
					dma_pp(dma_sel, DMA_CTRL_EN), dmacr);
		}
	}
}

static int afs_aurora_errors(struct AFHBA_DEV *adev)
{
	u32 stat = afhba_read_reg(adev, ASR(adev->ACR));

	if ((stat&AFHBA_AURORA_STAT_ERR) != 0){
		u32 ctrl = afhba_read_reg(adev, adev->ACR);
		afhba_write_reg(adev, adev->ACR, ctrl|AFHBA_AURORA_CTRL_CLR);
		if (++adev->aurora_error_count==1){
			dev_info(pdev(adev),
			"aurora%c initial s:0x%08x m:0x%08x e:0x%08x",
			adev->sfp-SFP_A + 'A',
			stat, AFHBA_AURORA_STAT_ERR, stat&AFHBA_AURORA_STAT_ERR);
		}else{
			dev_warn(pdev(adev),
			"aurora%c error: [%d] s:0x%08x m:0x%08x e:0x%08x",
			adev->sfp-SFP_A + 'A',
			adev->aurora_error_count,
			stat, AFHBA_AURORA_STAT_ERR, stat&AFHBA_AURORA_STAT_ERR);
		}
		stat = afhba_read_reg(adev, ASR(adev->ACR));
		if ((stat&AFHBA_AURORA_STAT_ERR) != 0){
			dev_err(pdev(adev),
			"aurora%c error: [%d] s:0x%08x m:0x%08x e:0x%08x NOT CLEARED",
			adev->sfp-SFP_A + 'A',
			adev->aurora_error_count,
			stat, AFHBA_AURORA_STAT_ERR, stat&AFHBA_AURORA_STAT_ERR);
			msleep(1000);
			return -1;
		}else{
			return 1;
		}
	}else{
		return 0;
	}
}

static void _afs_pcie_mirror_init(struct AFHBA_DEV *adev)
{
	int ireg;

	for (ireg = PCIE_CNTRL; ireg <= PCIE_BUFFER_CTRL; ++ireg){
		u32 local = afhba_read_reg(adev, ireg*sizeof(u32));
		switch(ireg){
		case PCIE_CONF:
			local |= adev->sfp << PCIE_CONF_AF_PORT_SHL;
		}
		PCI_REG_WRITE(adev, ireg, local);
	}
}

#define MSLEEP_TO 10

static int is_valid_z_ident(unsigned z_ident, char buf[], int maxbuf)
{
	if ((z_ident&0x21060000) == 0x21060000){
		snprintf(buf, maxbuf, "acq2106_%03d.comms%X",
				z_ident&0x0ffff, (z_ident&0x00f00000)>>20);
		return 1;
	}else if ((z_ident&0xe4330000) == 0xe4330000){
        snprintf(buf, maxbuf, "kmcu_%03d.comms%x",
                        z_ident&0x0ffff, (z_ident&0x00f00000)>>20);
        return 1;
	}else if ((z_ident&0x43000000) == 0x43000000){
        snprintf(buf, maxbuf, "kmcu_%03d.comms%x",
                        z_ident&0x0ffff, 0);
        return 1;
	}else{
		return 0;
	}
}

static int _afs_check_read(struct AFHBA_DEV *adev)
{
	struct AFHBA_STREAM_DEV* sdev = adev->stream_dev;
	unsigned z_ident1 = _afs_read_zynqreg(adev, Z_IDENT);
	unsigned z_ident2 = _afs_read_zynqreg(adev, Z_IDENT);

	if (z_ident2 == 0xffffffff || (z_ident2&~0x0ffff) == 0xdead0000){
		dev_err(pdev(adev), "ERROR reading Z_IDENT %08x, please reboot now", z_ident2);
		return -1;
	}else{
		char buf[80];
		int valid_id = is_valid_z_ident(z_ident2, buf, 80);
#define ZI_FMT	"[%d] Z_IDENT 1:0x%08x 2:0x%08x %s"

		if (valid_id){
			if (sdev->zi_report != ZI_GOOD){
				dev_info(pdev(adev), ZI_FMT,
						adev->idx, z_ident1, z_ident2,
						buf);
				sdev->zi_report = ZI_GOOD;
			}
			return 0;
		}else{
			if (sdev->zi_report != ZI_BAD){
				dev_info(pdev(adev), ZI_FMT,
					adev->idx, z_ident1, z_ident2,
					"ID NOT VALID");
				sdev->zi_report = ZI_BAD;
			}
			return 1;
		}
	}
}

static int _afs_comms_init(struct AFHBA_DEV *adev)
{
	struct AFHBA_STREAM_DEV* sdev = adev->stream_dev;
	int to = 0;

	afhba_write_reg(adev, adev->ACR, AFHBA_AURORA_CTRL_ENA);

	while(!afs_aurora_lane_up(adev)){
		msleep(to += MSLEEP_TO);
		if (to > aurora_to_ms){
			return 0;
		}
	}
	/* ... now make _sure_ it's up .. */
	msleep(MSLEEP_TO);
	afs_init_dma_clr(adev);
	_afs_pcie_mirror_init(adev);

	return sdev->comms_init_done = _afs_check_read(adev) == 0;
}

int afs_comms_init(struct AFHBA_DEV *adev)
{
	struct AFHBA_STREAM_DEV* sdev = adev->stream_dev;

	if (afs_aurora_lane_up(adev)){
		if (!adev->link_up){
			dev_info(pdev(adev), "aurora%c link up!", aurora_id(adev));
			adev->link_up = true;
		}
		if (!sdev->comms_init_done){
			sdev->comms_init_done = _afs_comms_init(adev);
		}
		afs_aurora_errors(adev);
		return sdev->comms_init_done;
	}else{
		if (adev->link_up){
			dev_info(pdev(adev), "aurora%c link down!", aurora_id(adev));
			adev->link_up = false;
		}
		dev_dbg(pdev(adev), "aurora lane down");
		return sdev->comms_init_done = false;
	}
}

int afs_comms_ready(struct AFHBA_DEV *adev)
{
	struct AFHBA_STREAM_DEV* sdev = adev->stream_dev;
	return adev->link_up && sdev->comms_init_done;
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

	dma_sync_single_for_device(dev, hb->pa+offset, mark_len, PCI_DMA_FROMDEVICE);
}


static int is_marked_empty(struct device *dev, struct HostBuffer *hb){
	u32 mark_len = 2 * sizeof(u32);
	u32 offset = hb->req_len - mark_len;
	u32 *pmark = (u32*)(hb->va + offset);
	int is_empty;

	dma_sync_single_for_cpu(dev, hb->pa+offset, mark_len, PCI_DMA_FROMDEVICE);

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
	if (adev->stream_dev->job.buffers_demand == 0){
		return;
	}
	if (dma_descriptor_ram){
		dev_warn(pdev(adev), "%s: buffer %02d %s",
				msg, ibuf, is_error? "ERROR": "WARNING");
	}else{
		u32 inflight_descr = DMA_PUSH_DESC_STA_RD(adev);
		struct HostBuffer*  inflight = hb_from_descr(adev, inflight_descr);
		u32 fifsta = DMA_DESC_FIFSTA_RD(adev);

		dev_warn(pdev(adev), "%s: buffer %02d  last descr:%08x [%02d] fifo:%08x %s",
				msg,
				ibuf,
				inflight_descr,
				inflight? inflight->ibuf: -1,
				fifsta,
				is_error? "ERROR": "WARNING");
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

static int _queue_full_buffer(struct AFHBA_DEV *adev, struct HostBuffer* hb, int nrx)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	struct JOB *job = &sdev->job;

	hb->esta = read_astatus2(adev);
	if (buffer_debug){
		report_inflight(adev, hb->ibuf, 0, "->FULL");
	}

	hb->bstate = BS_FULL;
	list_move_tail(&hb->list, &sdev->bp_full.list);
	job->buffers_received++;
	return nrx + 1;
}
static int queue_full_buffers(struct AFHBA_DEV *adev)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	struct HostBuffer* hb;
	struct HostBuffer* tmp;
	struct HostBuffer* first = 0;
	struct JOB *job = &sdev->job;
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
			if (ifilling >= max_empty_backlog_check){
				break; 	/* top 2 buffers empty: no action */
			}else{
				continue;  /* check skipped data. */
			}
		}else{
			if (ifilling > 1 && first && hb != first){
				if (is_marked_empty(&adev->pci_dev->dev, first)){

					if (dma_descriptor_ram && assume_stuck_buffers_are_ok){
						report_inflight(adev, first->ibuf, 0, "queuing dirty");
						nrx = _queue_full_buffer(adev, first, nrx);
					}else{
						report_stuck_buffer(adev, first->ibuf);
						return_empty(adev, first);
						if (stop_on_skipped_buffer){
							dev_warn(pdev(adev), "stop_on_skipped_buffer triggered");
							job->please_stop = PS_PLEASE_STOP;
						}
					}
					first = 0;
				}else{
					report_inflight(adev, first->ibuf, 0, "jackpot");
					nrx = _queue_full_buffer(adev, first, nrx);
					first = 0;
				}
			}

			nrx = _queue_full_buffer(adev, hb, nrx);
		}
	}

	if (nrx){
		if (ifilling > NBUFFERS){
			dev_warn(pdev(adev), "ifilling > NBUFFERS?");
			ifilling = 0;
		}
		job->catchup_histo[nrx]++;
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
	int order = getOrder(buffer_len);
	int ii;

	dev_dbg(pdev(adev), "afs_init_buffers() 01 order=%d", order);

	sdev->hbx = kzalloc(sizeof(struct HostBuffer)*nbuffers, GFP_KERNEL);
        INIT_LIST_HEAD(&sdev->bp_empties.list);
	INIT_LIST_HEAD(&sdev->bp_filling.list);
	INIT_LIST_HEAD(&sdev->bp_full.list);
	spin_lock_init(&sdev->job_lock);

	mutex_init(&sdev->list_mutex);
	mutex_lock(&sdev->list_mutex);

	sdev->buffer_len = buffer_len;
	dev_dbg(pdev(adev), "allocating %d buffers size:%d order:%d dev.dma_mask:%08llx",
			nbuffers, buffer_len, order, *adev->pci_dev->dev.dma_mask);

	for (hb = sdev->hbx, ii = 0; ii < nbuffers; ++ii, ++hb){
		void *buf = (void*)__get_free_pages(GFP_KERNEL|GFP_DMA32, order);

		if (!buf){
			dev_err(pdev(adev), "failed to allocate buffer %d", ii);
			break;
		}

		dev_dbg(pdev(adev), "buffer %2d allocated at %p, map it", ii, buf);

		hb->ibuf = ii;
		hb->pa = dma_map_single(&adev->pci_dev->dev, buf,
				buffer_len, PCI_DMA_FROMDEVICE);
		hb->va = buf;
		hb->len = buffer_len;

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

static irqreturn_t afs_cos_isr(int irq, void *data)
{
	struct AFHBA_DEV* adev = (struct AFHBA_DEV*)data;

	unsigned cr = afhba_read_reg(adev, HOST_PCIE_INTERRUPT_REG);
	afhba_write_reg(adev, HOST_PCIE_INTERRUPT_REG, cr);
	dev_info(pdev(adev), "afs_cos_isr %08x", cr);

	return IRQ_HANDLED;
}



static int hook_interrupts(struct AFHBA_DEV* adev)
{
	int rc = pci_enable_msi(adev->pci_dev);
	if (rc < 0){
		dev_err(pdev(adev), "pci_enable_msi_exact(%d) FAILED", 1);
		return rc;
	}
	rc = request_irq(adev->pci_dev->irq, afs_cos_isr, IRQF_SHARED, "afhba", adev);
	if (rc < 0) {
		pr_warn("afhba.%d: request_irq =%d failed!\n",
				adev->idx, adev->pci_dev->irq);
		pci_disable_msi(adev->pci_dev);
		return rc;
	}
	return 0;
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


int load_buffers(struct AFHBA_DEV* adev)
/* return 0 on success */
{
	struct AFHBA_STREAM_DEV* sdev = adev->stream_dev;

	queue_free_buffers(adev);
	if (dma_descriptor_ram > 1 && !sdev->job.dma_started){
		return validate_dma_descriptor_ram(adev, DMA_PUSH_DESC_RAM,
							sdev->push_ram_cursor, 1);
     	}else{
       		queue_free_buffers(adev);
       		return 0;
       	}
}

int start_job(struct AFHBA_DEV* adev)
/* returns 0 on success */
{
	int retry = 0;
	int load_ok = 0;

	afs_stop_dma(adev, DMA_PUSH_SEL);	/* belt+braces */
	afs_configure_streaming_dma(adev, DMA_PUSH_SEL);

	while(retry++ < max_dma_load_retry){
		if (load_buffers(adev) == 0){
			load_ok = 1;
			break;
		}
	}
	if (!load_ok){
		dev_err(pdev(adev), "ERROR dma load retry count exceeded");
		return 1;
	}
	afs_start_dma(adev, DMA_PUSH_SEL);

	spin_lock(&adev->stream_dev->job_lock);
	adev->stream_dev->job.dma_started = 1;
	spin_unlock(&adev->stream_dev->job_lock);
	return 0;
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
	int job_is_go_but_aurora_is_down = 0;
	unsigned loop_jiffies = 0;
	unsigned last_amon_jiffies = 0;

	sched_setscheduler(current, SCHED_FIFO, &param);
	afs_comms_init(adev);

	for ( ; !kthread_should_stop(); ++loop_count, loop_jiffies += WORK_TO){
		int timeout = wait_event_interruptible_timeout(
			sdev->work.w_waitq,
			test_and_clear_bit(WORK_REQUEST, &sdev->work.w_to_do),
			WORK_TO) == 0;

		if (!timeout || loop_count%10 == 0){
			dev_dbg(pdev(adev), "TIMEOUT? %d queue_free_buffers() ? %d",
			    timeout, job_is_go(job)  );
		}

		if (loop_jiffies - last_amon_jiffies > amon_jiffies){
			if (aurora_monitor && !afs_comms_init(adev) &&
			    job_is_go(job) && !job_is_go_but_aurora_is_down){
				dev_warn(pdev(adev), "job is go but aurora is down");
				job_is_go_but_aurora_is_down = 1;
			}else{
				job_is_go_but_aurora_is_down = 0;
			}
			last_amon_jiffies = loop_jiffies;
		}

	        if (job_is_go(job)){
	        	if (!job->dma_started){
	        		if (start_job(adev) != 0){
	        			job->please_stop = PS_PLEASE_STOP;
	        		}
	        	}else{
	        		queue_free_buffers(adev);
			}
		}

		if (job->buffers_demand > 0 && queue_full_buffers(adev) > 0){
			wake_up_interruptible(&sdev->return_waitq);
		}

		spin_lock(&sdev->job_lock);

	        if (sdev->job.on_pull_dma_timeout){
	        	sdev->job.on_pull_dma_timeout(adev);
	        }
	        if (sdev->job.on_push_dma_timeout){
	        	sdev->job.on_push_dma_timeout(adev);
	        }

		switch(job->please_stop){
		case PS_STOP_DONE:
			break;
		case PS_PLEASE_STOP:
			afs_stop_dma(adev, DMA_PUSH_SEL);
			job->please_stop = PS_STOP_DONE;
			job->dma_started = 0;
			if (dma_descriptor_ram > 1){
				validate_dma_descriptor_ram(
					adev, DMA_PUSH_DESC_RAM, sdev->push_ram_cursor, 99);
			}
			break;
/*
  		default:
			please_check_fifo = job_is_go(job) &&
				afs_dma_started(adev, DMA_PUSH_SEL);
*/
		}
		spin_unlock(&sdev->job_lock);

		if (please_check_fifo){
			check_fifo_status(adev);
			please_check_fifo = 0;
		}
	}

	return 0;
}


static void startWork(struct AFHBA_DEV *adev)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	sdev->work.w_task = kthread_run(afs_isr_work, adev, adev->name);
	sdev->work.mon_task = kthread_run(as_mon, adev, adev->mon_name);
}

static void stopWork(struct AFHBA_DEV *adev)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	if (sdev->work.w_task){
		kthread_stop(sdev->work.w_task);
	}
	if (sdev->work.mon_task){
		kthread_stop(sdev->work.mon_task);
	}
}

ssize_t afs_histo_read(
	struct file *file, char *buf, size_t count, loff_t *f_pos)
{
	unsigned *the_histo = PD(file)->private;
	int maxentries = PD(file)->private2;
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
	afs_dma_reset(adev, DMA_PUSH_SEL);
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

int afs_histo_open(struct inode *inode, struct file *file, unsigned *histo, int hcount)
{
	file->f_op = &afs_fops_histo;
	PD(file)->private = histo;
	PD(file)->private2 = hcount;
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



	for (ii = 0; ii < nbuffers; ++ii, ++hb){
		hb->bstate = BS_EMPTY;
		list_add_tail(&hb->list, &sdev->bp_empties.list);
	}

	sdev->init_descriptors(sdev);



	memset(sdev->data_fifo_histo, 0, DATA_FIFO_SZ);
	memset(sdev->desc_fifo_histo, 0, DESC_FIFO_SZ);

	mutex_unlock(&sdev->list_mutex);
	return 0;
}


void afs_stop_llc_push(struct AFHBA_DEV *adev)
{
	DEV_DBG(pdev(adev), "afs_stop_llc_push()");
	DEV_DBG(pdev(adev), "afs_dma_set_recycle(0)");
	msleep(1);
	afs_dma_set_recycle(adev, DMA_PUSH_SEL, 0);
	afs_dma_reset(adev, DMA_PUSH_SEL);
}

void afs_stop_llc_pull(struct AFHBA_DEV *adev)
{
	dev_info(pdev(adev), "afs_stop_llc_pull()");
	afs_dma_reset(adev, DMA_PULL_SEL);
}

void afs_stop_stream_push(struct AFHBA_DEV *adev)
{
	dev_info(pdev(adev), "afs_stop_stream_push()");
	afs_dma_reset(adev, DMA_PUSH_SEL);
}

void afs_stop_stream_pull(struct AFHBA_DEV *adev)
{
	dev_info(pdev(adev), "afs_stop_stream_pull()");
	afs_dma_reset(adev, DMA_PULL_SEL);
}

int push_dma_timeout(struct AFHBA_DEV *adev)
/* called with job_lock ON */
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	u32 dma_ctrl = DMA_CTRL_RD(adev);
	int action = 0;

	action = sdev->job.on_push_dma_timeout != 0 &&
				(dma_ctrl&DMA_CTRL_EN) == 0;
	if (action){
		struct XLLC_DEF* xllc_def = &sdev->job.push_llc_def;
		dev_err(pdev(adev), "DMA_CTRL_EN NOT SET attempt restart");
		afs_dma_reset(adev, DMA_PUSH_SEL);
		afs_load_llc_single_dma(adev, DMA_PUSH_SEL, xllc_def->pa, xllc_def->len);
		sdev->push_dma_timeouts++;
	}
	return 0;
}
long afs_start_ai_llc(struct AFHBA_DEV *adev, struct XLLC_DEF* xllc_def)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	struct JOB* job = &sdev->job;

	spin_lock(&sdev->job_lock);
	job->please_stop = PS_OFF;
	spin_unlock(&sdev->job_lock);
	sdev->onStopPush = afs_stop_llc_push;

	if (xllc_def->pa == RTM_T_USE_HOSTBUF){
		xllc_def->pa = sdev->hbx[0].pa;
	}
	afs_dma_reset(adev, DMA_PUSH_SEL);
	afs_load_llc_single_dma(adev, DMA_PUSH_SEL, xllc_def->pa, xllc_def->len);
	spin_lock(&sdev->job_lock);
	job->please_stop = PS_OFF;
	job->on_push_dma_timeout = push_dma_timeout;
	job->push_llc_def = *xllc_def;
	spin_unlock(&sdev->job_lock);
	return 0;
}
long afs_start_ao_llc(struct AFHBA_DEV *adev, struct XLLC_DEF* xllc_def)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;

	sdev->job.please_stop = PS_OFF;
	sdev->onStopPull = afs_stop_llc_pull;

	if (xllc_def->pa == RTM_T_USE_HOSTBUF){
		xllc_def->pa = sdev->hbx[0].pa;
	}
	afs_dma_reset(adev, DMA_PULL_SEL);
	afs_load_llc_single_dma(adev, DMA_PULL_SEL, xllc_def->pa, xllc_def->len);
	return 0;
}

int afs_dma_open(struct inode *inode, struct file *file)
{
	struct AFHBA_DEV *adev = PD(file)->dev;
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;

	int ii;

	dev_dbg(pdev(adev), "45: DMA open");

	/** @@todo protect with lock ? */
	if (sdev->pid == 0){
		sdev->pid = current->pid;
	}

	if (sdev->pid != current->pid){
		return -EBUSY;
	}

	sdev->shot++;

	if (sdev->buffer_len == 0) sdev->buffer_len = buffer_len;
	sdev->req_len = min(sdev->buffer_len, buffer_len);

	for (ii = 0; ii != nbuffers; ++ii){
		sdev->hbx[ii].req_len = sdev->req_len;
	}
	if (afs_reset_buffers(adev)){
		return -ERESTARTSYS;
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

	dev_dbg(pdev(adev), "afs_dma_release() 01 %s %d %p<-%p->%p",
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
	spin_lock(&sdev->job_lock);
	sdev->job.please_stop = PS_PLEASE_STOP;
	sdev->job.on_push_dma_timeout = 0;
	sdev->job.on_pull_dma_timeout = 0;
	sdev->job.buffers_demand = 0;
	spin_unlock(&sdev->job_lock);


	if (sdev->onStopPull){
		sdev->onStopPull(adev);
		sdev->onStopPull = 0;
	}
	if (sdev->onStopPush){
		sdev->onStopPush(adev);
		sdev->onStopPush = 0;
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

	if (sdev->onStopPush == 0){
		sdev->onStopPush = afs_stop_stream_push;
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
		int backlog = 0;
		struct StreamBufferDef sbd;

		list_for_each_entry_safe(hb, tmp, &sdev->bp_full.list, list){
			if (nbytes+SBDSZ > count){
				dev_dbg(pdev(adev), "quit nbytes %d count %lu",
				    nbytes, (long)count);
				break;
			}

			sbd.ibuf = IBUF_MAGIC|(backlog<<IBUF_IDX_SHL)|hb->ibuf;
			sbd.esta = hb->esta;

			if (copy_to_user(buf+nbytes, &sbd, SBDSZ)){
				rc = -EFAULT;
				goto read99;
			}
			dev_dbg(pdev(adev), "add my_buffers %d", hb->ibuf);

			list_move_tail(&hb->list, &PD(file)->my_buffers);
			hb->bstate = BS_FULL_APP;
			nbytes += SBDSZ;
			backlog++;
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

static unsigned int afs_dma_poll(struct file* file, poll_table *poll_table)
{
	struct AFHBA_DEV *adev = PD(file)->dev;
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	unsigned int mask = POLLOUT | POLLWRNORM;
	if (!list_empty(&sdev->bp_full.list)){
		mask |= POLLIN | POLLRDNORM;
	}else{
		poll_wait(file, &sdev->return_waitq, poll_table);
		if (!list_empty(&sdev->bp_full.list)){
			mask |= POLLIN | POLLRDNORM;
		}
	}
	return mask;
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

	if (!afs_dma_started(adev, DMA_PUSH_SEL)){
		afs_start_dma(adev, DMA_PUSH_SEL);
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


int fix_dma_buff_size(struct AB *ab, struct XLLC_DEF *xdef)
/* descriptors are power of 2 * 1K .. attempt to fit in 2 descs.. */
{
	int nblocks = ab->buffers[0].len/1024;
	int ii;
	for (ii = 0; ii < 16; ++ii){
		if (1<<ii > nblocks){
			int len1 = (1<<(ii-1))*1024;
			xdef[0] 	= ab->buffers[0];
			xdef[1].pa 	= xdef[0].pa + len1;
			xdef[1].len 	= xdef[0].len - len1;
			xdef[0].len 	= len1;
			xdef[2] 	= ab->buffers[1];
			xdef[3].pa 	= xdef[2].pa + len1;
			xdef[3].len 	= xdef[2].len - len1;
			xdef[2].len 	= len1;
			return 4;
		}else if (1<<ii == nblocks){
			xdef[0] = ab->buffers[0];
			xdef[1] = ab->buffers[1];
			return 2;
		}
	}
	printk("ERROR: fix_dma_descriptors BUFFER TOO LONG");
	return 0;
}
long afs_start_AI_AB(struct AFHBA_DEV *adev, struct AB *ab)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	struct JOB* job = &sdev->job;
	struct XLLC_DEF xdef[4];
	int nb;

	spin_lock(&sdev->job_lock);
	job->please_stop = PS_OFF;
	spin_unlock(&sdev->job_lock);
	sdev->onStopPush = afs_stop_llc_push;

	if (ab->buffers[0].pa == RTM_T_USE_HOSTBUF){
		ab->buffers[0].pa = sdev->hbx[0].pa;
	}
	if (ab->buffers[1].pa == RTM_T_USE_HOSTBUF){
		ab->buffers[1].pa = sdev->hbx[1].pa;
	}else if (ab->buffers[1].pa < buffer_len){
		/* pa is an offset in buffer 0 */
		ab->buffers[1].pa += ab->buffers[0].pa;
	}
	afs_dma_reset(adev, DMA_PUSH_SEL);
	if (use_llc_multi){
		afs_load_dram_descriptors_ll(adev, DMA_PUSH_SEL, ab->buffers, 2);
	}else{
		nb = fix_dma_buff_size(ab, xdef);
		afs_load_dram_descriptors(adev, DMA_PUSH_SEL, xdef, nb);
	}
	spin_lock(&sdev->job_lock);
	job->please_stop = PS_OFF;
	job->on_push_dma_timeout = 0;
	spin_unlock(&sdev->job_lock);
	return 0;
}


long afs_start_ABN(struct AFHBA_DEV *adev, struct ABN *abn, enum DMA_SEL dma_sel)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	struct JOB* job = &sdev->job;
	int ib;

	spin_lock(&sdev->job_lock);
	job->please_stop = PS_OFF;
	spin_unlock(&sdev->job_lock);
	if(dma_sel&DMA_PUSH_SEL){
		sdev->onStopPush = afs_stop_llc_push;
	}
	if (dma_sel&DMA_PULL_SEL){
		sdev->onStopPull = afs_stop_llc_pull;
	}

	dev_dbg(pdev(adev), "%s descriptors:%d %s", __FUNCTION__, abn->ndesc,
			abn->buffers[0].pa == RTM_T_USE_HOSTBUF? "RTM_T_USE_HOSTBUF": "USER_ADDR");

	if (abn->buffers[0].pa == RTM_T_USE_HOSTBUF){
		abn->buffers[0].pa = sdev->hbx[0].pa;

		for (ib = 1; ib < abn->ndesc; ++ib){
			abn->buffers[ib].pa = abn->buffers[ib-1].pa + PAGE_SIZE;
		}
	}

	for (ib = 0; ib < abn->ndesc; ++ib){
		dev_dbg(pdev(adev), "%s [%2d] pa:%08x len:%d", 
				__FUNCTION__, ib, abn->buffers[ib].pa, abn->buffers[ib].len);
	}

	afs_load_dram_descriptors_ll(adev, dma_sel, abn->buffers, abn->ndesc);

	spin_lock(&sdev->job_lock);
	job->please_stop = PS_OFF;

	if(dma_sel&DMA_PUSH_SEL){
		job->on_push_dma_timeout = 0;
	}
	if (dma_sel&DMA_PULL_SEL){
		job->on_pull_dma_timeout = 0;
	}
	spin_unlock(&sdev->job_lock);
	return 0;
}
long afs_dma_ioctl(struct file *file,
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
	case AFHBA_START_AI_LLC : {
		struct XLLC_DEF xllc_def;
		long rc;
		COPY_FROM_USER(&xllc_def, varg, sizeof(struct XLLC_DEF));
		rc =  afs_start_ai_llc(adev, &xllc_def);
		COPY_TO_USER(varg, &xllc_def, sizeof(struct XLLC_DEF));
		return rc;
	}
	case AFHBA_START_AO_LLC : {
		struct XLLC_DEF xllc_def;
		long rc;
		COPY_FROM_USER(&xllc_def, varg, sizeof(struct XLLC_DEF));
		rc = afs_start_ao_llc(adev, &xllc_def);
		COPY_TO_USER(varg, &xllc_def, sizeof(struct XLLC_DEF));
		return rc;
	}
	case AFHBA_START_AI_AB: {
		struct AB ab;
		long rc;
		COPY_FROM_USER(&ab, varg, sizeof(struct AB));
		rc = afs_start_AI_AB(adev, &ab);
		COPY_TO_USER(varg, &ab, sizeof(struct AB));
		return rc;
	}
	case AFHBA_START_AI_ABN:
	case AFHBA_START_AO_ABN:
	{
		struct ABN *abn = kmalloc(sizeof(struct ABN), GFP_KERNEL);
		long rc;
		COPY_FROM_USER(abn, varg, sizeof(struct ABN));
		if (cmd == AFHBA_START_AI_ABN){
			rc = afs_start_ABN(adev, abn, DMA_PUSH_SEL);
		}else{
			rc = afs_start_ABN(adev, abn, DMA_PULL_SEL);
		}
		COPY_TO_USER(varg, abn, sizeof(struct ABN));
		return rc;
	}
	default:
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
	int minor = PD(vma->vm_file)->minor;

	int ibuf = minor<=NBUFFERS_MASK? minor&NBUFFERS_MASK: 0;
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
	.poll = afs_dma_poll,
	.write = afs_dma_write,
	.unlocked_ioctl = afs_dma_ioctl,
	.mmap = afs_mmap_host
};

static struct file_operations afs_fops_dma_poll = {
	.open = afs_dma_open,
	.release = afs_dma_release,
	.read = afs_dma_read_poll,
	.poll = afs_dma_poll,
	.write = afs_dma_write,
	.unlocked_ioctl = afs_dma_ioctl,
	.mmap = afs_mmap_host
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
			inode, file,
			adev->stream_dev->data_fifo_histo, RTDMAC_DATA_FIFO_CNT);
	case MINOR_DESC_FIFO:
		return afs_histo_open(
			inode, file,
			adev->stream_dev->desc_fifo_histo, RTDMAC_DESC_FIFO_CNT);
	case MINOR_CATCHUP_HISTO:
		return afs_histo_open(
			inode, file,
			adev->stream_dev->job.catchup_histo, NBUFFERS);
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

static ssize_t show_zmod_id(
		struct device * dev,
		struct device_attribute *attr,
		char * buf)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);
	return sprintf(buf, "0x%08x\n", _afs_read_zynqreg(adev, Z_MOD_ID));
}

static DEVICE_ATTR(z_mod_id, S_IRUGO, show_zmod_id, 0);

static ssize_t show_z_ident(
		struct device * dev,
		struct device_attribute *attr,
		char * buf)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);
	return sprintf(buf, "0x%08x\n", _afs_read_zynqreg(adev, Z_IDENT));
}

static DEVICE_ATTR(z_ident, S_IRUGO, show_z_ident, 0);

static ssize_t show_acq_model(
		struct device * dev,
		struct device_attribute *attr,
		char * buf)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);
	unsigned model = _afs_read_zynqreg(adev, Z_IDENT) >> 16;
	model = model&0x2106;
	return sprintf(buf, "acq%04x\n", model);
}

static DEVICE_ATTR(acq_model, S_IRUGO, show_acq_model, 0);

static ssize_t show_acq_port(
		struct device * dev,
		struct device_attribute *attr,
		char * buf)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);
	unsigned comms = _afs_read_zynqreg(adev, Z_IDENT) >> 20;
	comms = comms&0xf;
	return sprintf(buf, "%X\n", comms);
}

static DEVICE_ATTR(acq_port, S_IRUGO, show_acq_port, 0);

static ssize_t show_acq_ident(
		struct device * dev,
		struct device_attribute *attr,
		char * buf)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);
	unsigned z_ident = _afs_read_zynqreg(adev, Z_IDENT);
	unsigned ident = z_ident&0x0ffff;
	unsigned model = (z_ident >> 16) & 0x2106;

	return sprintf(buf, "acq%04x_%03d\n", model, ident);
}

static DEVICE_ATTR(acq_ident, S_IRUGO, show_acq_ident, 0);

static ssize_t store_com_trg(
	struct device * dev,
	struct device_attribute *attr,
	const char * buf, size_t count)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);
	unsigned tv;

	if (sscanf(buf, "%x", &tv) == 1){
		_afs_write_comreg(adev, COM_SOFT_TRIGGER, COM_SOFT_TRIGGER_EN);
		afhba_write_reg(adev, HOST_TEST_REG, tv);   /* forces 1usec high time */
		afhba_read_reg(adev, HOST_TEST_REG);
		_afs_write_comreg(adev, COM_SOFT_TRIGGER, ~COM_SOFT_TRIGGER_EN);
		return strlen(buf);
	}else{
		return -1;
	}
}

static DEVICE_ATTR(com_trg, (S_IWUSR|S_IWGRP), 0, store_com_trg);


static const struct attribute *dev_attrs[] = {
	&dev_attr_com_trg.attr,		/* must be first */
	&dev_attr_z_mod_id.attr,
	&dev_attr_z_ident.attr,
	&dev_attr_acq_model.attr,
	&dev_attr_acq_port.attr,
	&dev_attr_acq_ident.attr,
	NULL
};


void afs_create_sysfs(struct AFHBA_DEV *adev)
{
	const struct attribute ** attrs = dev_attrs;
	int rc;

	if (adev->remote_com_bar == -1 ){
		++attrs;
	}
	rc = sysfs_create_files(&adev->class_dev->kobj, attrs);
	if (rc){
		dev_err(pdev(adev), "failed to create files");
		return;
	}
}


int afhba_stream_drv_init(struct AFHBA_DEV* adev)
{
	adev->stream_dev = kzalloc(sizeof(struct AFHBA_STREAM_DEV), GFP_KERNEL);

	dev_info(pdev(adev), "afhba_stream_drv_init %s name:%s idx:%d", REVID, adev->name, adev->idx);

	afs_init_buffers(adev);
	if (adev->peer == 0){
		hook_interrupts(adev);
	}
	startWork(adev);
	adev->stream_fops = &afs_fops;
	afs_init_procfs(adev);
	afs_create_sysfs(adev);
	return 0;
}
int afhba_stream_drv_del(struct AFHBA_DEV* adev)
{
	dev_info(pdev(adev), "afhba_stream_drv_del()");
	afs_init_dma_clr(adev);
	stopWork(adev);
	return 0;
}
