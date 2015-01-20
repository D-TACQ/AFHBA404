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

#include <linux/device.h>
#include <linux/delay.h>
#include <linux/interrupt.h>
#include <linux/fs.h>
#include <linux/ioctl.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/list.h>
#include <linux/pci.h>
#include <linux/time.h>
#include <linux/init.h>
#include <linux/timex.h>
#include <linux/vmalloc.h>
#include <linux/mm.h>
#include <linux/moduleparam.h>
#include <linux/mutex.h>

#include <asm/uaccess.h>  /* VERIFY_READ|WRITE */


#ifndef EXPORT_SYMTAB
#define EXPORT_SYMTAB
#include <linux/module.h>
#endif

#include "acq-fiber-hba.h"
#include "afhba_stream_drv.h"
#include "lk-shim.h"

/* 4 x 16 x 2 = 128MB/s. */
#define BUFFER_LEN	0x400000	/* 30 BUFFERS/SEC full WHACK */
#define NBUFFERS	64

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
		u32 descr = sdev->hb[ii].descr;

		descr &= ~AFDMAC_DESC_LEN_MASK;
		descr |= getAFDMAC_Order(sdev->buffer_len)<< AFDMAC_DESC_LEN_SHL;
		descr |= AFDMAC_DESC_EOT;
		sdev->hb[ii].descr = descr;
	}
}


#include "acq-fiber-hba.h"
#include "afhba_stream_drv.h"

#define RTDMAC_DATA_FIFO_CNT	0x1000
#define RTDMAC_DESC_FIFO_CNT	0x1000

#define DATA_FIFO_SZ	(RTDMAC_DATA_FIFO_CNT*sizeof(unsigned))
#define DESC_FIFO_SZ	(RTDMAC_DESC_FIFO_CNT*sizeof(unsigned))

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

int as_init_buffers(struct AFHBA_DEV* adev)
{
	struct AFHBA_STREAM_DEV* sdev = adev->stream_dev;
	int ii;
	int order = getOrder(BUFFER_LEN);
	struct HostBuffer *hb = sdev->hb;

        INIT_LIST_HEAD(&sdev->bp_empties.list);
	INIT_LIST_HEAD(&sdev->bp_filling.list);
	INIT_LIST_HEAD(&sdev->bp_full.list);
	spin_lock_init(&sdev->job_lock);

	mutex_init(&sdev->list_mutex);
	mutex_lock(&sdev->list_mutex);

	sdev->buffer_len = BUFFER_LEN;
	dbg(1, "allocating %d buffers size:%d order:%d dev.dma_mask:%08llx",
			NBUFFERS, BUFFER_LEN, order, *adev->pci_dev->dev.dma_mask);

	for (ii = 0; ii < NBUFFERS; ++ii, ++sdev->nbuffers, ++hb){
		void *buf = (void*)__get_free_pages(GFP_KERNEL|GFP_DMA32, order);

		if (!buf){
			err("failed to allocate buffer %d", ii);
			break;
		}

		dbg(3, "buffer %2d allocated at %p, map it", ii, buf);

		hb->ibuf = ii;
		hb->pa = dma_map_single(&adev->pci_dev->dev, buf,
				BUFFER_LEN, PCI_DMA_FROMDEVICE);
		hb->va = buf;
		hb->len = BUFFER_LEN;

		dbg(3, "buffer %2d allocated, map done", ii);

		if ((hb->pa & (AFDMAC_PAGE-1)) != 0){
			err("HB NOT PAGE ALIGNED");
			BUG();
		}

		hb->descr = hb->pa | 0 | AFDMAC_DESC_EOT | (ii&AFDMAC_DESC_ID_MASK);
		hb->bstate = BS_EMPTY;

		dbg(3, "[%d] %p %08x %d %08x",
		    ii, hb->va, hb->pa, hb->len, hb->descr);
		list_add_tail(&hb->list, &sdev->bp_empties.list);
	}
	sdev->init_descriptors = init_descriptors_ht;
	sdev->init_descriptors(sdev);
	init_waitqueue_head(&sdev->work.w_waitq);
	init_waitqueue_head(&sdev->return_waitq);

	mutex_unlock(&sdev->list_mutex);

	init_histo_buffers(sdev);
	return 0;
}

int afhba_stream_drv_init(struct AFHBA_DEV* adev)
{
	adev->stream_dev = kzalloc(GFP_KERNEL, sizeof(struct AFHBA_STREAM_DEV));
	as_init_buffers(adev);
	return 0;
}
int afhba_stream_drv_del(struct AFHBA_DEV* adev)
{
	return 0;
}
