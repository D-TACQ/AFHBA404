/*
 * afhba_stream_drv.h
 *
 *  Created on: 19 Jan 2015
 *      Author: pgm
 */

#ifndef AFHBA_STREAM_DRV_H_
#define AFHBA_STREAM_DRV_H_


/* idea by John: make it a multiple of 3 to handle 96ch align case */
#define NBUFFERS	66

#define NBUFFERS_FIFO	NBUFFERS

#define NBUFFERS_MASK	127

#define BUFFER_LEN	0x100000


#define NSTATES		4
#define N_DRV_STATES	3

#define AFDMAC_PAGE		0x400		/* 1K pages */
#define AFDMAC_LL_PAGE		64		/* page length in LL */

#define AFDMAC_DESC_ADDR_MASK	0xfffffc00	/* base address */

#define AFDMAC_DESC_WRITE	0x00000200	/* Write BIT */
#define AFDMAC_DESC_EOT		0x00000100	/* End Of Transfer interrupt */

#define AFDMAC_DESC_LEN_MASK	0x000000f0	/* length (pages) */
#define AFDMAC_DESC_LEN_SHL	4

#define AFDMAC_DESC_ID_MASK	0x0000000f	/* ID 0..15 */
#define AFDMAC_DESC_ID		0x0000000f	/* ID 0..15 */

#define AFDMAC_DESCR(pa, pages, id)	\
	(((pa)&AFDMAC_DESC_ADDR_MASK)|	\
	((((pages)-1) << AFDMAC_DESC_LEN_SHL)&AFDMAC_DESC_LEN_MASK)| \
	(id))

/* we wanted an elegant multi-vector solution, but we can only have one int */
#define MSI_DMA		0
#ifdef MSI_BLOCK_WORKS
#define MSI_UART	1
#else
#define MSI_UART	0
#endif

enum { PS_OFF, PS_PLEASE_STOP, PS_STOP_DONE };



struct AFHBA_STREAM_DEV {

	struct mutex list_mutex;

	struct BOTTLING_PLANT {
		struct list_head list;
		struct proc_dir_entry *proc;
	}
		bp[N_DRV_STATES];

#define bp_empties bp[BS_EMPTY]
#define bp_filling bp[BS_FILLING]
#define bp_full	   bp[BS_FULL]
/* BS_FULL_APP : in list in path buffer */


	int nbuffers;
	int buffer_len;
	struct HostBuffer *hbx;		/* streaming host buffers [nbuffers] */


	struct proc_dir_entry *proc_dir_root;

	struct JOB {
		unsigned buffers_demand;
		unsigned buffers_queued;
		unsigned buffers_received;
		unsigned ints;
		int please_stop;
		unsigned rx_buffers_previous;
		unsigned int_previous;
		unsigned rx_rate;
		unsigned int_rate;
		unsigned errors;
		unsigned buffers_discarded;
	}
		job;
	spinlock_t job_lock;

	unsigned *data_fifo_histo;
      	unsigned *desc_fifo_histo;

      	void (*init_descriptors)(struct AFHBA_STREAM_DEV *sdev);

	struct WORK {				/* ISR BH processing */
		wait_queue_head_t w_waitq;
		unsigned long w_to_do;
		struct task_struct* w_task;
#define WORK_REQUEST	0
		struct task_struct* mon_task;
	} work;

	wait_queue_head_t return_waitq;

	u32 dma_regs[DMA_REGS_COUNT];

	unsigned pid;		/* pid of dma_read process */
	int req_len;		/* request len .. not part of job ? */
	void (* onStop)(struct AFHBA_DEV *adev);
};
#define MIRROR(adev, ix) (adev->stream_dev->dma_regs[ix])

void _afs_write_dmareg(struct AFHBA_DEV *adev, int regoff, u32 value);
u32 _afs_read_dmareg(struct AFHBA_DEV *adev, int regoff);

#define DMA_CTRL_WR(adev, value) \
	_afs_write_dmareg(adev, DMA_CTRL, MIRROR(adev, DMA_CTRL) = value)

#define DMA_CTRL_CLR(adev, bits) \
	_afs_write_dmareg(adev, DMA_CTRL, MIRROR(adev, DMA_CTRL) &= ~(bits))

#define DMA_CTRL_SET(adev, bits) \
	_afs_write_dmareg(adev, DMA_CTRL, MIRROR(adev, DMA_CTRL) |= (bits))

#define DMA_CTRL_RD(adev) MIRROR(adev, DMA_CTRL)


#define DMA_TEST_WR(adev, value) \
	_afs_write_dmareg(adev, DMA_TEST, MIRROR(adev, DMA_CTRL)), value)

#define DMA_TEST_RD(adev) \
	_afs_read_reg(adev, DMA_TEST)

#define DMA_DATA_FIFSTA_RD(adev)   _afs_read_dmareg(adev, DMA_DATA_FIFSTA)
#define DMA_DESC_FIFSTA_RD(adev)   _afs_read_dmareg(adev, DMA_DESC_FIFSTA)
#define DMA_PUSH_DESC_STA_RD(adev) _afs_read_dmareg(adev, DMA_PUSH_DESC_STA)
#define DMA_PULL_DESC_STA_RD(adev) _afs_read_dmareg(adev, DMA_PULL_DESC_STA)


#define EMPTY1	0xee11ee11
#define EMPTY2  0x22ee22ee

#define RTDMAC_DATA_FIFO_CNT	0x1000
#define RTDMAC_DESC_FIFO_CNT	0x1000

#define HB_ENTRY(plist)	list_entry(plist, struct HostBuffer, list)

int afs_init_procfs(struct AFHBA_DEV *adev);
int afs_reset_buffers(struct AFHBA_DEV *adev);

#endif /* AFHBA_STREAM_DRV_H_ */
