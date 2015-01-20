/*
 * afhba_stream_drv.h
 *
 *  Created on: 19 Jan 2015
 *      Author: pgm
 */

#ifndef AFHBA_STREAM_DRV_H_
#define AFHBA_STREAM_DRV_H_

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
	struct HostBuffer *hb;

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

};

#endif /* AFHBA_STREAM_DRV_H_ */
