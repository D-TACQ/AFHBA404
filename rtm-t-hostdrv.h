/*
 * rtm-t-hostdrv.h
 *
 *  Created on: May 25, 2011
 *      Author: pgm
 */

#ifndef RTM_T_HOSTDRV_H_
#define RTM_T_HOSTDRV_H_

/* idea by John: make it a multiple of 3 to handle 96ch align case */
#define NBUFFERS	66

#define NBUFFERS_FIFO	NBUFFERS

#define NBUFFERS_MASK	127

#define BUFFER_LEN	0x100000

#define MINOR_DMAREAD	254
#define MINOR_REGREAD	253
#define MINOR_DATA_FIFO 252
#define MINOR_DESC_FIFO 251
#define MINOR_UART	250
#define MINOR_PBI	249
#define MINOR_FIFO	248

#define MINOR_REMOTE	247

#define LLC_AI	0x1
#define LLC_AO	0x2

struct RTM_T_DEV {
	struct PciMapping {
		int bar;
		u32 pa;
		void* va;
		unsigned len;
		struct resource *region;
		char name[32];
	} mappings[MAP_COUNT];

	struct RegsMirror {
		u32 ctrl;
		u32 H_FCR;
		u32 dio;
		u32 dio_read;
	} regs;

	struct HostBuffer {
		int ibuf;
		void *va;
		u32 pa;
		int len;
		int req_len;
		u32 descr;
		struct list_head list;
		enum BSTATE {
			BS_EMPTY, BS_FILLING, BS_FULL, BS_FULL_APP }
		bstate;
		u32 timestamp;
	} *hb;

#define NSTATES		4
#define N_DRV_STATES	3

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
	struct pci_dev *pci_dev;
	struct CLASS_DEVICE *class_dev;
	int idx;
	char name[16];
	char mon_name[16];
	char slot_name[16];
	int major;
	struct list_head list;



	struct WORK {				/* ISR BH processing */
		wait_queue_head_t w_waitq;
		unsigned long w_to_do;
		struct task_struct* w_task;
#define WORK_REQUEST	0
		struct task_struct* mon_task;
	} work;



	wait_queue_head_t return_waitq;

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

	char last_dio_bit_store[40];

	int lowlat_length;
	void (*init_descriptors)(struct RTM_T_DEV *tdev);

	int req_len;

	unsigned pid;		/* pid of dma_read process */
	int is_llc;		/* llc mode enabled on ACQ */
	int is_ao32;
	int is_fhba;
	void (* onStop)(struct RTM_T_DEV* tdev);
	struct platform_device *hba_sfp_i2c[2];
	int dma_page_size;
	int lldma_page_size;

	int buffer_len;		/* length of buffer for this device */
	int iop_push;		/* iop mode for this device 	    */
};


void rtd_write_reg(struct RTM_T_DEV *tdev, int regoff, u32 value);
u32 rtd_read_reg(struct RTM_T_DEV *tdev, int regoff);

void rtm_t_createSysfs(struct RTM_T_DEV *tdev);

void rtm_t_create_sysfs_class(struct RTM_T_DEV *tdev);
void rtm_t_remove_sysfs_class(struct RTM_T_DEV *tdev);

int rtm_t_reset_buffers(struct RTM_T_DEV* tdev);

struct RTM_T_DEV* rtm_t_lookupDev(struct device *dev);
struct RTM_T_DEV *rtm_t_lookupDeviceFromClass(struct CLASS_DEVICE *dev);

int create_sfp_i2c(struct RTM_T_DEV *td);
int remove_sfp_i2c(struct RTM_T_DEV *td);


extern int buffer_len;

enum { PS_OFF, PS_PLEASE_STOP, PS_STOP_DONE };

int /* __devinit */ rtm_t_uart_init(struct pci_dev *pci_dev, void *mapping);
void /* __devexit */ rtm_t_uart_remove(struct pci_dev *pci_dev);

void init_descriptors_ht(struct RTM_T_DEV *tdev);
void init_descriptors_ll(struct RTM_T_DEV *tdev);

int initProcFs(struct RTM_T_DEV *tdev);

int acqfhba_spi_master_init(struct RTM_T_DEV *tdev);

#endif /* RTM_T_HOSTDRV_H_ */
