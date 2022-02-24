/**  @file acq-fiber-hba.h
 *   @brief AFHBA404 device driver data structs.
 *
 * ------------------------------------------------------------------------- *
 *   Copyright (C) 2014 Peter Milne, D-TACQ Solutions Ltd                
 *                      <peter dot milne at D hyphen TACQ dot com>          
 *                         www.d-tacq.com
 *   Created on: 10 Aug 2014  
 *    Author: pgm                                                         
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

#ifndef ACQ_FIBER_HBA_H_
#define ACQ_FIBER_HBA_H_

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

#include <linux/uaccess.h>  /* VERIFY_READ|WRITE */
#include <linux/version.h>

#ifdef CONFIG_GPU
#include "gpumemdrv.h"
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5,9,0)
#define SCHEDULE_FIFO	sched_set_fifo_low(current)
#else
#define SCHEDULE_FIFO do { \
		struct sched_param param = { .sched_priority = 10 }; \
		sched_setscheduler(current, SCHED_FIFO, &param); \
} while(0)
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4,10,0)
#include <uapi/linux/sched/types.h>
#endif

#ifdef CONFIG_KERNEL_ASSERTS
/* kgdb stuff */
#define assert(p) KERNEL_ASSERT(#p, p)
#else
#define assert(p) do {	\
	if (!(p)) {	\
		printk(KERN_CRIT "BUG at %s:%d assert(%s)\n",	\
		       __FILE__, __LINE__, #p);			\
		BUG();	\
	}		\
} while (0)
#endif


#define MAXDEV	8	/* maximum expected devices .. */


struct AFHBA_DEV;


#include "afhba_minor.h"
#include "afhba_debugfs.h"
#include "rtm-t_ioctl.h"		/* retain previous API */

#define REGS_BAR	0		/* @@todo assumed */

#define MAP_COUNT_MAX	6

#define MAP_COUNT_4G1	2
#define MAP_COUNT_4G2	4

#define MAP_COUNT_4G4	6

extern struct list_head afhba_devices;

int afhba_registerDevice(struct AFHBA_DEV *tdev);
void afhba_deleteDevice(struct AFHBA_DEV *tdev);
struct AFHBA_DEV* afhba_lookupDevice(int major);
struct AFHBA_DEV *afhba_lookupDeviceFromClass(struct device *dev);
struct AFHBA_DEV* afhba_lookupDevicePci(struct pci_dev *pci_dev);
struct AFHBA_DEV* afhba_lookupDev(struct device *dev);
int afhba_release(struct inode *inode, struct file *file);

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
	u32 esta;		/* error status */
	u32 timestamp;
};

struct AFHBA_STREAM_DEV;

struct iommu_mapping {
	struct iommu_domain *iom_dom;
        struct list_head list;
        unsigned long iova;
        phys_addr_t paddr;
        size_t size;
        int prot;
};

struct AFHBA_DEV {
	char name[16];
	char mon_name[16];
	char slot_name[16];
	struct pci_dev *pci_dev;
	struct device *class_dev;
	int idx;
	int major;
	struct list_head list;
	int map_count;

	struct PciMapping {
		int bar;
		u32 pa;
		void* va;
		unsigned len;
		struct resource *region;
		char name[32];
	} mappings[MAP_COUNT_MAX];
	int remote_com_bar;

	struct AFHBA_DEV* peer;
	void* remote;
	enum SFP { SFP_A, SFP_B, SFP_C, SFP_D } sfp;
	unsigned ACR;		/* Aurora Control Reg Offset */

	struct proc_dir_entry *proc_dir_root;
	struct dentry* debug_dir;
	char *debug_names;

	struct HostBuffer *hb1;		/* single hb for LLC */

	struct AFHBA_STREAM_DEV* stream_dev;
	struct file_operations* stream_fops;

	struct platform_device *hba_sfp_i2c[2];

	int link_up;
	int aurora_error_count;
	int aurora_status_read_count;

#ifdef CONFIG_GPU
	//For GPU functionality
	struct gpumem gpumem;
#endif
	struct iommu_domain *iommu_dom;
	struct list_head iommu_map_list;
};

#define SZM1(field)	(sizeof(field)-1)


extern int afhba_stream_drv_init(struct AFHBA_DEV* adev);
extern int afhba_stream_drv_del(struct AFHBA_DEV* adev);
extern void afhba_create_sysfs_class(struct AFHBA_DEV *adev);
void afhba_remove_sysfs_class(struct AFHBA_DEV *adev);
void afhba_create_sysfs(struct AFHBA_DEV *adev);
void afhba_remove_sysfs(struct AFHBA_DEV *adev);

struct AFHBA_DEV_PATH {
	int minor;
	struct AFHBA_DEV *dev;
	struct list_head my_buffers;
	void* private;
	int private2;
};


#define PSZ	  (sizeof (struct AFHBA_DEV_PATH))
#define PD(file)  ((struct AFHBA_DEV_PATH *)(file)->private_data)
#define DEV(file) (PD(file)->dev)

#define pdev(adev) (&(adev)->pci_dev->dev)

#define LOC(adev) ((adev)->mappings[0].va)
#define REM(adev) ((adev)->remote)

#define REGS_BAR	0
#define REMOTE_BAR	1
#define REMOTE_BAR2	2
#define NO_BAR 		-1



/* REGS */
#define FPGA_REVISION_REG		0x0000	/* FPGA Revision Register */
#define HOST_PCIE_CONTROL_REG 		0x0004	/* Host PCIe Control / Status Register */
#define HOST_PCIE_INTERRUPT_REG 	0x0008	/* Host PCIe Interrupt Register Register */
#define PCI_CONTROL_STATUS_REG 		0x000C	/* PCI Control / Status Register */
#define PCIE_DEVICE_CONTROL_STATUS_REG 	0x0010	/* PCIe Control / Status Register */
#define PCIE_LINK_CONTROL_STATUS_REG 	0x0014	/* PCIe Link Control / Status Register */
#define PCIE_CONF_REG 			0x0018	/* Host PCIe Configuration Reg */
#define PCIE_BUFFER_CTRL_REG		0x001C
#define HOST_TEST_REG			0x0020
#define HOST_COUNTER_REG 		0x0024	/* Heart Beat Counter */
#define HOST_PCIE_DEBUG_REG 		0x0028	/* Host PCIe Debug Register */

#define HOST_PCIE_LATSTATS_1		0x0030
#define HOST_PCIE_LATSTATS_2		0x0034

#define HOST_SPI_FLASH_CONTROL_REG 	0x0040	/* SPI FLASH Control Register */
#define HOST_SPI_FLASH_DATA_REG 	0x0044	/* SPI FLASH Data Register */
#define HOST_MON_REG			0x0048  /* Temperature Monitor/Volts Alarms */

#define AURORA_STEP			0x0010  /* Duplicate blocks every STEP */
#define AURORA_CONTROL_REGA 		0x0080	/* Aurora Control Register */
#define AURORA_STATUS_REGA 		0x0084	/* Aurora Status Register */
#define SFP_I2C_DATA_REGA 		0x0088	/* SFP I2C Control and Data Register */
#define AURORA_STATUS2_REGA		0x008C

#define AURORA_CONTROL_REG(sfp)		(AURORA_CONTROL_REGA + AURORA_STEP*(sfp))
#define AURORA_STATUS_REG(sfp)		(AURORA_STATUS_REGA  + AURORA_STEP*(sfp))
#define AURORA_STATUS2_REG(sfp)		(AURORA_STATUS2_REGA + AURORA_STEP*(sfp))
#define SFP_I2C_DATA_REG(sfp)		(SFP_I2C_DATA_REGA   + AURORA_STEP*(sfp))

#define HOST_COMMS_FIFO_CONTROL_REG 	0x00C0	/* ACQ400 Receive Communications FIFO Control Register */
#define HOST_COMMS_FIFO_STATUS_REG 	0x00C4	/* ACQ400 Receive Communications FIFO Status Register */
#define ACQ400_COMMS_READ 		0x0400	/* ACQ400 Receive Communications FIFO data */


#define ASR(acr)	((acr)+4)
#define I2C(acr)	((acr)+8)


#define SFP_I2C_SCL1_R		0
#define SFP_I2C_SDA1_R		1
#define	SFP_I2C_SCL1_W		8
#define SFP_I2C_SDA1_W		9
#define SFP_I2C_SCL2_R		0
#define SFP_I2C_SDA2_R		1
#define	SFP_I2C_SCL2_W		8
#define SFP_I2C_SDA2_W		9

#define AFHBA_SPI_BUSY		(1<<31)
#define AFHBA_SPI_CTL_START	(1<<7)
#define AFHBA_SPI_CS		(1<<0)
#define AFHBA_SPI_HOLD		(1<<1)
#define AFHBA_SPI_WP		(1<<2)

#define AFHBA_AURORA_CTRL_ENA		(1<<31)
#define AFHBA_AURORA_CTRL_TXDIS		(1<<16)
#define AFHBA_AURORA_CTRL_CLR		(1<<7)
#define AFHBA_AURORA_CTRL_PWR_DWN	(1<<4)
#define AFHBA_AURORA_CTRL_LOOPBACK	(0x7)

#define AFHBA_AURORA_STAT_SFP_PRESENTn	(1<<31)
#define AFHBA_AURORA_STAT_SFP_LOS	(1<<30)
#define AFHBA_AURORA_STAT_SFP_TX_FAULT  (1<<29)
#define AFHBA_AURORA_STAT_HARD_ERR	(1<<6)
#define AFHBA_AURORA_STAT_SOFT_ERR	(1<<5)
#define AFHBA_AURORA_STAT_FRAME_ERR	(1<<4)
#define AFHBA_AURORA_STAT_CHANNEL_UP	(1<<1)
#define AFHBA_AURORA_STAT_LANE_UP	(1<<0)

#define AFHBA_AURORA_STAT_ERR \
	(AFHBA_AURORA_STAT_SFP_LOS|AFHBA_AURORA_STAT_SFP_TX_FAULT|\
	AFHBA_AURORA_STAT_HARD_ERR|AFHBA_AURORA_STAT_SOFT_ERR|\
	AFHBA_AURORA_STAT_FRAME_ERR)


#define HOST_MON_TEMP_SHL	20
#define HOST_MON_USR_TEMP	(1<<5)
#define HOST_MON_OVER_TEMP	(1<<4)
#define HOST_MON_VOLT_ALARM	0x7

/* BAR1 register definitions: enums are long-word offsets */
#define ZYNQ_BASE			0x0000

enum ZYNQ_REGS {
	Z_MOD_ID,		/* 0x0000 */
	Z_DMA_CTRL,		/* 0x0004 */
	Z_COMMS_HB,		/* 0x0008 */
	Z_AURORA_CTRL,		/* 0x000C */
	Z_AURORA_SR,		/* 0x0010 */
	Z_IDENT			/* 0x0014 */
};



#define PCIE_BASE			0x1000

enum PCIE_REGS {
	PCIE_CNTRL = 1,
	PCIE_INTR,
	PCI_CSR,
	PCIE_DEV_CSR,
	PCIE_LINK_CSR,
	PCIE_CONF,
	PCIE_BUFFER_CTRL
};

#define PCIE_CONF_AF_PORT_SHL		16

#define DMA_BASE			0x2000
enum DMA_REGS {
	/* 2000 */ DMA_TEST,
	/* 2004 */ DMA_CTRL,
	/* 2008 */ DMA_DATA_FIFSTA,
	/* 200c */ DMA_DESC_FIFSTA,
	/* 2010 */ DMA_PUSH_DESC_STA,
	/* 2014 */ DMA_PULL_DESC_STA,
	/* 2018 */ DMA_PUSH_DATACOUNT,
	/* 201c */ DMA_PUSH_GAP,
	/* 2020 */ DMA_PUSH_DESC_LEN,
	/* 2024 */ DMA_PULL_DESC_LEN,
	/* 2028 */ DMA_REGS_COUNT,
};



enum DMA_SEL {
	DMA_PUSH_SEL = 0x1,
	DMA_PULL_SEL = 0x2,
	DMA_BOTH_SEL = 0x3
};

inline static const char*  sDMA_SEL(enum DMA_SEL dma_sel)
{
	switch(dma_sel){
	case DMA_PUSH_SEL: return "PUSH";
	case DMA_PULL_SEL: return "PULL";
	case DMA_BOTH_SEL: return "PUSHPULL";
	default:	   return "none";
	}
}

#define DMA_PUSH_DESC_FIFO		0x2040
#define DMA_PULL_DESC_FIFO		0x2080



#define DMA_CTRL_PULL_SHL		16
#define DMA_CTRL_PUSH_SHL		0

#define DMA_CTRL_EN			0x0001
#define DMA_CTRL_FIFO_RST		0x0010
#define DMA_CTRL_LOW_LAT		0x0020
#define DMA_CTRL_RECYCLE		0x0040
#define DMA_CTRL_RAM			0x8000

#define DMA_DESCR_ADDR			0xfffffc00
#define DMA_DESCR_INTEN			0x00000100
#define DMA_DESCR_LEN			0x000000f0
#define DMA_DESCR_ID			0x0000000f

#define DMA_DATA_FIFO_EMPTY		0x0001
#define DMA_DATA_FIFO_FULL		0x0002
#define DMA_DATA_FIFO_UNDER		0x0004
#define DMA_DATA_FIFO_OVER		0x0008
#define DMA_DATA_FIFO_COUNT		0xfff0
#define DMA_DATA_FIFO_COUNT_SHL		4

#define DMA_DESCR_LEN_BYTES(descr)	((1<<((descr&DMA_DESCR_LEN)>>4))*1024)

#define DMA_PUSH_DESC_RAM		0x6000
#define DMA_PULL_DESC_RAM		0x7000


#define DMA_DIR_DESC_FIFO(dma_sel) \
	((dma_sel)==DMA_PUSH_SEL? DMA_PUSH_DESC_FIFO: DMA_PULL_DESC_FIFO)

#define DMA_DIR_DESC_LEN(dma_sel) \
	((dma_sel)==DMA_PUSH_SEL? DMA_PUSH_DESC_LEN: DMA_PULL_DESC_LEN)

#define DMA_DIR_DESC_RAM(dma_sel) \
	((dma_sel)==DMA_PUSH_SEL? DMA_PUSH_DESC_RAM: DMA_PULL_DESC_RAM)

static inline u32 dma_pp(enum DMA_SEL dma_sel, u32 bits)
{
	u32 xx = 0;
	if ((dma_sel&DMA_PUSH_SEL) != 0){
		xx |= bits << DMA_CTRL_PUSH_SHL;
	}
	if ((dma_sel&DMA_PULL_SEL) != 0){
		xx |= bits << DMA_CTRL_PULL_SHL;
	}
	return xx;
}


void afhba_write_reg(struct AFHBA_DEV *adev, int regoff, u32 value);
u32 afhba_read_reg(struct AFHBA_DEV *adev, int regoff);
/*
New "Special"  Host Register area at offset 0xf000

First Register is the Trigger Register - at 0xf000

Set the Trigger by writing a 1 to bit 0 to Set it back to 0 again afterwards similar to the SOFT_TRIGGER Bit on the A9 Side.
*/
#define COMMON_BASE			0xf000
enum COMMON_REGS {
	COM_SOFT_TRIGGER,
};

#define COM_SOFT_TRIGGER_EN		0x0001

static inline u32 read_astatus2(struct AFHBA_DEV *adev)
{
	return afhba_read_reg(adev, AURORA_STATUS2_REG(adev->sfp));
}

#ifdef CONFIG_GPU
extern void gpumem_init(struct AFHBA_DEV *adev);
extern int iommu_init(struct AFHBA_DEV *adev);
extern void afhba_free_iommu(struct AFHBA_DEV *adev);
extern void afhba_free_gpumem(struct AFHBA_DEV *adev);
extern long afhba_gpumem_lock(struct AFHBA_DEV *adev, unsigned long arg);
extern long afhba_gpumem_unlock(struct AFHBA_DEV *adev, unsigned long arg);
extern int addIommuMapProcFile(struct AFHBA_DEV *adev);
extern int addGpuMemProcFile(struct AFHBA_DEV *adev);
#endif

#endif /* ACQ_FIBER_HBA_H_ */
