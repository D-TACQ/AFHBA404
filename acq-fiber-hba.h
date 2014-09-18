/* ------------------------------------------------------------------------- *
 * acq-fiber-hba.h  		                     	                    
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

#include <asm/uaccess.h>  /* VERIFY_READ|WRITE */


#define MINOR_REGREAD	253
#define MINOR_REMOTE	247

#define MAXDEV	8	/* maximum expected devices .. */

#define acq200_debug afhba_debug
#include "acq200_debug.h"
#include "lk-shim.h"

struct AFHBA_DEV;



#define MAP_COUNT	2



extern struct list_head devices;

int afhba_registerDevice(struct AFHBA_DEV *tdev);
void afhba_deleteDevice(struct AFHBA_DEV *tdev);
struct AFHBA_DEV* afhba_lookupDevice(int major);
struct AFHBA_DEV *afhba_lookupDeviceFromClass(struct CLASS_DEVICE *dev);
struct AFHBA_DEV* afhba_lookupDevicePci(struct pci_dev *pci_dev);
struct AFHBA_DEV* afhba_lookupDev(struct device *dev);

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
};


struct AFHBA_DEV {
	char name[16];
	char mon_name[16];
	char slot_name[16];
	struct pci_dev *pci_dev;
	struct CLASS_DEVICE *class_dev;
	int idx;
	int major;
	struct list_head list;

	struct PciMapping {
		int bar;
		u32 pa;
		void* va;
		unsigned len;
		struct resource *region;
		char name[32];
	} mappings[MAP_COUNT];

	struct proc_dir_entry *proc_dir_root;
	struct dentry* debug_dir;
	char *debug_names;

	int nbuffers;
	struct HostBuffer **hb;
};


struct AFHBA_DEV_PATH {
	int minor;
	struct AFHBA_DEV *dev;
	struct list_head my_buffers;
	void* private;
};


#define PSZ	  (sizeof (struct AFHBA_DEV_PATH))
#define PD(file)  ((struct AFHBA_DEV_PATH *)(file)->private_data)
#define DEV(file) (PD(file)->dev)

#define pdev(adev) (&(adev)->pci_dev->dev)

#define LOC(adev) ((adev)->mappings[0].va)
#define REM(adev) ((adev)->mappings[1].va)

#include "afhba_debugfs.h"

/* REGS */
#define FPGA_REVISION_REG		0x0000	/* FPGA Revision Register */
#define HOST_PCIE_CONTROL_REG 		0x0004	/* Host PCIe Control / Status Register */
#define HOST_PCIE_INTERRUPT_REG 	0x0008	/* Host PCIe Interrupt Register Register */
#define PCI_CONTROL_STATUS_REG 		0x000C	/* PCI Control / Status Register */
#define PCIE_DEVICE_CONTROL_STATUS_REG 	0x0010	/* PCIe Control / Status Register */
#define PCIE_LINK_CONTROL_STATUS_REG 	0x0014	/* PCIe Link Control / Status Register */
#define HOST_TEST_REG 			0x0018	/* Host Test Register */
#define HOST_COUNTER_REG 		0x001C	/* Heart Beat Counter */
#define HOST_PCIE_DEBUG_REG 		0x0020	/* Host PCIe Debug Register */
#define HOST_SPI_FLASH_CONTROL_REG 	0x0040	/* SPI FLASH Control Register */
#define HOST_SPI_FLASH_DATA_REG 	0x0044	/* SPI FLASH Data Register */
#define AURORA_CONTROL_REG 		0x0080	/* Aurora Control Register */
#define AURORA_STATUS_REG 		0x0084	/* Aurora Status Register */
#define SFP_I2C_DATA_REG 		0x0088	/* SFP I2C Control and Data Register */
#define HOST_COMMS_FIFO_CONTROL_REG 	0x00C0	/* ACQ400 Receive Communications FIFO Control Register */
#define HOST_COMMS_FIFO_STATUS_REG 	0x00C4	/* ACQ400 Receive Communications FIFO Status Register */
#define ACQ400_COMMS_READ 		0x0400	/* ACQ400 Receive Communications FIFO data */


#endif /* ACQ_FIBER_HBA_H_ */
