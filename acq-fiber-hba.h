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

#endif /* ACQ_FIBER_HBA_H_ */
