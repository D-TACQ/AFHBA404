/* ------------------------------------------------------------------------- *
 * acq-fiber-hba.c  		                     	                    
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

#include "acq200_debug.h"
#include "lk-shim.h"

char afhba_driver_name[] = "afhba";
char afhba__driver_string[] = "D-TACQ ACQ-FIBER-HBA Driver for ACQ400";
char afhba__driver_version[] = "B1000";
char afhba__copyright[] = "Copyright (c) 2010/2014 D-TACQ Solutions Ltd";


struct class* afhba_device_class;

#ifndef EXPORT_SYMTAB
#define EXPORT_SYMTAB
#include <linux/module.h>
#endif

#define PCI_VENDOR_ID_XILINX      0x10ee
#define PCI_DEVICE_ID_XILINX_PCIE 0x0007
// D-TACQ changes the device ID to work around unwanted zomojo lspci listing */
#define PCI_DEVICE_ID_DTACQ_PCIE  0xadc1

#define PCI_SUBVID_DTACQ	0xd1ac
#define PCI_SUBDID_FHBA_4G	0x4101

int afhba_probe(struct pci_dev *dev, const struct pci_device_id *ent)
{
	return 0;
}
void afhba_remove (struct pci_dev *dev)
{
	/* @@todo ... work goes here */
	pci_disable_device(dev);
}
/*
 *
 * { Vendor ID, Device ID, SubVendor ID, SubDevice ID,
 *   Class, Class Mask, String Index }
 */
static struct pci_device_id afhba_pci_tbl[] = {
	{ PCI_VENDOR_ID_XILINX, PCI_DEVICE_ID_XILINX_PCIE,
		PCI_SUBVID_DTACQ, PCI_SUBDID_FHBA_4G, 0 },
	{ PCI_VENDOR_ID_XILINX, PCI_DEVICE_ID_DTACQ_PCIE,
		PCI_SUBVID_DTACQ, PCI_SUBDID_FHBA_4G, 0 },
        { }
};
static struct pci_driver afhba_driver = {
        .name     = afhba_driver_name,
        .id_table = afhba_pci_tbl,
        .probe    = afhba_probe,
//        .remove   = __devexit_p(afhba_remove),
	.remove = afhba_remove
};

int __init afhba_init_module(void)
{
	int rc;

	info("%s %s %s %s\n%s",
	     afhba_driver_name, afhba__driver_string,
	     afhba__driver_version,
	     __DATE__,
	     afhba__copyright);

	afhba_device_class = class_create(THIS_MODULE, "afhba");
	rc = pci_register_driver(&afhba_driver);
	return rc;
}

void afhba_exit_module(void)
{
	pci_unregister_driver(&afhba_driver);
}

module_init(afhba_init_module);
module_exit(afhba_exit_module);

MODULE_DEVICE_TABLE(pci, afhba_pci_tbl);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Peter.Milne@d-tacq.com");
MODULE_DESCRIPTION("D-TACQ ACQ-FIBER-HBA Driver for ACQ400");
