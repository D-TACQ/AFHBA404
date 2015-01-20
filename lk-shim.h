/* ------------------------------------------------------------------------- */
/** @file lk-shim.h hides API changes for a range of Linux Kernels.        */
/* ------------------------------------------------------------------------- */
/*   Copyright (C) 2010 Peter Milne, D-TACQ Solutions Ltd
 *                      <Peter dot Milne at D hyphen TACQ dot com>
                                                                               
    This program is free software; you can redistribute it and/or modify
    it under the terms of Version 2 of the GNU General Public License
    as published by the Free Software Foundation;
                                                                               
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
                                                                               
    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.                */
/* ------------------------------------------------------------------------- */

#include <linux/version.h>
/*
 LINUX_VERSION_CODE is embedded in the headers:
eg
./include/linux/version.h:#define LINUX_VERSION_CODE 132626

use KERNEL_VERSION(a,b,c) to select on different kernels

*/

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,18)
#error This kernel not supported. Must be 2.6.18 or higher

#elif LINUX_VERSION_CODE > KERNEL_VERSION(2,6,30)
#ifdef EXPORT_SYMTAB		/* main module */
//#pragma message "Kernel > 2.6.30 .. building current version"
#endif

#elif LINUX_VERSION_CODE == KERNEL_VERSION(2,6,18)
#ifdef EXPORT_SYMTAB		/* main module */
#pragma message "building for kernel 2.6.18"
//extern int pci_enable_msi_block(struct pci_dev *dev, unsigned int nvec);
#define pci_enable_msi_block(dev, nvec) pci_enable_msi(dev)

#define NVEC	1

#ifdef __RTM_T_HOSTDRV_C__
static irqreturn_t rtm_t_rx_isr(int irq, void *data);

static irqreturn_t __rtm_t_rx_isr(
	int irq, void *data, struct pt_regs *not_used)
{
	return rtm_t_rx_isr(irq, data);
}

#define RTM_T_RX_ISR	__rtm_t_rx_isr
#endif

#endif

#elif LINUX_VERSION_CODE == KERNEL_VERSION(2,6,20)
#ifdef EXPORT_SYMTAB		/* main module */
#pragma message "building for kernel 2.6.20"
//extern int pci_enable_msi_block(struct pci_dev *dev, unsigned int nvec);
#define pci_enable_msi_block(dev, nvec) pci_enable_msi(dev)
#endif

#else
#warning Unknown kernel version source code change may be required.

#endif

#ifndef RTM_T_RX_ISR
#define RTM_T_RX_ISR rtm_t_rx_isr
#endif

/* kernel compatibility ugliness */
#if (LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,27))

#define CLASS_DEVICE		device
#define CLASS_DEVICE_CREATE	device_create
#define CLASS_DEVICE_ATTR	DEVICE_ATTR
#define CLASS_DEVICE_CREATE_FILE DEVICE_CREATE_FILE
#else
#define ORIGINAL_CLASS_DEVICE_INTERFACE	1
#define CLASS_DEVICE		class_device
#define CLASS_DEVICE_CREATE	class_device_create

#endif

