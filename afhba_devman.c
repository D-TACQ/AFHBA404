/* ------------------------------------------------------------------------- *
 * afhba_devman.c  		                     	                    
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


#include "acq-fiber-hba.h"

int afhba_registerDevice(struct AFHBA_DEV *adev)
{
	dev_dbg(pdev(adev), "name %s", adev->name);
	list_add_tail(&adev->list, &afhba_devices);
	return 0;
}

void afhba_deleteDevice(struct AFHBA_DEV *adev)
{
	list_del(&adev->list);
	kfree(adev);
}
struct AFHBA_DEV* afhba_lookupDevice(int major)
{
	struct AFHBA_DEV *pos;

	list_for_each_entry(pos, &afhba_devices, list){
		if (pos->major == major){
			return pos;
		}
	}
	BUG();
	return 0;
}

struct AFHBA_DEV *afhba_lookupDeviceFromClass(struct device *dev)
{
	struct AFHBA_DEV *pos;

	list_for_each_entry(pos, &afhba_devices, list){
		if (pos->class_dev == dev){
			return pos;
		}
	}
	BUG();
	return 0;
}

struct AFHBA_DEV* afhba_lookupDevicePci(struct pci_dev *pci_dev)
{
	struct AFHBA_DEV *pos;

	list_for_each_entry(pos, &afhba_devices, list){
		if (pos->pci_dev == pci_dev){
			return pos;
		}
	}
	BUG();
	return 0;
}

struct AFHBA_DEV* afhba_lookupDev(struct device *dev)
{
	struct AFHBA_DEV *pos;

	list_for_each_entry(pos, &afhba_devices, list){
		if (&pos->pci_dev->dev == dev){
			return pos;
		}
	}
	BUG();
	return 0;
}
