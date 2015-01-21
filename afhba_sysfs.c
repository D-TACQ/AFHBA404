/* ------------------------------------------------------------------------- */
/* acq400_drv.c  D-TACQ ACQ400 FMC  DRIVER   
 * afhba_sysfs.c
 *
 *  Created on: 20 Jan 2015
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

#include "acq-fiber-hba.h"


static ssize_t show_dev(
	struct device * dev,
	struct device_attribute *attr,
	char * buf)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);
	if (adev){
		return sprintf(buf, "%d:0\n", adev->major);
	}else{
		return -ENODEV;
	}
}
static DEVICE_ATTR(dev, S_IRUGO, show_dev, 0);

static const struct attribute *class_attrs[] = {
	&dev_attr_dev.attr,
	NULL
};

void afhba_create_sysfs_class(struct AFHBA_DEV *adev)
{
	int rc;
	dev_dbg(pdev(adev), "01");

	rc = sysfs_create_files(&adev->class_dev->kobj, class_attrs);
	if (rc){
		dev_err(pdev(adev), "failed to create files");
		return;
	}
	rc = sysfs_create_link(
		&adev->class_dev->kobj, &adev->pci_dev->dev.kobj, "device");
	if (rc) {
		dev_err(pdev(adev), "failed to create symlink %s\n", "device");
	}
	dev_dbg(pdev(adev), "9");
}

void afhba_remove_sysfs_class(struct AFHBA_DEV *adev)
{
	device_remove_file(adev->class_dev, &dev_attr_dev);
}
