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
#include "afhba_stream_drv.h"


static ssize_t show_job(
	struct device *dev,
	struct device_attribute *attr,
	char *buf)
{
	struct AFHBA_DEV *adev = afhba_lookupDev(dev);
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	struct JOB *job = &sdev->job;
	int bstates[NSTATES] = { 0, };
	int ii;
	int data_rate = job->rx_rate*sdev->req_len;

	if (data_rate > 0x100000){
		data_rate /= 0x100000;
	}
	for (ii = 0; ii != sdev->nbuffers; ++ii){
		bstates[sdev->hbx[ii].bstate]++;
	}


	return sprintf(buf,
		"dev=%s idx=%d demand=%d queued=%d "
		"rx=%d rx_rate=%d int_rate=%d "
		"MBPS=%d "
		"BS_EMPTY=%-2d BS_FILLING=%-2d BS_FULL=%-2d BS_FULL_APP=%-2d "
		"STATUS=%s ERRORS=%d\n",
		       adev->name, adev->idx,
		       job->buffers_demand,
		       job->buffers_queued,
		       job->buffers_received,
		       job->rx_rate,  job->int_rate,
		       data_rate,
		       bstates[0], bstates[1], bstates[2], bstates[3],
		       job->please_stop==PS_PLEASE_STOP? "PLEASE_STOP":
		       job->please_stop==PS_STOP_DONE? "STOP_DONE": "",
		       job->errors
		);
}

static DEVICE_ATTR(job, S_IRUGO, show_job, 0);


static ssize_t store_reset_buffers(
	struct device * dev,
	struct device_attribute *attr,
	const char *buf, size_t count)
{
	unsigned mode;

	if (sscanf(buf, "%u", &mode) > 0){
		if (mode == 1){
			afs_reset_buffers(afhba_lookupDev(dev));
		}
		return count;
	}else{
		return -EPERM;
	}
}

static DEVICE_ATTR(reset_buffers, S_IWUGO, 0, store_reset_buffers);


extern int buffer_len;

static ssize_t store_buffer_len(
	struct device * dev,
	struct device_attribute *attr,
	const char * buf, size_t count)
{
	int ll_length;
	struct AFHBA_STREAM_DEV *sdev = afhba_lookupDev(dev)->stream_dev;

	if (sscanf(buf, "%d", &ll_length) == 1 && ll_length > 0){
		sdev->buffer_len = min(ll_length, buffer_len);
		return strlen(buf);

	}else{
		return -1;
	}
}

static ssize_t show_buffer_len(
	struct device * dev,
	struct device_attribute *attr,
	char * buf)
{
	struct AFHBA_STREAM_DEV *sdev = afhba_lookupDev(dev)->stream_dev;

	sprintf(buf, "%d\n", sdev->buffer_len);
	return strlen(buf);
}

static DEVICE_ATTR(buffer_len, S_IRUGO|S_IWUGO, show_buffer_len, store_buffer_len);


char* getFlags(u32 stat, char buf[], int maxbuf)
{
	int cursor = 0;

	cursor += sprintf(buf+cursor, "%c%s ",
		(stat & AFHBA_AURORA_STAT_SFP_PRESENTn)? '-': '+',
				"PRESENT");
	if ((stat & AFHBA_AURORA_STAT_SFP_LOS) != 0){
		cursor += sprintf(buf+cursor, "LOS ");
	}
	if ((stat & AFHBA_AURORA_STAT_SFP_TX_FAULT) != 0){
		cursor += sprintf(buf+cursor, "TX_FAULT ");
	}
	if ((stat & AFHBA_AURORA_STAT_HARD_ERR) != 0){
		cursor += sprintf(buf+cursor, "HARD_ERR ");
	}
	if ((stat & AFHBA_AURORA_STAT_SOFT_ERR) != 0){
		cursor += sprintf(buf+cursor, "SOFT_ERR ");
	}
	if ((stat & AFHBA_AURORA_STAT_FRAME_ERR) != 0){
		cursor += sprintf(buf+cursor, "FRAME_ERR ");
	}
	if ((stat & AFHBA_AURORA_STAT_CHANNEL_UP) != 0){
		cursor += sprintf(buf+cursor, "+CHANNEL_UP ");
	}
	if ((stat & AFHBA_AURORA_STAT_LANE_UP) != 0){
		cursor += sprintf(buf+cursor, "+LANE_UP ");
	}
	strcat(buf, "\n");
	assert(cursor < maxbuf);
	return buf;
}

#define AURORA(SFPN) \
static ssize_t store_aurora##SFPN(					\
	struct device * dev,						\
	struct device_attribute *attr,					\
	const char * buf,						\
	size_t count)							\
{									\
	u32 ctrl = simple_strtoul(buf, 0, 16);				\
	afhba_write_reg(afhba_lookupDev(dev), AURORA_CONTROL_REG, ctrl);  \
	return count;							\
}									\
									\
									\
static ssize_t show_aurora##SFPN(					\
	struct device * dev,						\
	struct device_attribute *attr,					\
	char * buf)							\
{									\
	char flags[80];							\
	u32 stat = afhba_read_reg(afhba_lookupDev(dev), AURORA_STATUS_REG); \
	return sprintf(buf, "0x%08x %s\n", stat, getFlags(stat, flags, 80)); \
}									\
									\
									\
static DEVICE_ATTR(aurora, S_IRUGO|S_IWUGO, show_aurora##SFPN, store_aurora##SFPN);

AURORA(0);

static const struct attribute *dev_attrs[] = {
	&dev_attr_buffer_len.attr,
	&dev_attr_job.attr,
	&dev_attr_reset_buffers.attr,
	&dev_attr_aurora.attr,
	NULL
};

void afhba_create_sysfs(struct AFHBA_DEV *adev)
{
	int rc = sysfs_create_files(&adev->pci_dev->dev.kobj, dev_attrs);
	if (rc){
		dev_err(pdev(adev), "failed to create files");
		return;
	}
}

void afhba_remove_sysfs(struct AFHBA_DEV *adev)
{
	sysfs_remove_files(&adev->pci_dev->dev.kobj, dev_attrs);
}

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
	sysfs_remove_files(&adev->pci_dev->dev.kobj, class_attrs);
}
