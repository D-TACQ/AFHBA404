/** @file afhba_sysfs.c
 *  @brief sysfs interface aka "knobs"
 */
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



static char* getDataFifoStat(u32 stat, char buf[], int maxbuf)
{
	int cursor = 0;


	cursor += sprintf(buf+cursor, "%4d ",
			(stat&DMA_DATA_FIFO_COUNT)>>DMA_DATA_FIFO_COUNT_SHL);

	if ((stat & DMA_DATA_FIFO_EMPTY) != 0){
		cursor += sprintf(buf+cursor, "EMPTY ");
	}
	if ((stat & DMA_DATA_FIFO_FULL) != 0){
		cursor += sprintf(buf+cursor, "FULL ");
	}
	if ((stat & DMA_DATA_FIFO_UNDER) != 0){
		cursor += sprintf(buf+cursor, "UNDER ");
	}
	if ((stat & DMA_DATA_FIFO_OVER) != 0){
		cursor += sprintf(buf+cursor, "EMPTY ");
	}
	strcat(buf, "\n");
	assert(cursor < maxbuf);
	return buf;
}

#define DATA_FIFO_STAT(DIR, SHL) 					\
static ssize_t show_data_fifo_stat_##DIR(				\
	struct device * dev,						\
	struct device_attribute *attr,					\
	char * buf)							\
{									\
	char flags[80];							\
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);			\
	u32 stat = (DMA_DATA_FIFSTA_RD(adev) >> SHL)&0xffff;		\
	getDataFifoStat(stat, flags, 80);				\
	return sprintf(buf, "0x%04x %s\n", stat, flags);		\
}									\
									\
static DEVICE_ATTR(data_fifo_stat_##DIR, (S_IRUSR|S_IRGRP), show_data_fifo_stat_##DIR, 0)

DATA_FIFO_STAT(pull, DMA_CTRL_PULL_SHL);
DATA_FIFO_STAT(push, DMA_CTRL_PUSH_SHL);

#define DESC_FIFO_STAT(DIR, SHL) 					\
static ssize_t show_desc_fifo_stat_##DIR(				\
	struct device * dev,						\
	struct device_attribute *attr,					\
	char * buf)							\
{									\
	char flags[80];							\
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);			\
	u32 stat = (DMA_DESC_FIFSTA_RD(adev) >> SHL)&0xffff;		\
	getDataFifoStat(stat, flags, 80);				\
	return sprintf(buf, "0x%04x %s\n", stat, flags);		\
}									\
									\
static DEVICE_ATTR(desc_fifo_stat_##DIR, (S_IRUSR|S_IRGRP), show_desc_fifo_stat_##DIR, 0)

DESC_FIFO_STAT(pull, DMA_CTRL_PULL_SHL);
DESC_FIFO_STAT(push, DMA_CTRL_PUSH_SHL);

static char* getDmaCtrl(u32 stat, char buf[], int maxbuf)
{
	int cursor = 0;
	buf[0] = '\0';

	if ((stat & DMA_CTRL_EN) != 0){
		cursor += sprintf(buf+cursor, "ENABLE ");
	}
	if ((stat & DMA_CTRL_FIFO_RST) != 0){
		cursor += sprintf(buf+cursor, "RESET ");
	}
	if ((stat & DMA_CTRL_LOW_LAT) != 0){
		cursor += sprintf(buf+cursor, "LOWLAT ");
	}
	if ((stat & DMA_CTRL_RECYCLE) != 0){
		cursor += sprintf(buf+cursor, "RECYCLE ");
	}
	strcat(buf, "\n");
	assert(cursor < maxbuf);
	return buf;
}

#define DMA_CTRL(DIR, SHL) 						\
static ssize_t show_dma_ctrl_##DIR(					\
	struct device * dev,						\
	struct device_attribute *attr,					\
	char * buf)							\
{									\
	char flags[80];							\
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);			\
	u32 stat = (DMA_CTRL_RD(adev) >> SHL)&0xffff;			\
	getDmaCtrl(stat, flags, 80);					\
	return sprintf(buf, "0x%04x %s\n", stat, flags);		\
}									\
									\
static DEVICE_ATTR(dma_ctrl_##DIR, (S_IRUSR|S_IRGRP), show_dma_ctrl_##DIR, 0)

DMA_CTRL(pull, DMA_CTRL_PULL_SHL);
DMA_CTRL(push, DMA_CTRL_PUSH_SHL);

static char* getDesc(u32 descr, char buf[], int maxbuf)
{
	int cursor = 0;

	cursor += sprintf(buf+cursor, "pa=%08x ", descr&DMA_DESCR_ADDR);

	if ((descr & DMA_DESCR_INTEN) != 0){
		cursor += sprintf(buf+cursor, "INTEN ");
	}
	cursor += sprintf(buf+cursor, "lenb=%08x ", DMA_DESCR_LEN_BYTES(descr));
	cursor += sprintf(buf+cursor, "id=%x\n", descr&DMA_DESCR_ID);
	assert(cursor < maxbuf);
	return buf;
}

#define DMA_LATEST(DIR, RD) 						\
static ssize_t show_dma_latest_##DIR(					\
	struct device * dev,						\
	struct device_attribute *attr,					\
	char * buf)							\
{									\
	char flags[80];							\
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);			\
	u32 descr = RD(adev);						\
	getDesc(descr, flags, 80);					\
	return sprintf(buf, "0x%08x %s\n", descr, flags);		\
}									\
									\
static DEVICE_ATTR(dma_latest_##DIR##_desc, (S_IRUSR|S_IRGRP), show_dma_latest_##DIR, 0)

DMA_LATEST(pull, DMA_PULL_DESC_STA_RD);
DMA_LATEST(push, DMA_PUSH_DESC_STA_RD);




static ssize_t store_reset_buffers(
	struct device * dev,
	struct device_attribute *attr,
	const char *buf, size_t count)
{
	unsigned mode;

	if (sscanf(buf, "%u", &mode) > 0){
		if (mode == 1){
			afs_reset_buffers(afhba_lookupDeviceFromClass(dev));
		}
		return count;
	}else{
		return -EPERM;
	}
}

static DEVICE_ATTR(reset_buffers, (S_IWUSR|S_IWGRP), 0, store_reset_buffers);


extern int buffer_len;

static ssize_t store_buffer_len(
	struct device * dev,
	struct device_attribute *attr,
	const char * buf, size_t count)
{
	int ll_length;
	struct AFHBA_STREAM_DEV *sdev = afhba_lookupDeviceFromClass(dev)->stream_dev;

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
	struct AFHBA_STREAM_DEV *sdev = afhba_lookupDeviceFromClass(dev)->stream_dev;

	sprintf(buf, "%d\n", sdev->buffer_len);
	return strlen(buf);
}

static DEVICE_ATTR(buffer_len, (S_IRUSR|S_IRGRP)|(S_IWUSR|S_IWGRP), show_buffer_len, store_buffer_len);


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

static ssize_t store_aurora(
	struct device * dev,
	struct device_attribute *attr,
	const char * buf, size_t count)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);
	int cr_off = AURORA_CONTROL_REG(adev->sfp);
	u32 ctrl = simple_strtoul(buf, 0, 16);

	afhba_write_reg(adev, cr_off, ctrl);
	return count;
}

static ssize_t show_aurora(
	struct device * dev,
	struct device_attribute *attr,
	char * buf)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);
	int sr_off = AURORA_STATUS_REG(adev->sfp);
	int cr_off = AURORA_CONTROL_REG(adev->sfp);
	char flags[80];

	u32 stat = afhba_read_reg(adev, sr_off);

       if ((stat&AFHBA_AURORA_STAT_ERR) != 0){
                u32 ctrl = afhba_read_reg(adev, cr_off);
                afhba_write_reg(adev, cr_off, ctrl|AFHBA_AURORA_CTRL_CLR);
                afhba_write_reg(adev, cr_off, ctrl);
        }
        return sprintf(buf, "0x%08x %s\n", stat, getFlags(stat, flags, 80)); \
}


static DEVICE_ATTR(aurora, (S_IRUGO)|(S_IWUSR|S_IWGRP), show_aurora, store_aurora);

static ssize_t show_aurora_ext(
	struct device * dev,
	struct device_attribute *attr,
	char * buf)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);

	return sprintf(buf, "0x%08x\n", read_astatus2(adev));
}
static DEVICE_ATTR(aurora_ext, (S_IRUGO), show_aurora_ext, 0);

static ssize_t store_comms_init(
	struct device * dev,
	struct device_attribute *attr,
	const char * buf, size_t count)
{
	int init;
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;

	if (sscanf(buf, "%d", &init) == 1 && init==1){
		sdev->comms_init_done = false;
		afs_comms_init(adev);
		return strlen(buf);
	}else{
		return -1;
	}
}

static ssize_t show_comms_init(
	struct device * dev,
	struct device_attribute *attr,
	char * buf)
{
	struct AFHBA_STREAM_DEV *sdev = afhba_lookupDeviceFromClass(dev)->stream_dev;

	return sprintf(buf, "%d\n", sdev->comms_init_done);
}

static DEVICE_ATTR(comms_init, (S_IRUGO)|(S_IWUSR|S_IWGRP), show_comms_init, store_comms_init);


static ssize_t show_inflight(
		struct device * dev,
		struct device_attribute *attr,
		char * buf)
{
	struct AFHBA_STREAM_DEV *sdev = afhba_lookupDeviceFromClass(dev)->stream_dev;
	struct JOB *job = &sdev->job;

	return sprintf(buf, "%d\n", job->buffers_queued-job->buffers_received);
}

static DEVICE_ATTR(inflight, (S_IRUGO), show_inflight, 0);

static ssize_t show_shot(
		struct device * dev,
		struct device_attribute *attr,
		char * buf)
{
	struct AFHBA_STREAM_DEV *sdev = afhba_lookupDeviceFromClass(dev)->stream_dev;
	return sprintf(buf, "%d\n", sdev->shot);
}

static DEVICE_ATTR(shot, (S_IRUGO), show_shot, 0);

#define _ERRLAT(yymask) 	((yymask) == 0x0ffff || (yymask) == 0)
#define ERRLAT(yy) 		(_ERRLAT((yy)&0x0ffff))

static ssize_t show_latstat(
		struct device * dev,
		struct device_attribute *attr,
		char * buf)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);
	unsigned ls1 = afhba_read_reg(adev, HOST_PCIE_LATSTATS_1);
	unsigned ls2 = afhba_read_reg(adev, HOST_PCIE_LATSTATS_2);
	int maxtry = 1;

	unsigned over5 = ls1 & 0x0ffff;
	while (over5 && (ERRLAT(ls2>>16) || ERRLAT(ls2))){
		yield();
		ls2 = afhba_read_reg(adev, HOST_PCIE_LATSTATS_2);
		if (++maxtry > 100){
			dev_warn(dev, "show_latstat maxtry exceeded");
			break;
		}
	}
	return sprintf(buf, "%5d %5d %5d %5d\n",
			ls1>>16, ls1&0x0ffff, ls2>>16, ls2&0x0ffff);
}

static DEVICE_ATTR(latstat, (S_IRUSR|S_IRGRP), show_latstat, 0);

static ssize_t show_fpga_rev(
		struct device * dev,
		struct device_attribute *attr,
		char * buf)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);
	return sprintf(buf, "0x%08x\n", afhba_read_reg(adev, FPGA_REVISION_REG));
}

static DEVICE_ATTR(fpga_rev, (S_IRUGO), show_fpga_rev, 0);


static ssize_t store_dma_test(
	struct device * dev,
	struct device_attribute *attr,
	const char * buf, size_t count)
{
	unsigned test;
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);

	if (sscanf(buf, "0x%x", &test) == 1 || sscanf(buf, "%d", &test) == 1){
		DMA_TEST_WR(adev, test);
		return strlen(buf);
	}else{
		return -1;
	}
}

static ssize_t show_dma_test(
	struct device * dev,
	struct device_attribute *attr,
	char * buf)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);

	sprintf(buf, "0x%08x\n", DMA_TEST_RD(adev));
	return strlen(buf);
}

static DEVICE_ATTR(dma_test, (S_IRUGO)|(S_IWUSR|S_IWGRP), show_dma_test, store_dma_test);


static ssize_t show_heartbeat(
		struct device * dev,
		struct device_attribute *attr,
		char * buf)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);
	return sprintf(buf, "0x%08x\n", afhba_read_reg(adev, HOST_COUNTER_REG));
}

static DEVICE_ATTR(heartbeat, (S_IRUGO), show_heartbeat, 0);


static ssize_t store_push_dma_timeouts(
	struct device * dev,
	struct device_attribute *attr,
	const char * buf, size_t count)
{
	struct AFHBA_STREAM_DEV *sdev = afhba_lookupDeviceFromClass(dev)->stream_dev;
	int clear;

	if (sscanf(buf, "%d", &clear) == 1 && clear == 1){
		sdev->push_dma_timeouts = 0;
		return strlen(buf);

	}else{
		return -1;
	}
}

static ssize_t show_push_dma_timeouts(
	struct device * dev,
	struct device_attribute *attr,
	char * buf)
{
	struct AFHBA_STREAM_DEV *sdev = afhba_lookupDeviceFromClass(dev)->stream_dev;

	sprintf(buf, "%d\n", sdev->push_dma_timeouts);
	return strlen(buf);
}

static DEVICE_ATTR(push_dma_timeouts, (S_IRUGO)|(S_IWUSR|S_IWGRP), show_push_dma_timeouts, store_push_dma_timeouts);

static ssize_t store_pull_dma_timeouts(
	struct device * dev,
	struct device_attribute *attr,
	const char * buf, size_t count)
{
	struct AFHBA_STREAM_DEV *sdev = afhba_lookupDeviceFromClass(dev)->stream_dev;
	int clear;

	if (sscanf(buf, "%d", &clear) == 1 && clear == 1){
		sdev->pull_dma_timeouts = 0;
		return strlen(buf);

	}else{
		return -1;
	}
}

static ssize_t show_pull_dma_timeouts(
	struct device * dev,
	struct device_attribute *attr,
	char * buf)
{
	struct AFHBA_STREAM_DEV *sdev = afhba_lookupDeviceFromClass(dev)->stream_dev;

	sprintf(buf, "%d\n", sdev->pull_dma_timeouts);
	return strlen(buf);
}

static DEVICE_ATTR(pull_dma_timeouts, (S_IRUGO)|(S_IWUSR|S_IWGRP), show_pull_dma_timeouts, store_pull_dma_timeouts);

static ssize_t store_host_test(
	struct device * dev,
	struct device_attribute *attr,
	const char * buf, size_t count)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);
	unsigned tv;

	if (sscanf(buf, "%x", &tv) == 1){
		afhba_write_reg(adev, HOST_TEST_REG, tv);
		return strlen(buf);
	}else{
		return -1;
	}
}

static ssize_t show_host_test(
	struct device * dev,
	struct device_attribute *attr,
	char * buf)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);

	sprintf(buf, "%08x\n", afhba_read_reg(adev, HOST_TEST_REG));
	return strlen(buf);
}

static DEVICE_ATTR(host_test, (S_IRUGO)|(S_IWUSR|S_IWGRP), show_host_test, store_host_test);

static ssize_t show_host_temp(
	struct device * dev,
	struct device_attribute *attr,
	char * buf)
{
	struct AFHBA_DEV *adev = afhba_lookupDeviceFromClass(dev);
        unsigned reg = afhba_read_reg(adev, HOST_MON_REG);
        int TC100 = (reg>>HOST_MON_TEMP_SHL)*50398/4096 - 27315;

	sprintf(buf, "%08x T:%d.%02d C %s %s %s\n",
			reg, TC100/100, TC100%100,
			(reg&HOST_MON_USR_TEMP)? "USR_TEMP ALARM":"",
			(reg&HOST_MON_OVER_TEMP)? "OVER TEMP ALARM": "",
			(reg&HOST_MON_VOLT_ALARM)? "VOLTAGE ALARM": "");
	return strlen(buf);
}
static DEVICE_ATTR(host_temp, (S_IRUGO), show_host_temp, 0);

static ssize_t show_streamer_pid(
        struct device * dev,
        struct device_attribute *attr,
        char * buf)
{
	struct AFHBA_STREAM_DEV *sdev = 
		afhba_lookupDeviceFromClass(dev)->stream_dev;

        sprintf(buf, "%d\n", sdev->pid);
        return strlen(buf);
}
static DEVICE_ATTR(streamer_pid, (S_IRUGO), show_streamer_pid, 0 );


static const struct attribute *dev_attrs[] = {
	&dev_attr_buffer_len.attr,
	&dev_attr_dma_test.attr,
	&dev_attr_inflight.attr,
	&dev_attr_reset_buffers.attr,
	&dev_attr_aurora.attr,
	&dev_attr_aurora_ext.attr,
	&dev_attr_data_fifo_stat_push.attr,
	&dev_attr_data_fifo_stat_pull.attr,
	&dev_attr_desc_fifo_stat_push.attr,
	&dev_attr_desc_fifo_stat_pull.attr,
	&dev_attr_dma_ctrl_push.attr,
	&dev_attr_dma_ctrl_pull.attr,
	&dev_attr_dma_latest_push_desc.attr,
	&dev_attr_dma_latest_pull_desc.attr,
	&dev_attr_comms_init.attr,
	&dev_attr_shot.attr,
	&dev_attr_latstat.attr,
	&dev_attr_fpga_rev.attr,
	&dev_attr_heartbeat.attr,
	&dev_attr_host_test.attr,
	&dev_attr_host_temp.attr,
        &dev_attr_streamer_pid.attr,
	&dev_attr_pull_dma_timeouts.attr,
	&dev_attr_push_dma_timeouts.attr,
	NULL
};



void afhba_create_sysfs(struct AFHBA_DEV *adev)
{
	int rc;
	rc = sysfs_create_files(&adev->class_dev->kobj, dev_attrs);
	if (rc){
		dev_err(pdev(adev), "failed to create files");
		return;
	}
}

void afhba_remove_sysfs(struct AFHBA_DEV *adev)
{
	sysfs_remove_files(&adev->class_dev->kobj, dev_attrs);
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
static DEVICE_ATTR(dev, (S_IRUSR|S_IRGRP), show_dev, 0);

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
