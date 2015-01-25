/* ------------------------------------------------------------------------- */
/* acq400_drv.c  D-TACQ ACQ400 FMC  DRIVER   
 * afs_procfs.c
 *
 *  Created on: 22 Jan 2015
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

#include <linux/seq_file.h>
#include <linux/proc_fs.h>

#include "acq-fiber-hba.h"
#include "afhba_stream_drv.h"

extern int nbuffers;
extern struct list_head afhba_devices;

struct proc_dir_entry *afs_proc_root;

static void *hb_seq_start(struct seq_file *sfile, loff_t *pos)
/* *pos == iblock. returns next TBLOCK* */
{
	if (*pos >= nbuffers){
		return NULL;
	}else{
		return pos;
	}
}

static void *hb_seq_next(struct seq_file *s, void *v, loff_t *pos)
{
	(*pos)++;
	if (*pos >= nbuffers){
		return NULL;
	}else{
		return pos;
	}
}



static void hb_seq_stop(struct seq_file *s, void *v) {}


static inline const char *BSTATE2S(enum BSTATE bstate)
{
	switch(bstate){
	case BS_EMPTY:		return "BS_EMPTY";
	case BS_FILLING:	return "BS_FILLING";
	case BS_FULL:		return "BS_FULL";
	case BS_FULL_APP:	return "BS_FULL_APP";
	default:		return "BS_ERROR";
	}
}


static int hb_seq_show(struct seq_file *sfile, void *v)
{
	struct file *file = (struct file *)sfile->private;
	struct AFHBA_DEV *adev = PDE_DATA(file_inode(file));
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	int ib = *(int*)v;
	struct HostBuffer *hb = sdev->hbx+ib;
	int len = 0;

	if (ib == 0){
		len = seq_printf(sfile, "ix va pa len req_len descr state\n");
	}
	len += seq_printf(sfile,
			 "[%02d] %p %08x %06x %06x %08x %s\n",
			 ib, hb->va, hb->pa, hb->len, hb->req_len,
			 hb->descr, BSTATE2S(hb->bstate));
	return len;
}

static int hbd_seq_show(struct seq_file *sfile, void *v)
{
	struct file *file = (struct file *)sfile->private;
	struct AFHBA_DEV *adev = PDE_DATA(file_inode(file));
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	int ib = *(int*)v;
	struct HostBuffer *hb = sdev->hbx+ib;
	int len = 0;

	len += seq_printf(sfile, "%08x\n", hb->descr);
	return len;
}




static int ab_seq_show(struct seq_file *sfile, void *v)
/* shows buffers that are FULL but NOT in FULL LIST */
{
	struct file *file = (struct file *)sfile->private;
	struct AFHBA_DEV *adev = PDE_DATA(file_inode(file));
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	int ib = *(int*)v;
	struct HostBuffer *hb = sdev->hbx+ib;
	int len = 0;

	if (ib == 0){
		len = seq_printf(sfile, "ix va pa len req_len descr state\n");
	}
	if (hb->bstate == BS_FULL_APP){
		len += seq_printf(sfile,
				 "[%02d] %p %08x %06x %06x %08x %s\n",
				 ib, hb->va, hb->pa, hb->len, hb->req_len,
				 hb->descr, BSTATE2S(hb->bstate));
	}
	return len;
}


static int __proc_open(
	struct inode *inode, struct file *file,
	struct seq_operations *_seq_ops)
{
	int rc = seq_open(file, _seq_ops);

	if (rc == 0){
		struct seq_file* seq_file =
			(struct seq_file*)file->private_data;
		seq_file->private = file;
	}

	return rc;
}

static int hb_proc_open(struct inode *inode, struct file *file)
{
	static struct seq_operations _seq_ops = {
		.start = hb_seq_start,
		.next  = hb_seq_next,
		.stop  = hb_seq_stop,
		.show  = hb_seq_show
	};
	return __proc_open(inode, file, &_seq_ops);
}
static int hbd_proc_open(struct inode *inode, struct file *file)
{
	static struct seq_operations _seq_ops = {
		.start = hb_seq_start,
		.next  = hb_seq_next,
		.stop  = hb_seq_stop,
		.show  = hbd_seq_show
	};
	return __proc_open(inode, file, &_seq_ops);
}

static int addHostBufferProcFiles(struct AFHBA_DEV *adev)
{
	static struct file_operations hb_proc_fops = {
		.owner = THIS_MODULE,
		.open = hb_proc_open,
		.read = seq_read,
		.llseek = seq_lseek,
		.release = seq_release
	};
	static struct file_operations hbd_proc_fops = {
		.owner = THIS_MODULE,
		.open = hbd_proc_open,
		.read = seq_read,
		.llseek = seq_lseek,
		.release = seq_release
	};
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	struct proc_dir_entry *hb_entry =
			proc_create_data("HostBuffers", S_IRUGO,
					sdev->proc_dir_root, &hb_proc_fops, adev);

	if (hb_entry){
		hb_entry = proc_create_data("HostDescriptors", S_IRUGO,
				sdev->proc_dir_root, &hbd_proc_fops, adev);
		if (hb_entry){
			return 0;
		}
	}

	dev_err(pdev(adev), "Failed to create entry");
	return -1;
}

static int ab_proc_open(struct inode *inode, struct file *file)
{
	static struct seq_operations _seq_ops = {
		.start = hb_seq_start,
		.next  = hb_seq_next,
		.stop  = hb_seq_stop,
		.show  = ab_seq_show
	};
	return __proc_open(inode, file, &_seq_ops);
}

static int addAppBufferProcFiles(struct AFHBA_DEV *adev)
{
	static struct file_operations ab_proc_fops = {
		.owner = THIS_MODULE,
		.open = ab_proc_open,
		.read = seq_read,
		.llseek = seq_lseek,
		.release = seq_release
	};
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	struct proc_dir_entry *ab_entry =
		proc_create_data("AppBuffers", S_IRUGO,
			sdev->proc_dir_root, &ab_proc_fops, adev);
	if (ab_entry){
		return 0;
	}

	dev_err(pdev(adev), "Failed to create entry");
	return -1;
}

int afs_init_procfs(struct AFHBA_DEV *adev)
{
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	int rc;

	if (!afs_proc_root){
		afs_proc_root = proc_mkdir("driver/afhba", NULL);
		assert(afs_proc_root);
	}

	sdev->proc_dir_root = proc_mkdir(adev->name, afs_proc_root);

	if ((rc = addHostBufferProcFiles(adev)) == 0 &&
	    (rc = addAppBufferProcFiles(adev))  == 0)
		return 0;

	return rc;
}
