/** @file afs_procfs.c
*   @brief procfs interface, simple diagnostics
*
 * * ------------------------------------------------------------------------- */
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
#include <linux/dma-mapping.h>
#include <linux/iommu.h>

extern int nbuffers;
extern struct list_head afhba_devices;

struct proc_dir_entry *afs_proc_root;


#if LINUX_VERSION_CODE >= KERNEL_VERSION(5,8,0)
#define PROC_OPS(_name, _open, _read, _llseek, _release) 	\
        static struct proc_ops _name = {			\
		.proc_open = _open,				\
		.proc_read = _read,				\
		.proc_lseek = _llseek,				\
		.proc_release = _release			\
	}
#else
#define PROC_OPS(_name, _open, _read, _llseek, _release) 	\
	static struct file_operations _name = {			\
		.owner = THIS_MODULE,				\
		.open = _open,					\
		.read = _read,					\
		.llseek = _llseek,				\
		.release = _release				\
	}
#endif

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

	if (ib == 0){
		seq_printf(sfile, "ix va pa len req_len descr state\n");
	}
	seq_printf(sfile,
			 "[%02d] %p %08x %06x %06x %08x %s\n",
			 ib, hb->va, hb->pa, hb->len, hb->req_len,
			 hb->descr, BSTATE2S(hb->bstate));
	return 0;
}

static int hbd_seq_show(struct seq_file *sfile, void *v)
{
	struct file *file = (struct file *)sfile->private;
	struct AFHBA_DEV *adev = PDE_DATA(file_inode(file));
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	int ib = *(int*)v;
	struct HostBuffer *hb = sdev->hbx+ib;

	seq_printf(sfile, "%08x\n", hb->descr);
	return 0;
}




static int ab_seq_show(struct seq_file *sfile, void *v)
/* shows buffers that are FULL but NOT in FULL LIST */
{
	struct file *file = (struct file *)sfile->private;
	struct AFHBA_DEV *adev = PDE_DATA(file_inode(file));
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	int ib = *(int*)v;
	struct HostBuffer *hb = sdev->hbx+ib;

	if (ib == 0){
		seq_printf(sfile, "ix va pa len req_len descr state\n");
	}
	if (hb->bstate == BS_FULL_APP){
		seq_printf(sfile,
				 "[%02d] %p %08x %06x %06x %08x %s\n",
				 ib, hb->va, hb->pa, hb->len, hb->req_len,
				 hb->descr, BSTATE2S(hb->bstate));
	}
	return 0;
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
	PROC_OPS(hb_proc_fops, hb_proc_open, seq_read, seq_lseek, seq_release);
	PROC_OPS(hbd_proc_fops, hbd_proc_open, seq_read, seq_lseek, seq_release);

	struct proc_dir_entry *hb_entry =
			proc_create_data("HostBuffers", S_IRUGO,
					adev->proc_dir_root, &hb_proc_fops, adev);

	if (hb_entry){
		hb_entry = proc_create_data("HostDescriptors", S_IRUGO,
				adev->proc_dir_root, &hbd_proc_fops, adev);
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
	PROC_OPS(ab_proc_fops, ab_proc_open, seq_read, seq_lseek, seq_release);

	struct proc_dir_entry *ab_entry =
		proc_create_data("AppBuffers", S_IRUGO,
			adev->proc_dir_root, &ab_proc_fops, adev);
	if (ab_entry){
		return 0;
	}

	dev_err(pdev(adev), "Failed to create entry");
	return -1;
}

static int job_proc_show(struct seq_file *m, void *v)
 {
	struct file *file = (struct file *)m->private;
	struct AFHBA_DEV *adev = PDE_DATA(file_inode(file));
	struct AFHBA_STREAM_DEV *sdev = adev->stream_dev;
	struct JOB *job = &sdev->job;
	int bstates[NSTATES] = { 0, };
	int ii;
	int data_rate = job->rx_rate*sdev->req_len;

	if (data_rate > 0x100000){
		data_rate /= 0x100000;
	}

	for (ii = 0; ii != nbuffers; ++ii){
		int bs = sdev->hbx[ii].bstate;
		if (bs < 0 || bs > NSTATES-1){
			dev_warn(pdev(adev), "bstate[%d] %d out of range", ii, bs);
		}else{
			bstates[bs]++;
		}
	}

        seq_printf(m,
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
        return 0;
 }

static int job_proc_open(struct inode *inode, struct file *file)
{
        return single_open(file, job_proc_show, file);
}
static int addJobProcFile(struct AFHBA_DEV *adev)
{
	PROC_OPS(job_proc_fops, job_proc_open, seq_read, seq_lseek, single_release);

	if (proc_create_data("Job", S_IRUGO,
			adev->proc_dir_root, &job_proc_fops, adev) != 0){
		return 0;
	}else{
		dev_err(pdev(adev), "Failed to create entry");
		return -1;
	}
}


/* @TODO: warning ASSUMES this table struct printout fits 4K */
static int iommu_proc_show(struct seq_file *m, void *v)
{
        struct file *file = (struct file *)m->private;
        struct AFHBA_DEV *adev = PDE_DATA(file_inode(file));
        struct iommu_mapping *cursor;

        list_for_each_entry(cursor, &adev->iommu_map_list, list){
                seq_printf(m, "iova 0x%08lx -> 0x%016llx len:%lx dir:%s\n",
                                cursor->iova, cursor->paddr, cursor->size,
                                        cursor->prot==IOMMU_WRITE? "IOMMU_WRITE":
                                        cursor->prot==IOMMU_READ?  "IOMMU_READ" :
                                        "IOMMU_ERROR");
        }
        return 0;
}
static int iommu_proc_open(struct inode *inode, struct file *file)
{
        return single_open(file, iommu_proc_show, file);
}

static int addIommuMapProcFile(struct AFHBA_DEV *adev)
{
	PROC_OPS(iommu_proc_fops, iommu_proc_open, seq_read, seq_lseek, single_release);

        if (proc_create_data("IOMMU_maps", S_IRUGO,
                        adev->proc_dir_root, &iommu_proc_fops, adev) != 0){
                return 0;
        }else{
                dev_err(pdev(adev), "Failed to create entry");
                return -1;
        }
}


int afs_init_procfs(struct AFHBA_DEV *adev)
{
	int rc;

	if (!afs_proc_root){
		afs_proc_root = proc_mkdir("driver/afhba", NULL);
		assert(afs_proc_root);
	}

	adev->proc_dir_root = proc_mkdir(adev->name, afs_proc_root);

	if ((rc = addHostBufferProcFiles(adev)) == 0 &&
	    (rc = addAppBufferProcFiles(adev))  == 0 &&
	    (rc = addIommuMapProcFile(adev))  	== 0 &&
#ifdef CONFIG_GPU
	    (rc = addGpuMemProcFile(adev))  	== 0 &&
#endif
	    (rc = addJobProcFile(adev) 		== 0) == 0)

	{
		return 0;
	}

	return rc;
}
