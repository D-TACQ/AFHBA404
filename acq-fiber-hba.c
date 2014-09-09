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




#include "acq-fiber-hba.h"


char afhba_driver_name[] = "afhba";
char afhba__driver_string[] = "D-TACQ ACQ-FIBER-HBA Driver for ACQ400";
char afhba__driver_version[] = "B1003";
char afhba__copyright[] = "Copyright (c) 2010/2014 D-TACQ Solutions Ltd";


struct class* afhba_device_class;

LIST_HEAD(devices);


const char* afhba_devnames[MAXDEV];


#ifndef EXPORT_SYMTAB
#define EXPORT_SYMTAB
#include <linux/module.h>
#endif

int afhba_debug = 0;
module_param(afhba_debug, int, 0644);

#define PCI_VENDOR_ID_XILINX      0x10ee
#define PCI_DEVICE_ID_XILINX_PCIE 0x0007
// D-TACQ changes the device ID to work around unwanted zomojo lspci listing */
#define PCI_DEVICE_ID_DTACQ_PCIE  0xadc1

#define PCI_SUBVID_DTACQ	0xd1ac
#define PCI_SUBDID_FHBA_4G	0x4101

#define REGS_BAR	0
#define REMOTE_BAR	1
#define NO_BAR 		-1

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
} G_hb;

static int MAP2BAR(struct AFHBA_DEV *tdev, int imap)
{
	switch(imap){
	case REMOTE_BAR:
	case REGS_BAR:
		return imap;
	default:
		return NO_BAR;
	}
}

static int minor2bar(int iminor)
{
	switch(iminor){
	default:
		err("bad call minor2bar %d", iminor); /* fallthru */
	case MINOR_REGREAD:
		return REGS_BAR;
	case MINOR_REMOTE:
		return REMOTE_BAR;
	}

}
#define VALID_BAR(bar)	((bar) != NO_BAR)


ssize_t bar_read(
	struct file *file, char *buf, size_t count, loff_t *f_pos, int BAR)
{
	struct AFHBA_DEV *tdev = PD(file)->dev;
	int ii;
	int rc;
	void *va = tdev->mappings[BAR].va;
	int len = tdev->mappings[BAR].len;

	dbg(2, "01 tdev %p va %p name %s", tdev, va, tdev->name);

	if (count > len){
		count = len;
	}
	if (*f_pos >= len){
		return 0;
	}else{
		va += *f_pos;
	}

	for (ii = 0; ii < count/sizeof(u32); ++ii){
		u32 reg = readl(va + ii*sizeof(u32));
		rc = copy_to_user(buf+ii*sizeof(u32), &reg, sizeof(u32));
		if (rc){
			return -1;
		}
	}
	*f_pos += count;
	return count;
}




ssize_t bar_write(
	struct file *file, const char *buf, size_t count, loff_t *f_pos,
	int BAR, int OFFSET)
{
	struct AFHBA_DEV *tdev = PD(file)->dev;
	int LEN = tdev->mappings[BAR].len;
	u32 data;
	void *va = tdev->mappings[BAR].va + OFFSET;
	int rc = copy_from_user(&data, buf, min(count, sizeof(u32)));

	int ii;
	if (rc){
		return -1;
	}
	if (*f_pos > LEN){
		return -1;
	}else if (count + *f_pos > LEN){
		count = LEN - *f_pos;
	}

	for (ii = 0; ii < count; ii += sizeof(u32)){
		u32 readval = readl(va+ii);
		dbg(2, "writing %p = 0x%08x was 0x%08x",
					va+ii, data+ii, readval);

		writel(data+ii, va+ii);
	}

	*f_pos += count;
	return count;
}

int afhba_open(struct inode *inode, struct file *file)
{
	struct AFHBA_DEV *tdev = afhba_lookupDevice(MAJOR(inode->i_rdev));

	dbg(2, "01");
	if (tdev == 0){
		return -ENODEV;
	}else{
		file->private_data = kmalloc(PSZ, GFP_KERNEL);
		PD(file)->dev = tdev;
		PD(file)->minor = MINOR(inode->i_rdev);
		INIT_LIST_HEAD(&PD(file)->my_buffers);

		dbg(2, "33: minor %d", PD(file)->minor);

		switch((PD(file)->minor)){
		case MINOR_REGREAD:
		case MINOR_REMOTE:
			return 0;
		default:
			dbg(2,"99 tdev %p name %s", tdev, tdev->name);
			return -ENODEV;
		}
	}
}
ssize_t afhba_read(struct file *file, char __user *buf, size_t count, loff_t *f_pos)
{
	return bar_read(file, buf, count, f_pos, minor2bar(PD(file)->minor));
}

ssize_t afhba_write(
	struct file *file, const char *buf, size_t count, loff_t *f_pos)
{
	return bar_write(file, buf, count, f_pos, REGS_BAR, 0);
}
int afhba_mmap_bar(struct file* file, struct vm_area_struct* vma)
{
	struct AFHBA_DEV *tdev = PD(file)->dev;
	int bar = minor2bar(PD(file)->minor);
	unsigned long vsize = vma->vm_end - vma->vm_start;
	unsigned long psize = tdev->mappings[bar].len;
	unsigned pfn = tdev->mappings[bar].pa >> PAGE_SHIFT;

	dbg(2, "%c vsize %lu psize %lu %s",
		'D', vsize, psize, vsize>psize? "EINVAL": "OK");

	if (vsize > psize){
		return -EINVAL;                   /* request too big */
	}
	if (io_remap_pfn_range(
		vma, vma->vm_start, pfn, vsize, vma->vm_page_prot)){
		return -EAGAIN;
	}else{
		return 0;
	}
}



int afhba_release(struct inode *inode, struct file *file)
{
	dbg(2, "01");
	kfree(file->private_data);
	return 0;
}

void afhba_map(struct AFHBA_DEV *tdev)
{
	struct pci_dev *dev = tdev->pci_dev;
	int imap;
	int nmappings = 0;

	for (imap = 0; nmappings < MAP_COUNT; ++imap){
		struct PciMapping* mp = tdev->mappings+imap;
		int bar = MAP2BAR(tdev, imap);

		if (VALID_BAR(bar)){
			sprintf(mp->name, "afhba.%d.%d", tdev->idx, bar);

			mp->pa = pci_resource_start(dev,bar)&
						PCI_BASE_ADDRESS_MEM_MASK;
			mp->len = pci_resource_len(dev, bar);
			mp->region = request_mem_region(
					mp->pa, mp->len, mp->name);
			mp->va = ioremap_nocache(mp->pa, mp->len);

			dbg(2, "BAR %d va:%p", bar, mp->va);
			++nmappings;
		}
	}
}

int nbuffers = 1;
int BUFFER_LEN = 0x100000;

static int getOrder(int len)
{
	int order;
	len /= PAGE_SIZE;

	for (order = 0; 1 << order < len; ++order){
		;
	}
	return order;
}


static void init_buffers(struct AFHBA_DEV* tdev)
{
	int ii;
	int order = getOrder(BUFFER_LEN);
	struct HostBuffer *hb = &G_hb;


	dbg(1, "allocating %d buffers size:%d dev.dma_mask:%08llx",
			nbuffers, BUFFER_LEN, *tdev->pci_dev->dev.dma_mask);

	for (ii = 0; ii < nbuffers; ++ii, ++tdev->nbuffers, ++hb){
		void *buf = (void*)__get_free_pages(GFP_KERNEL|GFP_DMA32, order);

		if (!buf){
			err("failed to allocate buffer %d", ii);
			break;
		}

		dbg(3, "buffer %2d allocated at %p, map it", ii, buf);

		hb->ibuf = ii;
		hb->pa = dma_map_single(&tdev->pci_dev->dev, buf,
				BUFFER_LEN, PCI_DMA_FROMDEVICE);
		hb->va = buf;
		hb->len = BUFFER_LEN;

		dbg(3, "buffer %2d allocated, map done", ii);

		hb->bstate = BS_EMPTY;

		info("[%d] %p %08x %d %08x",
		    ii, hb->va, hb->pa, hb->len, hb->descr);
	}
}
int afhba_probe(struct pci_dev *dev, const struct pci_device_id *ent)
{
	static struct file_operations afhba_fops = {
		.open = afhba_open,
		.release = afhba_release,
		.read = afhba_read,
		.write = afhba_write,
		.mmap = afhba_mmap_bar
	};

	struct AFHBA_DEV *tdev = kzalloc(sizeof(struct AFHBA_DEV), GFP_KERNEL);
	int rc;
	static int idx;
	static u64 dma_mask = DMA_BIT_MASK(32);


	tdev->pci_dev = dev;
	tdev->idx = idx++;
	dev->dev.dma_mask = &dma_mask;

	sprintf(tdev->name, "afhba.%d", tdev->idx);
	afhba_devnames[tdev->idx] = tdev->name;
	sprintf(tdev->mon_name, "afhba-mon.%d", tdev->idx);


	rc = register_chrdev(0, tdev->name, &afhba_fops);
	if (rc < 0){
		err("can't get major");
		kfree(tdev);
		return rc;
	}else{
		tdev->major = rc;
	}
	afhba_map(tdev);
	afhba_registerDevice(tdev);
	afhba_createDebugfs(tdev);

	tdev->class_dev = CLASS_DEVICE_CREATE(
			afhba_device_class,			/* cls */
			NULL,					/* cls_parent */
			tdev->idx,				/* "devt" */
			&tdev->pci_dev->dev,			/* device */
			tdev->name);

	rc = pci_enable_device(dev);
	dbg(1, "pci_enable_device returns %d", rc);

	init_buffers(tdev);
	return 0;
}
void afhba_remove (struct pci_dev *dev)
{
	struct AFHBA_DEV *tdev = afhba_lookupDevicePci(dev);

	if (tdev){
		afhba_removeDebugfs(tdev);
	}
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
	class_destroy(afhba_device_class);
	pci_unregister_driver(&afhba_driver);
}

module_init(afhba_init_module);
module_exit(afhba_exit_module);

MODULE_DEVICE_TABLE(pci, afhba_pci_tbl);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Peter.Milne@d-tacq.com");
MODULE_DESCRIPTION("D-TACQ ACQ-FIBER-HBA Driver for ACQ400");
