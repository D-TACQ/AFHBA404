/** @file acq-fiber-hba.c
 *  @brief instantiates **kernel device driver**
 *
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
char afhba__driver_version[] = "B1113";
char afhba__copyright[] = "Copyright (c) 2010/2014 D-TACQ Solutions Ltd";


struct class* afhba_device_class;

LIST_HEAD(afhba_devices);


const char* afhba_devnames[MAXDEV];


#ifndef EXPORT_SYMTAB
#define EXPORT_SYMTAB
#include <linux/module.h>
#endif

int afhba_debug = 0;
module_param(afhba_debug, int, 0644);

/* deprecated: only retained for load script compatibility */
int ll_mode_only = 1;
module_param(ll_mode_only, int, 0444);


#include "d-tacq_pci_id.h"

int afhba4_stream = 0;
module_param(afhba4_stream, int, 0444);

int afhba_nports = 4;
module_param(afhba_nports, int, 0444);

int bad_bios_bar_limit = 0;
module_param(bad_bios_bar_limit, int, 0644);

extern int buffer_len;

static int MAP2BAR(struct AFHBA_DEV *adev, int imap)
{
	return imap;
}

static int minor2bar(struct AFHBA_DEV *adev, int iminor)
{
	switch(iminor){
	default:
		dev_err(pdev(adev), "bad call minor2bar %d", iminor); /* fallthru */
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
	struct AFHBA_DEV *adev = PD(file)->dev;
	int ii;
	int rc;
	void *va = adev->mappings[BAR].va;
	int len = adev->mappings[BAR].len;

	dev_dbg(pdev(adev), "01 adev %p va %p name %s", adev, va, adev->name);

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
	struct AFHBA_DEV *adev = PD(file)->dev;
	int LEN = adev->mappings[BAR].len;
	u32 data;
	void *va = adev->mappings[BAR].va + OFFSET;
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
		dev_dbg(pdev(adev), "writing %p = 0x%08x was 0x%08x",
					va+ii, data+ii, readval);

		writel(data+ii, va+ii);
	}

	*f_pos += count;
	return count;
}

int afhba_open(struct inode *inode, struct file *file)
{
	struct AFHBA_DEV *adev = afhba_lookupDevice(MAJOR(inode->i_rdev));

	dev_dbg(pdev(adev), "01");
	if (adev == 0){
		return -ENODEV;
	}else{
		file->private_data = kmalloc(PSZ, GFP_KERNEL);
		PD(file)->dev = adev;
		PD(file)->minor = MINOR(inode->i_rdev);
		INIT_LIST_HEAD(&PD(file)->my_buffers);

		dev_dbg(pdev(adev), "33: minor %d", PD(file)->minor);

		switch((PD(file)->minor)){
		case MINOR_REGREAD:
		case MINOR_REMOTE:
			return 0;
		default:
			if (adev->stream_fops != 0){
				file->f_op = adev->stream_fops;
				return file->f_op->open(inode, file);
			}else{
				dev_err(pdev(adev),"99 adev %p name %s", adev, adev->name);
				return -ENODEV;
			}
		}
	}
}
ssize_t afhba_read(struct file *file, char __user *buf, size_t count, loff_t *f_pos)
{
	struct AFHBA_DEV *adev = DEV(file);
	return bar_read(file, buf, count, f_pos, minor2bar(adev, PD(file)->minor));
}

ssize_t afhba_write(
	struct file *file, const char *buf, size_t count, loff_t *f_pos)
{
	return bar_write(file, buf, count, f_pos, REGS_BAR, 0);
}
int afhba_mmap_bar(struct file* file, struct vm_area_struct* vma)
{
	struct AFHBA_DEV *adev = PD(file)->dev;
	int bar = minor2bar(adev, PD(file)->minor);
	unsigned long vsize = vma->vm_end - vma->vm_start;
	unsigned long psize = adev->mappings[bar].len;
	unsigned pfn = adev->mappings[bar].pa >> PAGE_SHIFT;

	dev_dbg(pdev(adev), "%c vsize %lu psize %lu %s",
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

int afhba_mmap_hb(struct file* file, struct vm_area_struct* vma)
{
	struct AFHBA_DEV *adev = PD(file)->dev;
	struct HostBuffer* hb = adev->hb1;
	unsigned long vsize = vma->vm_end - vma->vm_start;
	unsigned long psize = buffer_len;
	unsigned pfn = hb->pa >> PAGE_SHIFT;

	dev_dbg(pdev(adev), "%c vsize %lu psize %lu %s",
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
	struct AFHBA_DEV *adev = PD(file)->dev;
	dev_dbg(pdev(adev), "01");
	kfree(file->private_data);
	return 0;
}

void afhba_map(struct AFHBA_DEV *adev)
{
	struct pci_dev *dev = adev->pci_dev;
	int imap;
	int nmappings = 0;

	for (imap = 0; nmappings < adev->map_count; ++imap){
		struct PciMapping* mp = adev->mappings+imap;
		int bar = MAP2BAR(adev, imap);

		dev_dbg(pdev(adev), "[%d] ", imap);
		if (VALID_BAR(bar)){
			if (adev->peer != 0 && adev->peer->mappings[imap].va != 0){
				adev->mappings[imap] = adev->peer->mappings[imap];
			}else{
				snprintf(mp->name, SZM1(mp->name), "afhba.%d.%d", adev->idx, bar);

				mp->pa = pci_resource_start(dev,bar)&
						PCI_BASE_ADDRESS_MEM_MASK;
				mp->len = pci_resource_len(dev, bar);
				mp->region = request_mem_region(
					mp->pa, mp->len, mp->name);
				mp->va = ioremap(mp->pa, mp->len);

				dev_dbg(pdev(adev), "BAR %d va:%p", bar, mp->va);
			}
			++nmappings;
		}
	}

	dev_dbg(pdev(adev), "99 nmappings %d", nmappings);
}



static int getOrder(int len)
{
	int order;
	len /= PAGE_SIZE;

	for (order = 0; 1 << order < len; ++order){
		;
	}
	return order;
}


static void init_buffers(struct AFHBA_DEV* adev)
{
	int order = getOrder(buffer_len);
	int ii = 0;
	struct HostBuffer* hb = adev->hb1 = kmalloc(sizeof(struct HostBuffer)*1, GFP_KERNEL);
	void *buf = (void*)__get_free_pages(GFP_KERNEL|GFP_DMA32, order);

	if (!buf){
		dev_err(pdev(adev), "failed to allocate buffer %d", ii);
		return;
	}

	dev_dbg(pdev(adev), "buffer %2d allocated at %p, map it", ii, buf);

	hb->ibuf = 0;
	hb->pa = dma_map_single(&adev->pci_dev->dev, buf,
			buffer_len, PCI_DMA_FROMDEVICE);
	hb->va = buf;
	hb->len = buffer_len;

	dev_dbg(pdev(adev), "buffer %2d allocated, map done", ii);

	hb->bstate = BS_EMPTY;

	dev_info(pdev(adev), "hb1 [%d] %p %08x %d %08x", ii,
			adev->hb1->va, adev->hb1->pa,
			adev->hb1->len, adev->hb1->descr);
}


struct AFHBA_DEV *adevCreate(struct pci_dev *dev)
{
	static int idx;
	struct AFHBA_DEV *adev = kzalloc(sizeof(struct AFHBA_DEV), GFP_KERNEL);

	static u64 dma_mask = DMA_BIT_MASK(32);

	adev->pci_dev = dev;
	adev->idx = idx++;
	dev->dev.dma_mask = &dma_mask;
	adev->remote_com_bar = NO_BAR;

	return adev;
}

void adevDelete(struct AFHBA_DEV* adev)
{
	kfree(adev);
}

#include <linux/pci_regs.h>

#define AFHBA_PCI_CMD (PCI_COMMAND_MASTER | PCI_COMMAND_MEMORY)

void force_busmaster_mode(struct AFHBA_DEV* adev)
{
	u16 cmd;

	/* Workaround for PCI problem when BIOS sets MMRBC incorrectly. */
	pci_read_config_word( adev->pci_dev, PCI_COMMAND, &cmd);
	if ((cmd & AFHBA_PCI_CMD) != AFHBA_PCI_CMD) {
		u16 new_cmd = cmd | AFHBA_PCI_CMD;
		pci_write_config_word( adev->pci_dev, PCI_COMMAND, new_cmd);
		dev_info(pdev(adev), "ENABLE BUS MASTER transactions %04x to %04x",
							cmd, new_cmd);
	}
}

int _afhba_probe(struct AFHBA_DEV* adev, int remote_bar,
		int (*stream_drv_init)(struct AFHBA_DEV* adev))
{
	static struct file_operations afhba_fops = {
		.open = afhba_open,
		.release = afhba_release,
		.read = afhba_read,
		.write = afhba_write,
		//.mmap = afhba_mmap_bar
		.mmap = afhba_mmap_hb
	};

	int rc;

	snprintf(adev->name, SZM1(adev->name), "afhba.%d", adev->idx);
	afhba_devnames[adev->idx] = adev->name;
	snprintf(adev->mon_name, SZM1(adev->name), "afhba-mon.%d", adev->idx);


	rc = register_chrdev(0, adev->name, &afhba_fops);
	if (rc < 0){
		dev_err(pdev(adev), "can't get major");
		kfree(adev);
		return rc;
	}else{
		adev->major = rc;
	}

	afhba_map(adev);
	adev->remote = adev->mappings[remote_bar].va;
	init_buffers(adev);
	afhba_registerDevice(adev);
	afhba_createDebugfs(adev);

	rc = pci_enable_device(adev->pci_dev);
	if (rc != 0){
		dev_warn(pdev(adev), "pci_enabled_device returned %d", rc);
	}
	force_busmaster_mode(adev);

	dev_dbg(pdev(adev), "pci_enable_device returns %d", rc);
	dev_info(pdev(adev), "FPGA revision: %08x",
			afhba_read_reg(adev, FPGA_REVISION_REG));

	adev->class_dev = device_create(
			afhba_device_class,			/* cls */
			NULL,					/* cls_parent */
			adev->idx,				/* "devt" */
			&adev->pci_dev->dev,			/* device */
			adev->name);
	afhba_create_sysfs_class(adev);

	dev_dbg(pdev(adev), "calling afhba_create_sysfs()");
	afhba_create_sysfs(adev);

	dev_dbg(pdev(adev), "calling stream_drv_init()");
	stream_drv_init(adev);

	dev_dbg(pdev(adev), "99 rc %d", rc);
	return rc;
}

int null_stream_drv_init(struct AFHBA_DEV* adev)
{
	dev_warn(pdev(adev), "null_stream_drv_init STUB");
	return 0;
}
#define STREAM		afhba_stream_drv_init
#define NOSTREAM	null_stream_drv_init

int afhba2_probe(struct AFHBA_DEV *adev)
{
	int rc;
	dev_info(pdev(adev), "AFHBA 4G 2-port firmware detected");
	adev->map_count = MAP_COUNT_4G2;
	adev->remote_com_bar = MAP_COUNT_4G2 -1;
	adev->sfp = SFP_A;

	if ((rc = _afhba_probe(adev, REMOTE_BAR, STREAM)) != 0){
		dev_err(pdev(adev), "ERROR failed to create first device");
		return rc;
	}else{
		struct AFHBA_DEV *adev2 = adevCreate(adev->pci_dev);
		adev2->map_count = MAP_COUNT_4G2;
		adev2->peer = adev;
		adev2->sfp = SFP_B;

		if ((rc = _afhba_probe(adev2, REMOTE_BAR2, STREAM)) != 0){
			dev_err(pdev(adev2), "ERROR failed to create second device");
			return rc;
		}
		return rc;
	}
}
int afhba4_probe(struct AFHBA_DEV *adev)
{
	int (*_init)(struct AFHBA_DEV* adev) = afhba4_stream? STREAM: NOSTREAM;
	int rc;
	int ib;

	dev_info(pdev(adev), "AFHBA404 detected");
	adev->map_count = MAP_COUNT_4G4;
	adev->remote_com_bar = MAP_COUNT_4G4-1;
	if (bad_bios_bar_limit){
		dev_warn(pdev(adev), "limiting BAR count to bad_bios_bar_limit=%d",
				bad_bios_bar_limit);
		adev->map_count = bad_bios_bar_limit;
	}
	adev->sfp = SFP_A;
	adev->ACR = AURORA_CONTROL_REGA;
	rc = _afhba_probe(adev, REMOTE_BAR, _init);
	if (rc!=0) return rc;


	for (ib = 1; ib < afhba_nports; ++ib){
		struct AFHBA_DEV *adev2 = adevCreate(adev->pci_dev);
		adev2->map_count = MAP_COUNT_4G4;
		adev2->peer = adev;
		adev2->sfp = SFP_A+ib;
		adev2->ACR = AURORA_CONTROL_REGA + ib*0x10;
		if ((rc = _afhba_probe(adev2, REMOTE_BAR+ib, _init)) != 0){
			dev_err(pdev(adev2), "ERROR failed to create device %d", ib);
				return rc;
		}
	}
	return 0;
}
int afhba_mtca_probe(struct AFHBA_DEV *adev)
{
	int (*_init)(struct AFHBA_DEV* adev) = afhba4_stream? STREAM: NOSTREAM;

	dev_info(pdev(adev), "AFHBA404 detected %s", afhba4_stream? "STREAM": "NOSTREAM");
	adev->map_count = MAP_COUNT_4G1;
	adev->remote_com_bar = MAP_COUNT_4G1-1;
	if (bad_bios_bar_limit){
		dev_warn(pdev(adev), "limiting BAR count to bad_bios_bar_limit=%d",
				bad_bios_bar_limit);
		adev->map_count = bad_bios_bar_limit;
	}
	adev->sfp = SFP_A;
	adev->ACR = AURORA_CONTROL_REGA;
	return _afhba_probe(adev, REMOTE_BAR, _init);
}

int afhba_probe(struct pci_dev *dev, const struct pci_device_id *ent)
{
	struct AFHBA_DEV *adev = adevCreate(dev);

	dev_info(pdev(adev), "AFHBA: subdevice : %04x\n", ent->subdevice);
	switch(ent->subdevice){
	case PCI_SUBDID_FHBA_2G:
		dev_err(pdev(adev), "AFHBA 2G FIRMWARE detected %04x", ent->subdevice);
		adevDelete(adev);
		return -1;
	case PCI_SUBDID_FHBA_4G_OLD:
		dev_err(pdev(adev), "AFHBA 4G OBSOLETE FIRMWARE detected %04x", ent->subdevice);
		adevDelete(adev);
		return -1;
	case PCI_SUBDID_FHBA_4G:
		dev_info(pdev(adev), "AFHBA 4G single port firmware detected");
		adev->map_count = MAP_COUNT_4G1;
		adev->sfp = SFP_A;
		return _afhba_probe(adev, REMOTE_BAR, STREAM);
	case PCI_SUBDID_FHBA_4G2:
		return afhba2_probe(adev);
	case PCI_SUBDID_FHBA_4G4:
		return afhba4_probe(adev);
	case PCI_SUBDID_HBA_KMCU:
		dev_info(pdev(adev), "KMCU detected");
		return afhba_mtca_probe(adev);
	case PCI_SUBDID_HBA_KMCU2:
		dev_info(pdev(adev), "KMCU2 detected");
		return afhba_mtca_probe(adev);
	default:
		return -ENODEV;
	}
}
void afhba_remove (struct pci_dev *dev)
{
	struct AFHBA_DEV *adev = afhba_lookupDevicePci(dev);

	if (adev){
		afhba_stream_drv_del(adev);
		afhba_removeDebugfs(adev);
		afhba_remove_sysfs(adev);
		afhba_remove_sysfs_class(adev);
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
		PCI_SUBVID_DTACQ, PCI_SUBDID_FHBA_2G, 0 },
	{ PCI_VENDOR_ID_XILINX, PCI_DEVICE_ID_XILINX_PCIE,
		PCI_SUBVID_DTACQ, PCI_SUBDID_FHBA_4G_OLD, 0 },
	{ PCI_VENDOR_ID_XILINX, PCI_DEVICE_ID_DTACQ_PCIE,
		PCI_SUBVID_DTACQ, PCI_SUBDID_FHBA_4G, 0 },
	{ PCI_VENDOR_ID_XILINX, PCI_DEVICE_ID_DTACQ_PCIE,
		PCI_SUBVID_DTACQ, PCI_SUBDID_FHBA_4G2, 0 },
	{ PCI_VENDOR_ID_XILINX, PCI_DEVICE_ID_DTACQ_PCIE,
		PCI_SUBVID_DTACQ, PCI_SUBDID_FHBA_4G4, 0 },
	{ PCI_VENDOR_ID_XILINX, PCI_DEVICE_ID_DTACQ_PCIE,
		PCI_SUBVID_DTACQ, PCI_SUBDID_HBA_KMCU, 0 },
	{ PCI_VENDOR_ID_XILINX, PCI_DEVICE_ID_DTACQ_PCIE,
		PCI_SUBVID_DTACQ, PCI_SUBDID_HBA_KMCU2, 0 },
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

	printk(KERN_INFO "%s %s %s\n%s\n",
	     afhba_driver_name, afhba__driver_string,
	     afhba__driver_version,
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

EXPORT_SYMBOL_GPL(afhba_devices);

MODULE_DEVICE_TABLE(pci, afhba_pci_tbl);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Peter.Milne@d-tacq.com");
MODULE_DESCRIPTION("D-TACQ ACQ-FIBER-HBA Driver for ACQ400");
