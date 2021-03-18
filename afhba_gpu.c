/*
 * afhba404_gpu.c
 *
 *  Created on: 1 Mar 2021
 *      Author: pgm
 */

#include <linux/seq_file.h>
#include <linux/proc_fs.h>
#include <linux/poll.h>
#include <linux/version.h>
#include <linux/dma-mapping.h>
#include <linux/iommu.h>

#include "acq-fiber-hba.h"
#include "afhba_stream_drv.h"

#ifdef CONFIG_GPU


/*------------------------------------------------------------------------------
-   GPU DMA functions defined here:
-   Implementation helped by reading https://github.com/karakozov/gpudma
-  What currently works:
-    afhba_gpumem_lock pins the GPU memory and the afhba404 device can access.
-    the callback function correctly deallocates the memory.
-  What currently does not work:
-    gpumem_lock does not correctly create a record of what memory is pinned.
-    this stops gpumem_unlock and gpumem_state from working correctly.
-    gpumem_unlock not working means the driver is reliant on the free_nvp_callback
-      working properly.  This is called when the program using the GPU terminates.
------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------
-   free_nvp_callback:
-     callback function to release memory when the gpu dma is turned off.
-     this function is called when either:
-       the program deallocates the memory (cuMemFree, etc.)
-       the program terminates
-     good practices should not rely on this, but current implementation does
-   TODO: add iommu unmap calls to turn off the dma remapping from 32 to 64 bit
------------------------------------------------------------------------------*/
void free_nvp_callback(void *data)
{
	dev_info(0, "%s(): stub - free on release\n", __FUNCTION__);
}


void afhba_free_gpumem(struct AFHBA_DEV *adev)
{
	struct gpumem_t *entry;
	struct gpumem_t *cursor;
	list_for_each_entry_safe(entry, cursor, &adev->gpumem.table_list, list){
		int rc = nvidia_p2p_free_page_table(entry->page_table);
		if (rc == 0){
			dev_info(pdev(adev), "%s(): nvidia_p2p_free_page_table() - OK!\n", __FUNCTION__);
		}else{
			dev_err(pdev(adev), "%s(): Error in nvidia_p2p_free_page_table()\n", __FUNCTION__);
		}

		list_del(&entry->list);
		kfree(entry);
	}
}

int get_nv_page_size(int val)
{
    switch(val) {
    case NVIDIA_P2P_PAGE_SIZE_4KB: return 4*1024;
    case NVIDIA_P2P_PAGE_SIZE_64KB: return 64*1024;
    case NVIDIA_P2P_PAGE_SIZE_128KB: return 128*1024;
    }
    return 0;
}
/*------------------------------------------------------------------------------
-  gpu_pin:
-    Function to pin the specified gpu virtual address and get the physical address
-    Currently unused, this is done manually for channel A in afhba_gpumem_unlock
-    Should be correctly implemented to generalize the gpu memory pinning for
-      multiple channel usage.
-    TODO: get gpumem->table_list allocated correctly as a static struct?
------------------------------------------------------------------------------*/
int gpu_pin(struct AFHBA_DEV *adev, const char* name,
	struct nvidia_p2p_dma_mapping ** nv_dma_map,
	uint64_t addr, uint64_t size, size_t *ppin_size){
	// gpu_pin function is currently unused, this is done manually inside afhba_gpumem_lock
	// should be separated out to generalize the gpu memory pinning
	int error = 0;
	size_t pin_size = 0ULL;
	struct gpumem_t *entry = (struct gpumem_t*)kzalloc(sizeof(struct gpumem_t), GFP_KERNEL);
	if(!entry) {
		dev_err(pdev(adev), "%s(): Error allocate memory to mapping struct\n", __FUNCTION__);
		return -ENOMEM;
	}
	INIT_LIST_HEAD(&entry->list);
	strncpy(entry->name, name, sizeof(entry->name)-1);
	entry->virt_start = (addr & GPU_BOUND_MASK);
	pin_size = addr + size - entry->virt_start;
	if(!pin_size) {
		printk(KERN_ERR"%s(): Error invalid memory size!\n", __FUNCTION__);
		error = -EINVAL;
		goto do_free_mem;
	}else{
		*ppin_size = pin_size;
	}

	dev_info(pdev(adev), "%s %s addr=%llx, size=%llx, virt_start=%llx, pin_size=%lx",
			__FUNCTION__, name, addr, size, entry->virt_start, pin_size);

	error = nvidia_p2p_get_pages(0, 0, entry->virt_start, pin_size, &entry->page_table, free_nvp_callback, entry);
	if(error != 0) {
		dev_err(pdev(adev), "%s(): Error in nvidia_p2p_get_pages()\n", __FUNCTION__);
		error = -EINVAL;
		goto do_unlock_pages;
	}

	dev_info(pdev(adev),"%s %s Pinned GPU memory, physical address is %llx",
			__FUNCTION__, name, entry->page_table->pages[0]->physical_address);

	error = nvidia_p2p_dma_map_pages(adev->pci_dev, entry->page_table, nv_dma_map);

	if(error){
		dev_err(pdev(adev), "%s(): Error %d in nvidia_p2p_dma_map_pages()\n", __FUNCTION__,error);
		error = -EFAULT;
		goto do_unmap_dma;
	}

	dev_info(pdev(adev),"%s %s nvidia_dma_mapping: npages= %d\n",
			__FUNCTION__, name, (*nv_dma_map)->entries);

	list_add_tail(&entry->list, &adev->gpumem.table_list);
	return 0;

do_unmap_dma:
	nvidia_p2p_dma_unmap_pages(adev->pci_dev, entry->page_table, *nv_dma_map);
do_unlock_pages:
	nvidia_p2p_put_pages(0, 0, entry->virt_start, entry->page_table);
do_free_mem:
	kfree(entry);
	return (long) error;
}

#define MARK	printk(KERN_ALERT "DEBUG: %s %d\n",__FUNCTION__,__LINE__);


void gpumem_init(struct AFHBA_DEV *adev)
{
	struct gpumem* gdev = &adev->gpumem;

	gdev->proc = 0;
	sema_init(&gdev->sem, 1);
	INIT_LIST_HEAD(&gdev->table_list);
}

long __afhba_gpumem_lock(
		struct AFHBA_DEV *adev, const char* name,
		unsigned long iova, uint64_t addr, uint64_t size, unsigned prot)
{
	struct nvidia_p2p_dma_mapping *nv_dma_map = 0;
	size_t pin_size = 0ULL;

	dev_info(pdev(adev), "Original %s HostBuffer physical address is %lx.\n", name, iova);
	dev_info(pdev(adev), "Virtual %s GPU address is  %llx.\n", name, addr);

	if (gpu_pin(adev, name, &nv_dma_map, addr, size, &pin_size)){
		dev_err(pdev(adev), "%s(): Error in gpu_pin()", __FUNCTION__);
	   	return -EFAULT;
	}

	//  Enable iommu DMA remapping -> AFHBA404 card can only address 32-bit memory
	if (afhba_iommu_map(adev, iova, nv_dma_map->dma_addresses[0], pin_size, prot)){
		dev_err(pdev(adev), "iommu_map failed -- aborting.\n");
		return -EFAULT;
	}else{
		dev_info(pdev(adev), "iommu_map success %s iova %lx..%llx points to %llx",
				name, iova, iova+size, nv_dma_map->dma_addresses[0]);
	}
	return 0;
}

/*------------------------------------------------------------------------------
-   afhba_gpumem_lock:
-     called from an ioctl call, user provides virtual address in gpu memory.
-     nvidia_p2p_get_pages provides us with the physical address we want to
-     redirect the dma push address to.
------------------------------------------------------------------------------*/
long afhba_gpumem_lock(struct AFHBA_DEV *adev, unsigned long arg)
{
	struct gpudma_lock_t param;
	long rc;

	if (!adev->iom_dom){
		dev_err(pdev(adev), "%s(): NO IOMMU\n", __FUNCTION__);
		return -1;
	}

	if(copy_from_user(&param, (void *)arg, sizeof(struct gpudma_lock_t))) {
		dev_err(pdev(adev), "%s(): Error in copy_from_user()\n", __FUNCTION__);
		return -EFAULT;
	}

	if ((rc = __afhba_gpumem_lock(adev, "VI",
			adev->stream_dev->hbx[param.ind_ai].pa,
			param.addr_ai, param.size_ai, IOMMU_WRITE)) != 0){
		dev_err(pdev(adev), "afhba_gpumem_lock failed to lock %s %ld", "VI", rc);
		return rc;
	}
	if ((rc = __afhba_gpumem_lock(adev, "VO",
			adev->stream_dev->hbx[param.ind_ao].pa,
			param.addr_ao, param.size_ao, IOMMU_READ)) != 0){
		dev_err(pdev(adev), "afhba_gpumem_lock failed to lock %s %ld", "VO", rc);
		return rc;
	}
	return 0;
}

/*------------------------------------------------------------------------------
-   afhba_gpumem_unlock:
-     called from an ioctl call, user provides virtual address in gpu memory.
-     nvidia_p2p_put_pages allows us to unlock the addresses we previously locked
-     this currently does not work, the table of locked memory is not working right
------------------------------------------------------------------------------*/
long afhba_gpumem_unlock(struct AFHBA_DEV *adev, unsigned long arg)
{
    int error = -EINVAL;
    struct gpumem_t *entry = 0;
    struct gpudma_unlock_t param;
    struct list_head *pos, *n;

    if(copy_from_user(&param, (void *)arg, sizeof(struct gpudma_unlock_t))) {
        printk(KERN_ERR"%s(): Error in copy_from_user()\n", __FUNCTION__);
        error = -EFAULT;
        goto do_exit;
    }

    list_for_each_safe(pos, n, &adev->gpumem.table_list) {

        entry = list_entry(pos, struct gpumem_t, list);
        if(entry) {
            if(entry->handle == param.handle) {

                printk(KERN_ERR"%s(): param.handle = %p\n", __FUNCTION__, param.handle);
                printk(KERN_ERR"%s(): entry.handle = %p\n", __FUNCTION__, entry->handle);

                if(entry->virt_start && entry->page_table) {
                    error = nvidia_p2p_put_pages(0, 0, entry->virt_start, entry->page_table);
                    if(error != 0) {
                        printk(KERN_ERR"%s(): Error in nvidia_p2p_put_pages()\n", __FUNCTION__);
                        goto do_exit;
                    }
                    printk(KERN_ERR"%s(): nvidia_p2p_put_pages() - Ok!\n", __FUNCTION__);
                }

                list_del(pos);
                kfree(entry);
                break;
            } else {
                printk(KERN_ERR"%s(): Skip entry: %p\n", __FUNCTION__, entry->handle);
            }
        }
    }
    // TODO: add iommu domain unmapping here, to turn off the 32 to 64 bit address translation

do_exit:
    return (long) error;
}


/* @TODO: warning ASSUMES this table struct printout fits 4K */
static int gpu_proc_show(struct seq_file *m, void *v)
{
	struct file *file = (struct file *)m->private;
	struct AFHBA_DEV *adev = PDE_DATA(file_inode(file));
	struct gpumem_t *cursor;
	int ii;

	list_for_each_entry(cursor, &adev->gpumem.table_list, list){
		seq_printf(m, "%s 0x%016llx\n", cursor->name, cursor->virt_start);
		if (cursor->page_table){
			for (ii = 0; ii < cursor->page_table->entries; ii++) {
				seq_printf(m, "%s [%d] 0x%llx\n", cursor->name, ii,
					cursor->page_table->pages[ii]->physical_address);
			}
		}else{
			seq_printf(m, "%s ERROR: memory NOT pinned\n", cursor->name);
		}
	}
	return 0;
}
static int gpu_proc_open(struct inode *inode, struct file *file)
{
	return single_open(file, gpu_proc_show, file);
}

int addGpuMemProcFile(struct AFHBA_DEV *adev)
{
	static struct file_operations gpu_proc_fops = {
			.owner = THIS_MODULE,
			.open = gpu_proc_open,
			.read = seq_read,
			.llseek = seq_lseek,
			.release = single_release
	};
	if (proc_create_data("GPU_mem", S_IRUGO,
			adev->proc_dir_root, &gpu_proc_fops, adev) != 0){
		return 0;
	}else{
		dev_err(pdev(adev), "Failed to create entry");
		return -1;
	}
}

#endif /* CONFIG_GPU */

