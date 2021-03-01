

#ifndef GPUMEM_H
#define GPUMEM_H

//-----------------------------------------------------------------------------

#include <linux/cdev.h>
#include <linux/sched.h>
#include <linux/semaphore.h>

#include "nv-p2p.h"

//-----------------------------------------------------------------------------

struct gpumem_t {
	struct list_head list;
	void *handle;
	u64 virt_start;
	nvidia_p2p_page_table_t* page_table;
	char name[8];
};

struct iommu_mapping {
	struct list_head list;
	unsigned long iova;
	phys_addr_t paddr;
	size_t size;
	int prot;
};
//-----------------------------------------------------------------------------

struct gpumem {
    struct semaphore         sem;
    struct proc_dir_entry*   proc;
    struct list_head         table_list;
    struct list_head         map_list;
};

//-----------------------------------------------------------------------------

int get_nv_page_size(int val);

//-----------------------------------------------------------------------------

#endif
