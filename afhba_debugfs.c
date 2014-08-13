/* ------------------------------------------------------------------------- *
 * afhba_debugfs.c  		                     	                    
 * ------------------------------------------------------------------------- *
 *   Copyright (C) 2014 Peter Milne, D-TACQ Solutions Ltd                
 *                      <peter dot milne at D hyphen TACQ dot com>          
 *                         www.d-tacq.com
 *   Created on: 11 Aug 2014  
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
#include <linux/debugfs.h>

struct dentry* afhba_debug_root;

#define NUM_REG_CREATE(dir, va, reg)					\
	sprintf(pcursor, "0x%04lx", reg);				\
	debugfs_create_x32(pcursor, S_IRUGO|S_IWUGO, dir, va+(reg));    \
	pcursor += strlen(pcursor) + 1;					\
	assert(pcursor-pbase < 4096)

void afhba_createDebugfs(struct AFHBA_DEV* adev)
{
	char* pcursor, *pbase;
	int ireg;
	if (!afhba_debug_root){
		afhba_debug_root = debugfs_create_dir("afhba", 0);
		if (!afhba_debug_root){
			dev_warn(pdev(adev), "failed create dir afhba");
			return;
		}
	}
	adev->debug_dir.top = debugfs_create_dir(
			afhba_devnames[adev->idx], afhba_debug_root);
	if (!adev->debug_dir.top){
		dev_warn(pdev(adev), "failed create dir %s", afhba_devnames[adev->idx]);
		return;
	}
	pbase = pcursor = adev->debug_names = kmalloc(4096, GFP_KERNEL);

	adev->debug_dir.loc = debugfs_create_dir("LOC", adev->debug_dir.top);
	if (!adev->debug_dir.loc){
		dev_warn(pdev(adev), "failed create dir %s", "LOC");
		return;
	}
	for (ireg = 0; ireg < 32; ++ireg){
		NUM_REG_CREATE(adev->debug_dir.loc, LOC(adev), ireg*sizeof(u32));
	}
	NUM_REG_CREATE(adev->debug_dir.loc, LOC(adev), 0x100*sizeof(u32));

	adev->debug_dir.rem = debugfs_create_dir("REM", adev->debug_dir.top);
	if (!adev->debug_dir.rem){
		dev_warn(pdev(adev), "failed create dir %s", "REM");
		return;
	}
	for (ireg = 0; ireg < 32; ++ireg){
		NUM_REG_CREATE(adev->debug_dir.rem, REM(adev), ireg*sizeof(u32));
	}
	NUM_REG_CREATE(adev->debug_dir.rem, REM(adev), 0x100*sizeof(u32));
}
void afhba_removeDebugfs(struct AFHBA_DEV* adev)
{
	debugfs_remove_recursive(adev->debug_dir.top);
	kfree(adev->debug_names);
}
