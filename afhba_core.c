/* ------------------------------------------------------------------------- */
/* afhba_core.c AFHBA						     */
/*
 * afhba_core.c
 *
 *  Created on: 24 Jan 2015
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

/* this fragment of code is linked into >1 module
 * debugs individually switched from module params
 * do not use dev_dbg - multiple instances same file..
 */
int reg_access_verbose;
module_param(reg_access_verbose, int, 0644);

#define VBS1	(reg_access_verbose >= 1)
#define VBS2	(reg_access_verbose >= 2)

void afhba_write_reg(struct AFHBA_DEV *adev, int regoff, u32 value)
{
	void* va = adev->mappings[REGS_BAR].va + regoff;
	if (VBS1) dev_info(pdev(adev), "%p = %08x", va + regoff, value);
	writel(value, va);
	if (VBS2) dev_info(pdev(adev), "%p : %08x", va, readl(va + regoff));
}

u32 afhba_read_reg(struct AFHBA_DEV *adev, int regoff)
{
	void* va = adev->mappings[REGS_BAR].va + regoff;
	u32 rv = readl(va + regoff);
	if (VBS1) dev_info(pdev(adev), "%p = %08x", va + regoff, rv);
	return rv;
}

