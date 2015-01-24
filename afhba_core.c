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

void afhba_write_reg(struct AFHBA_DEV *adev, int regoff, u32 value)
{
	void* va = adev->mappings[REGS_BAR].va + regoff;
	dev_dbg(pdev(adev), "%p = %08x", va + regoff, value);
	writel(value, va);
	dev_dbg(pdev(adev), "%p : %08x", va, readl(va + regoff));
}

u32 afhba_read_reg(struct AFHBA_DEV *adev, int regoff)
{
	void* va = adev->mappings[REGS_BAR].va + regoff;
	u32 rv = readl(va + regoff);
	dev_dbg(pdev(adev), "%p = %08x", va + regoff, rv);
	return rv;
}
