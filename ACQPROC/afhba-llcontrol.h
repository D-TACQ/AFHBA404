/* ------------------------------------------------------------------------- */
/* afhba-llcontrol-common.h  D-TACQ ACQ400 FMC  DRIVER                                   
 * Project: AFHBA404
 * Created: 5 Mar 2018  			/ User: pgm
 * ------------------------------------------------------------------------- *
 *   Copyright (C) 2018 Peter Milne, D-TACQ Solutions Ltd         *
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
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.                *
 *
 * TODO 
 * TODO
/* ------------------------------------------------------------------------- */

#ifndef LLCONTROL_AFHBA_LLCONTROL_COMMON_H_
#define LLCONTROL_AFHBA_LLCONTROL_COMMON_H_

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sched.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>

unsigned difftime_us(void);
void* get_mapping(dev_t devnum, int *pfd);
void clear_mapping(int fd, void* hb);
void setAffinity(unsigned cpu_mask);
void goRealTime(void);



#endif /* LLCONTROL_AFHBA_LLCONTROL_COMMON_H_ */
