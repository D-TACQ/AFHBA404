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

//#define _GNU_SOURCE
#include <sched.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <sched.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>

#include "afhba-get_shared_mapping.h"


#define HB_FILE "/dev/rtm-t.%d"
//#define LOG_FILE	"afhba.%d.log"

#define HB1	"/dev/rtm-t.%d.data/hb01"


#include <time.h>

#define NS 1000000000
#define US 1000000
#define NSUS (NS/US)

extern int *shm;

extern void shm_connect();


#define SHM_INTS	128

#define SHM_LEN 	(SHM_INTS*sizeof(int))

#define SHM_SAMPLE	0



#endif /* LLCONTROL_AFHBA_LLCONTROL_COMMON_H_ */
