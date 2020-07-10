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

#define _GNU_SOURCE
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

#include "afhba-llcontrol.h"
#define HB_FILE 	"/dev/rtm-t.%lu"
#define LOG_FILE	"afhba.%d.log"

#define HB1		"/dev/rtm-t.%d.data/hb01"

#define HB_LEN  0x100000		/* 1MB HOST BUFFERSW */

#include <time.h>

#define NS 1000000000
#define US 1000000
#define NSUS (NS/US)

#ifndef DEF_NCHAN
#define DEF_NCHAN 	16
#endif
int nchan = DEF_NCHAN;
int spadlongs = 16;


unsigned difftime_us(void)
/* return delta time in usec */
{
	static struct timespec ts0;
	struct timespec ts1;
	unsigned dt;

	if (clock_gettime(CLOCK_MONOTONIC, &ts1) != 0){
		perror("clock_gettime()");
		exit(1);
	}
	if (ts0.tv_sec != 0){
		struct timespec tsd;
		tsd.tv_sec = ts1.tv_sec - ts0.tv_sec;
		tsd.tv_nsec = ts1.tv_nsec - ts0.tv_nsec;
		dt = tsd.tv_sec*US + tsd.tv_nsec/NSUS;
	}else{
		dt = 0;
	}
	ts0 = ts1;
	return dt;
}

void* get_mapping(dev_t devnum, int *pfd) {
	char fname[80];
	int fd;
	void *host_buffer;

	sprintf(fname, HB_FILE, devnum);
	fd = open(fname, O_RDWR);
	if (fd < 0){
		perror(fname);
		exit(errno);
	}

	host_buffer = mmap(0, HB_LEN, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
	if (host_buffer == (caddr_t)-1 ){
		perror( "mmap" );
	        exit(errno);
	}
	memset(host_buffer, 0, HB_LEN);
	if(pfd) *pfd = fd;
	return host_buffer;
}

void clear_mapping(int fd, void* hb)
{
	munmap(hb, HB_LEN);
	close(fd);
}
void setAffinity(unsigned cpu_mask)
{
	int cpu = 0;
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        for (cpu = 0; cpu < 32; ++cpu){
                if ((1<<cpu) &cpu_mask){
                        CPU_SET(cpu, &cpu_set);
                }
        }
        printf("setAffinity: %d,%d,%d,%d\n",
                        CPU_ISSET(0, &cpu_set), CPU_ISSET(1, &cpu_set),
                        CPU_ISSET(2, &cpu_set), CPU_ISSET(3, &cpu_set)
                        );

        int rc = sched_setaffinity(0,  sizeof(cpu_set_t), &cpu_set);
        if (rc != 0){
                perror("sched_set_affinity");
                exit(1);
        }
}


int sched_fifo_priority;

void goRealTime(void)
{
	if (sched_fifo_priority < 1){
		return;
	}else{
		struct sched_param p = {};
		p.sched_priority = sched_fifo_priority;



		int rc = sched_setscheduler(0, SCHED_FIFO, &p);

		if (rc){
			perror("failed to set RT priority");
		}
	}
}

