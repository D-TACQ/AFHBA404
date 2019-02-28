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

#include "afhba-get_shared_mapping.h"


#define HB_FILE 	"/dev/rtm-t.%d"
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


#define NSHORTS	(nchan+spadlongs*sizeof(unsigned)/sizeof(short))
#define VI_LEN 	(NSHORTS*sizeof(short))
#define SPIX	(nchan*sizeof(short)/sizeof(unsigned))			/* Scratch Pad IndeX */
/* ai_buffer is a local copy of host buffer */
#define CH01 (((volatile short*)ai_buffer)[0])
#define CH02 (((volatile short*)ai_buffer)[1])
#define CH03 (((volatile short*)ai_buffer)[2])
#define CH04 (((volatile short*)ai_buffer)[3])

#define TLATCH(lb) (&((volatile unsigned*)(lb))[SPIX])      /* actually, sample counter */
#define SPAD1	(((volatile unsigned*)ai_buffer)[SPIX+1])   /* user signal from ACQ */


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

extern int *shm;

extern void shm_connect();

int sched_fifo_priority = 1;

#define SHM_INTS	128

#define SHM_LEN 	(SHM_INTS*sizeof(int))

#define SHM_SAMPLE	0

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
	if(pfd) *pfd = fd;
	return host_buffer;
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


void goRealTime(void)
{
	struct sched_param p = {};
	p.sched_priority = sched_fifo_priority;



	int rc = sched_setscheduler(0, SCHED_FIFO, &p);

	if (rc){
		perror("failed to set RT priority");
	}
}

FILE* fp_log;

void write_action(void *data)
{
	fwrite(data, sizeof(short), NSHORTS, fp_log);
}

void setup_logging(int devnum)
{
	char logfile[80];
	sprintf(logfile, LOG_FILE, devnum);

	fp_log = fopen(logfile, "w");
	if (fp_log == 0){
		perror(logfile);
		exit(1);
	}

}

void check_tlatch_action(void *local_buffer)
{
	static unsigned tl0;
	static int errcount;
	short *ai_buffer = local_buffer;

	unsigned tl1 = *TLATCH(local_buffer);
	if (tl1 != tl0+1){
		if (++errcount < 100){
			printf("%d => %d\n", tl0, tl1);
		}else if (errcount == 100){
			printf("stop reporting at 100 errors ..\n");
		}
	}
	tl0 = tl1;
}

#endif /* LLCONTROL_AFHBA_LLCONTROL_COMMON_H_ */
