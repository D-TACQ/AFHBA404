/* ------------------------------------------------------------------------- *
 * afhba-llcontrol-example.c
 * ------------------------------------------------------------------------- *
 *   Copyright (C) 2014 Peter Milne, D-TACQ Solutions Ltd
 *                      <peter dot milne at D hyphen TACQ dot com>
 *                         www.d-tacq.com
 *   Created on: 18 September 2014
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

#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>

#define HB_FILE "/dev/afhba.0.loc"
#define HB_LEN	0x1000

#define LOG_FILE	"afhba.log"

void* host_buffer;
int fd;
int nsamples = 10000000;		/* 10s at 1MSPS */
int sched_fifo_priority = 0;

FILE* fp_log;

#define CH01 (((volatile short*)host_buffer)[0])
#define CH02 (((volatile short*)host_buffer)[1])
#define CH03 (((volatile short*)host_buffer)[2])
#define CH04 (((volatile short*)host_buffer)[3])
#define TLATCH (((volatile unsigned*)host_buffer)[2])	/* actually, sample counter */
#define SPAD1	(((volatile unsigned*)host_buffer)[3])   /* user signal from ACQ */

void get_mapping() {
	fd = open(HB_FILE, O_RDWR);
	if (fd < 0){
		perror("HB_FILE");
		exit(errno);
	}
	host_buffer = mmap(0, HB_LEN, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
	if (host_buffer == (caddr_t)-1 ){
		perror( "mmap" );
	        exit(errno);
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

void ui(int argc, char* argv[])
{
	// this ui is kind of basic ..
}

void setup()
{
	get_mapping();
	goRealTime();
	fp_log = fopen("llcontrol.log", "w");
	if (fp_log == 0){
		perror("llcontrol.log");
		exit(1);
	}
}

void run()
{
	unsigned tl0 = TLATCH;
	unsigned spad1_0 = SPAD1;
	unsigned spad1_1;
	unsigned tl1;
	int sample;
	int println = 0;

	for (sample = 0; sample < nsamples; ++sample, tl0 = tl1){
		while((tl1 = TLATCH) == tl0){
			sched_yield();
		}
		if (sample%10000 == 0){
			if (println == 0){
				printf("[%d] ", sample);
				println = 1;
			}
			printf("%u ", sample, tl1);
		}
		if (spad1_1 != spad1_0){
			if (println == 0){
				printf("[%d] ", sample);
				println = 1;
			}
			printf("%u => %u ", sample, spad1_0, spad1_1);
			spad1_0 = spad1_1;
		}
		if (println){
			printf("\n");
			println = 0;
		}
		fwrite(host_buffer, sizeof(short), 8, fp_log);
	}
}

close() {
	munmap(host_buffer, HB_LEN);
	close(fd);
	fclose(fp_log);
}
int main(int argc, char* argv[])
{
	ui(argc, argv);
	setup();
	run();
}
