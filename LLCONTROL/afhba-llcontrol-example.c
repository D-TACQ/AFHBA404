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

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


#include <sched.h>

#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>

/* Kludge alert */
typedef unsigned       u32;
typedef unsigned short u16;
typedef unsigned char  u8;


#include "../rtm-t_ioctl.h"
#define HB_FILE "/dev/rtm-t.%d"
#define HB_LEN	0x1000

#define LOG_FILE	"afhba.%d.log"

void* host_buffer;
int fd;
int nsamples = 10000000;		/* 10s at 1MSPS */
int samples_buffer = 1;			/* set > 1 to decimate max 16*64bytes */
int sched_fifo_priority = 1;
int verbose;
FILE* fp_log;
void (*G_action)(void*);
int devnum = 0;

short* local_buffer;
int *pollcats;

/* ACQ425 */

#define NCHAN	16
#define NSHORTS	32
#define VI_LEN 	(NSHORTS*sizeof(short))
#define SPIX	(NCHAN*sizeof(short)/sizeof(unsigned))

#define CH01 (((volatile short*)host_buffer)[0])
#define CH02 (((volatile short*)host_buffer)[1])
#define CH03 (((volatile short*)host_buffer)[2])
#define CH04 (((volatile short*)host_buffer)[3])
#define TLATCH (((volatile unsigned*)host_buffer)[SPIX])	/* actually, sample counter */
#define SPAD1	(((volatile unsigned*)host_buffer)[SPIX+1])   /* user signal from ACQ */


struct XLLC_DEF xllc_def = {
		.pa = RTM_T_USE_HOSTBUF,
		.len = VI_LEN
};

void get_mapping() {
	char fname[80];
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
}


void null_action(void *data)
{

}

void write_action(void *data)
{
	fwrite(data, sizeof(short), NSHORTS, fp_log);
}

void check_tlatch_action(void *local_buffer)
{
	static unsigned tl0;

	unsigned tl1 = TLATCH;
	if (tl1 != tl0+1){
		printf("%d => %d\n", tl0, tl1);
	}
	tl0 = tl1;
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


void ui(int argc, char* argv[])
{
        if (getenv("RTPRIO")){
		sched_fifo_priority = atoi(getenv("RTPRIO"));
        }
	if (getenv("VERBOSE")){
		verbose = atoi(getenv("VERBOSE"));
	}
	if (getenv("DEVNUM")){
		devnum = atoi(getenv("DEVNUM"));
	}
        if (getenv("AFFINITY")){
                setAffinity(strtol(getenv("AFFINITY"), 0, 0));
        }

	/* own PA eg from GPU */
	if (getenv("PA_BUF")){
		xllc_def.pa = strtoul(getenv("PA_BUF"), 0, 0);
	}
	if (argc > 1){
		nsamples = atoi(argv[1]);
	}
	if (argc > 2){
		samples_buffer = atoi(argv[2]);
	}
	G_action = null_action;
	if (getenv("ACTION")){
		if (strcmp(getenv("ACTION"), "check_tlatch") == 0){
			G_action = check_tlatch_action;
		}
	}
}

void setup()
{
	char logfile[80];
	sprintf(logfile, LOG_FILE, devnum);
	local_buffer = calloc(NSHORTS*nsamples, sizeof(short));
	pollcats = calloc(nsamples, sizeof(int));
	get_mapping();
	fp_log = fopen(logfile, "w");
	if (fp_log == 0){
		perror(logfile);
		exit(1);
	}

	xllc_def.len = samples_buffer*VI_LEN;
	if (xllc_def.len > 16*64){
		xllc_def.len = 16*64;
		samples_buffer = xllc_def.len/VI_LEN;
		fprintf(stderr, "WARNING: samples_buffer clipped to %d\n", samples_buffer);
	}

	if (ioctl(fd, AFHBA_START_AI_LLC, &xllc_def)){
		perror("ioctl AFHBA_START_AI_LLC");
		exit(1);
	}
}

#define TLX(lbp) (((volatile unsigned*)lbp)[SPIX])        /* actually, sample counter */


void run(void (*action)(void*))
{
	short* lbp = local_buffer;
	unsigned tl0 = 0xdeadbeef;	/* always run one dummy loop */
	unsigned spad1_0 = SPAD1;
	unsigned spad1_1;
	unsigned tl1;
	int sample;
	int println = 0;

	mlockall(MCL_CURRENT);
	memset(host_buffer, 0, VI_LEN);

	for (sample = 0; sample <= nsamples; 
			++sample, tl0 = tl1, lbp+=NSHORTS){
		/** atomic start */
		int pollcat;

		for(pollcat = 0; (tl1 = TLATCH) == tl0; ++pollcat){   
			;
		}
		memcpy(lbp, host_buffer, VI_LEN);
		pollcats[sample] = pollcat;
		/** atomic end */
		action(lbp);
	}

	if (verbose == 1){
		printf("[%10u] %10u\n", sample-1, tl1);
	}	
}


void staged_tlatch_report()
{
	short* lbp = local_buffer;

	if (verbose > 1){
		int sample;
		for (sample = 0, lbp = local_buffer; sample <= nsamples; sample += 10000, 
			lbp += 10000*VI_LEN/sizeof(short)){
			printf("[%10u] %10u\n", sample, TLX(lbp));
		}
	}
}

void write_log() {
	FILE *fp = fopen("llcontrol.log", "w");
	if (fp == 0){
		perror("llcontrol.log");
		return;
	}
	fwrite(local_buffer, sizeof(short), NSHORTS*nsamples, fp);
	fclose(fp);
}


void write_pollcats() {
	FILE *fp = fopen("pollcat.log", "w");
	if (fp == 0){
		perror("pollcat.log");
		return;
	}
	fwrite(pollcats, sizeof(int), nsamples, fp);
	fclose(fp);
}




void check_tlatch_action_post()
{
	unsigned tl0;
	unsigned tl1;
	short *lbp = local_buffer;
	int sample;

	tl0 = TLX(lbp);
	lbp += NSHORTS;
	for (sample = 1; sample < nsamples; ++sample, lbp += NSHORTS, tl0 = tl1){
		int err = 0;
		int prt = 0;

		tl1 = TLX(lbp);
		if (tl1 != tl0 + 1){
			err = prt = 1;
		}else if (verbose == 0 && sample%100000 == 0){
			prt = 1;
		}

		if (prt){
			printf("%10d %10d %10d %10d %s %d\n", 
				sample, tl0, tl1, tl1-tl0, err?"ERR":"GOOD", pollcats[sample]);
		}
	}
}

void close_llc() {
	staged_tlatch_report();
	check_tlatch_action_post();
	write_log();
	write_pollcats();
	munmap(host_buffer, HB_LEN);
	close(fd);
	fclose(fp_log);
}

static void* tight_loop(void* unused)
{
	unsigned i0;
	unsigned i1;

	for (i0 = i1 = 0; ++i1 > i0; i0 = i1){
		;
	}
}	

#include <signal.h>
#include <pthread.h>
void goPosixRT(void *(*work)(void *)) 
{
       	struct sched_param svparam = {.sched_priority = 90 };
	pthread_t svtid;
        pthread_attr_t svattr, clattr;
        sigset_t set;
        int sig;

        sigemptyset(&set);
        sigaddset(&set, SIGINT);
        sigaddset(&set, SIGTERM);
        sigaddset(&set, SIGHUP);
        pthread_sigmask(SIG_BLOCK, &set, NULL);

        pthread_attr_init(&svattr);
        pthread_attr_setdetachstate(&svattr, PTHREAD_CREATE_JOINABLE);
        pthread_attr_setinheritsched(&svattr, PTHREAD_EXPLICIT_SCHED);
        pthread_attr_setschedpolicy(&svattr, SCHED_FIFO);
        pthread_attr_setschedparam(&svattr, &svparam);

        errno = pthread_create(&svtid, &svattr, work, NULL);

	pthread_join(svtid, NULL);
}

static void* llc(void* unused)
{
	run(G_action);
	printf("posix done\n");
	printf("shot complete\n");
}

int main(int argc, char* argv[])
{
	ui(argc, argv);
	setup();
	if (getenv("RTPRIO") != 0){
		printf("running posix RT\n");
		goPosixRT(llc);
		close_llc();
		return 0;
	}else{
		printf("running regular\n");
		llc(0);
		printf("finished\n");
		close_llc();
	}
}
