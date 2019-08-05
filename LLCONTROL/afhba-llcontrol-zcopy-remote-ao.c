/* ------------------------------------------------------------------------- *
 * afhba-llcontrol-zcopy-remote-ao.c
 * simple llcontrol example, but feeds TWO AOs from AI same buffer, no copy
 * ------------------------------------------------------------------------- *
 *   Copyright (C) 2019 Sean Alsop, D-TACQ Solutions Ltd
 *                      <sean dot alsop at D hyphen TACQ dot com>
 *                         www.d-tacq.com
 *   Created on: 5 August 2019
 *    Author: Sean Alsop
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
short* ao_buffer;
void* ao_only_buffer;
int fd;
int ao_fd; // new file descriptor for ao box
int nsamples = 10000000;		/* 10s at 1MSPS */
int samples_buffer = 1;			/* set > 1 to decimate max 16*64bytes */
int sched_fifo_priority = 1;
int verbose;
FILE* fp_log;
void (*G_action)(void*);
int devnum = 0;
int ao_devnum = 1; // devnum for ao system.
int dummy_first_loop;
/* potentially good for cache fill, but sets initial value zero */

#define AO_OFFSET (HB_LEN/2)
// #define DEF_NCHAN 	16
#define DEF_NCHAN 	128
int nchan = DEF_NCHAN;
#define DEF_AO_ONLY_CHAN 4
int ao_only_chan = DEF_AO_ONLY_CHAN;
int spadlongs = 16;

#define NSHORTS (nchan+spadlongs*sizeof(unsigned)/sizeof(short))

int has_do32;


#define VI_LEN 	(NSHORTS*sizeof(short))
#define SPIX	(nchan*sizeof(short)/sizeof(unsigned))

#define CH01 (((volatile short*)host_buffer)[0])
#define CH02 (((volatile short*)host_buffer)[1])
#define CH03 (((volatile short*)host_buffer)[2])
#define CH04 (((volatile short*)host_buffer)[3])
#define TLATCH (&((volatile unsigned*)host_buffer)[SPIX])      /* actually, sample counter */
#define SPAD1	(((volatile unsigned*)host_buffer)[SPIX+1])   /* user signal from ACQ */

struct XLLC_DEF xllc_def = {
		.pa = RTM_T_USE_HOSTBUF,

};

// Create a new XLLC_DEF struct in order to pass it to ioctl.
struct XLLC_DEF ao_only_xllc_def = {
        .pa = RTM_T_USE_HOSTBUF,

};

#define AO_CHAN	32
#define VO_LEN  (AO_CHAN*sizeof(short) + (has_do32?sizeof(unsigned):0))
#define VO_ONLY_LEN  (ao_only_chan*sizeof(short))

void get_mapping() {
	char fname[80];
    char ao_fname[80];
	sprintf(fname, HB_FILE, devnum);
	fd = open(fname, O_RDWR);
	if (fd < 0){
		perror(fname);
		exit(errno);
	}

    sprintf(ao_fname, HB_FILE, ao_devnum);
    ao_fd = open(ao_fname, O_RDWR);
    if (ao_fd < 0){
        perror(ao_fname);
        exit(errno);
    }


	host_buffer = mmap(0, HB_LEN, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
	if (host_buffer == (void*)-1 ){
		perror( "mmap" );
	        exit(errno);
	}

    // ao_only_buffer = mmap(0, HB_LEN, PROT_READ|PROT_WRITE, MAP_SHARED, ao_fd, 0);
    // if (ao_only_buffer == (void*)-1 ){
    //     perror( "mmap" );
    //         exit(errno);
    // }

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

void write_action(void *data)
{
	fwrite(data, sizeof(short), NSHORTS, fp_log);
}

void check_tlatch_action(void *local_buffer)
{
	static unsigned tl0;

	unsigned tl1 = *TLATCH;
	if (tl1 != tl0+1){
		printf("%d => %d\n", tl0, tl1);
	}
	tl0 = tl1;
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
	/* own PA eg from GPU */
	if (getenv("PA_BUF")){
		xllc_def.pa = strtoul(getenv("PA_BUF"), 0, 0);
	}
	if (getenv("DO32")){
		has_do32 = atoi(getenv("DO32"));
	}
	if (getenv("DUMMY_FIRST_LOOP")){
		dummy_first_loop = atoi(getenv("DUMMY_FIRST_LOOP"));
	}
	if (getenv("NCHAN")){
		nchan = atoi(getenv("NCHAN"));
		fprintf(stderr, "NCHAN set %d\n", nchan);
	}
	if (getenv("SPADLONGS")){
		spadlongs = atoi(getenv("SPADLONGS"));
		fprintf(stderr, "SPADLONGS set %d\n", spadlongs);
	}
	xllc_def.len = VI_LEN;
    ao_only_xllc_def.len = VO_ONLY_LEN;

	if (argc > 1){
		nsamples = atoi(argv[1]);
	}
	if (argc > 2){
		samples_buffer = atoi(argv[2]);
	}
	G_action = write_action;
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
	get_mapping();
	goRealTime();
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
	xllc_def.len = VO_LEN;
	if (ioctl(fd, AFHBA_START_AO_LLC, &xllc_def)){
		perror("ioctl AFHBA_START_AO_LLC");
		exit(1);
	}

    xllc_def.len = VO_ONLY_LEN;

    // Use ioctl to communicate AO LLC to AHBA404
    // if (ioctl(ao_fd, AFHBA_START_AO_LLC, &ao_only_xllc_def)){
    if (ioctl(ao_fd, AFHBA_START_AO_LLC, &xllc_def)){
        perror("ioctl AFHBA_START_AO_LLC for AO only system.");
        exit(1);


    // ao_buffer = (short*)((void*)host_buffer+AO_OFFSET);

    }
    printf("AO BUF: %08x \n", xllc_def.len);
    printf("AO buf pa: 0x%08x len %d\n", ao_only_xllc_def.pa, ao_only_xllc_def.len);


}

void print_sample(unsigned sample, unsigned tl)
{
	if (sample%10000 == 0){
		printf("[%10u] %10u\n", sample, tl);
	}
}

void run(void (*action)(void*))
{
	short* local_buffer = calloc(NSHORTS, sizeof(short));
	unsigned tl0 = 0xdeadbeef;	/* always run one dummy loop */
	unsigned tl1;
	unsigned sample;
	int println = 0;

	mlockall(MCL_CURRENT);
	memset(host_buffer, 0, VI_LEN);
	if (!dummy_first_loop){
		*TLATCH = tl0;
	}

	for (sample = 0; sample <= nsamples; ++sample, tl0 = tl1){
		memcpy(local_buffer, host_buffer, VI_LEN);
		while((tl1 = *TLATCH) == tl0){
			sched_yield();
			memcpy(local_buffer, host_buffer, VI_LEN);
            // memcpy(ao_only_buffer, local_buffer, VI_LEN);
		}
		action(local_buffer);
		if (verbose){
			print_sample(sample, tl1);
		}
	}
}

void closedown() {
	munmap(host_buffer, HB_LEN);
	close(fd);
	fclose(fp_log);
}
int main(int argc, char* argv[])
{
	ui(argc, argv);
	setup();
	printf("ready for data\n");
	run(G_action);
	printf("finished\n");
	closedown();
}
