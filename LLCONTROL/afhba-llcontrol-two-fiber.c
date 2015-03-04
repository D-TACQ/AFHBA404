/* ------------------------------------------------------------------------- *
 * afhba-llcontrol-two-fiber.c
 * simple llcontrol example, TWO HBA, two buffers, CPU copy (realistic).
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
#include <string.h>
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
int nsamples = 10000000;		/* 10s at 1MSPS */
int samples_buffer = 1;			/* set > 1 to decimate max 16*64bytes */
int sched_fifo_priority = 1;
int verbose;
FILE* fp_log;
void (*G_action)(void*);

int dummy_first_loop;
/* potentially good for cache fill, but sets initial value zero */



/* ACQ425 */

#define NCHAN	16
#define NSHORTS	32
#define VI_LEN 	(NSHORTS*sizeof(short))
#define SPIX	(NCHAN*sizeof(short)/sizeof(unsigned))

#define CH01 (((volatile short*)host_buffer)[0])
#define CH02 (((volatile short*)host_buffer)[1])
#define CH03 (((volatile short*)host_buffer)[2])
#define CH04 (((volatile short*)host_buffer)[3])
#define TLATCH (&((volatile unsigned*)host_buffer)[SPIX])      /* actually, sample counter */
#define SPAD1	(((volatile unsigned*)host_buffer)[SPIX+1])   /* user signal from ACQ */

struct XLLC_DEF ai_def = {
		.pa = RTM_T_USE_HOSTBUF,
		.len = VI_LEN
};

#define AO_CHAN	32
#define VO_LEN  (AO_CHAN*sizeof(short) + sizeof(unsigned))

struct XLLC_DEF ao_def = {
		.pa = RTM_T_USE_HOSTBUF,
		.len = VI_LEN
};

struct Device {
	int devnum;
	char fname[80];
	int fd;
	void* mapping;
};

struct Device dev_ai = { .devnum = 0 };
struct Device dev_ao = { .devnum = 1 };
short* ao_buffer;
short *ai_buffer;

/* SPLIT single HB into 2
 * [0] : AI
 * [1] : AO
 */
void get_mapping(struct Device *device) {
	sprintf(device->fname, HB_FILE, device->devnum);
	device->fd = open(device->fname, O_RDWR);
	if (device->fd < 0){
		perror(device->fname);
		exit(errno);
	}

	device->mapping = mmap(0, HB_LEN*2,
			PROT_READ|PROT_WRITE, MAP_SHARED, device->fd, 0);
	if (device->mapping == (caddr_t)-1 ){
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
	if (getenv("DEV_AI")){
		dev_ai.devnum = atoi(getenv("DEV_AI"));
	}
	if (getenv("DEV_AO")){
		dev_ao.devnum = atoi(getenv("DEV_AO"));
	}
	/* own PA eg from GPU */
	if (getenv("PA_AI_BUF")){
		ai_def.pa = strtoul(getenv("PA_AI_BUF"), 0, 0);
	}
	if (getenv("PA_AO_BUF")){
		ao_def.pa = strtoul(getenv("PA_AO_BUF"), 0, 0);
	}
	if (getenv("DUMMY_FIRST_LOOP")){
		dummy_first_loop = atoi(getenv("DUMMY_FIRST_LOOP"));
	}
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
	sprintf(logfile, LOG_FILE, dev_ai.devnum);

	goRealTime();
	fp_log = fopen(logfile, "w");
	if (fp_log == 0){
		perror(logfile);
		exit(1);
	}


	get_mapping(&dev_ai);
	ai_buffer = host_buffer = dev_ai.mapping;

	get_mapping(&dev_ao);
	ao_buffer = dev_ao.mapping;

	ai_def.len = samples_buffer*VI_LEN;
	if (ai_def.len > 16*64){
		ai_def.len = 16*64;
		samples_buffer = ai_def.len/VI_LEN;
		fprintf(stderr, "WARNING: samples_buffer clipped to %d\n",
				samples_buffer);
	}

	if (ioctl(dev_ai.fd, AFHBA_START_AI_LLC, &ai_def)){
		perror("ioctl AFHBA_START_AI_LLC");
		exit(1);
	}
	printf("AI buf pa: 0x%08x\n", ai_def.pa);

	if (ioctl(dev_ao.fd, AFHBA_START_AO_LLC, &ao_def)){
		perror("ioctl AFHBA_START_AO_LLC");
		exit(1);
	}
	printf("AO buf pa: 0x%08x\n", ao_def.pa);
}

void print_sample(unsigned sample, unsigned tl)
{
	if (sample%10000 == 0){
		printf("[%10u] %10u\n", sample, tl);
	}
}

void control(short *ao, short *ai)
{
	int ii;
	for (ii = 0; ii < AO_CHAN; ii += 2){
		ao[ii] = ai[0];
		ao[ii+1] = ai[1];
	}
}

void run(void (*action)(void*))
{
	short* ai_buffer = calloc(NSHORTS, sizeof(short));
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
		memcpy(ai_buffer, host_buffer, VI_LEN);
		while((tl1 = *TLATCH) == tl0){
			sched_yield();
			memcpy(ai_buffer, host_buffer, VI_LEN);
		}
		control(ao_buffer, ai_buffer);
		action(ai_buffer);

		if (verbose){
			print_sample(sample, tl1);
		}
	}
}

close() {
	munmap(host_buffer, HB_LEN);
	close(dev_ai.fd);
	close(dev_ao.fd);
	fclose(fp_log);
}
int main(int argc, char* argv[])
{
	ui(argc, argv);
	setup();
	printf("ready for data\n");
	run(G_action);
	printf("finished\n");
}
