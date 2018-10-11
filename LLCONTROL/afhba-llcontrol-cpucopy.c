/* ------------------------------------------------------------------------- *
 * afhba-llcontrol-cpucopy.c
 * simple llcontrol example, ONE HBA, two buffers, CPU copy (realistic).
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

/** Description of Program
 *

 getMapping() from device driver : this is where the data appears
 goRealTime() isolate the process for best RT performance
 ioctl(fd, AFHBA_START_AI_LLC) : device driver starts the acquisition

 Then the control loop in run() is as follows:

 for (sample = 0; sample <= nsamples; ++sample..){
     poll for new data
     run the control algorithm on the new AI, store new AO
 }

*/




#include "afhba-llcontrol-common.h"


#define HB_LEN	0x1000



void* host_buffer;
int fd;
int nsamples = 10000000;		/* 10s at 1MSPS */
int samples_buffer = 1;			/* set > 1 to decimate max 16*64bytes */
int sched_fifo_priority = 1;
int verbose;
FILE* fp_log;
void (*G_action)(void*);
int devnum = 0;
int dummy_first_loop;
/** potentially good for cache fill, but sets initial value zero */
int G_POLARITY = 1;		
/** env POLARITY=-1 negates feedback this is usefult to know that the 
 *  software is in fact doing something 					 */


#define DEF_NCHAN 	16
int nchan = DEF_NCHAN;
int spadlongs = 16;

short* ao_buffer;
int has_do32;

int DUP1 = 0; 			/* duplicate AI[DUP1], default 0 */
short *AO_IDENT;

#define NSHORTS	(nchan+spadlongs*sizeof(unsigned)/sizeof(short))
#define VI_LEN 	(NSHORTS*sizeof(short))
#define SPIX	(nchan*sizeof(short)/sizeof(unsigned))
/* ai_buffer is a local copy of host buffer */
#define CH01 (((volatile short*)ai_buffer)[0])
#define CH02 (((volatile short*)ai_buffer)[1])
#define CH03 (((volatile short*)ai_buffer)[2])
#define CH04 (((volatile short*)ai_buffer)[3])
#define TLATCH (&((volatile unsigned*)ai_buffer)[SPIX])      /* actually, sample counter */
#define SPAD1	(((volatile unsigned*)ai_buffer)[SPIX+1])   /* user signal from ACQ */

struct XLLC_DEF xllc_def = {
		.pa = RTM_T_USE_HOSTBUF,

};

#define DEF_AO_CHAN	32
int aochan = DEF_AO_CHAN;
#define VO_LEN  (aochan*sizeof(short) + (has_do32?sizeof(unsigned):0))

#define DO_IX	(16)		/* longwords */


/* SPLIT single HB into 2
 * [0] : AI
 * [1] : AO
 */
void get_mapping() {
	char fname[80];
	sprintf(fname, HB_FILE, devnum);
	fd = open(fname, O_RDWR);
	if (fd < 0){
		perror(fname);
		exit(errno);
	}

	host_buffer = mmap(0, HB_LEN*2, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
	if (host_buffer == (caddr_t)-1 ){
		perror( "mmap" );
	        exit(errno);
	}
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


void write_action(void *data)
{
	fwrite(data, sizeof(short), NSHORTS, fp_log);
}

void check_tlatch_action(void *local_buffer)
{
	static unsigned tl0;
	static int errcount;
	short *ai_buffer = local_buffer;

	unsigned tl1 = *TLATCH;
	if (tl1 != tl0+1){
		if (++errcount < 100){
			printf("%d => %d\n", tl0, tl1);
		}else if (errcount == 100){
			printf("stop reporting at 100 errors ..\n");
		}
	}
	tl0 = tl1;
}

void control_dup1(short *ao, short *ai);
void (*G_control)(short *ao, short *ai) = control_dup1;

#define MV100   (32768/100)

short* make_ao_ident(int ao_ident)
{
        short* ids = calloc(aochan, sizeof(short));
        if (ao_ident){
                int ic;

                for (ic = 0; ic < aochan; ++ic){
                        ids[ic] = ic*MV100*ao_ident;
                }
        }
        return ids;
}

int FFNLUT;
void control_feedforward(short *ao, short *ai);

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
	if (getenv("AICHAN")){
		nchan = atoi(getenv("AICHAN"));
		fprintf(stderr, "AICHAN (nchan) set %d\n", nchan);
	}
	if (getenv("AOCHAN")){
		aochan = atoi(getenv("AOCHAN"));
		fprintf(stderr, "AOCHAN set %d\n", aochan);
	}
	if (getenv("POLARITY")){
		G_POLARITY = atoi(getenv("POLARITY"));
		fprintf(stderr, "G_POLARITY set %d\n", G_POLARITY);
	}
        if (getenv("AFFINITY")){
                setAffinity(strtol(getenv("AFFINITY"), 0, 0));
        }
        if (getenv("DUP1")){
                DUP1 = atoi(getenv("DUP1"));
                G_control = control_dup1;
        }
	if (getenv("FEED_FORWARD")){
		int ff = atoi(getenv("FEED_FORWARD"));
		if (ff){
			G_control = control_feedforward;
			FFNLUT = ff;
		}
	}

        {
                int ao_ident = 0;
                if (getenv("AO_IDENT")){
                        ao_ident = atoi(getenv("AO_IDENT"));
                }
                AO_IDENT = make_ao_ident(ao_ident);
        }

	if (getenv("SPADLONGS")){
		spadlongs = atoi(getenv("SPADLONGS"));
		fprintf(stderr, "SPADLONGS set %d\n", spadlongs);
	}
	xllc_def.len = VI_LEN;

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
	printf("AI buf pa: 0x%08x len %d\n", xllc_def.pa, xllc_def.len);



	xllc_def.pa += HB_LEN;
	xllc_def.len = VO_LEN;

	if (has_do32){
		int ll = xllc_def.len/64;
		xllc_def.len = ++ll*64;
	}
	if (ioctl(fd, AFHBA_START_AO_LLC, &xllc_def)){
		perror("ioctl AFHBA_START_AO_LLC");
		exit(1);
	}
	printf("AO buf pa: 0x%08x len %d\n", xllc_def.pa, xllc_def.len);

	ao_buffer = (short*)((void*)host_buffer+HB_LEN);

	if (has_do32){
		/* marker pattern for the PAD area for hardware trace */
		unsigned* dox = (unsigned *)ao_buffer;
		int ii;
		for (ii = 0; ii <= 0xf; ++ii){
		        dox[DO_IX+ii] = (ii<<24)|(ii<<16)|(ii<<8)|ii;
		}
	}
}

void print_sample(unsigned sample, unsigned tl)
{
	if (sample%10000 == 0){
		printf("[%10u] %10u\n", sample, tl);
	}
}

void copy_tlatch_to_do32(void *ao, void *ai)
{
	unsigned* dox = (unsigned *)ao;
	unsigned* tlx = (unsigned *)ai;

	dox[DO_IX] = tlx[SPIX];
}

void control_dup1(short *ao, short *ai)
{
        int ii;

        for (ii = 0; ii < aochan; ii++){
                ao[ii] = AO_IDENT[ii] + ai[DUP1];
        }

        if (has_do32){
                copy_tlatch_to_do32(ao, ai);
        }
}

#include <math.h>



short ff(int ii)
{
	static short* lut;
	if (lut == 0){
		int ii;
		lut = calloc(FFNLUT, sizeof(short));
		for (ii = 0; ii < FFNLUT; ++ii){
			lut[ii] = MV100 * sin((double)ii * 2*M_PI/FFNLUT);
		}
	}
	return lut[ii%FFNLUT];
}
void control_feedforward(short *ao, short *ai)
{
        int ii;
	static int cursor;

	short xx = ff(cursor++);

        for (ii = 0; ii < aochan; ii++){
                ao[ii] = AO_IDENT[ii] + xx;
        }

        if (has_do32){
                copy_tlatch_to_do32(ao, ai);
        }
}


void control_example2(short *ao, short *ai)
{
	int ii;
	for (ii = 0; ii < aochan; ii += 2){
		ao[ii] = G_POLARITY * ai[0];
		ao[ii+1] = (((ii&1) != 0? ii: -ii)*ai[0])/aochan;
	}
	if (has_do32){
		copy_tlatch_to_do32(ao, ai);
	}
}


void run(void (*control)(short *ao, short *ai), void (*action)(void*))
{
	short* ai_buffer = calloc(NSHORTS, sizeof(short));
	unsigned tl0 = 0xdeadbeef;	/* always run one dummy loop */
	unsigned tl1;
	unsigned sample;
	int println = 0;
	int pollcat = 0;

	mlockall(MCL_CURRENT);
	memset(host_buffer, 0, VI_LEN);
	if (!dummy_first_loop){
		TLATCH[0] = tl0;
	}

	for (sample = 0; sample <= nsamples; ++sample, tl0 = tl1, pollcat = 0){
		memcpy(ai_buffer, host_buffer, VI_LEN);
		while((tl1 = TLATCH[0]) == tl0){
			sched_yield();
			memcpy(ai_buffer, host_buffer, VI_LEN);
			++pollcat;
		}
		control(ao_buffer, ai_buffer);
		TLATCH[1] = pollcat;
		TLATCH[2] = difftime_us();
		action(ai_buffer);

		if (verbose){
			print_sample(sample, tl1);
		}
	}
}

void closedown(void) {
	munmap(host_buffer, HB_LEN);
	close(fd);
	fclose(fp_log);
}
int main(int argc, char* argv[])
{
	ui(argc, argv);
	setup();
	printf("ready for data\n");
	run(G_control, G_action);
	printf("finished\n");
	closedown();
}
