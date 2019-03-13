/* ------------------------------------------------------------------------- *
 * afhba-bufferAB-do32-threshold.c
 * simple llcontrol example, ONE HBA, bufferA bufferB, CPU copy (realistic).
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

 DO32=1 : copy tlatch (sample number to DO32
 DO32=2 : run a threshold crossing algorithm, AI[0:31] -> DO[0:31] 
*/


#include "afhba-llcontrol-common.h"

#define HB_LEN  0x100000		/* 1MB HOST BUFFERSW */

#define BUFFER_AB_OFFSET 0x040000	/* BUFFERB starts here */
#define XO_OFF  0x080000		/* XO buffer at this offset */

#define LOG_FILE	"afhba.%d.log"

const char* log_file = LOG_FILE;

void* host_buffer;

void* bufferAB[2];
int fd;
int nsamples = 10000000;		/* 10s at 1MSPS */
int samples_buffer = 1;			/* set > 1 to decimate max 16*64bytes */

int verbose;
FILE* fp_log;
void (*G_action)(void*);
int devnum = 0;
int dummy_first_loop;
/** potentially good for cache fill, but sets initial value zero */
int G_POLARITY = 1;		
/** env POLARITY=-1 negates feedback this is usefult to know that the 
 *  software is in fact doing something 					 */


short* xo_buffer;
int has_do32;

#define NSHORTS1 (nchan + spadlongs*sizeof(unsigned)/sizeof(short))
#undef NSHORTS
#define NSHORTS	(NSHORTS1*samples_buffer)
#define VI_LEN 	(NSHORTS*sizeof(short))


struct XLLC_DEF xllc_def = {
		.pa = RTM_T_USE_HOSTBUF,

};

#define VO_LEN  64

#define DO_IX   0


void control_none(short* xo, short* ai);
void control_thresholds(short* ao, short *ai);
void control_threshold_history(short* ao, short *ai);

void (*G_control)(short *ao, short *ai) = control_none;

#define MV100   (32768/100)


int mon_chan = 0;

void ui(int argc, char* argv[])
{
	if (getenv("LOG_FILE")){
		log_file = getenv("LOG_FILE");
	}
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
	if (getenv("POLARITY")){
		G_POLARITY = atoi(getenv("POLARITY"));
		fprintf(stderr, "G_POLARITY set %d\n", G_POLARITY);
	}
        if (getenv("AFFINITY")){
                setAffinity(strtol(getenv("AFFINITY"), 0, 0));
        }
        if (getenv("DO_THRESHOLDS")){
		G_control = control_thresholds;
	}
        if (getenv("CONTROL_THRESHOLD_HISTORY")){
        	mon_chan = atoi(getenv("CONTROL_THRESHOLD_HISTORY"));
        	G_control = control_threshold_history;
        	fprintf(stderr, "control set %s mon_chan %d\n",
        			"control_threshold_history", mon_chan);
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
		const char* acts = getenv("ACTION");
		if (strcmp(acts, "check_tlatch") == 0){
			G_action = check_tlatch_action;
		}
	}
}

void setup()
{
	char logfile[80];
	sprintf(logfile, log_file, devnum);
	host_buffer = get_mapping(devnum, &fd);
	goRealTime();
	struct AB ab_def;
	fp_log = fopen(logfile, "w");
	if (fp_log == 0){
		perror(logfile);
		exit(1);
	}

	ab_def.buffers[0].pa = xllc_def.pa;
	ab_def.buffers[1].pa = BUFFER_AB_OFFSET;
	ab_def.buffers[0].len =
	ab_def.buffers[1].len = VI_LEN;

	if (ioctl(fd, AFHBA_START_AI_AB, &ab_def)){
		perror("ioctl AFHBA_START_AI_AB");
		exit(1);
	}
	printf("AI buf pa: %c 0x%08x len %d\n", 'A',
			ab_def.buffers[0].pa, ab_def.buffers[0].len);
	printf("AI buf pa: %c 0x%08x len %d\n", 'B',
			ab_def.buffers[1].pa, ab_def.buffers[1].len);

	xllc_def.pa = ab_def.buffers[0].pa + XO_OFF;
	xllc_def.len = VO_LEN;

	if (ioctl(fd, AFHBA_START_AO_LLC, &xllc_def)){
		perror("ioctl AFHBA_START_AO_LLC");
		exit(1);
	}
	printf("AO buf pa:   0x%08x len %d\n", xllc_def.pa, xllc_def.len);

	xo_buffer = (short*)((void*)host_buffer+XO_OFF);
}

void print_sample(unsigned sample, unsigned tl)
{
	if (sample%10000 == 0){
		printf("[%10u] %10u\n", sample, tl);
	}
}

void control_thresholds(short *xo, short *ai)
/* set DO bit for corresponding AI > 0 */
{
	unsigned *do32 = (unsigned*)xo;
	int ii;
	static unsigned yy = 0;
	
	for (ii = 0; ii < 32; ++ii){
		if (ai[ii] > 100){
			yy |= 1<<ii;
		}else if (ai[ii] < -100){
			yy &= ~(1<<ii);
		}
	}

	do32[DO_IX] = yy;
}

void control_threshold_history(short *xo, short *ai)
/* set DO bit for corresponding AI[mon_chan][t] > 0 t=0:31 */
{
	unsigned *do32 = (unsigned*)xo;
	int ii;
	static unsigned yy = 0;

	for (ii = 0; ii < 32; ++ii){
		if (ai[ii*NSHORTS1 + mon_chan] > 100){
			yy |= 1<<ii;
		}else if (ai[ii*NSHORTS1 + mon_chan] < -100){
			yy &= ~(1<<ii);
		}
	}

	do32[DO_IX] = yy;
}
void control_none(short *xo, short *ai)
{
	unsigned* dox = (unsigned *)xo;
	unsigned* tlx = (unsigned *)ai;

	dox[DO_IX] = tlx[SPIX];
}


#define MARKER 0xdeadc0d1


void run(void (*control)(short *ao, short *ai), void (*action)(void*))
{
	short* ai_buffer = calloc(NSHORTS, sizeof(short));
	unsigned tl1;
	unsigned ib;
	int println = 0;
	int nbuffers = nsamples/samples_buffer;
	int ab = 0;
	int rtfails = 0;
	int pollcat = 0;

	mlockall(MCL_CURRENT);
	memset(bufferAB[0], 0, VI_LEN);
	memset(bufferAB[1], 0, VI_LEN);
	TLATCH(bufferAB[0])[0] = MARKER;
	TLATCH(bufferAB[1])[0] = MARKER;

	for (ib = 0; ib <= nbuffers; ++ib, tl1, ab = !ab, pollcat = 0){
		/* WARNING: RT: software MUST get there first, or we lose data */
		if (TLATCH(bufferAB[ab])[0] != MARKER){
			TLATCH(bufferAB[ab])[0] = MARKER;
			++rtfails;
		}

		while((tl1 = TLATCH(bufferAB[ab])[0]) == MARKER){
			sched_yield();
			++pollcat;
		}
		memcpy(ai_buffer, bufferAB[ab], VI_LEN);
		TLATCH(bufferAB[ab])[0] = MARKER;
		control(xo_buffer, ai_buffer);
		TLATCH(ai_buffer)[1] = ib != 0? pollcat: 0;
		TLATCH(ai_buffer)[2] = ib != 0? difftime_us(): 0;
		action(ai_buffer);

		if (verbose){
			print_sample(ib, tl1);
		}
	}
	if (rtfails){
		fprintf(stderr, "ERROR: rtfails:%d\n", rtfails);
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
