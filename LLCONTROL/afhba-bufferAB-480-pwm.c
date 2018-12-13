/* ------------------------------------------------------------------------- *
 * afhba-bufferAB-480-pwm.c
 * simple llcontrol example, ONE HBA, bufferA bufferB, CPU copy (realistic).
 * cpontrol for custom DIO482 pwm system
 * ------------------------------------------------------------------------- *
 *   Copyright (C) 2018 Peter Milne, D-TACQ Solutions Ltd
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

#define INSTRUMENT 1			/* instrument key values in external buffer */

#define HB_LEN  0x100000		/* 1MB HOST BUFFERSW */

#define BUFFER_AB_OFFSET 0x040000	/* BUFFERB starts here */
#define XO_OFF  0x080000		/* XO buffer at this offset */

#define LOG_FILE	"afhba.%d.log"

#define HTS_MIN_BUFFER	4096

const char* log_file = LOG_FILE;

void* host_buffer;

void* bufferAB[2];
int fd;
int nsamples = 10000000;		/* 10s at 1MSPS */
int samples_buffer;			/* set > 1 to decimate max 16*64bytes */
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
int spadlongs = 0;

short* bufferXO;
int has_do32;

#define NSHORTS1 	(nchan + spadlongs*sizeof(unsigned)/sizeof(short))
#define NSHORTS		(NSHORTS1*samples_buffer)
#define VI_LEN 		(NSHORTS*sizeof(short))
#define VI_LONGS	(VI_LEN/sizeof(unsigned))

#define EOB(buf)	(((volatile unsigned*)(buf))[VI_LONGS-1])



struct XLLC_DEF xllc_def_ai = {
		.pa = RTM_T_USE_HOSTBUF,

};
struct XLLC_DEF xllc_def_ao;



#define VO_LEN  (32*sizeof(long))

#define DO_IX   0

#define SHM_RTERR 		1
#define SHM_BCO 		2
#define SHM_POLLCATMIN 	3
#define SHM_POLLCATMAX 	4

#define SHM_CH0	5

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

	host_buffer = mmap(0, HB_LEN, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
	if (host_buffer == (caddr_t)-1 ){
		perror( "mmap" );
	        exit(errno);
	}
	bufferAB[0] = host_buffer;
	bufferAB[1] = host_buffer + BUFFER_AB_OFFSET;
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


void null_action(void* data)
{}

void write_action(void *data)
{
	fwrite(data, sizeof(short), NSHORTS, fp_log);
}


int control_none(short* xo, short* ai, short ai10);


int G_buffer_copy_overruns;

int control_check_overrun(short *xo, short* ai, short ai10);
int (*G_control)(short *ao, short *ai, short ai10) = control_check_overrun;


int control_check_mean(short *xo, short* ai, short ai10);

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
		xllc_def_ai.pa = strtoul(getenv("PA_BUF"), 0, 0);
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
	if (getenv("CONTROL_CHECK_MEAN")){
		G_control = control_check_mean;
	}

    if (getenv("AFFINITY")){
                setAffinity(strtol(getenv("AFFINITY"), 0, 0));
    }

	xllc_def_ai.len = VI_LEN;

	if (argc > 1){
		nsamples = atoi(argv[1]);
	}
	samples_buffer = HTS_MIN_BUFFER/nchan/2;
	fprintf(stderr, "nchan: %d samples_buffer = %d\n", nchan, samples_buffer);

	G_action = null_action;
	if (getenv("ACTION")){
		const char* acts = getenv("ACTION");
		if (strcmp(acts, "write_action") == 0){
			G_action = write_action;
		}
	}
}

void setup()
{
	char logfile[80];
	sprintf(logfile, log_file, devnum);
	get_mapping();
	get_shared_mapping(devnum, 1, &xllc_def_ao, (void**)&bufferXO);
	shm_connect();
	goRealTime();
	struct AB ab_def;
	fp_log = fopen(logfile, "w");
	if (fp_log == 0){
		perror(logfile);
		exit(1);
	}

	ab_def.buffers[0].pa = xllc_def_ai.pa;
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

	xllc_def_ao.len = VO_LEN;

	if (ioctl(fd, AFHBA_START_AO_LLC, &xllc_def_ao)){
		perror("ioctl AFHBA_START_AO_LLC");
		exit(1);
	}
	printf("AO buf pa:   0x%08x len %d\n", xllc_def_ao.pa, xllc_def_ao.len);

	bufferXO = (short*)((void*)host_buffer+XO_OFF);

}

void print_sample(unsigned sample, unsigned tl)
{
	if (sample%10000 == 0){
		printf("[%10u] %10u\n", sample, tl);
	}
}


int control_none(short *xo, short *ai, short ai10)
{
	unsigned* dox = (unsigned *)xo;
	unsigned* tlx = (unsigned *)ai;


}


#define MARKER 0xdeadc0d1


int control_check_overrun(short *xo, short* ai, short ai10)
{
	if (ai[0] != ai10){
		return ++G_buffer_copy_overruns;
	}else{
		return 0;
	}
}

int totals[DEF_NCHAN];
int control_check_mean(short *xo, short* ai, short ai10)
/* means aren't super relevant to high speed ADC, but they are simple to calculate .. */
{
	int tt;
	int ic;

	memset(totals, 0, sizeof(totals));

	if (control_check_overrun(xo, ai, ai10) == 0){

		for (tt = 0; tt < samples_buffer; ++tt){
			for (ic = 0; ic < nchan; ++ic){
				totals[ic] += ai[tt*nchan + ic];
			}
		}
	}

	for (ic = 0; ic < nchan; ++ic){
		totals[ic] /= samples_buffer;
	}
#ifdef INSTRUMENT
	memcpy(shm+SHM_CH0, totals, sizeof(totals));
#endif
}

void run(int (*control)(short *ao, short *ai, short ai10), void (*action)(void*))
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
	memset(bufferAB[1], 1, VI_LEN);
	EOB(bufferAB[0]) = MARKER;
	EOB(bufferAB[1]) = MARKER;

	for (ib = 0; ib <= nbuffers; ++ib, tl1, ab = !ab, pollcat = 0){
		/* WARNING: RT: software MUST get there first, or we lose data */
		if (EOB(bufferAB[ab]) != MARKER){
			EOB(bufferAB[ab]) = MARKER;
			++rtfails;
		}

		while((tl1 = EOB(bufferAB[ab])) == MARKER){
			sched_yield();
			++pollcat;
		}
		memcpy(ai_buffer, bufferAB[ab], VI_LEN);
		EOB(bufferAB[ab]) = MARKER;
		control(bufferXO, ai_buffer, ((short *)bufferAB[ab])[0]);
		action(ai_buffer);

#ifdef INSTRUMENT
		if (verbose){
			print_sample(ib, tl1);
		}
		shm[SHM_SAMPLE] = ib;
		shm[SHM_RTERR] = rtfails;
		shm[SHM_BCO] = G_buffer_copy_overruns;
		if (pollcat < shm[SHM_POLLCATMIN]){
			shm[SHM_POLLCATMIN] = pollcat;
		}else if (pollcat > shm[SHM_POLLCATMAX]){
			shm[SHM_POLLCATMAX] = pollcat;
		}
#endif
	}
	if (rtfails){
		fprintf(stderr, "ERROR:i BCO:%d  rtfails:%d out of %d buffers %d %%\n", G_buffer_copy_overruns, rtfails, nbuffers, rtfails*100/nbuffers);
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
