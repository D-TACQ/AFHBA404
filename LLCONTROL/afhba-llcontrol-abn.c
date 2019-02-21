/* ------------------------------------------------------------------------- *
 * afhba-llcontrol-abn.c
 * simple llcontrol example, ONE HBA, bufferABN (max=MAXABN)
 * allows inicreased rate. Eg samples may be bursted.
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

void* bufferABN[MAXABN];
int fd;

int NDESC = MAXABN;
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


#define DEF_NCHAN 	192			/* eg TCV THOMSON, 6 x ACQ423 */
int nchan = DEF_NCHAN;
int spadlongs = 16;

short* xo_buffer;
int has_do32;

#define PAGE_SIZE	0x1000

#define NSHORTS1 (nchan + spadlongs*sizeof(unsigned)/sizeof(short))
#define NSHORTS	(NSHORTS1*samples_buffer)
#define VI_LEN 	(NSHORTS*sizeof(short))
#define SPIX	(NSHORTS/2-spadlongs)

#define TLATCH(buf) (&((volatile unsigned*)(buf))[SPIX])      /* actually, sample counter */
#define SPAD1	(((volatile unsigned*)ai_buffer)[SPIX+1])   /* user signal from ACQ */

struct XLLC_DEF xllc_def = {
	.pa = RTM_T_USE_HOSTBUF,
};

#define VO_LEN  64

#define DO_IX   0

/* SPLIT single HB into 2
 * [0] : AI
 * [1] : AO
 */
void get_mapping() {
	char fname[80];
	int ib;
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
	for (ib = 0; ib < NDESC; ++ib){
		bufferABN[ib] = host_buffer + ib * PAGE_SIZE;
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
	static int call_count;
	short *ai_buffer = local_buffer;

	++call_count;

	if (tl0 == 0){
		tl0 = *TLATCH(ai_buffer);
	}else{
		unsigned tl1 = *TLATCH(ai_buffer);
		if (tl1 != tl0+samples_buffer){
			if (++errcount < 100){
				printf("ERROR:%5d %08x => %08x\n", call_count, tl0, tl1);
			}else if (errcount == 100){
				printf("stop reporting at 100 errors ..\n");
			}
		}
		tl0 = tl1;
	}
}

void control_none(short* xo, short* ai) {}
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
	if (getenv("NDESC")){
		NDESC = atoi(getenv("NDESC"));
		if (NDESC > MAXABN) NDESC = MAXABN;
		fprintf(stderr, "NDESC set %d", NDESC);
	}
	if (getenv("POLARITY")){
		G_POLARITY = atoi(getenv("POLARITY"));
		fprintf(stderr, "G_POLARITY set %d\n", G_POLARITY);
	}
        if (getenv("AFFINITY")){
                setAffinity(strtol(getenv("AFFINITY"), 0, 0));
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
	int ib;
	sprintf(logfile, log_file, devnum);
	get_mapping();
	goRealTime();
	struct ABN abn;
	fp_log = fopen(logfile, "w");
	if (fp_log == 0){
		perror(logfile);
		exit(1);
	}

	abn.ndesc = NDESC;
	abn.buffers[0].pa = xllc_def.pa;

	for (ib = 0; ib < NDESC; ++ib){
		if (xllc_def.pa != RTM_T_USE_HOSTBUF){
			abn.buffers[ib].pa = xllc_def.pa + ib*PAGE_SIZE;
		}
		abn.buffers[ib].len = VI_LEN;
	}


	if (ioctl(fd, AFHBA_START_AI_ABN, &abn)){
		perror("ioctl AFHBA_START_AI_ABN");
		exit(1);
	}
	for (ib = 0; ib < NDESC; ++ib){
		printf("AI buf pa: %2d 0x%08x len %d\n", ib,
			abn.buffers[ib].pa, abn.buffers[ib].len);
	}
}

void print_sample(unsigned sample, unsigned tl)
{
	if (sample%10000 == 0){
		printf("[%10u] %10u\n", sample, tl);
	}
}

#define MARKER 0xdeadc0d1


void run(void (*control)(short *ao, short *ai), void (*action)(void*))
{
	short* ai_buffer = calloc(NSHORTS, sizeof(short));
	unsigned tl1;
	unsigned ib;
	int println = 0;
	int nbuffers = nsamples/samples_buffer;
	int abn = 0;
	int rtfails = 0;
	int pollcat = 0;

	mlockall(MCL_CURRENT);

	for (abn = 0; abn < NDESC; ++abn){
		memset(bufferABN[abn], 0, VI_LEN);
		TLATCH(bufferABN[abn])[0] = MARKER;
	}

	for (ib = 0, abn = 0; ib <= nbuffers;
		++ib, tl1, abn = (abn+1) < NDESC? (abn+1): 0, pollcat = 0){
		/* WARNING: RT: software MUST get there first, or we lose data */
		if (TLATCH(bufferABN[abn])[0] != MARKER){
			TLATCH(bufferABN[abn])[0] = MARKER;
			++rtfails;
		}

		while((tl1 = TLATCH(bufferABN[abn])[0]) == MARKER){
			sched_yield();
			++pollcat;
		}
		memcpy(ai_buffer, bufferABN[abn], VI_LEN);
		TLATCH(bufferABN[abn])[0] = MARKER;
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
