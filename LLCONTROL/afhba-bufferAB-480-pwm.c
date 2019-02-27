/* ------------------------------------------------------------------------- *
 * afhba-bufferAB-480-pwm.c
 * simple llcontrol example, ONE HBA, bufferA bufferB, CPU copy (realistic).
 * cpontrol for custom DIO482 pwm system
 * ------------------------------------------------------------------------- *
 *   Copyright (C) 2018 Peter Milne, D-TACQ Solutions Ltd
 *                      <peter dot milne at D hyphen TACQ dot com>
 *                         www.d-tacq.com
 *   Created on: 18 December 2018
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

#include "pwm_internals.h"

#define INSTRUMENT 1			/* instrument key values in external buffer */

#define HB_LEN  0x100000		/* 1MB HOST BUFFERSW */

#define BUFFER_AB_OFFSET 0x040000	/* BUFFERB starts here */

#define LOG_FILE	"afhba.%d.log"

#define HTS_MIN_BUFFER	4096

const char* log_file = LOG_FILE;

void* host_buffer;

void* bufferAB[2];
int fd;
int nsamples = 10000000;		/* 10s at 1MSPS */
int samples_buffer;			/* set > 1 to decimate max 16*64bytes */

int verbose;
int debug;
FILE* fp_log;
void (*G_action)(void*);
int devnum = 0;

#define DEF_NCHAN 	16
int nchan = DEF_NCHAN;
int spadlongs = 0;

int has_do32;

#define MAX_DO32	32

float G_setpoint = 1000;

#define NSHORTS1 	(nchan + spadlongs*sizeof(unsigned)/sizeof(short))
#define NSHORTS		(NSHORTS1*samples_buffer)
#define VI_LEN 		(NSHORTS*sizeof(short))
#define VI_LONGS	(VI_LEN/sizeof(unsigned))

#define EOB(buf)	(((volatile unsigned*)(buf))[VI_LONGS-1])


#define DEBUG(str)  while(debug) { printf("%d %s say when\n", __LINE__, str); getchar(); break; }

struct XLLC_DEF xllc_def_ai = {
		.pa = RTM_T_USE_HOSTBUF,

};
struct XLLC_DEF xllc_def_ao;



#define VO_LEN  (32*sizeof(unsigned))

#define DO_IX   0

/* intermediate values are copied to a local shared memory for co-routines to observe */
#define SHM_RTERR 		1
#define SHM_BCO 		2
#define SHM_POLLCATMIN 	3
#define SHM_POLLCATMAX 	4

#define SHM_CH0	5				/* FIRST AO chan at this index */




void null_action(void* data)
{}

void write_action(void *data)
{
	fwrite(data, sizeof(short), NSHORTS, fp_log);
}


int control_none(unsigned* xo, short* ai, short ai10);


int G_buffer_copy_overruns;

int control_check_overrun(unsigned *xo, short* ai, short ai10);
int (*G_control)(unsigned *ao, short *ai, short ai10) = control_check_overrun;


int control_check_mean(unsigned *xo, short* ai, short ai10);

int control_proportional_control(unsigned *xo, short* ai, short ai10);

#define MV100   (32768/100)


int mon_chan = 0;

int G_chan_step = 1;
float G_phase_per_channel = 0;

void ui(int argc, char* argv[])
{
	const char* env;
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
	if (getenv("DEBUG")){
		debug = atoi(getenv("DEBUG"));
	}
	/* own PA eg from GPU */
	if (getenv("PA_BUF")){
		xllc_def_ai.pa = strtoul(getenv("PA_BUF"), 0, 0);
	}
	if (getenv("DO32")){
		has_do32 = atoi(getenv("DO32"));
	}
	if (getenv("NCHAN")){
		nchan = atoi(getenv("NCHAN"));
		fprintf(stderr, "NCHAN set %d\n", nchan);
	}
	if ((env = getenv("CHAN_STEP"))){
		G_chan_step = atoi(env);
	}
	if ((env = getenv("PHASE_PER_CHANNEL"))){
		G_phase_per_channel = atof(env);
	}
	if (getenv("CONTROL_CHECK_MEAN")){
		G_control = control_check_mean;
	}else if ((env = getenv("CONTROL_SETPOINT"))){
		G_setpoint = atof(env);
		G_control = control_proportional_control;
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
	host_buffer = get_mapping(devnum, &fd);
	get_shared_mapping(devnum, 1, &xllc_def_ao, (void**)&pbufferXO);
	DEBUG("get_shared_mapping");
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
	DEBUG("setup99");
	printf("AO buf pa:   0x%08x len %d\n", xllc_def_ao.pa, xllc_def_ao.len);
}

void print_sample(unsigned sample, unsigned tl)
{
	if (sample%10000 == 0){
		printf("[%10u] %10u\n", sample, tl);
	}
}


int control_none(unsigned *xo, short *ai, short ai10)
{
	unsigned* dox = (unsigned *)xo;
	unsigned* tlx = (unsigned *)ai;


}


#define MARKER 0xdeadc0d1


int control_check_overrun(unsigned *xo, short* ai, short ai10)
{
	if (ai[0] != ai10){
		return ++G_buffer_copy_overruns;
	}else{
		return 0;
	}
}

int totals[DEF_NCHAN];
int control_check_mean(unsigned *xo, short* ai, short ai10)
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
	return 0;
}

static float gain = 10.0/3000 *.01;		/* gain codes per pwm% */
static struct PWM_CTRL pwm[MAX_DO32];

void cpc_init()
{
	int ic;
	
	if (pwm[0].PWM_GP == 0){
		for (ic = 0; ic < MAX_DO32; ++ic){
			set(ic+1, pwm[ic]);
			pwm[ic].PWM_GP = 0;
			pbufferXO[ic] = pwm2raw(pwm[ic]);
		}
	}
}


int cpc(unsigned *xo, int actuals[], float duty[])
/* proportional control: run one step for each channel */
{
	int ic;
	static int cycle;

	if (cycle++ < 2){
		unsigned gp = GP_DEFAULT;
	
		for (ic = 0; ic < MAX_DO32; ++ic){
			set(ic+1, pwm[ic]);
			pwm[ic].PWM_GP = gp;
			pwm[ic].PWM_IC = gp-2;
			pbufferXO[ic] = pwm2raw(pwm[ic]);
		}
		return 0;
	}

	// assume first 16 channels have feedback
	for (ic = 0; ic < DEF_NCHAN; ic += 1){
		float error = G_setpoint - actuals[ic];
		float duty1 = duty[ic] + error*gain;

		pwm[ic] = set_duty(pwm[ic], duty1, ic*G_phase_per_channel);
		pbufferXO[ic*G_chan_step] = pwm2raw(pwm[ic]);
		duty[ic] = duty1;
	}
	return 0;
}

float dutys[DEF_NCHAN];

int control_proportional_control(unsigned *xo, short* ai, short ai10)
{
	if (control_check_mean(xo, ai, ai10) == 0){
		return cpc(xo, totals, dutys);
	}
}
void run(int (*control)(unsigned *ao, short *ai, short ai10), void (*action)(void*))
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

	DEBUG("run");

	for (ib = 0; nbuffers == 0 || ib <= nbuffers; ++ib, tl1, ab = !ab, pollcat = 0){
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
		control(pbufferXO, ai_buffer, ((short *)bufferAB[ab])[0]);
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
	if (G_control == control_proportional_control){

//		cpc_init();
	}
	printf("ready for data\n");
	run(G_control, G_action);
	printf("finished\n");
	closedown();
}
