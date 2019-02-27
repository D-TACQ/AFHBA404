/* ------------------------------------------------------------------------- *
 * afhba-llcontrol-multiuut.c
 * read incoming data from nuuts.
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

/* SPLIT single HB into 2
 * [0] : AI
 * [1] : AO
 */

#define AO_OFFSET (HB_LEN/2)

/* hardcoded for 4 devs */

struct Dev {
	int devnum;
	int fd;
	void* host_buffer;
	void* lbuf;
	struct XLLC_DEF xllc_def;
};

struct Dev devs[4] = {
		{ 0, }, { 1, }, {2, }, {3, }
};
/* limit with devmax */
int devmax = 4;

#define FORALL for (int id = 0; id < devmax; ++id)

#undef TLATCH
#define TLATCH(id) ((volatile unsigned*)(devs[id].lbuf)+SPIX)      /* actually, sample counter */


void* host_buffer;
int fd;
int nsamples = 10000000;		/* 10s at 1MSPS */
int samples_buffer = 1;			/* set > 1 to decimate max 16*64bytes */

int verbose;

void (*G_action)(void*);
int devnum = 0;
int dummy_first_loop;
/** potentially good for cache fill, but sets initial value zero */
int G_POLARITY = 1;		
/** env POLARITY=-1 negates feedback this is usefult to know that the 
 *  software is in fact doing something 					 */



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
	if (getenv("DEVMAX")){
		devmax = atoi(getenv("DEVMAX"));
	}
	/* own PA eg from GPU */
	if (getenv("PA_BUF")){
		unsigned pa_buf = strtoul(getenv("PA_BUF"), 0, 0);
		FORALL {
			devs[id].xllc_def.pa = pa_buf;
			pa_buf += VI_LEN;
		}

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

void _get_connected(struct Dev* dev)
{
	dev->host_buffer = get_mapping(dev->devnum, &dev->fd);
	dev->xllc_def.pa = RTM_T_USE_HOSTBUF;
	dev->xllc_def.len = samples_buffer*VI_LEN;
	memset(dev->host_buffer, 0, VI_LEN);
	dev->lbuf = calloc(VI_LEN, 1);
	if (ioctl(dev->fd, AFHBA_START_AI_LLC, &dev->xllc_def)){
		perror("ioctl AFHBA_START_AI_LLC");
		exit(1);
	}
	printf("[%d] AI buf pa: 0x%08x len %d\n", dev->devnum, dev->xllc_def.pa, dev->xllc_def.len);

}
void get_connected()
{
	int idev;
	for (idev = 0; idev < devmax; ++idev){
		_get_connected(&devs[idev]);
	}
}
void setup()
{
	setup_logging(devnum);
	host_buffer = get_mapping(devnum, &fd);
	goRealTime();
}

void print_sample(unsigned sample, unsigned tl)
{
	if (sample%10000 == 0){
		printf("[%10u] %10u\n", sample, tl);
	}
}


void run(void (*action)(void*))
{
	short* ai_buffer = calloc(NSHORTS, sizeof(short));
	unsigned tl0 = 0xdeadbeef;	/* always run one dummy loop */
	unsigned tl1;
	unsigned sample;
	int println = 0;
	int pollcat[4] = {};


	mlockall(MCL_CURRENT);
	FORALL TLATCH(id)[0] = tl0;

	for (sample = 0; sample <= nsamples; ++sample, tl0 = tl1, memset(pollcat, 0, sizeof(pollcat))){
		FORALL {
			memcpy(devs[id].lbuf, devs[id].host_buffer, VI_LEN);
			while((tl1 = TLATCH(id)[0]) == tl0){
				sched_yield();
				memcpy(devs[id].lbuf, devs[id].host_buffer, VI_LEN);
				++pollcat[id];
			}
		}

		FORALL {
			TLATCH(id)[1] = pollcat[id];
			TLATCH(id)[2] = difftime_us();
			action(devs[id].lbuf);
		}

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
	run(G_action);
	printf("finished\n");
	closedown();
}
