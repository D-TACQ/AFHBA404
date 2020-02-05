/* ------------------------------------------------------------------------- *
 * afhba-llcontrol-multiuut-4AI1AO1DX.c
 * read incoming data from 1..4 uuts, control standardized payload.
 * ------------------------------------------------------------------------- *
 *   Copyright (C) 2019 Peter Milne, D-TACQ Solutions Ltd
 *                      <peter dot milne at D hyphen TACQ dot com>
 *                         www.d-tacq.com
 *   Created on: 18 November 2019
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
 * Assume standardized ACQ2106+4xACQ424+AO32+DIO32 (128AI,32DI,32AO,32DO)
 *
 * UI
 *
 DEVMAX=N 	// 1..4 sets number of devs. AFHBA404 MUST be populated from port 0 up.

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

#define AO_BUFFER(dev) ((short*)((dev)->host_buffer+AO_OFFSET))

/* limit with devmax */
int devmax = 4;




void* host_buffer;
int fd;
int nsamples = 10000000;		/* 10s at 1MSPS */
int samples_buffer = 1;			/* set > 1 to decimate max 16*64bytes */

int verbose;
int bolodev = -1;			/* bolodev does LLC, but the phase will be different to ACQ424 SAR. So, don't wait for the TLATCH, just log the data that is there */


#define FORALL 	for (int id = 0; id < devmax; ++id)
#define IS_BOLO (id == bolodev)
void control_dup1(short *ao, short *ai);
void (*G_control)(short *ao, short *ai) = control_dup1;

void (*G_action)(void*);
int devnum = 0;
int dummy_first_loop;
/** potentially good for cache fill, but sets initial value zero */
int G_POLARITY = 1;		
/** env POLARITY=-1 negates feedback this is useful to show that the
 *  software is in fact doing something
 */

/** PAYLOAD : per box. */

int has_do32 = 1;
int ai_chan = 128;
int ao_chan = 32;
#define VO_LEN  (ao_chan*sizeof(short) + (has_do32?sizeof(unsigned):0))

int DUP1 = 0; 			/* duplicate AI[DUP1], default 0 */
short *AO_IDENT;		/* AO, identity output (offset = f(chan)) */


#define DO_IX	(16)		/* longwords */
#define WD_BIT 	0x00000001	/* bit d0 is the Watchdog */

#define BOLO_MODULES	2
#define BOLO_DATA_SIZE  (8 * 3 * sizeof(unsigned))
int bololen = BOLO_MODULES*BOLO_DATA_SIZE + 16*sizeof(unsigned);

void copy_tlatch_to_do32(void *ao, void *ai)
{
	unsigned* dox = (unsigned *)ao;
	unsigned* tlx = (unsigned *)ai;

	dox[DO_IX] = tlx[SPIX];
}


void control_dup1(short *ao, short *ai)
{
         int ii;
         short* ai_01 = devs[0].lbuf;

         for (ii = 0; ii < ao_chan; ii++){
                 ao[ii] = AO_IDENT[ii] + ai_01[DUP1];
         }

         if (has_do32){
                 copy_tlatch_to_do32(ao, ai);
         }
}

void ui(int argc, char* argv[])
{
	if (getenv("DEVMAX")){
		devmax = atoi(getenv("DEVMAX"));
		printf("DEVMAX set %d\n", devmax);
	}

        if (getenv("RTPRIO")){
		sched_fifo_priority = atoi(getenv("RTPRIO"));
        }
        if (getenv("AFFINITY")){
                setAffinity(strtol(getenv("AFFINITY"), 0, 0));
        }

	if (getenv("VERBOSE")){
		verbose = atoi(getenv("VERBOSE"));
	}
	if (getenv("BOLODEV")){
		bolodev = atoi(getenv("BOLODEV"));
		if (getenv("BOLOLEN")) {
			bololen = atoi(getenv("BOLOLEN"));
		}
	}


	if (getenv("DUMMY_FIRST_LOOP")){
		dummy_first_loop = atoi(getenv("DUMMY_FIRST_LOOP"));
	}

	nchan = ai_chan + has_do32*(sizeof(u32)/sizeof(short));

	if (getenv("POLARITY")){
		G_POLARITY = atoi(getenv("POLARITY"));
		fprintf(stderr, "G_POLARITY set %d\n", G_POLARITY);
	}

	spadlongs = has_do32? 15: 16;

	if (argc > 1){
		nsamples = atoi(argv[1]);
		fprintf(stderr, "nsamples set %d\n", nsamples);
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
        {
                int ao_ident = 0;
                if (getenv("AO_IDENT")){
                        ao_ident = atoi(getenv("AO_IDENT"));
                }
                AO_IDENT = make_ao_ident(ao_chan, ao_ident);
        }
}

void _get_connected(struct Dev* dev, unsigned vi_len)
{
	struct XLLC_DEF xo_xllc_def;

	dev->host_buffer = get_mapping(dev->devnum, &dev->fd);
	dev->xllc_def.pa = RTM_T_USE_HOSTBUF;
	dev->xllc_def.len = samples_buffer*vi_len;
	memset(dev->host_buffer, 0, vi_len);
	dev->lbuf = calloc(vi_len, 1);
	if (ioctl(dev->fd, AFHBA_START_AI_LLC, &dev->xllc_def)){
		perror("ioctl AFHBA_START_AI_LLC");
		exit(1);
	}
	printf("[%d] AI buf pa: 0x%08x len %d\n", dev->devnum, dev->xllc_def.pa, dev->xllc_def.len);

	xo_xllc_def = dev->xllc_def;
	xo_xllc_def.pa += AO_OFFSET;
	xo_xllc_def.len = VO_LEN;

	if (has_do32){
		int ll = xo_xllc_def.len/64;
		xo_xllc_def.len = ++ll*64;
	}
	if (ioctl(dev->fd, AFHBA_START_AO_LLC, &xo_xllc_def)){
		perror("ioctl AFHBA_START_AO_LLC");
		exit(1);
	}
	printf("AO buf pa: 0x%08x len %d\n", xo_xllc_def.pa, xo_xllc_def.len);

	if (has_do32){
		/* marker pattern for the PAD area for hardware trace */
		unsigned* dox = (unsigned *)AO_BUFFER(dev);
		int ii;
		for (ii = 0; ii <= 0xf; ++ii){
		        dox[DO_IX+ii] = (ii<<24)|(ii<<16)|(ii<<8)|ii;
		}
	}
}
void get_connected()
{
	FORALL {
		_get_connected(&devs[id], IS_BOLO? bololen: VI_LEN);
	}
}
void setup()
{
	setup_logging(devnum);
	get_connected();
	goRealTime();
}

void print_sample(unsigned sample, unsigned tl)
{
	if (sample%10000 == 0){
		printf("[%10u] %10u\n", sample, tl);
	}
}

void dio_watchdog(short *ao)
{
        unsigned* dox = (unsigned *)ao;

	dox[DO_IX] ^= WD_BIT;
}

void run(void (*control)(short *ao, short *ai), void (*action)(void*))
{
	unsigned tl0[4];
	int pollcat[4] = {};
	unsigned tl1;
	unsigned sample;

	mlockall(MCL_CURRENT);

	FORALL TLATCH(devs[id].host_buffer)[0] = 
	       TLATCH(devs[id].lbuf)[0] = 
	       tl0[id] = 0xdeadbeef; 
	if (dummy_first_loop){
		TLATCH(devs[0].host_buffer)[0] -= 3;	/* force first time pass thru */
	}

	for (sample = 0; sample <= nsamples; ++sample, memset(pollcat, 0, sizeof(pollcat))){
		FORALL {
			while(!IS_BOLO && (tl1 = TLATCH(devs[id].host_buffer)[0]) == tl0[id]){
				if (id == 0 && sample == 0) dio_watchdog(AO_BUFFER(devs+id));
				sched_fifo_priority>1 || sched_yield();
				++pollcat[id];
			}
			memcpy(devs[id].lbuf, devs[id].host_buffer, VI_LEN);
			tl0[id] = tl1;
		}

		FORALL {
			if (!IS_BOLO){
				control(AO_BUFFER(devs+id), devs[id].lbuf);
				/* TLATCH [1] is usecs from HW */
				TLATCH(devs[id].lbuf)[2] = pollcat[id];
				TLATCH(devs[id].lbuf)[3] = difftime_us();
			}
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
	run(G_control, G_action);
	printf("finished\n");
	closedown();
}
