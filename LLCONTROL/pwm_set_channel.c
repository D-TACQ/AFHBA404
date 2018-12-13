/*
 * pwm_set_channel.c
 *
 *  Created on: 13 Dec 2018
 *      Author: pgm
 */


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <assert.h>

#include "afhba-get_shared_mapping.h"

#define MAXCHAN	32

#define ICHAN(chan) ((chan)-1)

#define CHAN_ALL 0

struct PWM_CTRL {
	unsigned PWM_IS:1;
	unsigned PWM_GP:11;
	unsigned PWM_OC:10;
	unsigned PWM_IC:10;
};


int devnum = 0;
int ibuf = 0;

unsigned *pbuf;

void _query(int chan)
{
	struct PWM_CTRL pwm;

	memcpy(&pwm, &pbuf[ICHAN(chan)], sizeof(unsigned));
	printf("ch:%02d, is:%d gp:%4d, ic:%d oc:%d ", chan,
			pwm.PWM_IS, pwm.PWM_GP, pwm.PWM_IC, pwm.PWM_OC);
}

void query(int chan)
{
	if (chan == CHAN_ALL){
		int cc;
		for (cc = 1; cc <= MAXCHAN; ++cc){
			_query(cc);
		}
	}else{
		_query(chan);
	}
	printf("\n");
}


int main(int argc, char* argv[])
{

	struct PWM_CTRL pwm_ctrl = {};
	int chan;

	assert(sizeof(struct PWM_CTRL) == sizeof(unsigned));

	fprintf(stderr, "pwm_set_channel chan is group icount ocount ");
	get_shared_mapping(devnum, ibuf, 0, (void**)&pbuf);

	if (argc > 1) chan = atoi(argv[1]);
	if (argc > 2){
		pwm_ctrl.PWM_IS = strtoul(argv[2], 0, 0);
	}else{
		query(chan);
	}
}
