/*
 * pwm_set_duty.c
 * pwm_set_duty CH DUTY% DELAY%   (DUTY 0..100, DELAY 0..50)
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



#define ICHAN(chan) ((chan)-1)

#define CHAN_ALL 0

#include "pwm_internals.h"


int devnum = 0;
int ibuf = 1;

unsigned *pbuf;

#define GP_DEFAULT 800		/* 125M/800 = 150k, run at 2x78 kHz */

void _set(int chan, struct PWM_CTRL pwm)
{
	pbuf[ICHAN(chan)] = pwm2raw(pwm);
}

void set(int chan, struct PWM_CTRL pwm){
	if (chan == CHAN_ALL){
		int cc;
		for (cc = 1; cc <= PWM_MAXCHAN; ++cc){
			_set(cc, pwm);
		}
	}else{
		_set(chan, pwm);
	}
}
void _query(int chan)
{
	struct PWM_CTRL pwm = raw2pwm(pbuf[ICHAN(chan)]);

	printf("{ ch:%02d,is:%d,gp:%4d,ic:%d,oc:%d } ", chan,
			pwm.PWM_IS, pwm.PWM_GP, pwm.PWM_IC, pwm.PWM_OC);
}

void query(int chan)
{
	if (chan == CHAN_ALL){
		int cc;
		for (cc = 1; cc <= PWM_MAXCHAN; ++cc){
			_query(cc);
		}
	}else{
		_query(chan);
	}
	printf("\n");
}

unsigned limit(unsigned xx, unsigned _max)
{
	return xx <= _max? xx: _max;
}

unsigned alimit(const char *xxs, unsigned _max)
{
	return limit(strtoul(xxs, 0, 0), _max);
}

float flimit(float xx, unsigned _max)
{
	return xx <= _max? xx: _max;
}
float falimit(const char *xxs, unsigned _max)
{
        return flimit(atof(xxs), _max);
}


struct PWM_CTRL set_duty(struct PWM_CTRL pwm, float duty_pc, float delay_pc)
/* duty_pc 0..99.9%, delay_pc 0..50% */
{
	if (duty_pc + delay_pc < 100){
		pwm.PWM_IS = 0;
		pwm.PWM_IC = delay_pc*pwm.PWM_GP/100;
		if (pwm.PWM_IC == 0) pwm.PWM_IC = 1;
		pwm.PWM_OC = pwm.PWM_IC + duty_pc*pwm.PWM_GP/100;
	}else{
		pwm.PWM_IS = 1;
		pwm.PWM_OC = delay_pc*pwm.PWM_GP/100;
		pwm.PWM_IC = duty_pc*pwm.PWM_GP/100 + pwm.PWM_OC - pwm.PWM_GP;
	}
	fprintf(stderr, "duty %.1f delay %.1f GP %u IS %u IC %u OC %u\n", 
			duty_pc, delay_pc, pwm.PWM_GP, pwm.PWM_IS, pwm.PWM_IC, pwm.PWM_OC);
	return pwm;
}

int main(int argc, char* argv[])
{

	struct PWM_CTRL pwm_ctrl = {};
	int chan = CHAN_ALL;
	float duty = 50.0;
	float delay = 0.0;

	assert(sizeof(struct PWM_CTRL) == sizeof(unsigned));

	if (argc > 1 && strpbrk(argv[1], "h?H")){
		fprintf(stderr, "pwm_set_channel chan is group icount ocount\n");
		return 0;
	}

	if (getenv("DEVNUM")){
		devnum = atoi(getenv("DEVNUM"));
	}
	if (getenv("IBUF")){
		ibuf = atoi(getenv("IBUF"));
	}
	if (getenv("PWM_GP")){
		pwm_ctrl.PWM_GP = atoi(getenv("PWM_GP"));
	}else{
		pwm_ctrl.PWM_GP = GP_DEFAULT;
	}
	get_shared_mapping(devnum, ibuf, 0, (void**)&pbuf);

	if (argc > 1) chan = alimit(argv[1], PWM_MAXCHAN);
	if (argc > 2){
		duty = falimit(argv[2], 100);
		if (argc > 3){
			delay = falimit(argv[3], 50);
		}

		pwm_ctrl = set_duty(pwm_ctrl, duty, delay);
		set(chan, pwm_ctrl);
	}else{
		fprintf(stderr, "pwm_set_duty CHAN DUTY_PC DELAY_PC\n");
		query(chan);
	}
}
