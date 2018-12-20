/*
 * pwm_control.c : simple control loop. NB: this SHOULD run as part of the RT process, but
 * it's easy/lazy to prototype it as a regular user task.
 *
 *  Created on: 13 Dec 2018
 *      Author: pgm
 */


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <assert.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "afhba-get_shared_mapping.h"



#define ICHAN(chan) ((chan)-1)

#define CHAN_ALL 0

#include "pwm_internals.h"


#define LLC_VI_SHM 	"/dev/shm/afhba-llcontrol"
#define AI01		5				// first AO at [5] in VI
int devnum = 0;
int ibuf = 1;

unsigned *pbuf;

#define GP_DEFAULT 800		/* 125M/800 = 150k, run at 2x78 kHz */

int* get_feedback()
{
	int fd = open("/dev/shm/afhba-llcontrol", O_RDONLY);
	int *mapping;
	assert(fd != -1);
	mapping = mmap(0, 4096, PROT_READ, MAP_SHARED, fd, 0);
	assert(mapping != MAP_FAILED);
	return mapping + AI01 - 1;		// index channels from 1
}
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

void control_loop(int chan, float setpoint, float gain)
{
	int *ai = get_feedback();
	struct PWM_CTRL pwm_ctrl = {};
	float duty = 50;
	float delay = 0;

	if (getenv("PWM_GP")){
		pwm_ctrl.PWM_GP = atoi(getenv("PWM_GP"));
	}else{
		pwm_ctrl.PWM_GP = GP_DEFAULT;
	}

	while(1) {
		float error = setpoint - ai[chan];
		float duty1 = duty + error*gain;
		fprintf(stderr, "set:%.2f actual:%d err:%.2f duty %.2f -> duty1 %.2f\n", 
				setpoint, ai[chan], error, duty, duty1);
		pwm_ctrl = set_duty(pwm_ctrl, duty1, delay);
		set(chan, pwm_ctrl);
		duty = duty1;
		sleep(1);
	}
}
int main(int argc, char* argv[])
{
	int chan = CHAN_ALL;
	float setpoint = 1000;		/* codes, different modules have different cal */
	float gain = 10.0/3000 *.5;		/* gain codes per pwm% */
	assert(sizeof(struct PWM_CTRL) == sizeof(unsigned));

	fprintf(stderr, "default gain %f\n", gain);

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
	get_shared_mapping(devnum, ibuf, 0, (void**)&pbuf);

	if (argc > 1) chan = falimit(argv[1], PWM_MAXCHAN);
	if (argc > 2){
		setpoint = falimit(argv[2], 32767);
		if (argc > 3){
			gain = falimit(argv[3], gain);
			fprintf(stderr, "user gain %s %f\n", argv[3], gain);
		}
		control_loop(chan, setpoint, gain);
	}else{
		fprintf(stderr, "pwm_set_duty CHAN DUTY_PC DELAY_PC\n");
		query(chan);
	}
}
