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

#include "pwm_internals.h"


int devnum = 0;
int ibuf = 1;

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
