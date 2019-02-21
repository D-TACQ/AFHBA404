
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <assert.h>

#include "pwm_internals.h"

unsigned *pbufferXO;


void _set(int chan, struct PWM_CTRL pwm)
{
	pbufferXO[ICHAN(chan)] = pwm2raw(pwm);
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
	struct PWM_CTRL pwm = raw2pwm(pbufferXO[ICHAN(chan)]);

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
/*	
	fprintf(stderr, "duty %.1f delay %.1f GP %u IS %u IC %u OC %u\n",
			duty_pc, delay_pc, pwm.PWM_GP, pwm.PWM_IS, pwm.PWM_IC, pwm.PWM_OC);

*/
	assert(pwm.PWM_GP == GP_DEFAULT);
	assert(pwm.PWM_GP != 0);
	return pwm;
}
