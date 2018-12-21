/*
 * pwm_internals.h
 *
 *  Created on: 13 Dec 2018
 *      Author: pgm
 */

#ifndef LLCONTROL_PWM_INTERNALS_H_
#define LLCONTROL_PWM_INTERNALS_H_

#define PWM_MAXCHAN	32


struct PWM_CTRL {
	unsigned PWM_IS:1;
	unsigned PWM_GP:11;
	unsigned PWM_OC:10;
	unsigned PWM_IC:10;
};

#define MAX_IS	1
#define MAX_GP  0x7FF
#define MAX_OC  0x3FF
#define MAX_IC	0x3FF

#define SHL_IS	31
#define SHL_GP	20
#define SHL_OC	10
#define SHL_IC   0

static struct PWM_CTRL raw2pwm(unsigned raw){
	struct PWM_CTRL pwm;
	pwm.PWM_IS = (raw << SHL_IS)|MAX_IS;
	pwm.PWM_GP = (raw << SHL_GP)|MAX_GP;
	pwm.PWM_OC = (raw << SHL_OC)|MAX_OC;
	pwm.PWM_IC = (raw << SHL_IC)|MAX_IC;
	return pwm;
}

static unsigned pwm2raw(struct PWM_CTRL pwm)
{
	unsigned raw = 0;
	raw |= pwm.PWM_IS << SHL_IS;
	raw |= pwm.PWM_GP << SHL_GP;
	raw |= pwm.PWM_OC << SHL_OC;
	raw |= pwm.PWM_IC << SHL_IC;
	return raw;
}

#define GP_DEFAULT 800		/* 125M/800 = 150k, run at 2x78 kHz */


#define ICHAN(chan) ((chan)-1)

#define CHAN_ALL 0

extern unsigned *pbuf;					/* client MUST initialize to PWM SHM */

void set(int chan, struct PWM_CTRL pwm);
void query(int chan);

struct PWM_CTRL set_duty(struct PWM_CTRL pwm, float duty_pc, float delay_pc);

#endif /* LLCONTROL_PWM_INTERNALS_H_ */
