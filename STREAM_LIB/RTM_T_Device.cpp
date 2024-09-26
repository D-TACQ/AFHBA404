/*
 * RTM_T_Device.cpp
 *
 *  Created on: Apr 28, 2011
 *      Author: pgm
 */

/** @file RTM_T_Device.cpp
 *  @brief interface to RTM_T Device (historic name for AFHBA404)
 */

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>

#include "local.h"
#include "RTM_T_Device.h"




static unsigned calc_nbuffers(unsigned devnum);
static unsigned calc_maxlen(unsigned devnum);
static unsigned calc_transfer_buffers(unsigned devnum);

RTM_T_Device::RTM_T_Device(int _devnum) :
	devnum(_devnum%100), nbuffers(calc_nbuffers(devnum)),
	maxlen(calc_maxlen(devnum)),
	transfer_buffers(calc_transfer_buffers(devnum))
{
	char buf[120];

	sprintf(buf, "/dev/rtm-t.%d", devnum);
	names[MINOR_DMAREAD] = buf;
	_open(MINOR_DMAREAD);

	sprintf(buf, "/dev/rtm-t.%d.regs", devnum);
	names[MINOR_REGREAD] = buf;
	_open(MINOR_REGREAD);

	for (int ib = 0; ib < nbuffers; ++ib){
		sprintf(buf, "/dev/rtm-t.%d.data/hb%02d", devnum, ib);
		names[ib] = buf;

		_open(ib);
		void *va = mmap(0, maxlen, PROT_READ|PROT_WRITE,
				MAP_SHARED, handles[ib], 0);

		if (va == (caddr_t)-1 ){
			perror( "mmap" );
		        _exit(errno);
		}else{
			host_buffers[ib] = va;
		}
	}

	sprintf(buf, "/dev/rtm-t.%d.ctrl", devnum);
	names[CTRL_ROOT] = buf;
}

static int getKnob(const char* knob, unsigned* value)
{
        FILE *fp = fopen(knob, "r");
	if (!fp)
	{
		perror(knob);
		return -1;
	}
        int rc = fscanf(fp, "%u", value);
        fclose(fp);
        return rc;
}

#define PARAMETERS              "/sys/module/afhba/parameters/"
#define BUFFER_LEN              PARAMETERS "buffer_len"
#define NBUFFERS                PARAMETERS "nbuffers"
#define TRANSFER_BUFFERS        PARAMETERS "transfer_buffers"

static unsigned calc_transfer_buffers(unsigned devnum)
{
	unsigned transfer_buffers = 0;
	if (getKnob(TRANSFER_BUFFERS, &transfer_buffers) != 1){
		perror("getKnob " TRANSFER_BUFFERS " failed");
		exit(1);
	}
	return transfer_buffers;
}

static unsigned calc_nbuffers(unsigned devnum)
{
	unsigned nbuffers = 0;
	if (getKnob(NBUFFERS, &nbuffers) != 1){
		perror("getKnob " NBUFFERS " failed");
		exit(1);
	}
	return nbuffers;
}
static unsigned calc_maxlen(unsigned devnum)
{
        char knob[80];
	unsigned maxlen;
	snprintf(knob, 80, "/dev/rtm-t.%u.ctrl/buffer_len", devnum);
	if (getKnob(knob, &maxlen) == 1){
		return maxlen;
	}else if (getKnob(BUFFER_LEN, &maxlen) == 1){
		return maxlen;
        }else{
		perror("maxlen not set");
		exit(1);
	}
}
