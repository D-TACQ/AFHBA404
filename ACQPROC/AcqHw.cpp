/*
 * AcqHw.cpp  :ACQ device with hardware hooks
 *
 *  Created on: 1 Mar 2020
 *      Author: pgm
 */

extern "C" {
#include "../LLCONTROL/afhba-llcontrol-common.h"
}

#include "AcqSys.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string.h>

#include <stdio.h>		// because sprintf()

#include <assert.h>


struct Dev {
	int devnum;
	int fd;
	char* host_buffer;
	char* lbuf;
	struct XLLC_DEF xllc_def;

	Dev() {
		memset(this, 0, sizeof(Dev));
	}
};

int samples_buffer = 1;

#define AO_OFFSET 0x1000


void _get_connected(struct Dev* dev, unsigned vi_len)
{

}

extern int devnum;

ACQ_HW::ACQ_HW(string _name, VI _vi, VO _vo, VI _vi_offsets,
			VO _vo_offsets, VI& sys_vi_cursor, VO& sys_vo_cursor) :
				ACQ(_name, _vi, _vo, _vi_offsets,
							_vo_offsets, sys_vi_cursor, sys_vo_cursor),
				tl0(0xdeadbeef), dev(new Dev)
{

	// @@todo init dev.
	struct XLLC_DEF xo_xllc_def;
	dev->devnum = ::devnum++;

	dev->host_buffer = (char*)get_mapping(dev->devnum, &dev->fd);
	dev->xllc_def.pa = RTM_T_USE_HOSTBUF;
	dev->xllc_def.len = samples_buffer*vi.len();
	memset(dev->host_buffer, 0, vi.len());
	dev->lbuf = (char*)calloc(vi.len(), 1);
	if (ioctl(dev->fd, AFHBA_START_AI_LLC, &dev->xllc_def)){
		perror("ioctl AFHBA_START_AI_LLC");
		exit(1);
	}
	printf("[%d] AI buf pa: 0x%08x len %d\n", dev->devnum, dev->xllc_def.pa, dev->xllc_def.len);

	xo_xllc_def = dev->xllc_def;
	xo_xllc_def.pa += AO_OFFSET;
	xo_xllc_def.len = vo.len();

	if (vo.DO32){
		int ll = xo_xllc_def.len/64;
		xo_xllc_def.len = ++ll*64;
	}
	if (ioctl(dev->fd, AFHBA_START_AO_LLC, &xo_xllc_def)){
		perror("ioctl AFHBA_START_AO_LLC");
		exit(1);
	}
	printf("AO buf pa: 0x%08x len %d\n", xo_xllc_def.pa, xo_xllc_def.len);

	if (vo.DO32){
		/* marker pattern for the PAD area for hardware trace */
		unsigned* dox = (unsigned *)(dev->host_buffer+AO_OFFSET);
		int ii;
		for (ii = 0; ii <= 0xf; ++ii){
		        dox[vo_offsets.DO32/2+ii] = (ii<<24)|(ii<<16)|(ii<<8)|ii;
		}
	}
}

/* TLATCH now uses the dynamic set value */
#undef TLATCH
#define TLATCH	(dev->host_buffer[vi_offsets.SP32])

bool ACQ_HW::newSample(int sample)
/*< checks host buffer for new sample, if so copies to lbuf and reports true */
{
	if (nowait || TLATCH != tl0){
		memcpy(dev->lbuf, dev->host_buffer, vi.len());
		return true;
	}else if (sample == 0 && wd_mask){
		unsigned* dox = (unsigned *)(dev->host_buffer+AO_OFFSET);
		dox[vo_offsets.DO32/2] ^= wd_mask;
		return false;
	}else{
		return false;
	}
}
unsigned ACQ_HW::tlatch(void)
/*< returns latest tlatch from lbuf */
{
	return dev->lbuf[vi_offsets.SP32];
}
void ACQ_HW:: arm(int nsamples)
/*< prepare to run a shot nsamples long, arm the UUT. */
{

}
