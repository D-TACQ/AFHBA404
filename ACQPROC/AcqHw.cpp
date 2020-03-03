/*
 * AcqHw.cpp  :ACQ device with hardware hooks
 *
 *  Created on: 1 Mar 2020
 *      Author: pgm
 */

extern "C" {
#include "../local.h"
#include "../rtm-t_ioctl.h"
#include "afhba-llcontrol.h"

}

#include "AcqSys.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string.h>

#include <stdio.h>		// because sprintf()

#include <assert.h>

#include <sys/mman.h>

/** struct Dev : interface to AFHBA404 device driver. */
struct Dev {
	int devnum;
	int fd;
	char* host_buffer;
	char* lbuf;
	char* _lbuffer;
	struct XLLC_DEF xllc_def;

	Dev() {
		memset(this, 0, sizeof(Dev));
	}
};

int samples_buffer = 1;

#define AO_OFFSET 0x1000

#define XO_HOST	(dev->host_buffer+AO_OFFSET)


void _get_connected(struct Dev* dev, unsigned vi_len)
{

}

extern int devnum;

/* TLATCH now uses the dynamic set value */
#undef TLATCH
#define TLATCH	(*(unsigned*)(dev->host_buffer + vi_offsets.SP32))

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
	dev->_lbuffer = (char*)calloc(vi.len(), HBA::maxsam);
	dev->lbuf = dev->_lbuffer;
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
		dox = (unsigned *)(XO_HOST + vo_offsets.DO32);
		int ii;
		for (ii = 0; ii <= 0xf; ++ii){
		        dox[ii] = (ii<<24)|(ii<<16)|(ii<<8)|ii;
		}
	}
	TLATCH = 0xdeadbeef;
}

void HBA::start_shot()
{
	mlockall(MCL_CURRENT);
	goRealTime();
}

#define VITOSI(field, sz) \
	(vi.field && memcpy((char*)systemInterface.IN.field+vi_cursor.field, dev->lbuf+vi_offsets.field, vi.field*sz))

#define SITOVO(field, sz) \
	(vo.field && memcpy(XO_HOST+vo_offsets.field, (char*)systemInterface.OUT.field+vo_cursor.field, vo.field*sz))

void ACQ_HW::action(SystemInterface& systemInterface)
/**< copy SI to VO, copy VI to SI, advance local buffer pointer. */
{
	SITOVO(AO16, sizeof(short));
	SITOVO(DO32, sizeof(unsigned));

	VITOSI(AI16, sizeof(short));
	VITOSI(AI32, sizeof(unsigned));
	VITOSI(DI32, sizeof(unsigned));
	((unsigned*)dev->lbuf+vi_offsets.SP32)[2] = pollcount;
	pollcount = 0;
	VITOSI(SP32, sizeof(unsigned));
	dev->lbuf += vi.len();
}
ACQ_HW::~ACQ_HW() {
	const char* fname = (getName()+".dat").c_str();
	FILE *fp = fopen(fname, "w");
	if (fp == 0){
		perror(fname);
	}
	fwrite(dev->_lbuffer, vi.len(), HBA::maxsam, fp);
	fclose(fp);
	clear_mapping(dev->fd, dev->host_buffer);
}



/** checks host buffer for new sample, if so copies to lbuf and reports true */
bool ACQ_HW::newSample(int sample)
{
        unsigned tl1;

	if (nowait || (tl1 = TLATCH) != tl0){
		memcpy(dev->lbuf, dev->host_buffer, vi.len());
                tl0 = tl1;
		return true;
	}else if (sample == 0 && wd_mask){
		dox[0] ^= wd_mask;
		return false;
	}else{
		return false;
	}
}

/** returns latest tlatch from lbuf */
unsigned ACQ_HW::tlatch(void)
{
	return dev->lbuf[vi_offsets.SP32];
}
/** prepare to run a shot nsamples long, arm the UUT. */
void ACQ_HW:: arm(int nsamples)
{
	cerr << "ACQ_HW::arm: TODO" <<endl;
}
