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


#define PAGE_SIZE	0x1000


/** struct Dev : interface to AFHBA404 device driver. */
struct Dev {
	int devnum;
	int fd;
	char* host_buffer;
	struct LBUF {
		char* base;
		char* cursor;
	} lbuf_vi, lbuf_vo;

	struct XLLC_DEF xllc_def;

	Dev(int _devnum) {
		memset(this, 0, sizeof(Dev));
		devnum = _devnum;
	}
};

/* XO uses SAME kbuffer as AI, but 4K up */
#define AO_OFFSET 0x1000

#define XO_HOST	(dev->host_buffer+AO_OFFSET)


void _get_connected(struct Dev* dev, unsigned vi_len)
{

}

void HBA::start_shot()
{
	mlockall(MCL_CURRENT);
	goRealTime();
}


/** concrete model of ACQ2106 box. */
class ACQ_HW_BASE: public ACQ
{
	/** store raw data to file. */
	static void raw_store(const char* fname, const char* base, int len)
	{
		FILE *fp = fopen(fname, "w");
		if (fp == 0){
			perror(fname);
		}
		fwrite(base, len, HBA::maxsam, fp);
		fclose(fp);
	}

protected:
	Dev* dev;
	unsigned tl0;

	unsigned *dox;

	ACQ_HW_BASE(int devnum, string _name, VI _vi, VO _vo, VI _vi_offsets,
			VO _vo_offsets, VI& sys_vi_cursor, VO& sys_vo_cursor) :
		ACQ(devnum, _name, _vi, _vo, _vi_offsets,
				_vo_offsets, sys_vi_cursor, sys_vo_cursor),
				tl0(0xdeadbeef), dev(new Dev(devnum))
	{
		dev->host_buffer = (char*)get_mapping(dev->devnum, &dev->fd);
		dev->xllc_def.pa = RTM_T_USE_HOSTBUF;
		dev->xllc_def.len = G::samples_buffer*vi.len();
		dev->lbuf_vi.base = (char*)calloc(vi.len(), HBA::maxsam);
		dev->lbuf_vi.cursor = dev->lbuf_vi.base;
		dev->lbuf_vo.base = (char*)calloc(vo.len(), HBA::maxsam);
		dev->lbuf_vo.cursor = dev->lbuf_vo.base;
	}
	virtual ~ACQ_HW_BASE() {
		raw_store((getName()+"_VI.dat").c_str(), dev->lbuf_vi.base, vi.len());
		raw_store((getName()+"_VO.dat").c_str(), dev->lbuf_vo.base, vo.len());

		clear_mapping(dev->fd, dev->host_buffer);
	}

	virtual void arm(int nsamples);
	/**< prepare to run a shot nsamples long, arm the UUT. */
	virtual unsigned tlatch(void);
	/**< returns latest tlatch from lbuf */
};

class ACQ_HW: public ACQ_HW_BASE
{
	int sample;
public:
	ACQ_HW(int devnum, string _name, VI _vi, VO _vo, VI _vi_offsets,
			VO _vo_offsets, VI& sys_vi_cursor, VO& sys_vo_cursor);
	virtual ~ACQ_HW()
	{}
	virtual bool newSample(int sample);
	/**< checks host buffer for new sample, if so copies to lbuf and reports true */
	virtual void action(SystemInterface& systemInterface);
	/**< on newSample, copy VO from SI, copy VI to SI */
	virtual void action2(SystemInterface& systemInterface);
	/**< late action(), cleanup */
};

/* TLATCH now uses the dynamic set value */
#undef TLATCH
/** find sample count in VI. */
#define TLATCH0	((unsigned*)(dev->host_buffer + vi_offsets.SP32))[SPIX::TLATCH]

ACQ_HW::ACQ_HW(int devnum, string _name, VI _vi, VO _vo, VI _vi_offsets,
			VO _vo_offsets, VI& sys_vi_cursor, VO& sys_vo_cursor) :
		ACQ_HW_BASE(devnum, _name, _vi, _vo, _vi_offsets,
						_vo_offsets, sys_vi_cursor, sys_vo_cursor), sample(0)
{
	if (ioctl(dev->fd, AFHBA_START_AI_LLC, &dev->xllc_def)){
		perror("ioctl AFHBA_START_AI_LLC");
		exit(1);
	}
	if (G::verbose){
		printf("[%d] AI buf pa: 0x%08x len %d\n", dev->devnum, dev->xllc_def.pa, dev->xllc_def.len);
	}

	if (vo.len()){
		struct XLLC_DEF xo_xllc_def;
		if (vo.len()){
			xo_xllc_def = dev->xllc_def;
			xo_xllc_def.pa += AO_OFFSET;
			xo_xllc_def.len = vo.hwlen();

			if (vo.DO32){
				int ll = xo_xllc_def.len/64;
				xo_xllc_def.len = ++ll*64;
				dox = (unsigned *)(XO_HOST + vo_offsets.DO32);
			}
			if (ioctl(dev->fd, AFHBA_START_AO_LLC, &xo_xllc_def)){
				perror("ioctl AFHBA_START_AO_LLC");
				exit(1);
			}
			if (G::verbose){
				printf("[%d] AO buf pa: 0x%08x len %d\n", dev->devnum, xo_xllc_def.pa, xo_xllc_def.len);
			}

			if (vo.DO32){
				const char* hw_trace = getenv("DO32_HW_TRACE");
				if (hw_trace && atoi(hw_trace)){
				/* marker pattern for the PAD area for hardware trace */
					for (int ii = 0; ii <= 0xf; ++ii){
						dox[ii] = (ii<<24)|(ii<<16)|(ii<<8)|ii;
					}
				}
			}
		}
	}
	TLATCH0 = 0xdeadbeef;
}


/** copy VI.field to SI.field */
#define VITOSI(field) \
	(vi.field && \
	 memcpy(reinterpret_cast<char*>(systemInterface.IN.field+vi_cursor.field), dev->lbuf_vi.cursor+vi_offsets.field, \
			vi.field*sizeof(systemInterface.IN.field[0])))

/** copy SI.field to VO */
#define SITOVO(field) \
	(vo.field && \
	 memcpy(XO_HOST+vo_offsets.field, reinterpret_cast<char*>(systemInterface.OUT.field+vo_cursor.field), \
			 vo.field*sizeof(systemInterface.OUT.field[0])))

void ACQ_HW::action(SystemInterface& systemInterface)
/**< copy SI to VO, copy VI to SI, advance local buffer pointer. */
{
	VITOSI(AI16);

	if (G::verbose > 1){
		fprintf(stderr, "VITOSI(AI32) \"%s\" memcpy(%p, %p, %ld)\n", toString().c_str(),
				systemInterface.IN.AI32+vi_cursor.AI32,
				dev->lbuf_vi.cursor+vi_offsets.AI32,
				vi.AI32*sizeof(systemInterface.IN.AI32[0])
		);
	}
	VITOSI(AI32);
	VITOSI(DI32);
	((unsigned*)dev->lbuf_vi.cursor+vi_offsets.SP32)[SPIX::POLLCOUNT] = pollcount;
	VITOSI(SP32);
}

/** copy SI.field to XO archive. */
#define SITOVO2(field) \
	(vo.field && \
	 memcpy(dev->lbuf_vo.cursor+vo_offsets.field, (char*)systemInterface.OUT.field+vo_cursor.field, \
			 vo.field*sizeof(systemInterface.OUT.field[0])))

/** in slack time, copy SI.OUT to VO archive cursor.
 * @@todo make it optional in case it takes too long */
void ACQ_HW::action2(SystemInterface& systemInterface) {
	SITOVO(AO16);
	SITOVO(DO32);

	SITOVO2(AO16);
	SITOVO2(DO32);
	SITOVO2(CC32);
	if (++sample < HBA::maxsam){
		dev->lbuf_vi.cursor += vi.len();
	}
	pollcount = 0;
}



/** checks host buffer for new sample, if so copies to lbuf and reports true */
bool ACQ_HW::newSample(int sample)
{
    unsigned tl1;

	if (nowait || (tl1 = TLATCH0) != tl0){
		memcpy(dev->lbuf_vi.cursor, dev->host_buffer, vi.len());
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
unsigned ACQ_HW_BASE::tlatch(void)
{
	return *(unsigned*)(dev->lbuf_vi.cursor+vi_offsets.SP32);
}
/** prepare to run a shot nsamples long, arm the UUT. */
void ACQ_HW_BASE::arm(int nsamples)
{
	cerr << "ACQ_HW_BASE::arm: TODO" <<endl;
}


class ACQ_HW_MEAN: public ACQ_HW_BASE
{
	const int nmean;
	const int spix;
	unsigned *dox;
	int **raw;
	unsigned *tl0_array;

	int verbose;

	bool _newSample(int sample);
public:
	ACQ_HW_MEAN(int devnum, string _name, VI _vi, VO _vo, VI _vi_offsets,
			VO _vo_offsets, VI& sys_vi_cursor, VO& sys_vo_cursor, int nmean);
	virtual ~ACQ_HW_MEAN()
	{}
	virtual bool newSample(int sample);
	/**< checks host buffer for new sample, if so copies to lbuf and reports true */
	virtual void action(SystemInterface& systemInterface);
	/**< on newSample, copy VO from SI, copy VI to SI */
	virtual void action2(SystemInterface& systemInterface);
	/**< late action(), cleanup */
};


ACQ_HW_MEAN::ACQ_HW_MEAN(int devnum, string _name, VI _vi, VO _vo, VI _vi_offsets,
			VO _vo_offsets, VI& sys_vi_cursor, VO& sys_vo_cursor, int _nmean) :
		ACQ_HW_BASE(devnum, _name, _vi, _vo, _vi_offsets,
						_vo_offsets, sys_vi_cursor, sys_vo_cursor),
		nmean(_nmean),
		spix(vi_offsets.SP32/sizeof(unsigned)+SPIX::TLATCH),
		tl0_array(new unsigned[_nmean]),
		verbose(0)
{
	struct ABN abn;
	int ib;

	raw = new int* [nmean];
	abn.ndesc = nmean;
	abn.buffers[0].pa = dev->xllc_def.pa;

	for (ib = 0; ib < nmean; ++ib){
		if (dev->xllc_def.pa != RTM_T_USE_HOSTBUF){
			assert(0);			// this path not valid.
			abn.buffers[ib].pa = dev->xllc_def.pa + ib*PAGE_SIZE;
		}
		abn.buffers[ib].len = dev->xllc_def.len;
		raw[ib] = (int*)(dev->host_buffer + ib*PAGE_SIZE);
	}

	if (ioctl(dev->fd, AFHBA_START_AI_ABN, &abn)){
		perror("ioctl AFHBA_START_AI_ABN");
		exit(1);
	}
	printf("[%d] AI buf pa: 0x%08x len %d\n", dev->devnum, dev->xllc_def.pa, dev->xllc_def.len);

	if (getenv("ACQ_HW_MEAN_VERBOSE")){
		verbose = atoi(getenv("ACQ_HW_MEAN_VERBOSE"));
	}
	if (verbose){
		fprintf(stderr, "%s spix:%d\n", __FUNCTION__, spix);
	}
	for (ib = 0; ib < nmean; ++ib){
		tl0_array[ib] = 0xdeadbeef;
	}
}

bool ACQ_HW_MEAN::_newSample(int sample)
{
	for (int ib = 0; ib < nmean; ++ib){
		unsigned tl1;
		if ((tl1 = raw[ib][spix]) != tl0_array[ib]){
			tl0 = tl0_array[ib] = tl1;
			return true;
		}
	}
	return false;
}
bool ACQ_HW_MEAN::newSample(int sample)
/**< checks host buffer for new sample, if so copies to lbuf and reports true */
{
	if (nowait){
		return true;
	}else if (_newSample(sample)){
		if (verbose){
			fprintf(stderr, "TLATCH:%08x\n", tl0);
		}
		return true;
	}else{
		return false;
	}
}

void ACQ_HW_MEAN::action(SystemInterface& systemInterface)
/**< on newSample, copy VO from SI, copy VI to SI */
{
/** SIMPLIFY: supports AI32 ONLY!
 * COMPLEXIFY : re-scale as LJ 24 bit number so that HW=1 threshold is still valid, then OR the channel ID back in.
 * */

	for (int ic = 0; ic < vi.AI32; ++ic){
		int total = 0;

		for (int sam = 1; sam < nmean; ++sam){
			total += raw[sam][ic] >> 8;
		}
		total += raw[0][ic] >> 8;
		systemInterface.IN.AI32[vi_cursor.AI32+ic] = ((total/nmean) << 8) | (raw[0][ic]&0x00ff);
	}
}
void ACQ_HW_MEAN::action2(SystemInterface& systemInterface)
/**< late action(), cleanup */
{

}



ACQ* ACQ::factory(int devnum, string _name, VI _vi, VO _vo, VI _vi_offsets,
		VO _vo_offsets, VI& sys_vi_cursor, VO& sys_vo_cursor)
{
	static int HW = getenv("HW") != 0 ? atoi(getenv("HW")) : 0;

	switch(HW){
	case 1:
		return new ACQ_HW(devnum, _name, _vi, _vo, _vi_offsets, _vo_offsets, sys_vi_cursor, sys_vo_cursor);
	default:
		if (HW > 1){
			return new ACQ_HW_MEAN(devnum, _name, _vi, _vo, _vi_offsets, _vo_offsets, sys_vi_cursor, sys_vo_cursor, HW);
		} // else fall thru
	case 0:
		return new ACQ(devnum, _name, _vi, _vo, _vi_offsets, _vo_offsets, sys_vi_cursor, sys_vo_cursor);
	}
}

