/*
 * InlineDataHandler.cpp
 *
 *  Created on: 3 Aug 2020
 *      Author: pgm
 *
 *  export MUXAO=ao_dev,ao_count,ai_count,ai_start,ai_stride,wavelen
 *  ao_dev: ao device number 0..12
 *  ao_count : # ao channels
 *  ai_count : # ai channels
 *  ai_start : index to start ai slice at [0]
 *  ai_stride: subsample AI, eg for 2MSPS AI, 1SMSP AI, set 2
 *  wavelen  : max 256 for LLC output. Streaming typical 20000
 *
 *
 * export MUXAO=3,4,16,0,1,256
 *
 * ie the AO is a window of 4xAI channels, slides over 16 AI channels..
 *
 * Support scripts:
 * ./scripts/test_mux : changes mux setting on the fly (see getStartStride())
 * ./scripts/test_mux : sets environment above
 * ./scripts/monitor_hb00 : monitor the AO data in host buffers.
 */


#include <stdio.h>

#include <errno.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <sched.h>

#include "RTM_T_Device.h"
#include "local.h"

#include "rtm-t_ioctl.h"

#include "InlineDataHandler.h"

InlineDataHandler::InlineDataHandler() {
	// TODO Auto-generated constructor stub

}

InlineDataHandler::~InlineDataHandler() {
	// TODO Auto-generated destructor stub
}


struct ABN abn;
int ib;

int verbose;

class InlineDataHanderMuxAO_LLC : public InlineDataHandler {
	int ao_dev;
	int ao_count;
	int ai_count;
	int ai_start;
	int ai_stride;
	int wavelen;
	RTM_T_Device *dev;
	struct ABN abn;
	short** ao_va;

	void getStartStride() {
		FILE *fp = fopen("/dev/shm/amx_llc_start_stride", "r");
		if (fp){
			int _start, _stride;
			if (fscanf(fp, "%d,%d", &_start, &_stride) == 2 &&
					_start >= 0 && _start<(ai_count-ao_count) &&
					_stride >= 1 && _stride <= 100){
				ai_start = _start;
				ai_stride = _stride;
			}
			fclose(fp);
		}
	}
public:
	InlineDataHanderMuxAO_LLC(int _ao_dev, int _ao_count, int _ai_count, int _ai_start, int _ai_stride, int _wavelen) :
		ao_dev(_ao_dev),
		ao_count(_ao_count), ai_count(_ai_count), ai_start(_ai_start), ai_stride(_ai_stride),
		wavelen(_wavelen>=MAXABN? MAXABN: _wavelen)
	{
		printf("InlineDataHanderMuxAO_LLC ao_dev=%d ao_count=%d ai_count=%d ai_start=%d ai_stride=%d\n",
				ao_dev, ao_count, ai_count, ai_start, ai_stride);

		dev = new RTM_T_Device(ao_dev);

		for (int ib = 0; ib < wavelen; ++ib){
			abn.buffers[ib].pa = RTM_T_USE_HOSTBUF;
			abn.buffers[ib].len = _ao_count * sizeof(short);
		}
		abn.ndesc = wavelen;

		if (ioctl(dev->getDeviceHandle(), AFHBA_START_AO_ABN, &abn)){
			perror("ioctl AFHBA_START_AO_ABN");
			exit(1);
		}


		ao_va = new short* [wavelen];
		ao_va[0] = (short*)dev->getHostBufferMappingW();

		if (verbose){
			printf("%s [%d] va:%p pa:%08x\n", __FUNCTION__, 0, ao_va[0], abn.buffers[0].pa);
		}
		for (int ib = 1; ib < wavelen; ++ib){
			ao_va[ib] = ao_va[0] + (abn.buffers[ib].pa - abn.buffers[0].pa)/sizeof(short);

			if (verbose > 1){
				printf("%s [%d] va:%p pa:%08x\n", __FUNCTION__, ib, ao_va[ib], abn.buffers[ib].pa);
			}
		}
	}

	virtual void handleBuffer(int ibuf, const void *src, int len)
	/* take a slice ao_count out of AI buffer and distribute one per descriptor for LLC AO */
	{
		getStartStride();
		const int instep = ai_count*ai_stride;
		short* ai = (short*)src + ai_start;
		for (int ib = 0; ib < wavelen; ++ib){
			memcpy(ao_va[ib], ai, ao_count*sizeof(short));
			ai += instep;
		}
	}
};

InlineDataHandler* InlineDataHandler::factory(RTM_T_Device* ai_dev)
{
	printf("InlineDataHandler::factory() option InlineDataHanderMuxAO_LLC\n");
	if (getenv("MUXAO_VERBOSE")){
		verbose = atoi(getenv("MUXAO_VERBOSE"));
	}
	if (const char* value = getenv("MUXAO")){
		int pr[6];
		if (sscanf(value, "%d,%d,%d,%d,%d,%d", pr+0, pr+1, pr+2, pr+3, pr+4, pr+5) == 6){
			return new InlineDataHanderMuxAO_LLC(pr[0], pr[1], pr[2], pr[3], pr[4], pr[5]);
		}
	}
	return new InlineDataHandler;
}

