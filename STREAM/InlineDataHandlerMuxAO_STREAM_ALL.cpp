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
 * export MUXAO=3,4,16,0,1,10000
 *
 * echo 1,1 2,16 > /dev/shm/amx_ao_map
 */

/** @file InlineDataHandlerMuxAO_STREAM_ALL.cpp
 *  @brief Analog Multiplexer **AMX** instantiation of InlineDataHandler
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

#include <vector>

InlineDataHandler::InlineDataHandler() {
	// TODO Auto-generated constructor stub

}

InlineDataHandler::~InlineDataHandler() {
	// TODO Auto-generated destructor stub
}


#define AO_MAP_ZERO 0xffff


/* /dev/shm/amx_ao_from ai_mapping
 * aoch=aich                       // both index from 1 eg
 * 1=1
 * 2=17
 * unmapped channels are set AO_MAP_ZERO
 */
class InlineDataHanderMuxAO_STREAM : public InlineDataHandler {
	RTM_T_Device* ai_dev;
	int ao_dev;
	int ao_count;
	int ai_count;
	int ai_start;
	int ai_stride;
	int wavelen;
	RTM_T_Device *dev;
	unsigned * ao_ai_mapping;

	int ao_buf_ix;

	void updateMuxSelection()
	{
		FILE *fp = fopen("/dev/shm/amx_ao_map", "r");
		if (fp){
			int ai_ch, ao_ch;
			for (ao_ch = 0; ao_ch < ao_count; ++ao_ch){
				ao_ai_mapping[ao_ch] = AO_MAP_ZERO;
			}

			while (fscanf(fp, "%d,%d", &ao_ch, &ai_ch) == 2 &&
					ao_ch >= 1 && ao_ch <= ao_count &&
					ai_ch >= 1 &&  ai_ch <= ai_count){
				ao_ai_mapping[ao_ch-1] = ai_ch-1;
			}
			fclose(fp);
		}
	}
public:
	InlineDataHanderMuxAO_STREAM(int _ao_dev, int _ao_count, int _ai_count, int _ai_start, int _ai_stride, int _wavelen) :
		ao_dev(_ao_dev),
		ao_count(_ao_count), ai_count(_ai_count), ai_start(_ai_start), ai_stride(_ai_stride), wavelen(_wavelen),
		ao_buf_ix(0)
	{
		fprintf(stderr, "%s ao_dev:%d ao:%d ai:%d ai_start:%d ai_stride:%d wavelen:%d\n", __FUNCTION__,
				ao_dev, ao_count, ai_count, ai_start, ai_stride, wavelen);

		dev = new RTM_T_Device(ao_dev);
		memset(dev->getHostBufferMappingW(0), 0, dev->maxlen);
		memset(dev->getHostBufferMappingW(1), 0, dev->maxlen);

		ao_ai_mapping = new unsigned[ao_count];
		for (int ao_ch = 0; ao_ch < ao_count; ++ao_ch){
			ao_ai_mapping[ao_ch] = ao_ch;
		}
		if (ioctl(dev->getDeviceHandle(), AFHBA_AO_BURST_INIT, 0)){
			perror("ioctl AFHBA_AO_BURST_INIT");
			exit(1);
		}
	}

	virtual void handleBuffer(int ibuf, const void *src, int len)
	/* take a slice ao_count out of AI buffer and drop the slice into  */
	{
		const short* ai0 = (const short*)src;
		const short* ai = ai0;
		short* ao = (short*)dev->getHostBufferMappingW(ao_buf_ix); 
		const int instep = ai_count*ai_stride;
		updateMuxSelection();

		for (int sample = 0; sample < wavelen; ++sample, ai += instep, ao += ao_count){
			if ((const char*)(ai+ai_count) - (const char*)ai0 > dev->maxlen){
				fprintf(stderr, "%s ai overflow at sample:%d\n", __FUNCTION__, sample);
				exit(1);
			}
			for (int ao_ch = 0; ao_ch < ao_count; ++ao_ch){
				unsigned ai_ch = ao_ai_mapping[ao_ch];
				ao[ao_ch] = ai_ch==AO_MAP_ZERO? 0 : ai[ai_ch];
			}
		}
		if (ioctl(dev->getDeviceHandle(), AFHBA_AO_BURST_SETBUF, ao_buf_ix)){
			perror("ioctl AFHBA_AO_BURST_SETBUF");
			exit(1);
		}
		//ao_buf_ix = !ao_buf_ix;
	}
};

InlineDataHandler* InlineDataHandler::factory(RTM_T_Device* ai_dev)
{
	if (const char* value = getenv("MUXAO")){
		int pr[6];
		if (sscanf(value, "%d,%d,%d,%d,%d,%d", pr+0, pr+1, pr+2, pr+3, pr+4, pr+5) == 6){
			return new InlineDataHanderMuxAO_STREAM(pr[0], pr[1], pr[2], pr[3], pr[4], pr[5]);
		}
	}
	return new InlineDataHandler;
}

