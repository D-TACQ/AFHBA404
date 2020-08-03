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
 * export MUXAO=3,4,16,0,1,256
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


class InlineDataHanderMuxAO_STREAM : public InlineDataHandler {
	RTM_T_Device* ai_dev;
	int ao_dev;
	int ao_count;
	int ai_count;
	int ai_start;
	int ai_stride;
	int wavelen;
	RTM_T_Device *dev;

	int twizzle(int ibuf)
	{
		/* we probably don't want to be writing this AO buffer, we want to write the NEXT one?.
		 * maybe 2 ahead?.
		 *
		 */
		return dev->next(ibuf);
	}
public:
	InlineDataHanderMuxAO_STREAM(RTM_T_Device* _ai_dev,
			int _ao_dev, int _ao_count, int _ai_count, int _ai_start, int _ai_stride, int _wavelen) :
		ai_dev(_ai_dev),
		ao_dev(_ao_dev),
		ao_count(_ao_count), ai_count(_ai_count), ai_start(_ai_start), ai_stride(_ai_stride), wavelen(_wavelen)
	{
		dev = new RTM_T_Device(ao_dev);

		if (ioctl(dev->getDevnum(), RTM_T_START_STREAM_AO, &abn)){
			perror("ioctl RTM_T_START_STREAM_AO");
			exit(1);
		}
	}

	virtual void handleBuffer(int ibuf, const void *src, int len)
	/* take a slice ao_count out of AI buffer and drop the slice into  */
	{
		const short* ai = (const short*)src + ai_start;
		short* ao = (short*)dev->getHostBufferMappingW(twizzle(ibuf));
		const int instep = ai_count*ai_stride;
		for (int sample = 0; sample < wavelen; ++sample, ai += instep, ao += ao_count){
			memcpy(ao, ai, ai_count*sizeof(short));
		}
	}
};

InlineDataHandler* InlineDataHandler::factory(RTM_T_Device* ai_dev)
{
	if (const char* value = getenv("MUXAO")){
		int pr[5];
		if (sscanf(value, "%d,%d,%d,%d,%d,%d", pr+0, pr+1, pr+2, pr+3, pr+4, pr+5) == 5){
			return new InlineDataHanderMuxAO_STREAM(ai_dev, pr[0], pr[1], pr[2], pr[3], pr[4], pr[5]);
		}
	}
	return new InlineDataHandler;
}

