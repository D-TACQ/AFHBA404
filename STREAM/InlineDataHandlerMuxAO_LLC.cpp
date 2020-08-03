/*
 * InlineDataHandler.cpp
 *
 *  Created on: 3 Aug 2020
 *      Author: pgm
 */

#include "RTM_T_Device.h"
#include "InlineDataHandler.h"
#include <stdio.h>
#include <stdlib.h>

InlineDataHandler::InlineDataHandler() {
	// TODO Auto-generated constructor stub

}

InlineDataHandler::~InlineDataHandler() {
	// TODO Auto-generated destructor stub
}


class InlineDataHanderMuxAO_LLC : public InlineDataHandler {
	int ao_dev;
	int ao_count;
	int ai_count;
	int ai_start;
	int wavelen;
	RTM_T_Device *dev;

public:
	InlineDataHanderMuxAO_LLC(int _ao_dev, int _ao_count, int _ai_count, int _ai_start, int _wavelen) :
		ao_dev(_ao_dev),
		ao_count(_ao_count), ai_count(_ai_count), ai_start(_ai_start), wavelen(_wavelen)
	{
		dev = new RTM_T_Device(ao_dev);

	}
};

InlineDataHandler* InlineDataHandler::factory()
{
	if (const char* value = getenv("MUXAO")){
		int pr[5];
		if (sscanf(value, "%d,%d,%d,%d,%d", pr+0, pr+1, pr+2, pr+3, pr+4) == 5){
			return new InlineDataHanderMuxAO_LLC(pr[0], pr[1], pr[2], pr[3], pr[4]);
		}
	}
	return new InlineDataHandler;
}

