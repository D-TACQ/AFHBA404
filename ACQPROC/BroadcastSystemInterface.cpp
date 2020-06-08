/*
 * DefaultSystemInterface.cpp : sends a broad
 *
 *  Created on: 14 Mar 2020
 *      Author: pgm
 */

#include "AcqSys.h"
#include <assert.h>
#include <string.h>


#define PFN __PRETTY_FUNCTION__

SystemInterface::SystemInterface(const HBA& _hba) : hba(_hba)
/* make a gash SI to allow simulated operation. The real shm is customer specific */
{
	IN.AI16 = new short[AI16_count()];
	IN.AI32 = new int[AI32_count()];
	IN.DI32 = new unsigned[DI32_count()];    // needs to be bigger for PWM
	IN.SP32 = new unsigned[SP32_count()];

	OUT.AO16 = new short[AO16_count()];
	OUT.DO32 = new unsigned [DO32_count()];
	OUT.CC32 = new unsigned [CC32_count()];
}
SystemInterface::~SystemInterface()
{
	delete [] IN.AI16;
	delete [] IN.AI32;
	delete [] IN.DI32;
	delete [] IN.SP32;
	delete [] OUT.AO16;
	delete [] OUT.DO32;
	delete [] OUT.CC32;
}


int getenv(const char* key, int def)
{
	const char* val = getenv(key);
	if (val != 0){
		return atoi(val);
	}else{
		return def;
	}
}

class BroadcastSystemInterface: public SystemInterface {
	const int ntriggers;			/* opt: repeat trigger to instrument trigger overhead */
	int over;
	const char* trgfile;
	int th_chan[2];					/* channel index to threshold    */
	int thix;						/* threshold channel index 0|1	 */

public:
	BroadcastSystemInterface(const HBA& hba) :
		SystemInterface(hba), over(0), ntriggers(::getenv("NTRIGGERS", 1)), thix(0)
	{
		char* _trgfile = new char[80];
		snprintf(_trgfile, 80, "/dev/rtm-t.0.ctrl/com_trg");
		trgfile = _trgfile;
		th_chan[0] = ::getenv("THCHAN0", 0);
		th_chan[1] = ::getenv("THCHAN1", 0);
	}
	virtual ~BroadcastSystemInterface() {
		delete [] trgfile;
	}

	virtual void trigger() {
		FILE* fp = fopen(trgfile, "w");
		assert(fp);
		fprintf(fp, "1\n");
		fclose(fp);
	}

	virtual void ringDoorbell(int sample){
		int ai = IN.AI32[th_chan[thix]];

		if (over && ai < -2000){
			over = false;
		}else if (!over && ai > 2000) {
			for (int it = 0; it < ntriggers; ++it){
				trigger();
			}
			over = true;
			G::verbose && printf("%s over\n", PFN);
		}

		G::verbose > 1 && printf("%s[%d] %08x\n", PFN, sample, IN.AI32[0]);
		thix = !thix;
	}
};

SystemInterface& SystemInterface::factory(const HBA& hba)
{
	if (getenv("HW") && atoi(getenv("HW")) > 0){
		return * new BroadcastSystemInterface(hba);
	}else{
		return * new SystemInterface(hba);
	}
}

