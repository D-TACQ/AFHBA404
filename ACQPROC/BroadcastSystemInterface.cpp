/*
 * DefaultSystemInterface.cpp : sends a broad
 *
 *  Created on: 14 Mar 2020
 *      Author: pgm
 */

#include "AcqSys.h"
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

class BroadcastSystemInterface: public SystemInterface {
	const int ntriggers;			/* opt: repeat trigger to instrument trigger overhead */
	int over;
	const char* trgfile;

	static int get_ntriggers() {
		const char* val = getenv("NTRIGGERS");
		if (val != 0){
			return atoi(val);
		}else{
			return 1;
		}
	}
public:
	BroadcastSystemInterface(const HBA& hba) :
		SystemInterface(hba), over(0), ntriggers(get_ntriggers())
	{
		char* _trgfile = new char[80];
		snprintf(_trgfile, 80, "/dev/rtm-t.%d.ctrl/com_trg", hba.devnum);
		trgfile = _trgfile;
	}
	virtual ~BroadcastSystemInterface() {
		delete [] trgfile;
	}

	virtual void trigger() {
		FILE* fp = fopen(trgfile, "w");
		fprintf(fp, "1\n");
		fclose(fp);
	}

	virtual void ringDoorbell(int sample){
		if (over && IN.AI32[0] < -2000){
			over = false;
		}else if (!over && IN.AI32[0] > 2000) {
			for (int it = 0; it < ntriggers; ++it){
				trigger();
			}
			over = true;
			G::verbose && printf("%s over\n", PFN);
		}

		G::verbose > 1 && printf("%s[%d] %08x\n", PFN, sample, IN.AI32[0]);
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

