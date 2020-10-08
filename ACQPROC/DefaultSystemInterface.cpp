/*
 * DefaultSystemInterface.cpp
 *
 *  Created on: 14 Mar 2020
 *      Author: pgm
 */

#include "AcqSys.h"
#include <string.h>


class DummySingleThreadControlSystemInterface: public SystemInterface {

public:
	DummySingleThreadControlSystemInterface(const HBA& hba) :
		SystemInterface(hba)
	{}
	static int DUP1;

	virtual void ringDoorbell(int sample){
		G::verbose && printf("%s(%d)\n", PFN, sample);

		short xx = IN.AI16[DUP1];
		for (int ii = 0; ii < AO16_count(); ++ii){
			OUT.AO16[ii] = xx;
		}
		unsigned tl = tlatch();
		for (int ii = 0; ii < DO32_count(); ++ii){
                        if (ii > 0){
				OUT.DO32[ii] = 0xbe8bb8fa;
                        }else{
				OUT.DO32[ii] = tl;
                        }
		}
	}
};

int DummySingleThreadControlSystemInterface::DUP1;

SystemInterface& SystemInterface::factory(const HBA& hba)
{
	const char* key = getenv("SINGLE_THREAD_CONTROL");
	if (key){
		if (sscanf(key, "control_dup1=%d", &DummySingleThreadControlSystemInterface::DUP1) == 1 ||
		    strcmp(key, "control_dup1") == 0){
			return * new DummySingleThreadControlSystemInterface(hba);
		}
	}

	return * new SystemInterface(hba);
}

