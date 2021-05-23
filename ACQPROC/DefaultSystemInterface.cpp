/** @file DefaultSystemInterface.cpp
 *  @brief SystemInterface implementation example
 *  overloads ringDoorbell() to actually "do work".
 *  a real implementation should perform a shared memory interface with another process.
 *
 *  Created on: 14 Mar 2020
 *      Author: pgm
 *
 *  Demo Modes: dup1, duplicate AI0
 *      export SINGLE_THREAD_CONTROL=control_dup1
 *
 */

#include "AcqSys.h"
#include <string.h>



#define PWM_MAGIC	0xbe8bb8fa			// makes for a good display

class DummySingleThreadControlSystemInterface: public SystemInterface {

public:
	DummySingleThreadControlSystemInterface(const HBA& hba) :
		SystemInterface(hba)
	{
		if (G::verbose) printf("%s::%s DUP1:%d\n", __FILE__, PFN, DUP1);
	}
	static int DUP1;

	virtual void ringDoorbell(int sample){
		G::verbose && printf("%s(%d)\n", PFN, sample);

		short xx = IN.AI16[DUP1];
		for (int ii = 0; ii < AO16_count(); ++ii){
			OUT.AO16[ii] = xx;
		}
		unsigned tl = tlatch();
		for (int ii = 0; ii < DO32_count(); ++ii){
			OUT.DO32[ii] = tl;
		}
		for (int ii = 0; ii < PW32_count(); ++ii){
			for (int cc = 0; cc < PW32LEN; ++cc){
				OUT.PW32[ii][cc] = PWM_MAGIC;
			}
		}
	}
};

int DummySingleThreadControlSystemInterface::DUP1;

SystemInterface& SystemInterface::factory(const HBA& hba)
{
	if (G::verbose) printf("%s::%s\n", __FILE__, PFN);

	const char* key = getenv("SINGLE_THREAD_CONTROL");
	if (key){
		if (sscanf(key, "control_dup1=%d", &DummySingleThreadControlSystemInterface::DUP1) == 1 ||
		    strcmp(key, "control_dup1") == 0){
			return * new DummySingleThreadControlSystemInterface(hba);
		}
	}

	return * new SystemInterface(hba);
}

