/*
 * DefaultSystemInterface.cpp
 *
 *  Created on: 14 Mar 2020
 *      Author: pgm
 */

#include "AcqSys.h"
#include <string.h>
//#include <stdio.h>		// because sprintf()
#include <assert.h>




SystemInterface::SystemInterface(const HBA& _hba) : hba(_hba)
/* make a gash SI to allow simulated operation. The real shm is customer specific */
{
	IN.AI16 = (short*)calloc(4*192, sizeof(short));
	IN.AI32 = (int*)calloc(4*192, sizeof(int));
	IN.DI32 = (unsigned*)calloc(4*6, sizeof(unsigned));    // needs to be bigger for PWM
	IN.SP32 = (unsigned*)calloc(4*16, sizeof(unsigned));

	OUT.AO16 = (short*)calloc(4*192, sizeof(short));
	OUT.DO32 = (unsigned*)calloc(4*4, sizeof(unsigned));
	OUT.CC32 = (unsigned*)calloc(4*32, sizeof(unsigned));
}

class DummySingleThreadControlSystemInterface: public SystemInterface {

public:
	DummySingleThreadControlSystemInterface(const HBA& hba) :
		SystemInterface(hba)
	{}
	static int DUP1;

	virtual void ringDoorbell(int sample){
		G::verbose && printf("DummySingleThreadControlSystemInterface::ringDoorbell(%d)\n", sample);
		HBA& the_hba(HBA::instance());
		int imax = the_hba.vo.AO16;
		short xx = IN.AI16[DUP1];
		for (int ii = 0; ii < imax; ++ii){
			OUT.AO16[ii] = xx;
		}
		unsigned tl = tlatch();
		for (int ii = 0; ii < DO32_count(); ++ii){
			OUT.DO32[ii] = tl;
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

