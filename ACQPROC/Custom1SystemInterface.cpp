/*
 * DefaultSystemInterface.cpp
 *
 *  Created on: 14 Mar 2020
 *      Author: pgm
 */

#include "AcqSys.h"

#include <string.h>
#include <stdio.h>		// because sprintf()
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

class Custom1SingleThreadControlSystemInterface: public SystemInterface {
public:
	Custom1SingleThreadControlSystemInterface(const HBA& hba) :
		SystemInterface(hba)
	{}
	virtual void ringDoorbell(int sample){
		HBA& the_hba(HBA::instance());
		int imax = the_hba.vo.AO16;
		short xx = IN.AI16[0];
		for (int ii = 0; ii < imax; ++ii){
			OUT.AO16[ii] = xx;
		}
		unsigned tl = the_hba.uuts[0]->tlatch();
		for (int ii = 0; ii < the_hba.vo.DO32; ++ii){
			OUT.DO32[ii] = tl;
		}
	}
};

SystemInterface& SystemInterface::factory(const HBA& hba)
{
	fprintf(stderr, "SystemInterface::factory  CUSTOM INTERFACE\n");
	const char* key = getenv("SINGLE_THREAD_CONTROL");
	if (key){
		if (strcmp(key, "control_dup1")){
			return * new Custom1SingleThreadControlSystemInterface(hba);
		}
	}

	return * new SystemInterface(hba);
}

