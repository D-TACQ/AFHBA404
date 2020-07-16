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

