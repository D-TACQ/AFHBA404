/*
 * DefaultSystemInterface.cpp
 *
 *  Created on: 14 Mar 2020
 *      Author: pgm
 */

#include "AcqSys.h"
#include <string.h>

template <class T>
T* new_zarray(int nelems)
{
	T* nz_array = new T[nelems];
	memset(nz_array, 0, nelems*sizeof(T));
	return nz_array;
}


SystemInterface::SystemInterface(const HBA& _hba) : hba(_hba)
/* make a gash SI to allow simulated operation. The real shm is customer specific */
{
	IN.AI16 = new_zarray<short>(AI16_count());
	IN.AI32 = new_zarray<int>(AI32_count());
	IN.DI32 = new_zarray<unsigned>(DI32_count());    // needs to be bigger for PWM
	IN.SP32 = new_zarray<unsigned>(SP32_count());

	OUT.AO16 = new_zarray<short>(AO16_count());
	OUT.DO32 = new_zarray<unsigned>(DO32_count());
	OUT.CC32 = new_zarray<unsigned>(CC32_count());
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

class DummySingleThreadControlSystemInterface: public SystemInterface {

public:
	DummySingleThreadControlSystemInterface(const HBA& hba) :
		SystemInterface(hba)
	{}
	static int DUP1;

	virtual void ringDoorbell(int sample){
		G::verbose && printf("DummySingleThreadControlSystemInterface::ringDoorbell(%d)\n", sample);

		short xx = IN.AI16[DUP1];
		for (int ii = 0; ii < AO16_count(); ++ii){
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

