/**
 * @file HudpRelaySystemInterface.cpp
 * @brief SystemInterface implementation example
 *  overloads ringDoorbell() to actually "do work".
 *  a real implementation should perform a shared memory interface with another process.
 *
 *
 *  Created on: 29 Nov 2022
 *      Author: pgm
 */

#include "AcqSys.h"
#include <string.h>


class Actor {
public:
	virtual ~Actor() {}
	virtual void operator() (void) {

	}
};

class PullHostTrigger: public Actor {
	char trigger_knob[80];
	PullHostTrigger(const HBA& hba) {

	}

public:

	void action(void){
		;
	}

	static Actor& factory(const HBA& hba, int host_pull_trigger) {
		if (host_pull_trigger){
			return * new PullHostTrigger(hba);
		}else{
			return * new Actor();
		}
	}
};

class HudpRelaySingleThreadControlSystemInterface: public SystemInterface {

public:
	HudpRelaySingleThreadControlSystemInterface(const HBA& hba, int host_pull_trigger, int _dup1) :
		SystemInterface(hba),
		trigger(PullHostTrigger::factory(hba, host_pull_trigger)),
		DUP1(_dup1)
	{
		if (G::verbose) printf("%s::%s DUP1:%d\n", __FILE__, PFN, DUP1);
	}
	const int DUP1;		// Duplicate this input
	Actor& trigger;   	// HOST initiates pull trigger

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
		/* HP32 HudP relay. First 4 elems are SPAD[0-3], not for ext hardware but for Wireshark
		 * Subsequent elements could go to AO16, but only odd values, even values are packet index for debug
		 * nb: this is all DEMO/DEBUG aid, a real system will be different!
		 */
		for (int ii = 0; ii < HP32_count() && ii < 4; ++ii){
			OUT.HP32[ii] = IN.SP32[ii];			/* distinctive HudP pattern: AI->AO (assumed), first 4 SPAD elements */
		}
		for (int ii = 4; ii < HP32_count(); ++ii){
			OUT.HP32[ii] = xx + (ii<<16);			/* distinctive HudP pattern: AI->AO (assumed), ii: "channel index number" */
		}

		trigger();
	}
};

SystemInterface& SystemInterface::factory(const HBA& hba)
{
	if (G::verbose) printf("%s::%s\n", __FILE__, PFN);

	char* key = getenv("SINGLE_THREAD_CONTROL");
	if (key){
		int dup1 = 0;
		int host_pull_trigger = 0;
		if (sscanf(key, "host_pull_trigger=%d,%d", &host_pull_trigger, &dup1) >= 1 ||
		    strcmp(key, "control_dup1") == 0){                                          // previous default, just go with it
			return * new HudpRelaySingleThreadControlSystemInterface(hba, dup1, host_pull_trigger);
		}

	}

	return * new SystemInterface(hba);
}

