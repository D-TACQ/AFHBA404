/** @file BroadcastSystemInterface.cpp
 *  @brief sends a broadcast trigger to all fiber optic links.
 *
 *  Created on: 14 Mar 2020
 *      Author: pgm
 */

#include "AcqSys.h"
#include <assert.h>
#include <string.h>



class BroadcastSystemInterface: public SystemInterface {
	const int ntriggers;			/* opt: repeat trigger to instrument trigger overhead */
	int over;
	const char* trgfile;
	int th_chan[2];					/* channel index to threshold    */
	int thix;						/* threshold channel index 0|1	 */
	FILE* fp_log;

public:
	BroadcastSystemInterface(const HBA& hba) :
		SystemInterface(hba), over(0), ntriggers(::getenv("NTRIGGERS", 1)), thix(0), fp_log(0)
	{
		char* _trgfile = new char[80];
		snprintf(_trgfile, 80, "/dev/rtm-t.0.ctrl/com_trg");
		trgfile = _trgfile;
		th_chan[0] = ::getenv("THCHAN0", 0);
		th_chan[1] = ::getenv("THCHAN1", 0);
		if (getenv("SILOG")){
			fp_log = fopen(getenv("SILOG"), "w");
		}
	}
	virtual ~BroadcastSystemInterface() {
		delete [] trgfile;
		if (fp_log){
			fclose(fp_log);
		}
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

		if (fp_log){
			fwrite(IN.AI32, sizeof(int), AI32_count(), fp_log);
		}
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

