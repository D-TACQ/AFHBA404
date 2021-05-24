/** @file acqproc.cpp
 * @brief main program with UI, instantiates system.
 * acqproc_th.cpp  : test harness
 *
 *  Created on: 27 Feb 2020
 *      Author: pgm
 */


extern "C" {
#include "afhba-llcontrol.h"
extern int sched_fifo_priority;
}

#include <signal.h>


#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "AcqSys.h"


int getenv(const char* key, int def, int (*cvt)(const char* key))
{
	const char* value = getenv(key);
	if (value){
		return cvt(value);
	}else{
		return def;
	}
}

namespace G {
	int verbose = ::getenv("VERBOSE", 0);
	int dummy_first_loop= ::getenv("DUMMY_FIRST_LOOP", 0);		/**< possible speed up by filling cache first loop */
	int MAXLOG = ::getenv("MAXLOG", 1000000); 					/**< avoid oom */
	int samples_buffer = 1;										/**< number of samples in each VI buffer (default:1) */
	int nsamples = 2;											/**< samples to capture (default:2, typ 200000) */
	int maxpoll = 0;
};



const char* ui(int argc, char* argv[])
{
	const char* config_file;
	const char* key;

    if ((key = getenv("RTPRIO"))){
		sched_fifo_priority = atoi(key);
    }
    if ((key = getenv("AFFINITY")) && strtol(key, 0, 0) != 0){
                setAffinity(strtol(key, 0, 0));
    }
    if ((key = getenv("MAXPOLL")) && strtol(key, 0, 0) != 0){
                G::maxpoll = strtol(key, 0, 0);
    }

	if (argc > 1){
		config_file = argv[1];
	}else{
		fprintf(stderr, "USAGE acqproc_th CONFIG_FILE NSAMPLES [SAMPLES_BUFFER]\n");
		exit(1);
	}
	if (argc > 2){
		G::nsamples = atoi(argv[2]);
		fprintf(stderr, "nsamples set %d\n", G::nsamples);
	}
	if (argc > 3){
		G::samples_buffer = atoi(argv[3]);
	}
	return config_file;
}

bool closedown_request;

/* user may ask for a VERY LONG capture, but we STILL want to store the log files on ctrl-c */
void sigint_handler(int s)
{
	static int ctrl_c_count;
	printf("<ctrl-c> :: cleanup, store logs\n");
	closedown_request = true;
	if (++ctrl_c_count >= 3){
		printf("<ctrl-c> :: third call, just drop out\n");
		exit(1);
	}
}

void configure_ctrl_c_closedown() {
	struct sigaction sigIntHandler;

	sigIntHandler.sa_handler = sigint_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;

	sigaction(SIGINT, &sigIntHandler, NULL);
}


int main(int argc, char* argv[])
{
	const char* config_file = ui(argc, argv);

	HBA hba = HBA::create(config_file, G::nsamples==0? 100: G::nsamples>G::MAXLOG? G::MAXLOG: G::nsamples);


	hba.dump_config();
	configure_ctrl_c_closedown();

	SystemInterface& si(SystemInterface::factory(hba));

	hba.start_shot();

	si.trigger();

	try {
		for (int sample = 0; !closedown_request && (G::nsamples == 0 || sample < G::nsamples); ++sample){
			hba.processSample(si, sample);
		}
	} catch (int error) {
		fprintf(stderr, "ERROR:%d\n", error);
	}
}



