/*
 * acqproc_th.cpp  : test harness
 *
 *  Created on: 27 Feb 2020
 *      Author: pgm
 */


extern "C" {
#include "afhba-llcontrol.h"
extern int sched_fifo_priority;
}

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "AcqSys.h"

#define NSAM	2		// typ: 200000

namespace G {
	int nsamples = 2;
	int verbose;
	int dummy_first_loop;
	int samples_buffer;
};


const char* ui(int argc, char* argv[])
{
	const char* config_file;

        if (getenv("RTPRIO")){
		sched_fifo_priority = atoi(getenv("RTPRIO"));
        }
        if (getenv("AFFINITY")){
                setAffinity(strtol(getenv("AFFINITY"), 0, 0));
        }

	if (getenv("VERBOSE")){
		G::verbose = atoi(getenv("VERBOSE"));
	}

	if (getenv("DUMMY_FIRST_LOOP")){
		G::dummy_first_loop = atoi(getenv("DUMMY_FIRST_LOOP"));
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


void run_shot(HBA& hba, SystemInterface& systemInterface)
{
	for (int sample = 0; sample < G::nsamples; ++sample){
		hba.processSample(systemInterface, sample);
	}
}
int main(int argc, char* argv[])
{
	const char* config_file = ui(argc, argv);

	HBA hba = HBA::create(config_file, G::nsamples);

	hba.dump_config();

	SystemInterface si(SystemInterface::factory());

	hba.start_shot();
	run_shot(hba, si);
}

