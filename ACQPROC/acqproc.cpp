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

namespace G {
	int nsamples = 2;		/**< samples to capture (default:2, typ 200000) */
	int verbose;
	int dummy_first_loop;		/**< possible speed up by filling cache first loop */
	int samples_buffer = 1;		/**< number of samples in each VI buffer (default:1) */
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
	if ((key = getenv("VERBOSE"))){
		G::verbose = atoi(key);
	}
	if ((key = getenv("DUMMY_FIRST_LOOP"))){
		G::dummy_first_loop = atoi(key);
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

int main(int argc, char* argv[])
{
	const char* config_file = ui(argc, argv);

	HBA hba = HBA::create(config_file, G::nsamples);

	hba.dump_config();

	SystemInterface& si(SystemInterface::factory(hba));

	hba.start_shot();
	si.trigger();

	for (int sample = 0; sample < G::nsamples; ++sample){
		hba.processSample(si, sample);
	}
}



