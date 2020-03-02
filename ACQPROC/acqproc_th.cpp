/*
 * acqproc_th.cpp  : test harness
 *
 *  Created on: 27 Feb 2020
 *      Author: pgm
 */


#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "AcqSys.h"

#define NSAM	2		// typ: 200000

void fake_a_shot(HBA& hba, SystemInterface& systemInterface)
{
	for (int sample = 0; sample < NSAM; ++sample){
		hba.processSample(systemInterface, sample);
	}
}
int main(int argc, char* argv[])
{
	const char* config_file;
	if (argc > 1){
		config_file = argv[1];
	}
	HBA hba = HBA::create(config_file, NSAM);

	hba.dump_config();

	SystemInterface si;

	fake_a_shot(hba, si);
}

