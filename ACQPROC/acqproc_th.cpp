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

void fake_a_shot(HBA& hba)
{
	for (int sample = 0; sample < NSAM; ++sample){
		for (auto uut : hba.uuts){
			uut->newSample(sample);
		}
	}
}
int main(int argc, char* argv[])
{
	const char* config_file;
	if (argc > 1){
		config_file = argv[1];
	}
	HBA hba = HBA::create(config_file);

	hba.dump_config();

	fake_a_shot(hba);
}

