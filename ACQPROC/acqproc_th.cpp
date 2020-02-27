/*
 * acqproc_th.cpp  : test harness
 *
 *  Created on: 27 Feb 2020
 *      Author: pgm
 */




#include "AcqSys.h"



int main(int argc, char* argv[])
{
	const char* config_file = "configs/pcs1.json";


	HBA hba = HBA::create(config_file);

	hba.dump_data("runtime_config.json");
}

