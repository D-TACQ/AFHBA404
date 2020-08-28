/*
 * amx_dummy.cpp
 *
 *  Created on: 28 Aug 2020
 *      Author: pgm
 */




#include <stdio.h>

#include <errno.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>



#include <sched.h>

//using namespace std;

#include "RTM_T_Device.h"
#include "local.h"
#include "popt.h"

#include "rtm-t_ioctl.h"
/* default implementation is NULL */



int main(int argc, char* argv[])
{
	int devnum = 0;
	int verbose = 0;

	if (getenv("RTM_DEVNUM")){
		devnum = atol(getenv("RTM_DEVNUM"));
	}
	if (getenv("VERBOSE")){
		verbose = atol(getenv("VERBOSE"));
	}
	RTM_T_Device* dev = new RTM_T_Device(devnum);

	int fp = dev->getDeviceHandle();

	struct AO_BURST ao_burst;
	ao_burst.id = AO_BURST_ID;
	ao_burst.nbuf = 1;

	int rc = ioctl(fp, AFHBA_AO_BURST_INIT, &ao_burst);
	if (rc != 0){
		perror("AFHBA_AO_BURST_INIT");
		return 1;
	}

	while(1){
		rc = ioctl(fp, AFHBA_AO_BURST_SETBUF, 0);
		if (rc != 0){
			perror("AFHBA_AO_BURST_SETBUF");
			return 1;
		}
		if (verbose > 1){
			fprintf(stderr, "hit me\n");
			getchar();
		}else{
			usleep(20000);
		}
	}

	return 0;
}
