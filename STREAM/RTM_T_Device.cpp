/*
 * RTM_T_Device.cpp
 *
 *  Created on: Apr 28, 2011
 *      Author: pgm
 */

using namespace std;

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>

#include "local.h"
#include "RTM_T_Device.h"

RTM_T_Device::RTM_T_Device(int _devnum, int _nbuffers) :
	devnum(_devnum%100), nbuffers(_nbuffers)
{
	char buf[120];

	sprintf(buf, "/dev/rtm-t.%d", devnum);
	names[MINOR_DMAREAD] = *new string(buf);
	_open(MINOR_DMAREAD);

	sprintf(buf, "/dev/rtm-t.%d.regs", devnum);
	names[MINOR_REGREAD] = * new string(buf);
	_open(MINOR_REGREAD);

	for (int ib = 0; ib < nbuffers; ++ib){
		sprintf(buf, "/dev/rtm-t.%d.data/hb%02d", devnum, ib);
		names[ib] = buf;

		_open(ib);
		void *va = mmap(0, MAXLEN, PROT_READ|PROT_WRITE,
				MAP_SHARED, handles[ib], 0);

		if (va == (caddr_t)-1 ){
			perror( "mmap" );
		        _exit(errno);
		}else{
			host_buffers[ib] = va;
		}
	}

	sprintf(buf, "/dev/rtm-t.%d.ctrl", devnum);
	names[CTRL_ROOT] = * new string(buf);
}

