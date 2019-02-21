/*
 * afhba-get_shared_mapping.c
 *
 *  Created on: 13 Dec 2018
 *      Author: pgm
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sched.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>

#include "afhba-get_shared_mapping.h"
#include "../rtm-t_ioctl.h"

void get_shared_mapping(int devnum, int ibuf, struct XLLC_DEF* xllc_def, void** pbuf)
/*
 * inputs: devnum, ibuf
 * outputs: xllc_def : buffer definition for driver, pbuf : pbuffer pointer for app
 * example of setting up a mapping using a "3rd party buffer"
 * "3rd party buffer" means: the buffer va and pa were provided by some other source, eg GPU
 * In our example, while the buffer actually comes from the afhba404 driver, it's a shared buffer
 * and we get to set the pa directly in xllc_def
 */
{
	char fname[80];
	void *va;
	int fd;
	FILE *fp;

	sprintf(fname, "/dev/rtm-t.%d.data/hb%02d", devnum, ibuf);
	fd = open(fname, O_RDWR);
	if (fd < 0){
		perror(fname);
		exit(errno);
	}
	va = mmap(0, HB_LEN, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
	if (va == (caddr_t)-1 ){
		perror( "mmap" );
	    exit(errno);
	}else{
		if (pbuf != 0){
			*pbuf = va;
		}
	}
	if (xllc_def == 0){
		return;
	}
	sprintf(fname, "/proc/driver/afhba/afhba.%d/HostBuffers", devnum);
	fp = fopen(fname, "r");
	if (fp == 0){
		perror("fname");
		exit(errno);
	}else{
		char aline[128];
		int ib;
		unsigned pa;

		while (fgets(aline, sizeof(aline)-1, fp)){
// [00] ffff880005d00000 05d00000 100000 100000 05d000a0 BS_EMPTY
			if (sscanf(aline, "[%02d] %p %x", &ib, &va, &pa) == 3 && ib == ibuf){
				printf("found: ib=%d pa=0x%08x\n", ib, pa);
				xllc_def->pa = pa;
				xllc_def->len = HB_LEN;
				return;
			}
		}
	}
	fprintf(stderr, "ERROR: devnum:%d buffer%d NOT FOUND\n", devnum, ibuf);
	exit(1);
}

