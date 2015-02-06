/* ------------------------------------------------------------------------- */
/* rtm-t-stream-disk.cpp RTM-T PCIe Host Side test app	             	     */
/* ------------------------------------------------------------------------- */
/*   Copyright (C) 2010 Peter Milne, D-TACQ Solutions Ltd
 *                      <Peter dot Milne at D hyphen TACQ dot com>
                                                                               
    This program is free software; you can redistribute it and/or modify
    it under the terms of Version 2 of the GNU General Public License
    as published by the Free Software Foundation;
                                                                               
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
                                                                               
    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.                */
/* ------------------------------------------------------------------------- */


/** @file rtm-t-stream-disk.cpp D-TACQ PCIe RTM_T test app.
 * Continuous streaming to disk using PCIe
 * see bin/stream-to-ramdisk for example usage
 * */



#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>


#include <sched.h>

//using namespace std;

#include "RTM_T_Device.h"
#include "local.h"
#include "popt.h"

#include "rtm-t_ioctl.h"

#define DIAG(args...)
//#define DIAG(args...)	fprintf(stderr, args)

/* default: never completes */
int MAXITER	       = 0xffffffff;
int MAXITER_MASK       = 0x7fffffff;


int NELEMS = RTM_T_Device::MAXBUF;
int USLEEP = 0;
int VERBOSE = 0;

int SSIZE = sizeof(short) * 96;

int acq200_debug = 0;

int maxlen = RTM_T_Device::MAXLEN;

#define PARAMETERS		"/sys/module/afhba/parameters/"
#define BUFFER_LEN 		PARAMETERS "buffer_len"
#define NBUFFERS		PARAMETERS "nbuffers"
#define TRANSFER_BUFFERS 	PARAMETERS "transfer_buffers"

const char* OUTROOT = "/mnt";


static RTM_T_Device *dev;

int CYCLE;
int RECYCLE = 0;		/* accumulate by default */
int NO_OVERWRITE = 0;		/* refuse to allow buffer overwrite */
int MAXINT = 999999;
int PUT_DATA = 1;		/* output name of data buffer, not id */
int NBUFS = 0;			/* !=) ? stop after this many buffers */


/* number of buffers to transfer - set on command line or use module knob. */
unsigned transfer_buffers;

static double htime(void){
	static struct timeval t0;
	static int t0_valid;
	struct timeval t1, td;
	struct timeval carry = { 0, 0 };

	if (!t0_valid){
		gettimeofday(&t0, 0);
		t0_valid = 1;
	}

	gettimeofday(&t1, 0);
	
	if (t1.tv_usec < t0.tv_usec){
		carry.tv_usec = 1000000;
		carry.tv_sec = 1;
	}
	td.tv_usec =	t1.tv_usec	+ carry.tv_usec	- t0.tv_usec;
	td.tv_sec =	t1.tv_sec	- carry.tv_sec	- t0.tv_sec;

	double tds = td.tv_sec + (double)(td.tv_usec)/1000000;
	return tds;
}

static int write_meta(int fd, int ibuf, int nbuf)
{
	unsigned long long nsamples = (unsigned long long)nbuf * maxlen / SSIZE;
	
	char buf[128];
	snprintf(buf, 128, 
		"IBUF=%d\n" "NBUF=%d\n" "NSAMPLES=%llu\n" "HTIME=%.3f\n",
		 ibuf, nbuf, nsamples, htime());
	write(fd, buf, strlen(buf));
}

#define O_MODE	(O_WRONLY|O_CREAT|O_TRUNC)
#define PERM	(S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH)

static void process(int ibuf, int nbuf){
	if (VERBOSE){
		fprintf(stderr, "%02d\n", ibuf);
	}

	char buf[80];

	
	sprintf(buf, "%s/%06d/", OUTROOT, CYCLE);
	mkdir(buf, 0777);
	sprintf(buf, "%s/%06d/%d.%02d", OUTROOT, CYCLE, dev->getDevnum(), ibuf);

	if (NO_OVERWRITE){
		struct stat stat_buf;
		int rc = stat(buf, &stat_buf);
		DIAG("stat:rc %d errno %d\n", rc, errno);
		if (rc == 0){
			err("OVERRUN: NO_OVERWRITE SET and \"%s\" exists",
					buf);
			exit(1);
		}else if (errno != ENOENT){
			err("OVERRUN: NO_OVERWRITE SET and \"%s\" exists",
					buf);
			perror("buf");
			exit(errno);
		}else{
			;	/* ENOENT - that's good! */
		}
	}
	int outfp = open(buf, O_MODE, PERM);

	if (outfp == -1){
		perror(buf);
		_exit(errno);
	}
	write(outfp, dev->getHostBufferMapping(ibuf), maxlen);

	strcat(buf, ".id");
	int out_meta = open(buf, O_MODE, PERM);
	write_meta(out_meta, ibuf, nbuf);

	close(out_meta);		
	close(outfp);		/* close data last - we monitor this one */

	if (PUT_DATA){
		char *cp = strstr(buf, ".id");
		if (cp){
			*cp = '\0';
		}
	}
	puts(buf);

	if (USLEEP){
		usleep(USLEEP);
	}
}


static int stream()
{
	unsigned iter = 0;
	int id_buf[16];
	int fp = dev->getDeviceHandle();
	int ifirst = MAXINT;
	int nbuf = 0;

	int rc = ioctl(fp, RTM_T_START_STREAM_MAX, &transfer_buffers);
	if (rc != 0){
		perror("ioctl RTM_T_START_STREAM failed");
		exit(errno);
	}

	while (iter < MAXITER){
		DIAG("CALLING read\n");
		int nread = read(fp, id_buf, NELEMS*sizeof(int));
		int ibuf;

		DIAG("	read returned %d\n", nread);

		if (nread > 0){
			for (ibuf = 0; ibuf < nread/sizeof(int); ++ibuf){
				int nwrite =  sizeof(int)*1;

				if (id_buf[ibuf] <= ifirst ){
					ifirst = id_buf[ibuf];
					if (RECYCLE == 0){
						++CYCLE;
					}else{
						if (++CYCLE >= RECYCLE){
							CYCLE=0;
						}
					}
				}
				DIAG("CALLING process\n");
				process(id_buf[ibuf], ++nbuf);
			
				dbg(2, "write [%d] %d\n", ibuf, id_buf[ibuf]);

				DIAG("CALLING write\n");
				if (write(fp, id_buf+ibuf, nwrite) != nwrite ){
					perror("write failed");
					return -1;
				}
				if (NBUFS && nbuf >= NBUFS){
					printf("NBUFS %d reached, quitting now\n", NBUFS);
					goto all_done;
				}
			}
		}else{
			perror("read error");
			return nread;
		}

		iter = ++iter&MAXITER_MASK;
	}
all_done:
	DIAG("all done\n");
	return 0;
}

static int getKnob(const char* knob, unsigned* value)
{
	FILE *fp = fopen(knob, "r");
	int rc = fscanf(fp, "%u", value);
	fclose(fp);
	return rc;
}


static void calc_maxlen(int devnum)
{
	char knob[80];
	FILE *fp;
/*
	snprintf(knob, 80, "/dev/rtm-t.%d.ctrl/lowlat", devnum);
	FILE *fp = fopen(knob, "r");
	if (fp){
		int ll_maxlen;
		int nc = fscanf(fp, "%d", &ll_maxlen);
		fclose(fp);

		if (nc == 1 && ll_maxlen > 0){
			maxlen = ll_maxlen;
			info("maxlen set LL %d\n", maxlen);
			return;
		}
	}
	fclose(fp);
*/
	snprintf(knob, 80, "/dev/rtm-t.%d.ctrl/buffer_len", devnum);
	fp = fopen(knob, "r");
	if (fp){
		int nc = fscanf(fp, "%d", &maxlen);
		fclose(fp);

		if (nc == 1 && maxlen > 0){
			info("maxlen set LL %d\n", maxlen);
			return;
		}
	}
	fclose(fp);

	fp = fopen(BUFFER_LEN, "r");
	if (!fp){
		perror(BUFFER_LEN);
		exit(errno);
	}
	if (fscanf(fp, "%d", &maxlen) == 1){
		info("maxlen set %d", maxlen);
	}else{
		err("maxlen not set");
	}
	fclose(fp);
}

static void run_stop_monitor(char *monitor)
{
	/* actually, this is unnecessary since capture will stop on timeout
	 * however, it could be useful for a rapid exit.
	 * it's anticipated that monitor is an expect script blocking on the
	 * statemon service from ACQ.
	 */
	int ppid = getpid();

	if (fork() == 0){
		char arg1[20];
		sprintf(arg1, "%d", ppid);
		info("monitor: %s %s\n", monitor, arg1);
		int rc = execlp(monitor, basename(monitor), arg1);

		if (rc){
			perror("execlp failed");
		}
	}
}

void setRtPrio(int prio)
{
	struct sched_param prams = {
	};

	prams.sched_priority = prio;

	int rc = sched_setscheduler(0, SCHED_FIFO, &prams);
	if (rc != 0){
		perror("sched_setscheduler()");
	}
}


static void init_defaults(int argc, char* argv[])
{
	int devnum = 0;

	unsigned nbuffers = RTM_T_Device::MAXBUF;

	getKnob(NBUFFERS, &nbuffers);
	info("using %d buffers\n", nbuffers);
	
	if (getenv("RTM_DEVNUM")){
		devnum = atol(getenv("RTM_DEVNUM"));
	}
	dev = new RTM_T_Device(devnum, nbuffers);
	
	if (getenv("RTM_MAXITER")){
		MAXITER = atol(getenv("RTM_MAXITER"));
		info("MAXITER set %d\n", MAXITER);
	}
	if (getenv("RTM_NELEMS")){
		NELEMS = atol(getenv("RTM_NELEMS"));
		info("NELEMS set %d\n", NELEMS);
	}
	if (getenv("RTM_DEBUG")){
		acq200_debug = atol(getenv("RTM_DEBUG"));
		info("DEBUG set %d\n", acq200_debug);
	}
	if (getenv("RTM_USLEEP")){
		USLEEP = atol(getenv("RTM_USLEEP"));
		info("USLEEP set %d\n", USLEEP);
	}
	if (getenv("RTM_VERBOSE")){
		VERBOSE = atol(getenv("RTM_VERBOSE"));
		info("VERBOSE set %d\n", VERBOSE);
	}
	if (getenv("RTM_PUT_DATA")){
		PUT_DATA=atoi(getenv("RTM_PUT_DATA"));
	}
	if (getenv("SSIZE")){
		SSIZE = atoi(getenv("SSIZE"));
		info("SSIZE set %d\n", SSIZE);
	}

	if (getenv("RTPRIO")){
		setRtPrio(atoi(getenv("RTPRIO")));
	}
	if (getenv("OUTROOT")){
		OUTROOT=getenv("OUTROOT");
	}
	if (getenv("NBUFS")){
		NBUFS=atoi(getenv("NBUFS"));
	}
	if (getenv("RECYCLE")){
		RECYCLE=atol(getenv("RECYCLE"));
	}
	if (getenv("NO_OVERWRITE")){
		NO_OVERWRITE=atol(getenv("NO_OVERWRITE"));
	}
	setvbuf(stdout, 0, _IOLBF, 0);

	calc_maxlen(devnum);


	char* monitor;
	if ((monitor = getenv("KILL_ON_STOP")) != 0){
		run_stop_monitor(monitor);
	}

	getKnob(TRANSFER_BUFFERS, &transfer_buffers);
	if (argc > 1){
		unsigned tb = strtoul(argv[1], 0, 0);
		if (tb > 0){
			transfer_buffers = tb;
		}else{
			err("stream-disk [NBUFFERS]");
		}
	}
	info("streaming %u buffers", transfer_buffers);
}

int main(int argc, char* argv[])
{
	init_defaults(argc, argv);
	return stream();
}
