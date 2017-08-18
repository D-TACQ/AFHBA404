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
int CONCAT = 0;

int SSIZE = sizeof(short) * 96;

int acq200_debug = 0;

const char* OUTROOT = "/mnt";


static RTM_T_Device *dev;

int CYCLE;
int RECYCLE = 0;		/* accumulate by default */
int NO_OVERWRITE = 0;		/* refuse to allow buffer overwrite */
int MAXINT = 999999;
int PUT_DATA = 1;		/* output name of data buffer, not id */
int NBUFS = 0;			/* !=) ? stop after this many buffers */
int PUT4KPERFILE = 0;		/* fake 4MB output for pv to measure at /1000 */

struct SEQ {
	unsigned long errors;
	unsigned long buffers;
}
	SEQ;

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
	unsigned long long nsamples = (unsigned long long)nbuf * dev->maxlen / SSIZE;

	char buf[128];
	snprintf(buf, 128, 
		"IBUF=%d\n" "NBUF=%d\n" "NSAMPLES=%llu\n" "HTIME=%.3f\n",
		 ibuf, nbuf, nsamples, htime());
	write(fd, buf, strlen(buf));
}

void fail_if_exists(char *buf)
{
	struct stat stat_buf;
	int rc = stat(buf, &stat_buf);
	DIAG("stat:rc %d errno %d\n", rc, errno);
	if (rc == 0){
		err("OVERRUN: NO_OVERWRITE SET and \"%s\" exists", buf);
		exit(1);
	}else if (errno != ENOENT){
		err("OVERRUN: NO_OVERWRITE SET and \"%s\" exists", buf);
		perror("buf");
		exit(errno);
	}else{
		;	/* ENOENT - that's good! */
	}
}

#define O_MODE	(O_WRONLY|O_CREAT|O_TRUNC)
#define PERM	(S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH)

int icat;
int outfp;

#define OUTROOTFMT	"%s/%06d/"
#define OUTFMT2		"%s/%06d/%d.%02d"
#define OUTFMT3		"%s/%06d/%d.%03d"

const char* outfmt;

char buf4k[4096];


int succ(int ib) {
	++ib;
	return ib >= dev->nbuffers? 0: ib;
}

static void process(int ibuf, int nbuf){
	if (VERBOSE == 1){
		if (ibuf%10 == 0){
			fprintf(stderr, "%c", ibuf==0? '\r': '.');
			fflush(stderr);
		}
	}else if (VERBOSE == 2){
		static int cycle;
		if (ibuf == 0){
			fprintf(stderr, "\r%06d", cycle++);
			fflush(stderr);
		}
	}else if (VERBOSE > 2){
		fprintf(stderr, "%02d\n", ibuf);
	}

	static char data_fname[80];
	static char old_fname[80];
	char buf[80];
	static int _ibuf = -1;
	
	if (_ibuf != -1){
		if (succ(_ibuf) != ibuf){
			SEQ.errors++;
			if (SEQ.errors < 10){
				if (succ(succ(_ibuf)) == ibuf || succ(ibuf) == _ibuf){
					fprintf(stderr, "WARNING potential buffer cycle detected %u/%u skip %d -> %d\n",
						SEQ.errors, SEQ.buffers, _ibuf, ibuf);
				}else{
					fprintf(stderr, "ERROR: buffer %u/%u skip %d -> %d\n",
									SEQ.errors, SEQ.buffers, _ibuf, ibuf);
				}
			}
		}
		SEQ.buffers++;
	}
	_ibuf = ibuf;

	if (icat == 0){
		sprintf(buf, OUTROOTFMT, OUTROOT, CYCLE);
		mkdir(buf, 0777);
		
		sprintf(data_fname, outfmt, OUTROOT, CYCLE, 
				dev->getDevnum(), ibuf);

		outfp = open(data_fname, O_MODE, PERM);
		if (outfp == -1){
			perror(buf);
			_exit(errno);
		}
		strcpy(buf, data_fname);
		strcat(buf, ".id");
		int out_meta = open(buf, O_MODE, PERM);
		write_meta(out_meta, ibuf, nbuf);
		close(out_meta);
	}
	if (NO_OVERWRITE){
		fail_if_exists(data_fname);
	}

	int rc = write(outfp, dev->getHostBufferMapping(ibuf), dev->maxlen);
	if (rc != dev->maxlen){
		perror("write fail");
		exit(1);
	}

	if (++icat > CONCAT){
		close(outfp);		/* close data last - we monitor this one */
		if (PUT_DATA){
			if (strlen(old_fname)){
				if (PUT4KPERFILE){
					strcpy(buf4k, old_fname);
					strcat(buf4k, "\n");
					fwrite(buf4k, 1, 4096, stdout);
					fflush(stdout);
				}else{
					puts(old_fname);
				}
			}
			strcpy(old_fname, data_fname);
		}else{
			puts(buf);
		}
		icat = 0;
	}

	if (USLEEP){
		usleep(USLEEP);
	}
}


static int stream()
{
	unsigned iter = 0;
	int id_buf[NELEMS];
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
			int nb = nread/sizeof(int);
			if (nb > 1){
				fprintf(stderr, "buffer_backlog:%d\n", nb);
			}
			for (ibuf = 0; ibuf < nb; ++ibuf){
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
			goto on_error;
		}

		iter = ++iter&MAXITER_MASK;
	}
on_error:
all_done:
	fprintf(stderr, "rtm-t-stream-disk finish %u seq errors in %u buffers\n", SEQ.errors, SEQ.buffers);
	DIAG("all done\n");
	return 0;
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
		int rc = execlp(monitor, basename(monitor), arg1, (char*)NULL);

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

	if (getenv("RTM_DEVNUM")){
		devnum = atol(getenv("RTM_DEVNUM"));
	}
	dev = new RTM_T_Device(devnum);
	
	outfmt = dev->nbuffers > 99? OUTFMT3: OUTFMT2;

	
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
	if (getenv("CONCAT")){
		CONCAT=atoi(getenv("CONCAT"));
	}
	if (getenv("PUT4KPERFILE")){
		PUT4KPERFILE = 1;
	}
	setvbuf(stdout, 0, _IOLBF, 0);

	char* monitor;
	if ((monitor = getenv("KILL_ON_STOP")) != 0){
		run_stop_monitor(monitor);
	}

	transfer_buffers = dev->transfer_buffers;
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
