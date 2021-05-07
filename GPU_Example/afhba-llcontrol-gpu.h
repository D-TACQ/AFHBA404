/*-------------------------------------
*       afhba-llcontrol-gpu.h :
*            Common header file for the cuda kernels and programs
*
*
*
*/


#ifndef CONTROL_HEADER_H
#define CONTROL_HEADER_H



#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
//#include <dirent.h>
//#include <signal.h>
//#include <pthread.h>
//#include <math.h>
#include <stdint.h>
#include <stdlib.h>
//#include <time.h>
//#include <string>

//#include <unistd.h>
//#include <fcntl.h>
//#include <string.h>
//#include <errno.h>
//#include <sys/uio.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/mman.h>
//#include <stdbool.h>

#include "afhba-llcontrol-common.h"
#include "pwm_internals.h"

#define HB_LEN 0x100000
#define BUFFER_AB_OFFSET 0x040000
#define HTS_MIN_BUFFER 4096
#define NCHAN 16

#define NSAMP     1000000 // How many buffer loops to go through
#define NCHAN     16
#define NSHORTS		2048
#define VI_LEN 		(NSHORTS*sizeof(short))
#define VI_LONGS	(VI_LEN/sizeof(unsigned))
#define VO_LEN  (32*sizeof(unsigned))
#define EOB(buf) (((volatile unsigned *)(buf))[VI_LONGS-1])
#define BOB(buf) (((volatile unsigned *)(buf))[0])

#define DTACQ_TIMEOUT 1
#define MISSED_BUFFER 2
#define SAMPLE_SUCCESS 0

#define MARKER 0xdeadc0d1
#define MAX_DO32 32

// Some constants for the control algorithm
#define DT_ACQ482 1e-7

// Global structs to pass information around
extern struct gpudma_lock_t lock;

extern short * tdata_cpu;
extern short * tdata_gpu;
extern int nsamples;

// Dummy functions for the CUDA kernel wrappers and common functions:
void vecAdd_wrapper(float * a, float * b, float * c, int n);
void llcontrol_gpu_example(void * volatile ai_buffer_ptr, 
                           unsigned * volatile ao_buffer_ptr,
                           short * total_data,
                           int nCycles);


#endif
