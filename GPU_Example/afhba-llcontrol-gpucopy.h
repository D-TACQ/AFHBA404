/*
 * afhba-llcontrol-gpucopy.h
 *
 *  Created on: 19 Feb 2021
 *      Author: Sean Alsop
 */

#ifndef GPU_EXAMPLE_AFHBA_LLCONTROL_GPUCOPY_H_
#define GPU_EXAMPLE_AFHBA_LLCONTROL_GPUCOPY_H_


#include "afhba-llcontrol-common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>


#define NCHAN     128
#define AI_CHAN	  128
#define AO_CHAN   64
#define MxN	  (AI_CHAN*AO_CHAN)
#define SPAD_LEN  32
#define NSHORTS   (NCHAN * 2) + SPAD_LEN
#define VI_LEN 	    (NSHORTS*sizeof(short))
#define VI_LONGS	(VI_LEN/sizeof(unsigned))
#define VO_LEN  (AO_CHAN*sizeof(int))
#define EOB(buf) (((volatile unsigned *)(buf))[VI_LONGS-1])
#define BOB(buf) (((volatile unsigned *)(buf))[0])

extern int AO_THREADS;
extern int PROFILE;

// @@todo .. is this even necessary?
#define SLEEP_NS	5

void llcontrol_gpu_A_matrix_wrapper(void * volatile ai_buffer_ptr,
                           unsigned * volatile ao_buffer_ptr,
                           short * total_data,
			   float* AMX,
                           int nCycles);




#endif /* GPU_EXAMPLE_AFHBA_LLCONTROL_GPUCOPY_H_ */
