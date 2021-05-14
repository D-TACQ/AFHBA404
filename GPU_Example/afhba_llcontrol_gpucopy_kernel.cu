#include "afhba-llcontrol-gpucopy.h"

#define NSEC_PER_CLK  1		// SWAG
#define DEBUG_PERIODIC_STATUS 		0
#define MULVERBOSE			0x0c
#define REDVERBOSE			0x20
#define VERBOSE 			0
#define REPORT_MISSED_SAMPLE_ERROR	0   /* this is a VALUABLE feature, @@todo but it's firing bogusly */

#define HAS_DO32			0
#define USE_SHARED_A			0   /* shared is supposed to be faster, but for our low thread case, it's not ! */
#define USE_SHARED_R			1   /* R case HAS to use shared mem */

//#define COLS				2		// fastest test


#ifndef COLS
#define COLS	AI_CHAN					// full house
#endif

__device__ int stop;

/**
 * nsleep() busy wait .. it appears that hammering an aread of global to detect a change, can block the change, so avoid by busy waiting ..
 */
__device__ void nsleep(unsigned nsec) {
	long long start_clock = clock64();
	long long clock_count = nsec/NSEC_PER_CLK;
	//	printf("nsleep: nsec:%u clock_count:%llu\n", nsec, clock_count);
	for (unsigned pollcat = 0; clock64() - start_clock  < clock_count; ) {
		if (++pollcat&0xfffff == 0){
			printf("nsleep pc:%u start:%llu now:%llu end:%llu\n", pollcat, start_clock, clock64() - start_clock,  clock_count);
		}
	}
}

/**
 * wait_sample() .. return when new sample has arrived.
 * 
 * @@todo: test for missing samples. A valuable test, but in some builds it fails with a false positive: the data is right but it reports ERROR ???
 */
__device__ int wait_sample(int ii, unsigned* tlp, unsigned tl0, short* pai0)
{
	unsigned tl;
#if REPORT_MISSED_SAMPLE_ERROR	
	unsigned tl0p1 = tl0+1;
#endif	
	for (unsigned pollcat = 0; (tl = *tlp) == tl0; ){
		if ((++pollcat&0x0fffff) == 0){
			printf("ii:%10d pollcat:%08x nothing to see at %p %08x %04x %04x %04x %04x\n",
					ii, pollcat, tlp, *tlp, pai0[0]&0xffff, pai0[1]&0xffff, pai0[2]&0xffff, pai0[3]&0xffff );
			if (ii > 0){
				printf("QUITTING on data flow stop\n");
				stop = 1;
				return 0;
			}
		}else{
			nsleep(SLEEP_NS);
		}
	}
#if REPORT_MISSED_SAMPLE_ERROR	
	if (tl0p1 != tl){
		printf("ERROR: wait_sample() %d missing tl tl0:%u wanted:%u got:%u %s\n", ii, tl0, tl0p1, tl, tl0p1 == tl? "EQ": "NE");
		stop = 1;
	}
#endif	
	return tl;
}

#ifdef NAIVE_CPU_IMPLEMENTATION
void cpu_A_Matrix() {
	short AO[AO_CHAN];
	short AI[AI_CHAN];
#ifdef TWO_D_MATRIX_OK	
	float AMX[AO_CHAN][AI_CHAN];
#else
	float AMX[AO_CHAN*AI_CHAN];
#endif	
	
	for (int ao = 0; ao < AO_CHAN; ++ao){
		for (int ai = 0; ai < AI_CHAN; ++ai){
#ifdef TWO_D_MATRIX_OK			
			AO[ao] += AMX[ao][ai] * AI[ai];
#else			
			AO[ao] += AMX[ao*AI_CHAN+ai] * AI[ai];
#endif			
		}
	}
}
#endif

/**
 * llcontrol_gpu_A_matrix() .. kernel for matrix output
 * 
 * AO[64] = A.AI[128]   
 * 
 * This works with AO_THREADS=32 for 32 AO, and with AO_THREADS=1 (fully serial)
 */
__global__ void llcontrol_gpu_A_matrix(void * volatile ai_buffer_ptr,
		unsigned * volatile ao_buffer_ptr,
		short * total_data,
		float* AMX,
		int nCycles,
		int profile){
	unsigned * tlatch = &((unsigned*)ai_buffer_ptr)[AI_CHAN/2+HAS_DO32];
	short * pai0 = (short*)ai_buffer_ptr;
	unsigned * pvi = (unsigned*)ai_buffer_ptr;
	short * pao0 = (short*)ao_buffer_ptr;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x;
	bool proc0 = (index==0);
	int wait = !profile && proc0;
	
	__shared__ float sAI[AI_CHAN];

#if USE_SHARED_A	
	__shared__ float sAMX[AI_CHAN*AO_CHAN];		// local A matrix, init from global
	
	for (int ii = index; ii < AI_CHAN*AO_CHAN; ii += stride){
		sAMX[ii] = AMX[ii];
	}
#define MY_AMX	sAMX
#else
#define MY_AMX  AMX
#endif
	
	//printf("%2d Starting data loop now! %d cycles NCHAN %d blk:%d dim:%d tid:%d\n", proc_number, nCycles, NCHAN, blockIdx.x, blockDim.x, threadIdx.x);

	unsigned tl0 = *tlatch;
	volatile unsigned tl;

	for (int sam = 0; !stop && sam < nCycles; sam++) {
		if (wait){
			tl = wait_sample(sam, tlatch, tl0, pai0);
		}
		__syncthreads();
		for (int ai = index; ai < AI_CHAN; ai += stride){
			sAI[ai] = pai0[ai];
		}
		/* to the calculation here. IDEALLY, ao_stride==AO_CHAN ie one thread per AO, 		 
		 * but plan to test with smaller #threads to prove GPU goodness
		 */
		for (int ao = index; ao < AO_CHAN; ao += stride){
			int ao_result = 0;
		
			for (int ai = 0; ai < COLS; ++ai){
				ao_result += MY_AMX[ao*AI_CHAN+ai]*sAI[ai];
			}

			if (ao_result > 0x7fff){
				ao_result = 0x7fff;
			}else if (ao_result < -0x7fff){
				ao_result = -0x7fff;
			}
			pao0[ao] = (short)ao_result;
		}
		
		
#if DEBUG_PERIODIC_STATUS     
		if (proc0 && sam%40000 == 0){
			printf("Cycle: %10d tl:%10u tl0 %10u\n", sam, tl, tl0);
			for (int iw = 0; iw < 80; ++iw){
				printf("%08x%c", pvi[iw], iw%16==15? '\n': ' ');
			}
		}
#endif
		__syncthreads();
	}
	return;
}



#ifdef MY_AMX
#undef MY_AMX
#endif

#define REDCOLS		16

/**
 * llcontrol_gpu_A_matrix_reduce() .. N way parallel kernel for matrix output
 * This didn't fit: too big for _shared_ memory:
 *    Assume AO_CHAN * AI_CHAN threads, single pass multiplication, then binary reduction for dot product.
 * So now we try to partition the matrix with >AI_CHAN threads, 
 * 	result is an output [AO][REDCOLS] that does fit in _shared_ memory, followed by a limited reduction to single column.
 * 
 * AO[64] = A.AI[128]   
 * 
 * This works with <<<AI_CHAN>>> ie 8192 threads.
 */
__global__ void llcontrol_gpu_A_matrix_reduce(void * volatile ai_buffer_ptr,
		unsigned * volatile ao_buffer_ptr,
		short * total_data,
		float* AMX,
		int nCycles,
		int profile){
	unsigned * tlatch = &((unsigned*)ai_buffer_ptr)[AI_CHAN/2+HAS_DO32];
	short * pai0 = (short*)ai_buffer_ptr;	
	short * pao0 = (short*)ao_buffer_ptr;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = blockDim.x;
	bool proc0 = (index==0);
	int wait = !profile && proc0;

#if USE_SHARED_R
	__shared__ float sAMX[AI_CHAN*AO_CHAN];		// local A matrix, init from global
	
	for (int ii = index; ii < AI_CHAN*AO_CHAN; ii += stride){
#if VERBOSE & 0x1		
		printf("init sAMX[%3d]\n", ii);
#endif		
		sAMX[ii] = AMX[ii];
	}
#define MY_AMX sAMX
#else
#define MY_AMX AMX
#endif	

	__shared__ float sAI[AI_CHAN];			// local copy of AI data, init once
	__shared__ float sAO[AO_CHAN][REDCOLS];
	
	unsigned tl0 = *tlatch;
	volatile unsigned tl;

	for (int sam = 0; !stop && sam < nCycles; sam++) {
		if (wait){
			tl = wait_sample(sam, tlatch, tl0, pai0);	// only index==0 need do this.
		}
		__syncthreads();
				
		for (int ai = index; ai < AI_CHAN; ai += stride){
#if VERBOSE & 0x2			
			printf("tid:%03d ai:%d\n", ai, ai);
#endif			
			sAI[ai] = pai0[ai];				// cache AI from ACQ2106
		}
		__syncthreads();
		
		/* multiply */
		for (int ii = index; ii < AO_CHAN*REDCOLS; ii += stride){
			int ao = ii/REDCOLS;
			int rc = ii%REDCOLS;
			int ai = rc*AI_CHAN/REDCOLS;
			
			sAO[ao][rc] = MY_AMX[ao*AI_CHAN+ai]*sAI[ai];
#if VERBOSE & 0x4		
			printf("tid:%03d mul AO[%2d][%d]  = AMX[%2d][%3d] * AI[%3d]\n",
						ii, ao, rc, ao, ai+0,        ai+0);
			__syncthreads();
#endif			
			for (int jj = 1; jj < AI_CHAN/REDCOLS; ++jj){				
				sAO[ao][rc] += MY_AMX[ao*AI_CHAN+ai+jj]*sAI[ai+jj];
#if VERBOSE & 0x8
				printf("tid:%03d mac AO[%2d][%d] += AMX[%2d][%3d] * AI[%3d]\n",
						ii, ao, rc, ao, ai+jj,        ai+jj);
#endif				
			}
		}
		__syncthreads();
		/* reduce */		
		for(int ao = index; ao < AO_CHAN; ao += stride){
			for (int cols = REDCOLS/2; cols; cols /= 2){
				for (int ii = 0; ii < cols; ++ii){	
					sAO[ao][ii] += sAO[ao][cols*2-1-ii];
#if VERBOSE & 0x20			
					printf("tid:%03d red sAO[%2d][%d] += sAO[%2d][%d]\n", ao, ao, ii, ao, cols*2-1-ii);
#endif					
				}
			}
		}

		/* output as saturated short */
		for (int ao = index; ao < AO_CHAN; ao += stride){
			int ao_result = sAO[ao][0];
			if (ao_result > 0x7fff){
				ao_result = 0x7fff;
			}else if (ao_result < -0x7fff){
				ao_result = -0x7fff;
			}
			pao0[ao] = (short)ao_result;
#if VERBOSE & 0x40			
			printf("tid:%03d out pAO[%2d] sAO[%2d][%d] %04x\n", ao, ao, ao, 0, (short)ao_result);
#endif	
		}
		
#if DEBUG_PERIODIC_STATUS     
		if (proc0 && sam%40000 == 0){
			printf("Cycle: %10d tl:%10u tl0 %10u\n", sam, tl, tl0);
			for (int iw = 0; iw < 80; ++iw){
				printf("%08x%c", sAI[iw], iw%16==15? '\n': ' ');
			}
		}
#endif
	}
	return;
}

void llcontrol_gpu_A_matrix_wrapper(void * volatile ai_buffer_ptr,
		unsigned * volatile ao_buffer_ptr,
		short * total_data,
		float* AMX,
		int nCycles){
	//Wrapper to call the CUDA kernel
	if (COLS < AO_CHAN){
		printf("Reduced column count %d\n", COLS);
	}
	if (VERBOSE){
		printf("VERBOSE=0x%x\n", VERBOSE);
	}
	if (REDUCE_ALGO){		
		printf("NEW STUFF split AMX row into REDCOLS:%d\n", REDCOLS);
		printf("AI:%d AO:%d THREADS:%d USE_SHARED:%d\n", AI_CHAN, AO_CHAN, AO_CHAN*REDCOLS, USE_SHARED_R);
		int B = 1, T = AO_CHAN*REDCOLS;
		if (AO_CHAN*REDCOLS > 1024){
			B = 1+ AO_CHAN*REDCOLS/1024;
			T = 1024;
		}
		llcontrol_gpu_A_matrix_reduce<<<B,T>>>(ai_buffer_ptr, ao_buffer_ptr, total_data, AMX, nCycles, PROFILE);
		//llcontrol_gpu_A_matrix_reduce<<<1,1>>>(ai_buffer_ptr, ao_buffer_ptr, total_data, AMX, nCycles, PROFILE);
	}else{
		printf("AI:%d AO:%d THREADS:%d COLS:%d USE_SHARED:%d\n", AI_CHAN, AO_CHAN, AO_CHAN, COLS, USE_SHARED_A);
		llcontrol_gpu_A_matrix<<<1,AO_CHAN>>>(ai_buffer_ptr, ao_buffer_ptr, total_data, AMX, nCycles, PROFILE);
	}
	return;
}
