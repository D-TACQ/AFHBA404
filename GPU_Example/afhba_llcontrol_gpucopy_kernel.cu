#include "afhba-llcontrol-gpucopy.h"

#define NSEC_PER_CLK  1		// SWAG
#define DEBUG_PERIODIC_STATUS 		0
#define VERBOSE 			1
#define REPORT_MISSED_SAMPLE_ERROR	0   /* this is a VALUABLE feature, @@todo but it's firing bogusly */

#define HAS_DO32			0

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
	int proc_number = blockIdx.x*blockDim.x + threadIdx.x;
	int ao_stride = blockDim.x;
	bool proc0 = (proc_number==0);
	int wait = !profile && proc0;
	
	__shared__ float sAI[AI_CHAN];

	//printf("%2d Starting data loop now! %d cycles NCHAN %d blk:%d dim:%d tid:%d\n", proc_number, nCycles, NCHAN, blockIdx.x, blockDim.x, threadIdx.x);

	unsigned tl0 = *tlatch;
	volatile unsigned tl;

	for (int ii = 0; !stop && ii < nCycles; ii++) {
		if (wait){
			tl = wait_sample(ii, tlatch, tl0, pai0);
		}
		__syncthreads();
		for (int ai = proc_number; ai < AI_CHAN; ai += blockDim.x){
			sAI[ai] = pai0[ai];
		}
		/* to the calculation here. IDEALLY, ao_stride==AO_CHAN ie one thread per AO, 		 
		 * but plan to test with smaller #threads to prove GPU goodness
		 */
		for (int ao = proc_number; ao < AO_CHAN; ao += ao_stride){
			int ao_result = 0;
		
			for (int ai = 0; ai < COLS; ++ai){
				ao_result += AMX[ao*AI_CHAN+ai]*sAI[ai];
			}

			if (ao_result > 0x7fff){
				ao_result = 0x7fff;
			}else if (ao_result < -0x7fff){
				ao_result = -0x7fff;
			}
			pao0[ao] = (short)ao_result;
		}
		
		
#if DEBUG_PERIODIC_STATUS     
		if (proc0 && ii%40000 == 0){
			printf("Cycle: %10d tl:%10u tl0 %10u\n", ii, tl, tl0);
			for (int iw = 0; iw < 80; ++iw){
				printf("%08x%c", pvi[iw], iw%16==15? '\n': ' ');
			}
		}
#endif
		__syncthreads();
	}
	return;
}



#define REDCOLS		4

/**
 * llcontrol_gpu_A_matrix_p() .. N way parallel kernel for matrix output
 * This didn't fit: too big for _shared_ memory:
 *    Assume AO_CHAN * AI_CHAN threads, single pass multiplication, then binary reduction for dot product.
 * So now we try to partition the matrix with >AI_CHAN threads, 
 * 	result is an output [REDCOLS][AO] that does fit in _shared_ memory, followed by a limited reduction to single column.
 * 
 * AO[64] = A.AI[128]   
 * 
 * This works with <<<AI_CHAN>>> ie 8192 threads.
 */
__global__ void llcontrol_gpu_A_matrix_p(void * volatile ai_buffer_ptr,
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
	
	
	__shared__ float sAMX[AI_CHAN][AO_CHAN];		// local A matrix, init from global

	__shared__ float sAI[AI_CHAN];			// local copy of AI data, init once
	__shared__ float sAO[REDCOLS][AO_CHAN];
	
	unsigned tl0 = *tlatch;
	volatile unsigned tl;

	for (int sam = 0; !stop && sam < nCycles; sam++) {
		if (wait){
			tl = wait_sample(sam, tlatch, tl0, pai0);	// only index==0 need do this.
		}
		__syncthreads();
		
		for (int ai = index; ai < AI_CHAN; ai += stride){
			sAI[ai] = pai0[ai];				// cache AI from ACQ2106
		}
#if 1		
		/* multiply */
		for (int ii = index; ii < AO_CHAN*REDCOLS; ii += REDCOLS){
			for (int jj = 0; jj < AI_CHAN/REDCOLS; ++jj){
				int ai = jj+ii;
				int ao = ii/REDCOLS;
				sAO[jj%REDCOLS][ao] = sAMX[ao][ai]*sAI[ai];
			}
		}
		
		/* reduce */
		int cols = REDCOLS/2;
		for (int ii = index; ii < AO_CHAN*cols; ++ii){
			for(int ao = 0; ao < AO_CHAN; ++ao){
				sAO[ii][ao] += sAO[cols-1-ii][ao];
			}
		}
		
		/* output as saturated short */
		for (int ao = index; ao < AO_CHAN; ++ao){
			int ao_result = sAO[ao][0];
			if (ao_result > 0x7fff){
				ao_result = 0x7fff;
			}else if (ao_result < -0x7fff){
				ao_result = -0x7fff;
			}
			pao0[ao] = (short)ao_result;
		}
#endif	
		
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
	if (REDUCE_ALGO){
		printf("NEW STUFF\n");
		printf("AI:%d AO:%d THREADS:%d\n", AI_CHAN, AO_CHAN, AO_CHAN*REDCOLS);
		llcontrol_gpu_A_matrix_p<<<1,AO_CHAN*REDCOLS>>>(ai_buffer_ptr, ao_buffer_ptr, total_data, AMX, nCycles, PROFILE);
	}else{
		printf("AI:%d AO:%d THREADS:%d COLS:%d\n", AI_CHAN, AO_CHAN, AO_CHAN, COLS);
		llcontrol_gpu_A_matrix<<<1,AO_CHAN>>>(ai_buffer_ptr, ao_buffer_ptr, total_data, AMX, nCycles, PROFILE);
	}
	return;
}
