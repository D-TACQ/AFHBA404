#include "afhba-llcontrol-gpucopy.h"

#define NSEC_PER_CLK  1		// SWAG
#define DEBUG_PERIODIC_STATUS 0

__device__ int stop;

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

__device__ int wait_sample(int ii, unsigned* tlp, unsigned tl0, short* pai0)
{
	unsigned tl;
	unsigned tl0p1 = tl0+1;
	
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
	if (tl0p1 != tl){
		printf("ERROR: wait_sample() %d missing tl tl0:%u wanted:%u got:%u %s\n", ii, tl0, tl0p1, tl, tl0p1 == tl? "EQ": "NE");
		stop = 1;
	}
	return tl;
}

__global__ void llcontrol_gpu_A_matrix(void * volatile ai_buffer_ptr,
		unsigned * volatile ao_buffer_ptr,
		short * total_data,
		float* AMX,
		int nCycles,
		int profile){
	unsigned * tlatch = &((unsigned*)ai_buffer_ptr)[AI_CHAN/2+1];
	short * pai0 = (short*)ai_buffer_ptr;
	unsigned * pvi = (unsigned*)ai_buffer_ptr;
	short * pao0 = (short*)ao_buffer_ptr;
	int proc_number = blockIdx.x*blockDim.x + threadIdx.x;
	int ao_stride = blockDim.x;
	bool proc0 = (proc_number==0);
	int wait = !profile && proc0;
	
	__shared__ float sAMX[AI_CHAN*AO_CHAN];
	if (proc0){
		for (int ii = 0; ii < AI_CHAN*AO_CHAN; ++ii){
			sAMX[ii] = AMX[ii];
		}
	}

	__shared__ float sAI[AI_CHAN];

	printf("%2d Starting data loop now! %d cycles NCHAN %d blk:%d dim:%d tid:%d\n", proc_number, nCycles, NCHAN, blockIdx.x, blockDim.x, threadIdx.x);

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
		
			for (int ai = 0; ai < AI_CHAN; ++ai){
				ao_result += sAMX[ao*AI_CHAN+ai]*sAI[ai];
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

void llcontrol_gpu_A_matrix_wrapper(void * volatile ai_buffer_ptr,
		unsigned * volatile ao_buffer_ptr,
		short * total_data,
		float* AMX,
		int nCycles){
	//Wrapper to call the CUDA kernel
	llcontrol_gpu_A_matrix<<<1,AO_THREADS>>>(ai_buffer_ptr, ao_buffer_ptr, total_data, AMX, nCycles, PROFILE);
	return;
}
