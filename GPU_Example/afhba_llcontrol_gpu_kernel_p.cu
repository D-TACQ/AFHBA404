#include "afhba-llcontrol-gpucopy.h"

// hello

#define NSEC_PER_CLK  10		// SWAG


__device__ void nsleep(unsigned nsec) {
	clock_t start_clock = clock();
        clock_t clock_count = nsec/NSEC_PER_CLK;
        while (clock() - start_clock  < clock_count) {
		;
        }
}

__device__ int wait_sample(int ii, unsigned* tlp, unsigned tl0, short* pai0)
{
	unsigned tl;
	for (int pollcat = 0; (tl = *tlp) == tl0; ){
		if ((++pollcat&0xffff) == 0){
                	printf("ii:%10d pollcat:%08x nothing to see at %p %08x %04x %04x %04x %04x\n",
                                        ii, pollcat, tlp, *tlp, pai0[0]&0xffff, pai0[1]&0xffff, pai0[2]&0xffff, pai0[3]&0xffff );
                }else{
			nsleep(1000);
		}
	}
	return tl;
}

__global__ void llcontrol_gpu_dummy(void * volatile ai_buffer_ptr,
                              unsigned * volatile ao_buffer_ptr,
                              short * total_data,
                              int nCycles){
  unsigned * tlatch = &((unsigned*)ai_buffer_ptr)[NCHAN/2+1];
  short * pai0 = (short*)ai_buffer_ptr;
  unsigned * pvi = (unsigned*)ai_buffer_ptr;
  short * pao0 = (short*)ao_buffer_ptr;
  int proc_number = blockIdx.x*blockDim.x + threadIdx.x;
  bool proc0 = (proc_number==0);

  printf("Starting data loop now! %d cycles NCHAN %d blk:%d dim:%d tid:%d\n", nCycles, NCHAN, blockIdx.x, blockDim.x, threadIdx.x);

  unsigned tl0 = *tlatch;
  unsigned tl;
#if 0
  for (int ii = 0; ii < 16; ii++) {
      printf("ai[%d] %p 0x%04x\n", ii, pai0+ii, pai0[ii]);
  }
#endif

  for (int ii = 0; ii < nCycles; ii++) {
	if (proc0){
		tl = wait_sample(ii, tlatch, tl0, pai0);
	}
      	pao0[0] = pai0[0] *.5 ;

	for (int ic = 1; ic < 32; ++ic){
		pao0[ic] = pai0[32+ic-1] * 1.02;
	}
#if 1      
      if (ii%40000 == 0){
		printf("Cycle: %10d tl:%10u tl0 %10u\n", ii, tl, tl0);
	      	for (int iw = 0; iw < 80; ++iw){
			printf("%08x%c", pvi[iw], iw%16==15? '\n': ' ');
	      	}
      }
#endif      
      tl0 = tl;
  }



  __syncthreads();
  return;
}

//  if (proc0) printf("Terminating GPU Kernel.\n");
//  return;
//
//}


void llcontrol_gpu_example_dummy(void * volatile ai_buffer_ptr,
                           unsigned * volatile ao_buffer_ptr,
                           short * total_data,
                           int nCycles){
  //Wrapper to call the CUDA kernel
  llcontrol_gpu_dummy<<<1,1>>>(ai_buffer_ptr, ao_buffer_ptr, total_data, nCycles);
  return;
}
