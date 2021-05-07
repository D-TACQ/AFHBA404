#include "afhba-llcontrol-gpucopy.h"

// hello

#define NSEC 200


__global__ void llcontrol_gpu_dummy(void * volatile ai_buffer_ptr,
                              unsigned * volatile ao_buffer_ptr,
                              short * total_data,
                              int nCycles){
  unsigned * tlatch = &((unsigned*)ai_buffer_ptr)[NCHAN/2+1];
  short * pai0 = (short*)ai_buffer_ptr;
  unsigned * pvi = (unsigned*)ai_buffer_ptr;
  short * pao0 = (short*)ao_buffer_ptr;

  printf("Starting data loop now! %d cycles NCHAN %d\n", nCycles, NCHAN);

  unsigned tl0 = *tlatch;
  unsigned tl;
#if 0
  for (int ii = 0; ii < 16; ii++) {
      printf("ai[%d] %p 0x%04x\n", ii, pai0+ii, pai0[ii]);
  }
#endif

  for (int ii = 0; ii < nCycles; ii++) {
      int pollcat;
      for (pollcat = 0; (tl = *tlatch) == tl0; ){
	     	if ((++pollcat&0xfffff) == 0){
			printf("ii:%10d pollcat:%08x nothing to see at %p %08x %04x %04x %04x %04x\n", 
					ii, pollcat, tlatch, *tlatch, pai0[0]&0xffff, pai0[1]&0xffff, pai0[2]&0xffff, pai0[3]&0xffff );
		}
		{
		clock_t start_clock = clock();
		clock_t clock_offset = 0;
		clock_t clock_count = NSEC/10;
		while (clock_offset < clock_count) {
			clock_offset = clock() - start_clock;
	    	}
		}
      }
      for (int ic = 0; ic < 32; ++ic){
	pao0[ic] = *pai0;
      }
#if 0      
      if (ii%10000 == 0){
		printf("Cycle: %10d tl:%10u tl0 %10u pollcat:%d\n", ii, tl, tl0, pollcat);
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
