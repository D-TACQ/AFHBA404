#include "afhba-llcontrol-gpucopy.h"

// hello

__global__ void llcontrol_gpu_dummy(void * volatile ai_buffer_ptr,
                              unsigned * volatile ao_buffer_ptr,
                              short * total_data,
                              int nCycles){
  unsigned * tlatch = &((unsigned*)ai_buffer_ptr)[NCHAN/2];
  unsigned * pai0 = (unsigned*)ai_buffer_ptr;

  printf("Starting data loop now! %d cycles\n", nCycles);

  unsigned tl0 = *tlatch;
  unsigned tl;

  unsigned ai0 = *pai0;
  unsigned ai;
  for (int ii = 0; ii < 5; ii++) {
      printf("ai[%d] %p 0x%08x\n", ii, pai0+ii, pai0[ii]);
  }

  for (int ii = 0; ii < nCycles; ii++) {
      int pollcat;
      for (pollcat = 0; (tl = *tlatch) == tl0 && (ai = *pai0) == ai0; ++pollcat){
          ;
      }
      printf("Cycle: %d tl:%u tl0 %u pollcat:%d\n", ii, tl, tl0, pollcat);
      tl0 = tl;
      ai0 = ai;
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
