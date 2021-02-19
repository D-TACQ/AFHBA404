/**********************************************************
*   afhba-llcontrol-gpu-kernels.cu:
*      Contains:
*         CUDA Kernels (__global__)
*         CUDA __device__ functions
*         Wrappers to call these kernels from .cpp files
*
***********************************************************/

#include "afhba-llcontrol-gpu.h"

// vecAdd is a toy CUDA kernel to make sure CUDA is working properly
__global__ void vecAdd(float * a, float * b, float * c, int n) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i<n) c[i] = a[i] + b[i];
}

void vecAdd_wrapper(float * a, float * b, float * c, int n) {
  vecAdd<<<ceil(n/256.0),256>>>(a,b,c,n);
  cudaDeviceSynchronize();
  return;
}

// Device functions
__device__ int wait_sample(void * buffer_ptr) {
    // Watches data buffer for sample to appear
    // Best to run this in serial, with only one CUDA threads
    unsigned cnt = 0;
    long long start;
    long long end;
    start = clock64();
    if (EOB(buffer_ptr) != MARKER) {
      return MISSED_BUFFER;
    }
  
  while(EOB(buffer_ptr)==MARKER) {
    ++cnt;
    if (cnt > (unsigned)3 << 30) {
      end = clock64();
      printf("Timeout: %e seconds\n", (end-start)/1.531e9);
      return DTACQ_TIMEOUT;
    }
  }
  return SAMPLE_SUCCESS;
}

__device__ void sort_samples(unsigned * ai_buffer, unsigned * total_data, int cycle) {
  // Assumes number of time-points in buffer is equal to number of CUDA threads
  // Copies the buffer data to the global storage array
  // The variable type conversion is for speed, copying a single 32-bit int is
  //   faster than copying two 16-bit shorts, so the data is copied in 
  //   sets of two.
  for (int i = 0; i < NCHAN/2; i++) {
    total_data[cycle*NSHORTS/2 + i + (NCHAN/2)*threadIdx.x] = ai_buffer[(NCHAN/2)*threadIdx.x + i];
  }
  return;
}

__device__ struct PWM_CTRL set_pwm(unsigned gp, unsigned icnt, unsigned ocnt, unsigned iste) {
    struct PWM_CTRL pwm;
    if (gp > MAX_GP || icnt > MAX_IC || ocnt > MAX_OC || iste > MAX_IS) {
      gp = 0;
      icnt = 0;
      ocnt = 0;
      iste = 0;
      if (threadIdx.x == 0) printf("PWM Failed\n");
    }
    pwm.PWM_GP = gp;
    pwm.PWM_IC = icnt;
    pwm.PWM_OC = ocnt;
    pwm.PWM_IS = iste;
    return pwm;
  }
  
  __device__ static unsigned pwm2raw_gpu(struct PWM_CTRL pwm) {
    unsigned raw = 0;
    raw |= pwm.PWM_IS << SHL_IS;
    raw |= pwm.PWM_GP << SHL_GP;
    raw |= pwm.PWM_OC << SHL_OC;
    raw |= pwm.PWM_IC << SHL_IC;
    return raw;
  }

__global__ void llcontrol_gpu(void * volatile ai_buffer_ptr, 
                              unsigned * volatile ao_buffer_ptr,
                              short * total_data,
                              int nCycles){
  // Assumed to launch with number of threads equal to number of time-samples
  // that will show up in the AI_BUFFER.
  int proc_number = blockIdx.x*blockDim.x + threadIdx.x;
  bool proc0 = (proc_number==0);
  bool pwm_thread = (proc_number < 32); // does the thread have a PWM channel
  int res;
  int rt_status[4];  

  void * bufferAB[2];
  int ab = 0;
  bufferAB[0] = ai_buffer_ptr;
  bufferAB[1] = (void *)( (char *)ai_buffer_ptr + BUFFER_AB_OFFSET );
  if (proc0){
      EOB(bufferAB[0]) = MARKER;
      EOB(bufferAB[1]) = MARKER;
  }

  struct PWM_CTRL pwm;
  if (pwm_thread){ // Set 0's to ao_buffer for this demonstration.
      pwm = set_pwm(0,0,0,0);
      ao_buffer_ptr[proc_number] = pwm2raw_gpu(pwm);
  }

  if (proc0) printf("GPU entering loop, ready for data.\n");

  for (int i = 0; i < nCycles; i++) { // Simple loop, just stores digitizer data
    
    // Wait for next sample to come in, this is done in serial
    if (proc0){
        res = wait_sample(bufferAB[ab]);
        rt_status[res]++;
    }
    __syncthreads();

    // copy the data to the global storage array. this is done in parallel.
    // passed as unsigned pointer (not short) for speed reasons.
    sort_samples((unsigned *)bufferAB[ab], (unsigned *)total_data, i);

    if (proc0) EOB(bufferAB[ab]) = MARKER;

    ab = !ab;
    __syncthreads();

  }

  if (proc0) printf("Terminating GPU Kernel.\n");
  return;

}

void llcontrol_gpu_example(void * volatile ai_buffer_ptr, 
                           unsigned * volatile ao_buffer_ptr,
                           short * total_data,
                           int nCycles){
  //Wrapper to call the CUDA kernel
  llcontrol_gpu<<<1,128>>>(ai_buffer_ptr, ao_buffer_ptr, total_data, nCycles);
  return;
}