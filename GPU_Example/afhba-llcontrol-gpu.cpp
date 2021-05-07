/* ------------------------------------------------------------------------- 
 * afhba-llcontrol-gpu.cpp
 * simple llcontrol example, ONE HBA, bufferA bufferB, GPU copy (realistic).
 * control for custom DIO482 pwm system
 * Based on afhba-bufferAB-480-pwm.c by Peter Milne
 * and gpudma by Vladimir Karakozov [ https://github.com/karakozov/gpudma ]
 * -------------------------------------------------------------------------             
*/ 

#include "afhba-llcontrol-gpu.h"

struct XLLC_DEF xllc_def_ai;
struct XLLC_DEF xllc_def_ao;
struct gpudma_lock_t lock;

int fd;
int devnum = 0;
int samples_buffer;

// Pointers for digitizer log
int nsamples;
short * tdata_cpu;
short * tdata_gpu;
int tdata_size;

int cudaTest() { // Simple test to make sure CUDA works
  int n = 1000;
  float eps = 1e-10;
  int size = n*sizeof(float);
  float *gpu_A, *gpu_B, *gpu_C;
  float *A = (float *)malloc(size);
  float *B = (float *)malloc(size);
  float *C = (float *)malloc(size);
  float *Ctest = (float *)malloc(size);

  for (int ix = 0; ix < n; ix++){
      A[ix] = (float)rand();
      B[ix] = (float)rand();
      Ctest[ix] = A[ix] + B[ix];
  }

  cudaMalloc((void **) &gpu_A, size);
  cudaMemcpy(gpu_A, A, size, cudaMemcpyHostToDevice);
  cudaMalloc((void **) &gpu_B, size);
  cudaMemcpy(gpu_B, B, size, cudaMemcpyHostToDevice);
  cudaMalloc((void **) &gpu_C, size);

  vecAdd_wrapper(gpu_A, gpu_B, gpu_C, n);
  
  cudaMemcpy(C,gpu_C,size,cudaMemcpyDeviceToHost);
#if 0
  for (int ix = 0; ix < n; ix++){
    if (abs(C[ix] - Ctest[ix])>eps){
      return 0;
    }
  }
 #endif

  cudaFree(gpu_A); cudaFree(gpu_B); cudaFree(gpu_C);
  return 1;

}

// CUDA Error Evaluation
void checkError(CUresult status)
{
  if (status != CUDA_SUCCESS) {
    const char *perrstr = 0;
    CUresult ok = cuGetErrorString(status,&perrstr);
    if(ok == CUDA_SUCCESS) {
        if(perrstr) {
            fprintf(stderr, "info: %s\n", perrstr);
        } else {
            fprintf(stderr, "info: unknown error\n");
        }
    }
    exit(0);
  }
}

bool wasError(CUresult status)
{
    if(status != CUDA_SUCCESS) {
        const char *perrstr = 0;
        CUresult ok = cuGetErrorString(status,&perrstr);
        if(ok == CUDA_SUCCESS) {
            if(perrstr) {
                fprintf(stderr, "info: %s\n", perrstr);
            } else {
                fprintf(stderr, "info: unknown error\n");
            }
        }
        return true;
    }
    return false;
}

void prepare_gpu() { // Allocates memory for CPU-GPU communication
  nsamples = NSAMP;
  // tdata is the total log of digitizer data, initialize as zeros
  tdata_size = NSHORTS*(nsamples+1)*sizeof(short);
  tdata_cpu = (short *)malloc(tdata_size);
  memset(tdata_cpu,0x00,tdata_size);
  cudaMalloc((void **) &tdata_gpu, tdata_size);
  cudaMemcpy(tdata_gpu,tdata_cpu,tdata_size,cudaMemcpyHostToDevice);
}

void output_gpu_data(){
  // Spits tdata out as a file
  cudaMemcpy(tdata_cpu, tdata_gpu, tdata_size, cudaMemcpyDeviceToHost);
  FILE * fp_out = fopen("acq2106_gpu.log","w");
  fwrite(tdata_cpu,sizeof(short),NSHORTS*nsamples,fp_out);
  fclose(fp_out);
}

int get_mapping_gpu(){ // Allocates memory for AFHBA404 datastream

  int res = -1;
  unsigned int flag = 1;
  CUresult status;
  size_t size_ai;
  size_t size_ao;
  CUcontext  context;

  CUdeviceptr dptr_ai = 0;
  CUdeviceptr dptr_ao = 0;

  //Open the AFHBA404 device:
  char fname[80];
  sprintf(fname, HB_FILE, devnum);

  fd = open(fname, O_RDWR);
  if (fd<0){
    perror(fname);
    exit(errno);
  }

  //Get and print information about the CUDA card being used:
  checkError(cuInit(0));

  int  total = 0;
  checkError(cuDeviceGetCount(&total));
  fprintf(stderr,"Total CUDA devices: %d\n",total);

  CUdevice device;
  checkError(cuDeviceGet(&device,0));

  char name[256];
  checkError(cuDeviceGetName(name,256,device));
  fprintf(stderr,"Device 0 taken: %s\n",name);

  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device);
  cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device);
  fprintf(stderr,"Compute capability: %d.%d\n",major,minor);

  int clockRate = 0;
  cuDeviceGetAttribute(&clockRate,CU_DEVICE_ATTRIBUTE_CLOCK_RATE,device);
  fprintf(stderr,"Clock Rate (kHz) is: %d\n",clockRate);

  size_t global_mem = 0;
  checkError( cuDeviceTotalMem(&global_mem, device));
  fprintf(stderr, "Global memory: %llu MB\n", (unsigned long long)(global_mem >> 20));
  if(global_mem > (unsigned long long)4*1024*1024*1024L)
        fprintf(stderr, "64-bit Memory Address support\n");

  //Now get into the meat of things:

  checkError(cuCtxCreate(&context,0,device));
  size_ai = HB_LEN;
  size_ao = HB_LEN; // This is 'over-allocating', but we have plenty of memory

  status = cuMemAlloc(&dptr_ai,size_ai);
  if (wasError(status)){
    return 1;
  }
  status = cuMemAlloc(&dptr_ao,size_ao);
  if (wasError(status)){
    return 1;
  }
  fprintf(stderr,"Allocate AI memory address: 0x%llx\n", (unsigned long long) dptr_ai);
  fprintf(stderr,"Allocate AO memory address: 0x%llx\n", (unsigned long long) dptr_ao);

  status = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dptr_ai);
  if (wasError(status)){
    cuMemFree(dptr_ai);
    return 1;
  }
  status = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dptr_ao);
  if (wasError(status)){
    cuMemFree(dptr_ao);
    return 1;
  }

  lock.addr_ai = dptr_ai;
  lock.size_ai = size_ai;
  lock.ind_ai = 0;

  lock.addr_ao = dptr_ao;
  lock.size_ao = size_ao;
  lock.ind_ao = 1;

  res = ioctl(fd, AFHBA_GPUMEM_LOCK, &lock);
  if (res<0){
    fprintf(stderr,"Error in AFHBA_GPUMEM_LOCK.\n");
    goto do_free_attr;
  }

  return 0;

  do_free_attr:
      flag = 0;
      cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,dptr_ai);
      cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,dptr_ao);

      cuMemFree(dptr_ai);
      cuMemFree(dptr_ao);

  do_free_context:
    cuCtxDestroy(context);

    close(fd);
    return 1;

}


int setup() {

  if(get_mapping_gpu()) {
      printf("Error in get_mapping_gpu, exiting.\n");
      return 1;
  }

  get_shared_mapping(devnum, 1, &xllc_def_ao, (void**)&pbufferXO);
  
  samples_buffer = HTS_MIN_BUFFER/NCHAN/2;

  struct AB ab_def;
  printf("BUFFER_AB_OFFSET=%x, VI_LEN=%lx.\n",BUFFER_AB_OFFSET,VI_LEN);

  ab_def.buffers[0].pa = xllc_def_ai.pa;
  ab_def.buffers[1].pa = BUFFER_AB_OFFSET;
  ab_def.buffers[0].len = 
  ab_def.buffers[1].len = VI_LEN;

  if (ioctl(fd, AFHBA_START_AI_AB, &ab_def)){
      perror("ioctl AFHBA_START_AI_AB");
      exit(1);
  }

  printf("AI buf pa: %c 0x%08x len %d\n", 'A',
			ab_def.buffers[0].pa, ab_def.buffers[0].len);
  printf("AI buf pa: %c 0x%08x len %d\n", 'B',
			ab_def.buffers[1].pa, ab_def.buffers[1].len);

  xllc_def_ao.len = VO_LEN;

  if (ioctl(fd, AFHBA_START_AO_LLC, &xllc_def_ao)){
		perror("ioctl AFHBA_START_AO_LLC");
		exit(1);
	}
  printf("AO buf pa:   0x%08x len %d\n", xllc_def_ao.pa, xllc_def_ao.len);
  return 0;

}

void run_llcontrol() {
  prepare_gpu();

  llcontrol_gpu_example((void *)lock.addr_ai, (unsigned *)lock.addr_ao, tdata_gpu, nsamples);

  sleep(3);
  printf("If kernel reports working, then GPU is ready to go.\n");

  cudaDeviceSynchronize(); // Wait for kernel to finish

  output_gpu_data(); 
}

int main() {

  // Run a simple CUDA kernel to verify GPU is working.
  printf("Testing CUDA...");
  if (cudaTest()){
      printf("CUDA worked!\n");
  } else {
      printf("CUDA failed :(, terminating program.\n");
      return 0;
  }
  
  if (setup()) {
      printf("Failure in setup(), ending program.\n");
  }

  run_llcontrol();

  // Terminate
  return 0;
}
