/*
 * afhba-llcontrol-gpucopy.c
 *
 *  Created on: 19 Feb 2021
 *      Author: Sean Alsop
 *
 *  A simple gpu example for a standard acq2106 LLC system.
 */

// Don't include the gpu header file as there is PWM specific code in there.
#include "afhba-llcontrol-gpu.h"

#include "afhba-llcontrol-gpucopy.h"
#include "afhba-llcontrol-common.h"

struct XLLC_DEF xllc_def_ai;
struct XLLC_DEF xllc_def_ao;

struct gpudma_lock_t lock;

int fd;
int devnum = 0;
int samples_buffer = 1;
int nsamples = 10000000;		/* 10s at 1MSPS */
int verbose;
//int VI_LEN = 256; // TODO: Make this variable.


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

  printf("DEBUG 1\n");

  lock.addr_ai = dptr_ai;
  lock.size_ai = size_ai;
  lock.ind_ai = 0;

  lock.addr_ao = dptr_ao;
  lock.size_ao = size_ao;
  lock.ind_ao = 1;

  printf("DEBUG 2: Pre-ioctl\n");

	printf("%p \n", lock.addr_ai);
	printf("%p \n", lock.size_ai);
	printf("%p \n", lock.ind_ai);

	printf("%p \n", lock.addr_ao);
	printf("%p \n", lock.size_ao);
	printf("%p \n", lock.ind_ao);

  res = ioctl(fd, AFHBA_GPUMEM_LOCK, &lock);
  if (res<0){
    fprintf(stderr,"Error in AFHBA_GPUMEM_LOCK.\n");
    goto do_free_attr;
  }
  printf("DEBUG 3: Post-ioctl\n");
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
	printf("Inside setup\n");
	if (get_mapping_gpu()) {
		printf("Error in get_mapping_gpu, exiting.\n");
		return 1;
	}
	printf("Finished get_mapping_gpu");

	get_shared_mapping(devnum, 1, &xllc_def_ao, (void**)&pbufferXO);
	xllc_def_ai.len = samples_buffer * VI_LEN;


	printf("%p \n", lock.addr_ai);
	printf("%p \n", lock.size_ai);
	printf("%p \n", lock.ind_ai);

	printf("%p \n", lock.addr_ao);
	printf("%p \n", lock.size_ao);
	printf("%p \n", lock.ind_ao);

	return 0;

}


int run_llcontrol_gpu(){
	return 0;
}


int closedown(){
	return 0;
}


int main(int argc, char *argv[]) {
	printf("Starting now...\n");

	printf("Starting setup now.\n");
	setup();

	printf("ready for data\n");
	run_llcontrol_gpu();

	printf("finished\n");
	closedown();
}
