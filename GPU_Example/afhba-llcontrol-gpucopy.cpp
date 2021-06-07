/*
 * afhba-llcontrol-gpucopy.c
 *
 *  Created on: 19 Feb 2021
 *      Author: Sean Alsop
 *
 *  A simple gpu example for a standard acq2106 LLC system.
 */

#include "afhba-llcontrol-gpucopy.h"
#include "afhba-llcontrol-common.h"

struct XLLC_DEF xllc_def_ai;
struct XLLC_DEF xllc_def_ao;

struct gpudma_lock_t lock;

int fd;
int devnum = 0;
int samples_buffer = 1;
int nsamples = 10000000;		/* 10s at 1MSPS */
float GAIN = 1.0;
unsigned MX;
enum MX { MX_EMPTY, MX_DIAGONAL, MX_COL0, MX_FULL0, MX_FULL1 } MX_BLOCKS = MX_DIAGONAL;
short * tdata_cpu;
short * tdata_gpu;
float* AMX_host;
float* AMX_gpu;
int tdata_size;
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
	exit(1);
	return 1;

}





int setup() {
	if (get_mapping_gpu()) {
		printf("Error in get_mapping_gpu, exiting.\n");
		return 1;
	}

	get_shared_mapping(devnum, 0, &xllc_def_ai, 0);
	get_shared_mapping(devnum, 1, &xllc_def_ao, 0);
	xllc_def_ai.len = samples_buffer * VI_LEN;

	if (ioctl(fd, AFHBA_START_AO_LLC, &xllc_def_ao)){
		perror("ioctl AFHBA_START_AO_LLC");
		exit(1);
	}

	if (ioctl(fd, AFHBA_START_AI_LLC, &xllc_def_ai)){
		perror("ioctl AFHBA_START_AI_LLC");
		exit(1);
	}



	return 0;

}


void prepare_gpu() { // Allocates memory for CPU-GPU communication
	// tdata is the total log of digitizer data, initialize as zeros
#if 0
	tdata_size = NSHORTS*(nsamples+1)*sizeof(short);
	tdata_cpu = (short *)malloc(tdata_size);
	memset(tdata_cpu,0x00,tdata_size);
	cudaMalloc((void **) &tdata_gpu, tdata_size);
	cudaMemcpy(tdata_gpu,tdata_cpu,tdata_size,cudaMemcpyHostToDevice);
#endif
	AMX_host = (float*)calloc(AI_CHAN*AO_CHAN, sizeof(float));

	switch(MX){
	case MX_DIAGONAL:
		printf("load matrix MX_DIAGONAL %.2f\n", GAIN);
		for (int ao = 0; ao < AO_CHAN; ++ao){
			AMX_host[ao*AI_CHAN + ao] = GAIN;
		}
		break;
	case MX_FULL0:
		printf("load matrix MX_FULL_AO %.2f\n", GAIN);
		for (int ao = 0; ao < AO_CHAN; ++ao){
			for (int ai = 0; ai < AO_CHAN; ++ai){
				AMX_host[ao*AI_CHAN + ai] = GAIN;
			}
		}
		break;
	case MX_FULL1:
		printf("load matrix MX_FULL %.2f\n", GAIN);
		for (int ao = 0; ao < AO_CHAN; ++ao){
			for (int ai = 0; ai < AI_CHAN; ++ai){
				AMX_host[ao*AI_CHAN + ai] = GAIN;
			}
		}
		break;
	case MX_EMPTY:
	default:
		break;
	}
	// todo .. make it global
	cudaMalloc((void **) &AMX_gpu, AI_CHAN*AO_CHAN*sizeof(float));
	cudaMemcpy(AMX_gpu, AMX_host, AI_CHAN*AO_CHAN*sizeof(float), cudaMemcpyHostToDevice);
}


int run_llcontrol_gpu(){
	prepare_gpu();
	llcontrol_gpu_A_matrix_wrapper((void *)lock.addr_ai, (unsigned *)lock.addr_ao, tdata_gpu, AMX_gpu, nsamples);
	unsigned int microseconds = 1000;
	usleep(microseconds);
	cudaDeviceSynchronize(); // Wait for kernel to finish
	return 0;
}


int closedown(){
	return 0;
}

int PROFILE = 0;
int REDUCE_ALGO = 0;		// REDuce Columns

void ui(int argc, char *argv[])
{
	if (argc > 1){
		nsamples = atoi(argv[1]);
	}
	if (const char* val = getenv("GAIN")){
		GAIN = atof(val);
	}
	/* 0: empty 1: diagnonal, 2: full house AO_CHAN, 3: full house */
	if (const char* val = getenv("MX")){
		MX = strtoul(val, 0, 16);
	}
	if (const char* val = getenv("PROFILE")){
		PROFILE = atoi(val);
	}
	if (const char* val = getenv("REDUCE_ALGO")){
		REDUCE_ALGO = atoi(val);
	}
}

int main(int argc, char *argv[]) {
	printf("Starting now...\n");
	ui(argc, argv);

	printf("Starting setup now.\n");
	setup();

	printf("ready for data\n");
	run_llcontrol_gpu();

	printf("finished\n");
	closedown();
}
