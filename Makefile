obj-m := afhba.o

SRC := $(shell pwd)

CC = $(CROSS_COMPILE)gcc
#CC = g++
CPP = g++
NVCC = nvcc

CFLAGS += -g -I$(CUDADIR)/include
#CFLAGS += -O3 -I$(CUDADIR)/include
LDLIB += -lm -lrt -lpopt -lpthread -lcuda -L$(CUDADIR)/lib64 -lcudart
# ONLY BUILD FOR COMPUTE CAPABILITY 6.1
NVCCFLAGS += -gencode=arch=compute_61,code=sm_61
NVCCFLAGS += -gencode=arch=compute_61,code=compute_61
NVCCFLAGS += -rdc=true

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) -c $<

%.o : %.cpp
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) -c $<

all: apps

APPS := afhba-llcontrol-gpu afhba-llcontrol-gpucopy 

apps: $(APPS)

afhba-llcontrol-gpu: afhba-llcontrol-gpu.o afhba-get_shared_mapping.o pwm_internals.o afhba-llcontrol-gpu-kernels.o
	nvcc -arch=sm_61 -o $@ $^ $(LDLIB) 

afhba-llcontrol-gpucopy: afhba-llcontrol-gpucopy.o afhba-get_shared_mapping.o pwm_internals.o afhba-llcontrol-gpu-kernels.o
	nvcc -arch=sm_61 -o $@ $^ $(LDLIB) 


clean:
	rm -f *.mod* *.o *.ko modules.order Module.symvers $(APPS)
