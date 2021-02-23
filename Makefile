obj-m := afhba.o

SRC := $(shell pwd)

CC = $(CROSS_COMPILE)gcc
#CC = g++
CPP = g++

CFLAGS += -g -I$(CUDADIR)/include
#CFLAGS += -O3 -I$(CUDADIR)/include
LDLIB += -lm -lrt -lpopt -lpthread -lcuda -L$(CUDADIR)/lib64 -lcudart
# ONLY BUILD FOR COMPUTE CAPABILITY 6.1
NVCCFLAGS += -gencode=arch=compute_61,code=sm_61
NVCCFLAGS += -gencode=arch=compute_61,code=compute_61
NVCCFLAGS += -rdc=true


all: apps
#EXTRA_CFLAGS += -I/usr/src/nvidia-418.39/nvidia/
EXTRA_CFLAGS += -I/usr/src/nvidia-460.32.03/nvidia/
NVIDIA_SRC_DIR += -I/usr/src/nvidia-460.32.03/nvidia/
ccflags-y += $(EXTRA_CFLAGS)
KBUILD_EXTRA_SYMBOLS := /home/dt100/NVIDIA-Linux-x86_64-460.32.03/kernel/Module.symvers
# default build is the local kernel.
# build other kernels like this example:
# make KRNL=2.6.20-1.2948.fc6-i686 ARCH=i386
# make KRNL=2.6.18-194.26.1.el5 ARCH=i386

afhba-objs = acq-fiber-hba.o \
	afhba_devman.o afhba_debugfs.o afhba_stream_drv.o afhba_sysfs.o \
	afhba_core.o  \
	afs_procfs.o

afhbaspi-objs = afhba_spi.o afhba_core.o

afhbasfp-objs =  afhba_i2c_bus.o afhba_core.o

modules:
	@ echo "Picking NVIDIA driver sources from NVIDIA_SRC_DIR=$(NVIDIA_SRC_DIR). If that does not meet your expectation, you might have a stale driver still around and that might cause problems."
	make -C $(KHEADERS) M=$(SRC)  modules

modules-dbg:
	make -C $(KHEADERS) M=$(SRC)  modules EXTRA_CFLAGS="-g"

.PHONY: gdb-cmd
gdb-cmd:
	@echo "add-symbol-file afhba.ko $$(sudo cat /sys/module/afhba/sections/.text)"

APPS := mmap xiloader
apps: $(APPS) stream functional_tests llc_support acqproc


flasherase:
	cd mtd-utils && $(MAKE)
	cp mtd-utils/flash_erase .

mmap:
	cc -o mmap mmap.c -lpopt

xiloader:
	cc -o xiloader xiloader.c -lpopt

llc_support:
	cd LLCONTROL && $(MAKE)

acqproc:
	cd ACQPROC && $(MAKE)

acqproc_clean:
	cd ACQPROC && $(MAKE) clean

stream:
	cd STREAM && $(MAKE)

functional_tests:
	cd FUNCTIONAL_TESTS && $(MAKE)

spi_support: flasherase
	make -C $(KHEADERS) M=$(LDRV)/spi  obj-m="spi-bitbang.o" modules
	make -C $(KHEADERS) M=$(LDRV)/mtd  obj-m="mtd.o" modules
	make -C $(KHEADERS) M=$(LDRV)/mtd/devices obj-m=m25p80.o modules
	cp ./linux/drivers/mtd/devices/m25p80.ko .
	cp ./linux/drivers/spi/spi-bitbang.ko .
	cp ./linux/drivers/mtd/mtd.ko .

spi_clean:
	make -C $(KHEADERS) M=$(LDRV)/spi clean
	make -C $(KHEADERS) M=$(LDRV)/mtd clean
	make -C $(KHEADERS) M=$(LDRV)/mtd/devices clean
	cd mtd-utils && $(MAKE) clean

llc_clean:
	cd LLCONTROL && $(MAKE) clean

stream_clean:
	cd STREAM && $(MAKE) clean

functional_tests_clean:
	cd FUNCTIONAL_TESTS && $(MAKE) clean

clean: llc_clean stream_clean functional_tests_clean acqproc_clean
	rm -f *.mod* *.o *.ko modules.order Module.symvers $(APPS) .*.o.cmd

DC := $(shell date +%y%m%d%H%M)
package: clean spi_clean
	git tag $(DC)
	(cd ..;tar cvzf AFHBA/release/afhba-$(DC).tgz \
		--exclude=release --exclude=SAFE AFHBA/* )

