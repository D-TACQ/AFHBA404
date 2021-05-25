# Makefile for afhba404
# supports both host llc (default, most users) and gpu llc (special!)
# to build everything for host: make
# to clean everything for host: make clean
#
# to build everything for gpu: GPU=1 make
# to clean everything for gpu: GPU=1 make clean
#

obj-m += afhba.o
#obj-m += afhbaspi.o
#obj-m += afhbasfp.o

SRC := $(shell pwd)
LDRV:= $(SRC)/linux/drivers
EXTRA_CFLAGS += -I/usr/src/nvidia-460.32.03/nvidia/
KBUILD_EXTRA_SYMBOLS := /home/dt100/NVIDIA-Linux-x86_64-460.32.03/kernel/Module.symvers
CONFIG_MODULE_SIG=n

EXTRA_CFLAGS += -DCONFIG_SPI


ifeq ($(GPU),1)
# enable next two lines for GPU
EXTRA_CFLAGS += -DCONFIG_GPU
EXTRA_GPU = afhba_gpu.o
GPU_APPS = gpu_apps
GPU_APPS_CLEAN = gpu_apps_clean
endif


# default build is the local kernel.
# build other kernels like this example:
# make KRNL=2.6.20-1.2948.fc6-i686 ARCH=i386
# make KRNL=2.6.18-194.26.1.el5 ARCH=i386

all: modules apps
#llc_support

flash: spi_support

flash_clean: spi_clean llc_clean

KRNL ?= $(shell uname -r)
# FEDORA:
KHEADERS := /lib/modules/$(KRNL)/build


afhba-objs = acq-fiber-hba.o \
	afhba_devman.o afhba_debugfs.o afhba_stream_drv.o afhba_sysfs.o \
	afhba_core.o  $(EXTRA_GPU) \
	afs_procfs.o

afhbaspi-objs = afhba_spi.o afhba_core.o

afhbasfp-objs =  afhba_i2c_bus.o afhba_core.o

modules:
	make -C $(KHEADERS) M=$(SRC)  modules

modules-dbg:
	make -C $(KHEADERS) M=$(SRC)  modules EXTRA_CFLAGS="-g"

.PHONY: gdb-cmd
gdb-cmd:
	@echo "add-symbol-file afhba.ko $$(sudo cat /sys/module/afhba/sections/.text)"

APPS := mmap xiloader
apps: $(APPS) stream functional_tests llc_support acqproc $(GPU_APPS)


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

gpu_apps:
	cd GPU_Example && $(MAKE)

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

acqproc_clean:
	cd ACQPROC && $(MAKE) clean


functional_tests_clean:
	cd FUNCTIONAL_TESTS && $(MAKE) clean

gpu_apps_clean:
	cd GPU_Example && $(MAKE) clean

clean: llc_clean stream_clean functional_tests_clean acqproc_clean $(GPU_APPS_CLEAN)
	rm -f *.mod* *.o *.ko modules.order Module.symvers $(APPS) .*.o.cmd

DC := $(shell date +%y%m%d%H%M)
package: clean spi_clean
	git tag $(DC)
	(cd ..;tar cvzf AFHBA/release/afhba-$(DC).tgz \
		--exclude=release --exclude=SAFE AFHBA/* )

# example remote build
viper:
	rsync -va -t GPU_Example/ dt100@viper:PROJECTS/AFHBA404/GPU_Example
	ssh dt100@viper 'cd PROJECTS/AFHBA404/GPU_Example; make'


