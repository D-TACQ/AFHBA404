obj-m += afhba.o
obj-m += afhbaspi.o
obj-m += afhbasfp.o

SRC := $(shell pwd)


EXTRA_CFLAGS += -DCONFIG_SPI

# default build is the local kernel.
# build other kernels like this example:
# make KRNL=2.6.20-1.2948.fc6-i686 ARCH=i386
# make KRNL=2.6.18-194.26.1.el5 ARCH=i386

all: modules apps spi_support

KRNL ?= $(shell uname -r)
# FEDORA:
# KHEADERS := /lib/modules/$(KRNL)/build
# SCI LIN:
KHEADERS := /usr/src/kernels/$(KRNL)/

afhba-objs = acq-fiber-hba.o \
	afhba_devman.o afhba_debugfs.o afhba_stream_drv.o afhba_sysfs.o \
	afhba_core.o  \
	afs_procfs.o 

afhbaspi-objs = afhba_spi.o afhba_core.o

afhbasfp-objs =  afhba_i2c_bus.o afhba_core.o
	
modules: 
	make -C $(KHEADERS) M=$(SRC)  modules


APPS := mmap
apps: $(APPS)

mmap:
	cc -o mmap mmap.c -lpopt

spi_support:
	make -C $(KHEADERS) M=$(KHEADERS)/drivers/spi obj-m="spi.o spi_bitbang.o" modules
	make -C $(KHEADERS) M=$(SRC)/mtd/devices obj-m=m25p80.o modules
	
clean:
	rm -f *.mod* *.o *.ko modules.order Module.symvers $(APPS) .*.o.cmd

DC := $(shell date +%y%m%d%H%M)
package: clean
	git tag $(DC)
	(cd ..;tar cvf AFHBA/release/afhba-$(DC).tar \
		--exclude=release --exclude=SAFE AFHBA/* )
	
