obj-m := afhba.o

SRC := $(shell pwd)


ifeq ($(ACQ_FHBA_SPI),1)
SPI_MODS=acq-fiber-hba-spi.o
SPI_SUPPORT=spi_support
EXTRA_CFLAGS += -DSPI_SUPPORT
else
SPI_SUPPORT=
endif


# default build is the local kernel.
# build other kernels like this example:
# make KRNL=2.6.20-1.2948.fc6-i686 ARCH=i386
# make KRNL=2.6.18-194.26.1.el5 ARCH=i386

all: modules apps

KRNL ?= $(shell uname -r)
# FEDORA:
# KHEADERS := /lib/modules/$(KRNL)/build
# SCI LIN:
KHEADERS := /usr/src/kernels/$(KRNL)/

afhba-objs = acq-fiber-hba.o afhba_devman.o afhba_debugfs.o afhba_stream_drv.o

modules: 
	make -C $(KHEADERS) M=$(SRC)  modules


APPS := mmap
apps: $(APPS)

mmap:
	cc -o mmap mmap.c -lpopt
	
clean:
	rm -f *.mod* *.o *.ko modules.order Module.symvers $(APPS) .*.o.cmd

DC := $(shell date +%y%m%d%H%M)
package: clean
	git tag $(DC)
	(cd ..;tar cvf AFHBA/release/afhba-$(DC).tar \
		--exclude=release --exclude=SAFE AFHBA/* )
	
