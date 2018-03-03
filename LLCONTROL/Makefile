obj-m := afhba.o

SRC := $(shell pwd)


ifeq ($(ACQ_FHBA_SPI),1)
SPI_MODS=acq-fiber-hba-spi.o
SPI_SUPPORT=spi_support
EXTRA_CFLAGS += -DSPI_SUPPORT
else
SPI_SUPPORT=
endif

CFLAGS += -O2
LDLIBS += -lm

# default build is the local kernel.
# build other kernels like this example:
# make KRNL=2.6.20-1.2948.fc6-i686 ARCH=i386
# make KRNL=2.6.18-194.26.1.el5 ARCH=i386

all: apps


APPS := afhba-llcontrol-example afhba-llcontrol-zcopy \
	afhba-llcontrol-testharness check-tlatch afhba-llcontrol-cpucopy \
	afhba-llcontrol-do32-threshold \
	afhba-llcontrol-two-fiber
	
apps: $(APPS)

	
afhba-llcontrol-example: afhba-llcontrol-example.c
	cc -o afhba-llcontrol-example afhba-llcontrol-example.c -lpopt	-lpthread

afhba-llcontrol-testharness: afhba-llcontrol-testharness.cpp
	$(CXX) -o afhba-llcontrol-testharness afhba-llcontrol-testharness.cpp -lpopt	

clean:
	rm -f *.mod* *.o *.ko modules.order Module.symvers $(APPS)


	
