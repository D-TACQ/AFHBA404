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
LDLIBS += -lm -lrt -lpopt -lpthread

# default build is the local kernel.
# build other kernels like this example:
# make KRNL=2.6.20-1.2948.fc6-i686 ARCH=i386
# make KRNL=2.6.18-194.26.1.el5 ARCH=i386

all: apps


APPS := afhba-llcontrol-example afhba-llcontrol-zcopy \
	afhba-llcontrol-testharness check-tlatch afhba-llcontrol-cpucopy \
	afhba-llcontrol-do32-threshold \
	afhba-bufferAB-do32-threshold \
	afhba-llcontrol-two-fiber \
	afhba-bufferAB-480-pwm pwm_set_channel pwm_set_duty pwm_control \
	afhba-llcontrol-abn
	
apps: $(APPS)

	
afhba-llcontrol-example: afhba-llcontrol-example.c
	cc -o afhba-llcontrol-example afhba-llcontrol-example.c -lpopt	-lpthread

afhba-bufferAB-480-pwm: afhba-bufferAB-480-pwm.o pwm_internals.o afhba-get_shared_mapping.o shm.o
	$(CC) -o $@ $^ $(LDLIBS)

#pwm_set_channel: pwm_set_channel.o afhba-get_shared_mapping.o pwm_internals.o
#	cc -o pwm_set_channel pwm_set_channel.o afhba-get_shared_mapping.o pwm_internals.o -lrt -lpopt	-lpthread

pwm_set_channel: pwm_set_channel.o afhba-get_shared_mapping.o pwm_internals.o
	$(CC) -o $@ $^ $(LDLIBS)

pwm_set_duty: pwm_set_duty.o afhba-get_shared_mapping.o pwm_internals.o
	$(CC) -o $@ $^ $(LDLIBS)

pwm_control: pwm_control.o afhba-get_shared_mapping.o pwm_internals.o
	$(CC) -o $@ $^ $(LDLIBS)


afhba-llcontrol-testharness: afhba-llcontrol-testharness.cpp
	$(CXX) -o afhba-llcontrol-testharness afhba-llcontrol-testharness.cpp -lpopt	



clean:
	rm -f *.mod* *.o *.ko modules.order Module.symvers $(APPS)


	
