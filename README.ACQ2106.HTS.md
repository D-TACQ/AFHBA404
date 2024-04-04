README.ACQ2106.HTS

[Current HOWTO guide, with automated scripting:](https://github.com/D-TACQ/AFHBA404/releases/download/v2.7/hts_streaming_r2.pdf)

High Throughput Streaming

Use Case:

ACQ2106+4xACQ425-16-2000 

Initial:
========
./scripts/load
dmesg -c  ** see bootlog at end
There's automated AMON (Aurora MONitor) polling to handle link up, link down.

Examples stream to a ramdisk. To set up the ramdisk:
./scripts/mount-ramdisk



Running a shot:
===============

The data rate is too high for a single AFHBA, use two cables, two AFHBA

./scripts/stream-to-ramdisk
./scripts/stream-to-ramdisk 1
SIMULATE=0 ./scripts/hts-test-harness-AI4

With data validation:
./scripts/stream-checkramp
./scripts/stream-checkramp 1
SIMULATE=1 ./scripts/hts-test-harness-AI4

Reduced payload - eg with 1 AI
set.site 1 data32=1

HBFARM=1 ./scripts/stream-checkramp
HBFARM=1 ./scripts/stream-checkramp 1
INTCLKDIV=50 SIMULATE=1 ./scripts/hts-test-harness-AI1

=> full rate from one module. 

Checklist: 

1. Is AFHBA Present: eg

[root@tatooine AFHBA]# lspci -v | grep Xilinx -A 10
01:00.0 RAM memory: Xilinx Corporation Device adc1
	Subsystem: Device d1ac:4102
	Flags: bus master, fast devsel, latency 0, IRQ 50
	Memory at f7e10000 (32-bit, non-prefetchable) [size=4K]
	Memory at f7e00000 (32-bit, non-prefetchable) [size=64K]
	Capabilities: [40] Power Management version 3
	Capabilities: [48] MSI: Enable+ Count=4/16 Maskable- 64bit+
	Capabilities: [58] Express Endpoint, MSI 00
	Capabilities: [100] Device Serial Number 68-4a-86-56-99-a3-68-0d
	Kernel driver in use: afhba

02:00.0 RAM memory: Xilinx Corporation Device adc1
	Subsystem: Device d1ac:4102
	Flags: bus master, fast devsel, latency 0, IRQ 54
	Memory at f7d10000 (32-bit, non-prefetchable) [size=4K]
	Memory at f7d00000 (32-bit, non-prefetchable) [size=64K]
	Capabilities: [40] Power Management version 3
	Capabilities: [48] MSI: Enable+ Count=4/16 Maskable- 64bit+
	Capabilities: [58] Express Endpoint, MSI 00
	Capabilities: [100] Device Serial Number 6c-68-aa-20-f9-55-08-0d
	Kernel driver in use: afhba

Device Serial Number 6c-68-aa-20-f9-55-08-0d:
Unique ID : 6c-68-aa-20-f9-55-08
AFHBA firmware revision: 0d


2. After loading the device driver, but before running a shot

Check the bootlog using dmesg

Example Bootlog with 2 x AFHBA in one chassis
=============================================
afhba D-TACQ ACQ-FIBER-HBA Driver for ACQ400 B1026 Jun 22 2015
Copyright (c) 2010/2014 D-TACQ Solutions Ltd
afhba 0000:01:00.0: AFHBA: subdevice : 4102
afhba 0000:01:00.0: hb1 [0] ffff880037b00000 37b00000 1048576 00000000
afhba 0000:01:00.0: FPGA revision: ac01000d
afhba 0000:01:00.0: afhba_stream_drv_init(1003)
afhba 0000:01:00.0: irq 50 for MSI/MSI-X
afhba 0000:01:00.0: irq 51 for MSI/MSI-X
afhba 0000:01:00.0: irq 52 for MSI/MSI-X
afhba 0000:01:00.0: irq 53 for MSI/MSI-X
afhba 0000:01:00.0: request_irq afhba.0-line 51 OK
afhba 0000:01:00.0: request_irq afhba.0-ppnf 52 OK
afhba 0000:01:00.0: request_irq afhba.0-spare 53 OK
afhba 0000:02:00.0: AFHBA: subdevice : 4102
afhba 0000:01:00.0: aurora link up!
afhba 0000:02:00.0: hb1 [0] ffff8800d6000000 d6000000 1048576 00000000
afhba 0000:01:00.0: [0] Z_IDENT 1:0x21a60010 2:0x21a60010 acq2106_016.commsA
afhba 0000:01:00.0: aurora initial s:0x00002267 m:0x60000070 e:0x00000060
afhba 0000:02:00.0: FPGA revision: ac01000d
afhba 0000:02:00.0: afhba_stream_drv_init(1003)
afhba 0000:02:00.0: irq 54 for MSI/MSI-X
afhba 0000:02:00.0: irq 55 for MSI/MSI-X
afhba 0000:02:00.0: irq 56 for MSI/MSI-X
afhba 0000:02:00.0: irq 57 for MSI/MSI-X
afhba 0000:02:00.0: request_irq afhba.1-line 55 OK
afhba 0000:02:00.0: request_irq afhba.1-ppnf 56 OK
afhba 0000:02:00.0: request_irq afhba.1-spare 57 OK
afhba 0000:02:00.0: aurora link up!
afhba 0000:02:00.0: [1] Z_IDENT 1:0x21b60010 2:0x21b60010 acq2106_016.commsB
afhba 0000:02:00.0: aurora initial s:0x00002b67 m:0x60000070 e:0x00000060

Did the ramdisk mount work?
===========================
[root@tatooine AFHBA]# mount
..
dram on /mnt type ramfs (rw)



3. Check interrupt assignments worked:

Example Interrupt listing:
=========================
[root@endor ~]# grep afhba /proc/interrupts  | cut -c-20,70-
  98:       9278     0  IR-PCI-MSI-edge      afhba.0-dma
  99:          0     0  IR-PCI-MSI-edge      afhba.0-line
 100:          0     0  IR-PCI-MSI-edge      afhba.0-ppnf
 101:          0     0  IR-PCI-MSI-edge      afhba.0-spare
 102:       9278     0  IR-PCI-MSI-edge      afhba.1-dma
 103:          0     0  IR-PCI-MSI-edge      afhba.1-line
 104:          0     0  IR-PCI-MSI-edge      afhba.1-ppnf
 105:          0     0  IR-PCI-MSI-edge      afhba.1-spare


Example Link Fail and restart
=============================

afhba 0000:0c:00.0: aurora link down!
afhba 0000:0c:00.0: job is go but aurora is down
afhba 0000:0c:00.0: afs_stop_stream_push()
afhba 0000:06:00.0: afs_stop_stream_push()
afhba 0000:0c:00.0: aurora link up!
afhba 0000:0c:00.0: [0] Z_IDENT 1:0xdeadc0de 2:0x21b60007 acq2106_007.commsB

stream-to-ramdisk processes will time out. 
The controller ./scripts/hts-test-harness-AI4 can be <ctrl-C> aborted and re-run without problem.


Example Stream Start
====================

[root@tatooine AFHBA]# ./scripts/stream-to-ramdisk
checking aurora is up ..
0x00002907 +PRESENT +CHANNEL_UP +LANE_UP 
stream to /mnt/afhba.0
init_defaults using 64 buffers



FAQ:
===

1. Runs for a few seconds and FAILS?
In our experience, if we forget to mount the ramdisk, data is streamed to the local disk instead. The high data rate overwhelms the disk and the computer can lock up completely.

You may see a message like "RT Throttling set" .. that's a sure sign that the disk system is working too hard..

