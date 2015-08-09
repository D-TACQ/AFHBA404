README.ACQ2106

What is it?

Examples of Low Latency Control LLC using

ACQ2106 : 1U appliance with 2 x MGT fiber-optic links
AFHBA : Acq Fiber Host Bus Adapter with 1 x MGT fiber-optic link

Theory of operation:

ADC => Z7030 -> SFP -> AFHBA -> PCI-Express-1x -> HOST DRAM

DAC <= Z7030 <- SFP <- AFHBA <- PCI-Express-1x <- HOST DRAM


Hardware:

Box0  (D-TACQ tester, acq2106_003):
ACQ2106
ACQ425-16-1000 in Site 1
AO424-32      in Site 4


Box1 (SHH "Pizza" System) :
ACQ2106
ACQ425-16-1000 in site 1

Box2 (SHH "AO" System) :
ACQ2106
AO424-32 in site 4
DIO432 in site 5.


Getting started:

1. Linux-rt system
Build and load device driver.


The Tests

1. Basic Input Only.
scripts/llc-test-harness-intclk-commsA
LLCONTROL/afhba-llcontrol-example

2. Box0 Input Output Test
scripts/llc-test-harness-intclk-AI1-AO4
LLCONTROL/afhba-llcontrol-zcopy

=> AO uses same HB as AI. So AO picks AI input data automatically.
Useful to show best possible performance, free of software overhead.

scripts/llc-test-harness-intclk-AI1-AO4
LLCONTROL/afhba-llcontrol-cpucopy

Duplicates AI01 -> AO01..AO31 and AI-02 -> AO02..AO32

3. Box1 Box2 Input Output Test
./scripts/llc-test-harness-intclk-AI1-AO4  acq2106_003
./scripts/llc-test-harness-intclk-commsA   acq2106_004
./LLCONTROL/afhba-llcontrol-two-fiber


4. Alternate Target Test.
Box1 A -> HOST hoth
Box2 B -> HOST tattoine

hoth:
scripts/llc-test-harness-intclk-commsA
LLCONTROL/afhba-llcontrol-example

tatooine:
COMMS_ONLY=YES COMMS=B scripts/llc-test-harness-intclk-commsA
LLCONTROL/afhba-llcontrol-example

Also works for alternate target single box:

Box1 A -> HOST hoth afhba.0
Box2 B -> HOST hoth afhba.1

./scripts/llc-test-harness-intclk-commsA
./LLCONTROL/afhba-llcontrol-example

COMMS_ONLY=YES COMMS=B scripts/llc-test-harness-intclk-commsA
DEVNUM=1 ./LLCONTROL/afhba-llcontrol-example


5. Single box, AI+AO+DO Test
DO32=1 ./LLCONTROL/afhba-llcontrol-cpucopy
./scripts/llc-test-harness-intclk-AI1-AO4-DO5


6. Two box, AI, AO+DO Test

Use z_ident to identify which box is tied to which AFHBA device:

[dt100@tatooine AFHBA]$ cat /dev/rtm-t.0.ctrl/z_ident
0x21a60004
[dt100@tatooine AFHBA]$ cat /dev/rtm-t.1.ctrl/z_ident
0x21b60003


ie 
AFHBA device 0 is connected to acq2106_004 portA
AFHBA device 1 is connected to acq2106_003 portB

ACQ2106_004 has AI in site 1
ACQ2106_003 has AO in site 4, DO in site 5

DEV_AI=0 DEV_AO=1 DO32=1 ./LLCONTROL/afhba-llcontrol-two-fiber
INTCLKDIV=250 /scripts/llc-test-harness-intclk-AI1-AO4-DO5-twobox

=> works, but because the clocks are async, there's a lot of jitter.


7. Two box, AI, AO+DO test, External Clock
DEV_AI=0 DEV_AO=1 DO32=1 ./LLCONTROL/afhba-llcontrol-two-fiber
LLC_CLK=ext EXTCLKDIV=4 ./scripts/llc-test-harness-intclk-AI1-AO4-DO5-twobox

NB: assumes the SAME 1MHz SYSCLK is applied to the CLK input of each box.
IDEALLY, we'd have EXT TRIG as well:

DEV_AI=0 DEV_AO=1 DO32=1 ./LLCONTROL/afhba-llcontrol-two-fiber

We've tested this to 250kHz, latency ~15 usec.


Iterative tests:
For reliability testing, run LLC in a loop.

./scripts/forxv 1000 ./scripts/llc-runner

To make this work, please review ./scripts/llc-runner to confirm that it 
is running the scenario you require.
NB: ./scripts/llc-test-harness-intclk-commsA 
has been especially refactored to allow it to be called externally

forxv ./scripts/llc-runner iteration 7914

first run LLC ./LLCONTROL/afhba-llcontrol-example 50000
now run TST ./scripts/llc-test-harness-intclk-commsA
ready for data
action init_comms
[         0]          0 	0 => 0 
action init_acq2106A
setting internal clock / 100
sitelist: 1
sites: 1
action start_stream

[     10000]      10353 
[     20000]      20704 
[     30000]      31041 
[     40000]      41374 
[     50000]      51725 
finished


