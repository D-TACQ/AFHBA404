README file for afhba llc

What is it? :

Demonstration of very low latency transfer from ADC to host memory.
Provides a test harness suitable for GA-style low latency control.

Hardware:
ACQ420FMC-4-1000 : 4 channels, 16 bit, 1MSPS digitizer
zc706 evaluation board
ACQ-FIBER-HBA : HOST-side host bus adapter.

Theory of operation:

ADC => Z7045 -> SFP -> AFHBA -> PCI-Express 1x -> HOST

ADC: 4 channels, 1MSPS, very low latency output
Z7045: ZYNQ series FPGA with ARM processor and MGT serial links
SFP : fiber optic link, running "Aurora" protocol at 5Gbps.
AFHBA: D-TACQ host bus adapter, Spartan 6.

Repetion Rate : 1MSPS
Latency: 
actually, we can't yet measure the overall latency, but we calcute that,
in the parts under our control

ADC => Z7045 -> SFP -> AFHBA -> 

Latency from ADC clock to data transfer is under 2usec.


Getting Started:

1. On the HOST linux-rt system:

1.1 Plug the AFHBA into your linux-rt system, and check that it still boots.

1.2 Check that the AFHBA has been correctly enumerated:

[dt100@helium ~]$ uname -a
Linux helium 3.10.33-rt32.43.el6rt.x86_64 #1 SMP PREEMPT RT Wed Jul 23 09:49:57 CEST 2014 x86_64 x86_64 x86_64 GNU/Linux
[dt100@helium ~]$ lspci | grep Xil
01:00.0 RAM memory: Xilinx Corporation Device adc1


1.3 Then you'll need to load and build our host side device driver.
I'm still working on this.

2. Power up the SANDWICH and check that you can log in to it.
2.1 Connect SANDWICH to Ethernet.
2.2 Connect to the console
3.3 Power up and log in

This is described in
http://www.d-tacq.com/resources/d-tacq-4G-acq4xx-UserGuide-r6.pdf
#5 "Power Up Guide"

3.4 It's _essential_  that SANDWICH is visible on Ethernet from HOST
let IP = SANDWICH IP ADDRESS

3. Connect fiber-optics

4. Install FPGA:

wget http://www.d-tacq.com/swrel/ZC706_TOP_01_ff.bit.gz
zc706_002> sha1sum ZC706_TOP_01_ff.bit.gz
cb8f5b6e7f018eb8ed2224a77fead21274a04f2a  ZC706_TOP_01_ff.bit.gz

scp ZC706_TOP_01_ff.bit.gz root@IP:/mnt/fpga.d

And reboot. Use a web browser to check status:
http://IP/d-tacq/#fpga
xiloader r1.01 (c) D-TACQ Solutions
eoh_location set 0
Xilinx Bitstream header.
built with tool version   : 22
generated from filename   : ZC706_TOP_01_ff
part                      : 7z045ffg900
date                      : 2014/09/18
time                      : 17:04:52
bitstream data starts at  : 96
bitstream data size       : 13321404

5. Install HOST SIDE SOFTWARE

wget http://www.d-tacq.com/swrel/afhba-1409182241.tar

On the host:

5.1 tar xvf afhba-1409182241.tar; cd AFHBA
5.2 make all

Then, load the driver. Later, you might want to automate this:

5.3 sudo ./load1

5.4 Configure scripts:
Change default IP in scripts/llc-test-harness (or set on command line)
REMIP=${1:-10.12.196.129}

5.5 TERMINAL SESSION 1: Configure the link
cd scripts
[root@helium scripts]# TEST=1 ./afhba-init
0x00000100
test LOC/0x0018 write 0x1 read 0x00000100
test LOC/0x0018 write 0x2 read 0x00000100
test LOC/0x0018 write 0x4 read 0x00000100
test LOC/0x0018 write 0x8 read 0x00000100
test LOC/0x0018 write 0x10 read 0x00000100
test LOC/0x0018 write 0x20 read 0x00000100
test LOC/0x0018 write 0x40 read 0x00000100
test LOC/0x0018 write 0x80 read 0x00000100
test LOC/0x0018 write 0x100 read 0x00000100
test LOC/0x0018 write 0x10000000 read 0x00000100
test LOC/0x0018 write 0x20000000 read 0x00000100
test LOC/0x0018 write 0x40000000 read 0x00000100
test LOC/0x0018 write 0x80000000 read 0x00000100
0x00000100
0x12345678
test REM/0x0104 write 0x1 read 0x00000001
test REM/0x0104 write 0x2 read 0x00000002
test REM/0x0104 write 0x4 read 0x00000004
test REM/0x0104 write 0x8 read 0x00000008
test REM/0x0104 write 0x10 read 0x00000010
test REM/0x0104 write 0x20 read 0x00000020
test REM/0x0104 write 0x40 read 0x00000040
test REM/0x0104 write 0x80 read 0x00000080
test REM/0x0104 write 0x100 read 0x00000100
test REM/0x0104 write 0x10000000 read 0x10000000
test REM/0x0104 write 0x20000000 read 0x20000000
test REM/0x0104 write 0x40000000 read 0x40000000
test REM/0x0104 write 0x80000000 read 0x80000000


afhba-init needs to run once per boot. 
You can run it without the TEST option once you know it works.

5.6 Check again that the HOST can see the ACQ 
ping IP

5.7 TERMINAL SESSION 2: run the "llc app":

please ignore the POSIX error at the beginning. 
It seems to work OK without RT priority.
Suggestions welcome!

[dt100@helium AFHBA]$ sudo ./afhba-llcontrol-example
failed to set RT priority: Invalid argument

... leave it hanging

5.8 TERMINAL SESSION 1:
[root@helium scripts]# ./llc-test-harness 
buffer PA 0x37c00000
0x00000060
0xdead0003
sitelist: 1
sites: 1
/usr/local/bin/procServ: spawning daemon process: 4421

now view data perhaps using ./mapsample
CTRL-C to quit

Leave it there ..

5.9 There should be lots of activity in TERMINAL SESSION 2:

[dt100@helium AFHBA]$ sudo ./afhba-llcontrol-example
failed to set RT priority: Invalid argument
[0] 0 0 => 81 
[sample] tlatch
[10000] 10000 
[20000] 20000 
[30000] 30000 
[40000] 40000 
[50000] 50000 
[60000] 60000 
[70000] 70000 
[80000] 80000 

...

[9950000] 9950000 
[9960000] 9960000 
[9970000] 9970000 
[9980000] 9980000 
[9990000] 9990000 

The correspondance between SAMPLE and TLATCH is unbelievably good ..

20141028: PGM notes : that was due to a coding error, the second value was simply "SAMPLE". Sorry 

The data format:

short ch[4], unsigned tlatch [other stuff ..]

./afhba-llcontrol-example creates a data log, dump as follows:

[dt100@helium AFHBA]$ ./scripts/dumplog  | head
ffff fff0 ffe5 ffe3 00000016 00000051
ffff fff0 ffe5 ffe3 00000018 00000051
ffff fff0 ffe5 ffe4 00000019 00000051
ffff fff0 ffe5 ffe3 0000001a 00000051
ffff fff1 ffe5 ffe3 0000001b 00000051
ffff fff0 ffe5 ffe3 0000001c 00000051
ffff fff0 ffe5 ffe3 0000001d 00000051

New stuff: October 28 2014


Monitor ./llc-test-harness
Start a new terminal as root on the HOST:
tail -f /var/log/messages
Oct 28 16:11:57 helium set_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0204 0x60
Oct 28 16:11:57 helium set_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0204 0x10
Oct 28 16:11:57 helium set_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0204 0x60
Oct 28 16:11:57 helium get_sys: /proc/sys/debug/afhba/afhba.0/BUF/pa 0x37400000
Oct 28 16:11:57 helium set_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0204 0x60
Oct 28 16:11:57 helium get_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0204 0x00000060
Oct 28 16:11:57 helium set_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0240 0x37400000
Oct 28 16:11:57 helium get_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0240 0xdead0003
Oct 28 16:11:57 helium set_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0204 0x61
Oct 28 16:11:57 helium rem_cmd: 10.12.196.129 4221 trg=1,2,1
Oct 28 16:11:58 helium rem_cmd: 10.12.196.129 4221 clk=0,0,0
Oct 28 16:11:59 helium rem_cmd: 10.12.196.129 4221 clkdiv=250
Oct 28 16:11:59 helium rem_cmd: 10.12.196.129 4220 spad=1,7,0
Oct 28 16:12:00 helium rem_cmd: 10.12.196.129 4220 run0 1
Oct 28 16:12:01 helium rem_cmd: 10.12.196.129 4220 streamtonowhered start
Oct 28 16:12:16 helium set_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0204 0x60
Oct 28 16:12:16 helium set_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0204 0x10
Oct 28 16:12:16 helium set_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0204 0x60
Oct 28 16:12:16 helium rem_cmd: 10.12.196.129 4220 streamtonowhered stop


TESTING EXTERNAL CLOCK.

Connect External Clock 0-5V, 1MHz to CLK input on TERM01.

Run the llcontrol example in the normal way:
[dt100@helium AFHBA400]$ sudo RTPRIO=20 ./afhba-llcontrol-example 100000

And run the test harness
[root@helium scripts]# LLC_CLK=ext ./llc-test-harness 
buffer PA 0x37400000
0x00000060
0xdead0003
sitelist: 1
sites: 1
/usr/local/bin/procServ: spawning daemon process: 28827

now view data perhaps using ./mapsample
CTRL-C to quit

What did the test harness do?

Oct 28 17:24:50 helium set_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0204 0x60
Oct 28 17:24:50 helium set_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0204 0x10
Oct 28 17:24:50 helium set_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0204 0x60
Oct 28 17:24:50 helium get_sys: /proc/sys/debug/afhba/afhba.0/BUF/pa 0x37400000
Oct 28 17:24:50 helium set_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0204 0x60
Oct 28 17:24:50 helium get_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0204 0x00000060
Oct 28 17:24:50 helium set_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0240 0x37400000
Oct 28 17:24:50 helium get_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0240 0xdead0003
Oct 28 17:24:50 helium set_sys: /proc/sys/debug/afhba/afhba.0/REM/0x0204 0x61
Oct 28 17:24:50 helium rem_cmd: 10.12.196.129 4221 trg=1,1,1
Oct 28 17:24:50 helium rem_cmd: 10.12.196.129 4221 clk=1,2,1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Selects front panel CLOCK
Oct 28 17:24:51 helium rem_cmd: 10.12.196.129 4221 clkdiv=1
Oct 28 17:24:52 helium rem_cmd: 10.12.196.129 4220 spad=1,7,0
Oct 28 17:24:53 helium rem_cmd: 10.12.196.129 4220 run0 1
Oct 28 17:24:54 helium rem_cmd: 10.12.196.129 4220 streamtonowhered start

Controller result:
[dt100@helium AFHBA400]$ sudo RTPRIO=20 ./afhba-llcontrol-example 100000
ready for data
[         0]          1 	0 => 0 
[     10000]      10137 
[     20000]      20247 
[     30000]      30356 
[     40000]      40458 



TESTING EXTERNAL TRG

Connect External Trigger, 0-5V, 1Hz to TRG input on TERM01

Run the llcontrol example in the normal way:
[dt100@helium AFHBA400]$ sudo RTPRIO=20 ./afhba-llcontrol-example 100000
...

Run the test harness eg
LLC_TRG=ext ./llc-test-harness

[dt100@helium AFHBA400]$ sudo RTPRIO=20 ./afhba-llcontrol-example 100000
ready for data
[         0]          1 	0 => 0 
[     10000]      10019 
[     20000]      20022 
[     30000]      30025 
[     40000]      40032 
[     50000]      50040 
[     60000]      60045 
[     70000]      70051 
[     80000]      80058 
[     90000]      90065 
finished

We know it's using the EXTERNAL TRIGGER, because the software hangs when there's no trigger connected..




REAL TIME PERFORMANCE.

If the controller software is achieving perfect real time, then the SAMPLE will match TLATCH.

This isn't happening!, which just goes to show, you can have an RTOS, but still not have hard real-time.
To achieve perfect RT, users will still have to assign the control thread to core that has interrupts disabled.

The latch count gives an idea of how far off we are:

It's also easy to vary the sample rate to judge this.

eg:

100kHz:

 [dt100@helium AFHBA400]$ sudo RTPRIO=20 ./afhba-llcontrol-example 100000

[root@helium scripts]# LLC_CLK=ext EXTCLKDIV=10 ./llc-test-harness 
buffer PA 0x37400000

[dt100@helium AFHBA400]$ sudo RTPRIO=20 ./afhba-llcontrol-example 100000
ready for data
[         0]          1 	0 => 0 
[     10000]      10002 
[     20000]      20002 
[     30000]      30002 
[     40000]      40002 
[     50000]      50002 
[     60000]      60002 
[     70000]      70002 
[     80000]      80002 
[     90000]      90002 

This is probably as good as it gets.

500kHz:
[root@helium scripts]# LLC_CLK=ext EXTCLKDIV=2 ./llc-test-harness
[dt100@helium AFHBA400]$ sudo RTPRIO=20 ./afhba-llcontrol-example 100000
ready for data
[         0]          1 	0 => 0 
[     10000]      10040 
[     20000]      20065 
[     30000]      30091 
[     40000]      40115 
[     50000]      50141 
[     60000]      60165 

[dt100@helium AFHBA400]$ sudo RTPRIO=20 ./afhba-llcontrol-example 100000
[root@helium scripts]# LLC_CLK=ext EXTCLKDIV=1 ./llc-test-harness 
buffer PA 0x37400000
0x00000060
0xdead0003

ready for data
[         0]          1 	0 => 0 
[     10000]      10138 
[     20000]      20253 
[     30000]      30361 
[     40000]      40473 
[     50000]      50582 
[     60000]      60693 


=> Almost no missing samples at 100kHz, 0.4% at 500kHz, 1.4% at 1MHz.
In reality, I'd have thought if the control algorithm can't handle missing 1.4% of samples, it's probably not robust enough anyway?


CLOCK MONITORING

There is extensive clock monitoring available using EPICS (CSS gui available).
Contact D-TACQ for info.













