# AFHBA404 LLC

## What is it?

* Demonstration of very low latency transfer from ADC to host memory.
* Demonstration of very low latency transfer from HOST to DAC.
* Provides a test harness suitable for GA-style low latency control.

## Hardware
|                 |                              |
|-----------------|------------------------------|
| ACQ2106+MGT400+ |	Carrier with fiber-comms     |
| NxACQ424ELF+	  |	N x 32 channels AI           |
| AO424ELF		  | 1 x 32 channels AO           |
| AFHBA404		  | 4 port host-side bus adapter |

AFHBA supports 4 ports 6Gbps/ SFP+ transceivers.

Demo below uses one port only.

## Theory of operation

<pre>
ADC => Z7030 -> SFP -> AFHBA404 -> PCI-Express 4x -> HOST
</pre>

ADC: 32 .. 128 channels, 1MSPS, very low latency output
Z7030: ZYNQ series FPGA with ARM processor and MGT serial links
SFP : fiber optic link, running "Aurora" protocol at 5Gbps.
AFHBA404: D-TACQ host bus adapter, Kintex.

Repetition Rate : 1MSPS
Latency: 

View on scope.

## Getting Started

### On the HOST linux-rt system
NB: we've been using Ubuntu 14.04. Partly from customer request, mainly from 
provision of a kernel with dynamic debug enabled.
Linux 3.10.33-rt should still work.

<pre>
dt100@jakku:~$ uname -a
Linux jakku 3.16.0-57-generic #77~14.04.1-Ubuntu SMP
</pre>

1. Plug the AFHBA into your linux-rt system, and check that it still boots.

2. Check that the AFHBA has been correctly enumerated:
<pre>
dt100@jakku:~$ lspci -v | grep -A 10 Xil
01:00.0 Memory controller: Xilinx Corporation Device adc1
	Subsystem: Device d1ac:4104
</pre>

3. Build and load host side driver:
https://github.com/petermilne/AFHBA404

### Power up the ACQ2106 and check that you can log in to it
1. Connect ACQ2106 to Ethernet.
2. Connect to the console
3. Power up and log in

This is described in
http://www.d-tacq.com/resources/d-tacq-4G-acq4xx-UserGuide.pdf
#5 "Power Up Guide"

3.4 It's _essential_  that ACQ2106 is visible on Ethernet from HOST
let IP = ACQ2106 IP ADDRESS

### Connect fiber-optics
Example connects PORTB on ACQ2106 to PORTD on AFHBA404
Any combination will do

## Install HOST SIDE SOFTWARE

git clone from 
https://github.com/petermilne/AFHBA404.git

or select the "Download Zip" option and unzip:
https://github.com/petermilne/AFHBA404/archive/master.zip
wget http://www.d-tacq.com/swrel/afhba-1409182241.tar

On the host:

<pre>
cd AFHBA404
make all
</pre>

Then, load the driver.
Later, you might want to automate this:

<pre>
sudo ./loadNIRQ
</pre>

## Run a shot

1. Connect a signal to AI01. Connect a scope to AI01, AO01.

2. TERMINAL SESSION 1: "PCS"
<pre>
cd LLCONTROL
DEVNUM=3 NCHAN=128 ./afhba-llcontrol-zcopy
</pre>

3. TERMINAL SESSION 2: "Session Management":
<pre>
INTCLKDIV=660 ./scripts/llc-test-harness-intclk-AI1234-AO5 acq2106_021
</pre>

4. TERMINAL SESSION 3: "TRIGGER"
Optional COMMS "over the fiber" trigger 
<pre>
RPORT=B ./scripts/remote_soft_trigger  acq2106_021
</pre>

.. the cool thing about the COMMS trigger is that it will operate on all 
connected ports at the same time.
