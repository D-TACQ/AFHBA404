## AFHBA404

### 1. Hardware
AHFBA404 is a PCI-express Gen2 4 lane card to fit standard PC enclosure
On the front panel: 4xSFP marked A,B,C,D

AFHBA404 is used to connect 1 to 4 ACQ2106 units to a HOST PC, establishing a 
fiber-optic link that may be used in one of two ways:

#### 1.1 HTS High Throughput Stream

Data is transferred in large blocks at maximum 400MB/s per link.
(max 4MB, min 4K), typically to disk archive.

In theory it's possible to run all 4 links at full speed, but to date
D-TACQ has not tested a PC that is capable of handling this.

More typical usage:
* 1 x 400MB/s, single box, full rate to local disk (NVMe SSD recommended)
* 4 x 100MB/s, connect 4 x ACQ2106 to single host.

AFHBA404 may also be used to send a common trigger to multiple ACQ2106.

See README.ACQ2106.HTS and STREAM/ for details.

#### 1.2 LLC Low Latency Control

Classic Plasma Control System.
Data is transferred in "small" blocks, one sample at a time to HOST DRAM
with minimum latency. An LLC system can be implemented in one or multiple
ACQ2106 boxes.

Typical use case:

* ACQ2106+4xACQ424ELF-32+2xAO424ELF32 : 128AI, 64AO, run in an LLC loop at 100kSPS.

For modern use of LLC, see [ACQPROC-README](./ACQPROC-README.md)

### 2. Device driver

Download from git:
<pre>
mkdir PROJECTS
cd PROJECTS
git clone https://github.com/D-TACQ/AFHBA404.git
cd AFHBA404
</pre>
Now refer to INSTALL
<pre>
make
sudo ./scripts/install-hotplug
</pre>

### 3. Check AFHBA404 firmware
<pre>
[dt100@brotto AFHBA404]$ sudo ./scripts/afhba404-get-ident 
brotto Device d1ac:4104 Device Serial Number af-ba-40-40-11-00-41-0b

af-ba-40-40-    : Device ID
11 	        : Comms code TT
00-41		: Serial Number (Decimal!)
0b		: Firmware Revision VV (hex)  

NB: for use with ACQ2106, ALL current units should show:
VV >= 0x0b. 
TT == 0x11.
</pre>
The matching ACQ2106 FPGA firmware has COMMS CODE 9011, eg
<pre>
acq2106_154> grep filename /tmp/fpga_status 
generated from filename   : ACQ2106_TOP_09_09_09_09_09_09_9011

ACQ2106_TOP_ : Idents ACQ2106
09_[09_]..   : Matches 6 modules sites, in this case ACQ423
9011         : COMMS MODULE, 90: SFP comms, 11: 6Gbps with error detection/correction
</pre>
If your AFHBA404 or ACQ2106 does NOT meet the requirements,
please contact D-TACQ for support.

### 4. Check Connected Devices
<pre>
[dt100@brotto AFHBA404]$ sudo ./scripts/get-ident-all 
brotto 0 acq2106_110 A
brotto 1 acq2106_110 B
brotto 2 acq2106_085 A

Lists the attached UUT's.

Linux Device numbers 0,1,2,3 map to local ports A,B,C,D
brotto 0 acq2106_110 A

hostname: brotto 	(the HOST PC)
DEVNUM  : 0      	(localport A)
UUT     : acq2106_110 	(remote host name)
RPORT   : A             (remote port A)

NB: the LINK is used for DATA only. Control is via Gigabit Ethernet
</pre>

Ideally, the HOSTPC has a DNS system (/etc/hosts if no DNS) that will allow
network software to connect to the UUT by name.

For further details, please refer to
* INSTALL
* README.ACQ2106.HTS
* LLCONTROL/README.AFHBA404
* Code Documentation: https://d-tacq.github.io/AFHBA404/html/index.html



