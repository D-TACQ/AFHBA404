# text input to match afhba404-acq2106-llc-fat.pdf
some users report difficulty in cut and paste from pdf.

# 2 INSTALL

## Load Device Driver
```
[dt100@brotto AFHBA404]$ sudo ./scripts/loadNIRQ
```
## Check AFHBA404 serial number and Firmware version
```
[dt100@brotto AFHBA404]$ sudo ./scripts/afhba404-get-ident
brotto Device d1ac:4104 Device Serial Number af-ba-40-40-11-00-41-0b
```
## Check UUTS connected to the link.
```
[dt100@brotto AFHBA404]$ sudo ./scripts/get-ident-all
brotto 0 acq2106_110 A
brotto 1 acq2106_110 B
```
In this case, our UUT is acq2106_110, and, for demonstration purposes we connected TWO fiber optic links:
```
DEVNUM=0, LPORT=A, UUT=ACQ2106_110, RPORT=A
DEVNUM=1, LPORT=B, UUT=ACQ2106_111, RPORT=B
Where:
DEVNUM : Linux device number 01,2,3 maps to LPORT : local PORT (SFP) A,B,C,D
UUT    : Unit Under Test: hostname, ideally, this is the network hostname as well
RPORT : remove PORT (SFP) A or B (C, D are NOT available for use at this time)
We demonstrate operation on either link. The scripting we demonstrate has been tested on
ALL of LPORT A,B,C,D and RPORT A,B
```

# 8 Running LLC

## Control Script Window
```
cd PROJECTS/AFHBA404
SOFT_TRIGGER=1 AISITES=1,2,3,4 AOSITES=5,6 XO_COMMS=A ./scripts/llc-test-harness-AI123-AO56 acq2106_110
AI123-AO56 acq2106_110
setting external clock / 10
sites: 1 2 3 4
y: run the shot, n: abort
```

## Control Program Window
```
cd PROJECTS/AFHBA404/
DEVNUM=0 AOCHAN=64 DUP1=0 AICHAN=128 ./LLCONTROL/afhba-llcontrol-cpucopy 100000
AICHAN (nchan) set 128
AOCHAN set 64
failed to set RT priority: Operation not permitted
AI buf pa: 0x65100000 len 320
AO buf pa: 0x65101000 len 128
ready for data
```

## Control Script Window
```
y
start transient
```

## Control Program Window
```
finished
[dt100@brotto AFHBA404]$ ls -l afhba.0.log
-rw-rw-r--. 1 dt100 dt100 32000320 Sep 6 14:20 afhba.0.log
```

# 11 Analyse the Data
## Plot Window
```
cd PROJECTS/ACQ400/HAPI/acq400_hapi
./user_apps/analysis/host_demux.py --src $HOME/PROJECTS/AFHBA404/afhba.0.log \
    --nchan=160 --egu=0 --pchan=1,33,65,97,129 acq2106_110
```

## Install kst 
pykst apparently only work on Linux. Surprising..
```
sudo yum install kst
wget https://kst-plot.kde.org/pykst.tgz
tar xvzf pykst.tgz
cd pyKst
sudo python2.7 setup.py install
sudo yum install PySide
sudo yum install python2-pyside.x86_64
```

