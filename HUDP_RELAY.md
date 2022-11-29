# HUDP_RELAY

## HUDP_RELAY

- Is a standard MGT/PCIe LLC system with HUDP add on
- A UUT with regular XI, XO payload:
 - sends LLC data to HOST using MGT/PCIe using a single Vector VI
 - pulls LLC data from HOST using MGT/PCIe using a single Vector VO
  - VI is a single DMA transaction with an integer number of whole PCIe TLP's length 64byte.
  - VI uses the SPAD feature to both add instrumentation and to pad the vector length to a whole number of TLP's
  - VO is a single DMA transition with an integer number of whole PCIe TLP's length 64b.
  - VO uses the PAD (aka "TCAN") feature to pad the vector length to a whole number of TLP's. The maximum length of PAD is 16xu32.
- With HUDP_RELAY, VO is deliberately extended using PAD to include additional "Relay" data.
- HUDP Hardware UDP processing unit is set to relay a slice of VO to remote client[s]


## Installation

### Fetch D-TACQ Sofware

- we assume that your HOST computer has kernel-devel, g++, python3 installed
 - in our examples, the HOST is "brotto":
 
 ```
 [dt100@brotto AFHBA404]$ cat /etc/redhat-release 
CentOS Linux release 7.4.1708 (Core) 
[dt100@brotto AFHBA404]$ head -n1 /proc/meminfo 
MemTotal:       16216156 kB
[dt100@brotto AFHBA404]$ grep "model name" /proc/cpuinfo 
model name	: Intel(R) Xeon(R) CPU E3-1220 v5 @ 3.00GHz
model name	: Intel(R) Xeon(R) CPU E3-1220 v5 @ 3.00GHz
model name	: Intel(R) Xeon(R) CPU E3-1220 v5 @ 3.00GHz
model name	: Intel(R) Xeon(R) CPU E3-1220 v5 @ 3.00GHz
 
 ```
  
- we assume a user with sudo privaleges. In our examples, nominally this user is "dt100"
```
cd ~; mkdir PROJECTS; cd PROJECTS;
git clone https://www.d-tacq.com/D-TACQ/acq400_hapi
git clone https://www.d-tacq.com/D-TACQ/AFHBA404

```

### Build AFHBA404 support

Follow instructions in INSTALL

#### One Time

```
cd AFHBA404
make
sudo ./scripts/install-hotplug
```

#### Every Boot

```
sudo ./scripts/loadNIRQ
sudo ./scripts/get-ident-all

pushd ../acq400_hapi; source ./setpath; popd
```

### ACQPROC

- ACQPROC uis a framework for running LLC, it features data-driven configuration.
- The easiest way to get a system-specific config file is to auto-create it, eg

```
dt100@brotto AFHBA404]$  ./HAPI/lsafhba.py 
0 HostComms(host='brotto', dev='0', uut='acq2106_354', cx='A')
