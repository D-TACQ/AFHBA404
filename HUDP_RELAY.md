# HUDP_RELAY

## HUDP_RELAY

- Is a standard MGT/PCIe LLC system with HUDP add on
- A UUT with regular XI, XO payload:
  - sends LLC data to HOST using MGT/PCIe using a single Vector VI
    - VI is a single DMA transaction with an integer number of whole PCIe TLP's length 64byte.
    - VI uses the SPAD feature to both add instrumentation and to pad the vector length to a whole number of TLP's  
  - pulls LLC data from HOST using MGT/PCIe using a single Vector VO
    - VO is a single DMA transition with an integer number of whole PCIe TLP's length 64b.
    - VO uses the PAD (aka "TCAN") feature to pad the vector length to a whole number of TLP's. The maximum length of PAD is 16xu32.
- With HUDP_RELAY, VO is deliberately extended using PAD to include additional "Relay" data.
- The HUDP Hardware UDP processing unit is set to relay a slice of VO to remote client[s]

- In "Normal" LLC, all processes are driven by the sample clock
  - On the clock, the ADC array samples simultaneously, the AGGREGATOR gathers  data to VI and sends VI to the host
  - On the clock, the UUT "pulls" VO from the HOST, and the DISTRIBUTOR scatters data to the outputs.
  
- In a PULL_HOST_TRIGGER system, a software action on the HOSTP triggers the VO fetch.


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
  
- we assume a user with sudo privileges. In our examples, nominally this user is "dt100"
```
cd ~; mkdir PROJECTS; cd PROJECTS;
git clone https://www.d-tacq.com/D-TACQ/acq400_hapi
git clone https://www.d-tacq.com/D-TACQ/AFHBA404

```

- in our testing, we have a private (Ethernet 1000LX network to another computer "naboo" as the data receiver
  - naboo: 10.12.198.254 
  - UUT:  10.12.198.100    on HUDP port on MGTD


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

### ACQPROC Configuration

- ACQPROC is a framework for running LLC, it features data-driven configuration.
- The easiest way to get a system-specific config file is to auto-create it, eg

```
dt100@brotto AFHBA404]$  ./HAPI/lsafhba.py --save_config ACQPROC/configs/mast_raw.json
0 HostComms(host='brotto', dev='0', uut='acq2106_354', cx='A')
```

- the raw config file ACQPROC/configs/mast_raw.json looks like this:

```json
{
    "AFHBA": {
        "UUT": [
            {
                "DEVNUM": 0,
                "name": "acq2106_354",
                "type": "pcs",
                "sync_role": "master",
                "COMMS": "A",
                "VI": {
                    "AI16": 32,
                    "DI32": 3,
                    "SP32": 13,
                    "AISITES": [
                        1
                    ],
                    "DIOSITES": [
                        4,
                        5,
                        6
                    ],
                    "NXI": 4
                },
                "VO": {
                    "AO16": 64,
                    "DO32": 3,
                    "AOSITES": [
                        2,
                        3
                    ],
                    "DIOSITES": [
                        4,
                        5,
                        6
                    ],
                    "NXO": 5
                }
            }
        ]
    }
}

```

- We then copy and modify the file to add the site specific features:
  - DO32 are ALL OUTPUTS
  - We specify a HUDP_RELAY section of 12 * u32
  
```
diff -urN ACQPROC/configs/mast_raw.json ACQPROC/configs/mast_HP32_12.json
--- ACQPROC/configs/mast_raw.json	2022-11-29 06:58:47.778833411 +0000
+++ ACQPROC/configs/mast_HP32_12.json	2022-11-29 06:58:38.831921558 +0000
@@ -24,6 +24,8 @@
                 "VO": {
                     "AO16": 64,
                     "DO32": 3,
+		    "DO_BYTE_IS_OUTPUT" : [ "1,1,1,1", "1,1,1,1", "1,1,1,1" ],
+		    "HP32": 12,
                     "AOSITES": [
                         2,
                         3

```

- creating a new config file ACQPROC/configs/mast_HP32_12.json

```json
{
    "AFHBA": {
        "UUT": [
            {
                "DEVNUM": 0,
                "name": "acq2106_354",
                "type": "pcs",
                "sync_role": "master",
                "COMMS": "A",
                "VI": {
                    "AI16": 32,
                    "DI32": 3,
                    "SP32": 13,
                    "AISITES": [
                        1
                    ],
                    "DIOSITES": [
                        4,
                        5,
                        6
                    ],
                    "NXI": 4
                },
                "VO": {
                    "AO16": 64,
                    "DO32": 3,
		    "DO_BYTE_IS_OUTPUT" : [ "1,1,1,1", "1,1,1,1", "1,1,1,1" ],
		    "HP32": 12,
                    "AOSITES": [
                        2,
                        3
                    ],
                    "DIOSITES": [
                        4,
                        5,
                        6
                    ],
                    "NXO": 5
                }
            }
        ]
    }
}

```

## Running the system

- First Time Action:
  -

```bash
cd ACQPROC; pushd ../acq400_hapi; source ./setpath; popd
SITECLIENT_TRACE=1 ./scripts/acqproc_multi.sh ACQPROC/configs/mast_HP32_12.json configure_uut
SITECLIENT_TRACE=1 ./user_apps/acq2106/hudp_setup.py --rx_ip=10.12.198.254 --tx_ip 10.12.198.100 --run0='notouch' --play0='notouch' --hudp_relay=140 acq2106_354 none
```

- Per Shot action, run the shot:
  - on UDP Rx (naboo)
  
```bash
nc -ul 10.12.198.254 53676 | pv > hudp.raw
```
  
  - on HOST
  
```bash
NOCONFIGURE=1 SITECLIENT_TRACE=1 ./scripts/acqproc_multi.sh ACQPROC/configs/mast_hudp_32lw.json
```

- Analysis

```bash
hexdump blah blah
```


- Host Pull Trigger Action

```bash
NOCONFIGURE=1 SINGLE_THREAD_CONTROL=host_pull_trigger=1,0 ./scripts/acqproc_multi.sh ACQPROC/configs/mast_hudp_32lw.json

```



 