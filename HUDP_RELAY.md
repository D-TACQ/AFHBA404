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
  - We specify a HUDP_RELAY section of 12 * u32 (ie 6xAO420, 4x16 bit)
    - NB: the PAD section cannot current exceed 64b, and already includes 3*u32 from 3*DIO
  
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
  - acqproc dummy run

```bash
./ACQPROC/acqproc ACQPROC/configs/mast_HP32_12.json 

# create runtime.json: a view of ACQPROC's internal model of the system. 
# We want to know the offset of the HP32 slice in the vector:
[dt100@brotto AFHBA404]$ grep -A 4 VO_OFFSETS runtime.json 
                    "VO_OFFSETS": {
                        "AO16": 0,
                        "DO32": 128,
                        "HP32": 140
                    },
```

 - the HP32 slice is at byte offset 140 in the VO vector. use as the argument for hudp_relay below ..
 - we set CLK to 10k to keep the speed down..

```bash
cd ~/PROJECTS/AFHBA404; pushd ../acq400_hapi; source ./setpath; popd
SITECLIENT_TRACE=1 CLK=10000 ./scripts/acqproc_multi.sh ACQPROC/configs/mast_HP32_12.json configure_uut
SITECLIENT_TRACE=1 ./user_apps/acq2106/hudp_setup.py --rx_ip=10.12.198.254 --tx_ip 10.12.198.100 --run0='notouch' --play0='notouch' --hudp_relay=140 acq2106_354 none
```

- Per Shot action, run the shot:
  - on UDP Rx (naboo)
  
```bash
nc -ul 10.12.198.254 53676 | pv > hudp.raw
```
  
  - on HOST
  
```bash
NOCONFIGURE=1 SITECLIENT_TRACE=1 THE_ACQPROC=./ACQPROC/acqproc_hpr CLK=10000 POST=10000 \
./scripts/acqproc_multi.sh ACQPROC/configs/mast_HP32_12.json
```
- During the shot, on Rx Host

```
[dt100@naboo ~]$ nc -ul 10.12.198.254 53676 | pv > hudp.raw
29.8MiB 0:11:04 [ 508kiB/s] [                                                     <=>
```

- During the shot, on PCSHOST

```
[dt100@brotto AFHBA404]$ SITECLIENT_TRACE=1 ../acq400_hapi/user_apps/acq2106/hudp_setup.py --rx_ip=10.12.198.254 --tx_ip 10.12.198.100 --run0='notouch' --play0='notouch' --hudp_relay=140 acq2106_354 none
Siteclient(acq2106_354, 4220) >MODEL
Siteclient(acq2106_354, 4220) <acq2106sfp
Siteclient(acq2106_354, 4220) >is_tiga
Siteclient(acq2106_354, 4220) <none
Siteclient(acq2106_354, 4220) >has_mgt
Siteclient(acq2106_354, 4220) <12 13

```

- Analysis

```bash
[dt100@naboo ~]$ hexdump -e '13/4 "%08x," "\n"' hudp.raw  | more
00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,00000000,
*
00000001,0000005e,22222222,33333333,0003e5d8,0004e5d8,0005e5d8,0006e5d8,0007e5d8,0008e5d8,0009e5d8,000ae5d8,00000000,
00000002,000000c2,22222222,33333333,0003e6c1,0004e6c1,0005e6c1,0006e6c1,0007e6c1,0008e6c1,0009e6c1,000ae6c1,00000000,
00000003,00000126,22222222,33333333,0003e7ae,0004e7ae,0005e7ae,0006e7ae,0007e7ae,0008e7ae,0009e7ae,000ae7ae,00000000,
00000004,0000018a,22222222,33333333,0003e89c,0004e89c,0005e89c,0006e89c,0007e89c,0008e89c,0009e89c,000ae89c,00000000,
00000005,000001ee,22222222,33333333,0003e98a,0004e98a,0005e98a,0006e98a,0007e98a,0008e98a,0009e98a,000ae98a,00000000,
00000006,00000252,22222222,33333333,0003ea7e,0004ea7e,0005ea7e,0006ea7e,0007ea7e,0008ea7e,0009ea7e,000aea7e,00000000,
00000007,000002b6,22222222,33333333,0003eb6e,0004eb6e,0005eb6e,0006eb6e,0007eb6e,0008eb6e,0009eb6e,000aeb6e,00000000,
```


- Host Pull Trigger Action

```bash
NOCONFIGURE=1 SITECLIENT_TRACE=1 THE_ACQPROC=./ACQPROC/acqproc_hpr CLK=10000 POST=10000 \
SINGLE_THREAD_CONTROL=host_pull_trigger=1,0 ./scripts/acqproc_multi.sh ACQPROC/configs/mast_HP32_12.json

```

  - @@WORTODO: we fixed the HPT to be half the rate, but the rate is still the same, looks like we still have CLOCK selected.
  

- Queries? Please contact peter.milne@d-tacq.com


 