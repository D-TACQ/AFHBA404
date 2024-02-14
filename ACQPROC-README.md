# ACQPROC-README

### ACQPROC : Low Latency Control configured by data file, any number of UUT's, any IO combo:
- a single compile-once executable supports any number (4 per AFHBA404) UUT's (ACQ2106), any payload.
- the system is defined by a config file (json), and the system autoconfigures itself.
- we're using a SHM to gather all IO data from the UUT's in a single AI,DI,AO,DO vectors, so that the PCS algorithm is independent of the actual set of hardware in use. 
eg 
  - it could be 4 boxes with 32AI each, or one box with 32AI
  - it could be 2 boxes each 96AI, 64AO or one box with 192AI and another with 128AO.
.. and the same algorithm will work with both
 it's assumed that the PCS is another process on other cores that interacts with the SHM.

- logic to handle special cases - eg 
  - BOLO8 (data not in phase), 
  - WATCHDOG (a pulse output before trigger).
  
- acqproc outputs the aggregate configuration as a config file (json), this publishes indices into shared memory vectors that we expect would be useful for the PCS and offsets of salient fields in the individual VI, VO vectors of each UUT, used by our post-shot analysis tools - the goal is that post-shot analysis is automatic, rather than depending on large numbers of command line constants. 

- a skeleton config file can be autocreated on any given system, this will give an instant working system, and is the recommended way to proceed

```
pushd ../acq400_hapi; source ./setpath; popd

./HAPI/lsafhba --save_config=myexperiment.json
```

- scripted tools (usually python) use the config file to configure the units via Ethernet, other tools use the input config file and in particular the output config file to drive data analysis.

- the config file is a common configuration interface between pre-shot configuration (python), in-shot realtime (C++, CUDA) and post-shot analysis (python).

- For a NEW PCS:
  - Define the config file
  - Subclass the shared memory interface SystemInterface to connect to your system.
    - Users can create a custom subclass to implement shared memory, comms
    - in particular, overload SystemInterface::ringDoorBell();
      - in ringDoorBell():
        - all latest inputs are in IN.*, use them and
        - leave outputs in OUT.* ..
    - the framework will take care of the rest.

  
```C++
struct SystemInterface {

public:
	/** ONE vector each type, all VI from all UUTS are split into types and
	 *   aggregated in the appropriate vectors. 
	 */
	struct Inputs {
		short *AI16;
		int *AI32;
		unsigned *DI32;
		unsigned *SP32;
	} IN;
	/**< ONE vector each type, scatter each type to appropriate VO all UUTS	 */
	struct Outputs {
		short* AO16;
		unsigned *DO32;
		unsigned *CC32;			/* calc values from PCS .. NOT outputs. */
	} OUT;

	SystemInterface(const HBA& _hba);

	virtual void ringDoorbell(int sample)
	/* alert PCS that there is new data .. implement by subclass. */

```

  - In a Single Thread implementation:
     - Inputs, Outputs are created from simple heap allocations
     - method ringDoorbell() is called when there is new data.
     - Single Thread implementations can perform IO processing inside ringDoorbell(), leaving updated Outputs for the system to handle.
     - examples https://github.com/D-TACQ/AFHBA404/blob/master/ACQPROC/DefaultSystemInterface.cpp
     - a "real" case will be more complex, and it may be an interface to another framework
      - eg the Subclass could be a SIMULINK wrapper.

  - In a Multi Thread implementation:
     - a subclass of SystemInterface could place Inputs Outputs in Shared Memory SHM
     - method ringDoorbell() is called when there is new data.
     - in a Multi Thread implementation, ringDoorBell() implements an IPC signal to the main task running in another thread of control, probably on other core[s]
     - so far, we didn't provide a Multi Thread example.
  - Late Binding: 
    -  different implementations of SystemInterface are combined at link time. 
    - See Makefile for explicit examples
    - New implementations should probably extend the Makefile. Or, we could provide acqproc as a solib.




## Glossary:
- PCS: Plasma Control System
- SHM: Shared Memory
- UUT: Unit Under Test, in our case, an ACQ2106 DAQ Appliance with variable module payload.
- AFHBA404: Host Bus Adapter with 4 fiber link ports, supporting up to 4UUT's.
- acqproc: compile once, configuration driven data handling engine supporting 4UUT's as the IO processor for a PCS

### Json definition
- AFHBA: root of the data. Represents an AFHBA404 (or multiple AFHBA404s)
- UUT: represents one ACQ2106 on one AFHBA404 port
- DEVNUM: device number of the AFHBA link port. The numbers are global to the OS, so for example 
  - DEVNUM=0 : PortA on first AFHBA404 card
  - DEVNUM=4 : PortB on second AFHBA404 card
  - DEVNUM defaults to 0 and by default counts up once per entry. System designer can insert a DEVNUM field in any UUT declaration.
- type: pcs, bolo [other specialized types]
- sync_role: fpmaster, slave 
  - represents the clocking configuration 
### Vectors
- VI : Vector Input: UUT sends a sample every clock, this is the VI, comprised of AI16, AI32, DI32, SP32
- VO : Vector Output: UUT fetches an output sample every clock, this is the VO, comprising AO16, DO32, CP32

### Fields in Vectors
- AI16 : Analog Input, 16 bit.(eg ACQ424ELF-32)
- AI32 : Analog Input, 32 bit (eg BOLO8)
- DI32 : Digital Input, 32 bit (eg DIO432ELF)
- SP32 : Scratch Pad, 32 bit unsigned. SPAD[0] is the TLATCH or sample count.
- AO16 : Analog Output, 16 bit, (eg AO424ELF-32)
- DO32 : Digital Output, 32 bit, (eg DI423ELF)
- CP32 : Calc Value, 32 bit, an intermediate calc result from the PCS. This is NOT sent to the UUT, but is stored in the raw output.
- HP32 : HudP remote XO values, 32 bit

### Rules for Vectors

- the DMAC has a granularity of 64 bytes, therefore the length of ALL vectors MUST be a multiple of 64 bytes.
- On VI, this achieved by setting the Scratch Pad Length. While SPAD is a convenient way to convey metadata (eg sample count, time in usecs etc), its primary purpose is the pad out the VI to the next 64 byte boundary.
- On VO this is achieved by a PAD or "TCAN" value.
- Both SPAD and TCAN are configured automatically by the acqproc setup scripts.




## sample config file, pcs1.json
```json
{
    "AFHBA": {
        "UUT": [
...
           	{ 	
           	    "name" : "acq2106_555",  
			    "type": "bolo",
              	"VI" : {
                  		"AI32" : 48,  
				        "SP32" : 16
               }
            }
        ]
    }
}
```
## special customization. ***OPTIONAL***, system will use sensible defaults:
- COMMS : global to the UUT definition, specify COMMS A or B
- WD_BIT : global to the UUT definition, selects Watchdog Bit in DO vector. (assumes DO present)
  - example: configs/pcs1.json https://github.com/D-TACQ/AFHBA404/blob/master/ACQPROC/configs/pcs1.json
- AISITES : local to the VI definition, specify exact AI modules in set.
- DISITES : local to the VI definition, specify exact DI modules in set.
  - for when inbound data is farmed to different HOSTS, and the HOST doesn't need to see ALL the data (unusual)
- AOSITES : local to the VO definition, specify exact AO module in set.
- DOSITES : local to the VO definition, specify exact AO modules in set.
  - for when outbound data is sources from different HOSTS. (more likely, for spreading processing load among hosts)
  
```json
{
    "AFHBA": {
        "DEVNUM": 0,
        "UUT": [
             {
                "name": "acq2106_130",
                "type": "pcs",
                "sync_role": "fpmaster",
				"COMMS": "B",
                "VI": {
                    "AI16": 128,
                    "SP32": 16					
                },
                "VO": {
                    "AO16": 32,
                    "AOSITES": [6]
                }
            }
        ]
    }
}

```
 
- DO_BYTE_IS_OUTPUT : array of output definitions, one per DO module.

```json
{
    "AFHBA": {
        "UUT": [
                {
                "name" : "acq2106_276",
                "type": "pcs",
                "VI" : {
                    "AI16" : 192,
                    "SP32" : 16
                    },
                "VO": {
                    "AO16" : 0
                }
                },
                {
                "name": "acq2106_277",
                "type": "pcs,nowait,pwm",
                "VI": {
                    "AI16" : 0
                },
                "VO": {
                    "AO16": 128,
                    "DO32": 1,
                    "DO_BYTE_IS_OUTPUT" : [ "1,1,1,0" ],
                    "PW32": 1
                    }
                }
            ]
        }
}

```

## Typical Dummy Run:
```
[pgm@hoy5 AFHBA404]$ sudo ./ACQPROC/acqproc ./ACQPROC/configs/pcs1.json 
NOTICE: port 3 is bolo in non-bolo set, set nowait
HBA0 VI:1216 VO:204
	[0] acq2106_123 VI:320 VO:68 Offset of SPAD IN VI :260
 System Interface Indices 0,0 WD mask: 0x80000000
	[1] acq2106_124 VI:320 VO:68 Offset of SPAD IN VI :260
 System Interface Indices 128,15
	[2] acq2106_125 VI:320 VO:68 Offset of SPAD IN VI :260
 System Interface Indices 256,30
	[3] acq2106_555 VI:256 VO:0 Offset of SPAD IN VI :192
 System Interface Indices 384,45
 new sample: 0 acq2106_123
 new sample: 0 acq2106_124
 new sample: 0 acq2106_125
 new sample: 0 acq2106_555
 new sample: 1 acq2106_123
 new sample: 1 acq2106_124
 new sample: 1 acq2106_125
 new sample: 1 acq2106_555
```

## runtime.json : computed system configuration

acqproc outputs an augmented version of the configuration file.
This includes all intermediate calculations performed on the vectors (counts, offsets)
This is firstly to let a human check that the file has been interpreted in a way that meets expectation, and secondly, could be a machine-readable input for the PCS algorithm, rather than relying on manually defined offsets and counts.

### AFHBA
First, a reflection of the original config file:
```json
{
    "AFHBA": {
        "UUT": [
            {
                "VI": {
                    "AI16": 128,
                    "DI32": 1,
                    "SP32": 15
                },
                "VO": {
                    "AO16": 32,
                    "DO32": 1
                },
                "WD_BIT": 31,
                "name": "acq2106_123",
                "type": "pcs"
            },
            {
                "VI": {
                    "AI16": 128,
                    "DI32": 1,
                    "SP32": 15
                },
                "VO": {
                    "AO16": 32,
                    "DO32": 1
                },
                "name": "acq2106_124",
                "type": "pcs"
            },
            {
                "VI": {
                    "AI16": 128,
                    "DI32": 1,
                    "SP32": 15
                },
                "VO": {
                    "AO16": 32,
                    "DO32": 1
                },
                "name": "acq2106_125",
                "type": "pcs"
            },
            {
                "VI": {
                    "AI32": 48,
                    "SP32": 16
                },
                "name": "acq2106_555",
                "type": "bolo"
            }
        ]
    },
```
### GLOBAL_INDICES
Then, a configuration structure with GLOBAL_INDICES, indices into the type-specific SHM vectors
```json
   "SYS": {
        "UUT": {
            "GLOBAL_INDICES": [
                {
                    "VI": {
                        "AI16": 0,
                        "DI32": 0,
                        "SP32": 0
                    },
                    "VO": {
                        "AO16": 0,
                        "DO32": 0
                    }
                },
                {
                    "VI": {
                        "AI16": 128,
                        "DI32": 1,
                        "SP32": 15
                    },
                    "VO": {
                        "AO16": 32,
                        "DO32": 1
                    }
                },
                {
                    "VI": {
                        "AI16": 256,
                        "DI32": 2,
                        "SP32": 30
                    },
                    "VO": {
                        "AO16": 64,
                        "DO32": 2
                    }
                },
                {
                    "VI": {
                        "AI32": 0,
                        "SP32": 45
                    },
                    "VO": {}
                }
            ],
```
### LOCAL
And LOCAL, byte offsets into the raw data files emitted by a PCS run:
```json
           "LOCAL": [
                {
                    "VI_OFFSETS": {
                        "AI16": 0,
                        "DI32": 256,
                        "SP32": 260
                    },
                    "VO_OFFSETS": {
                        "AO16": 0,
                        "DO32": 64
                    },
                    "VX_LEN": {
                        "VI": 320,
                        "VO": 68
                    }
                },
                {
                    "VI_OFFSETS": {
                        "AI16": 0,
                        "DI32": 256,
                        "SP32": 260
                    },
                    "VO_OFFSETS": {
                        "AO16": 0,
                        "DO32": 64
                    },
                    "VX_LEN": {
                        "VI": 320,
                        "VO": 68
                    }
                },
                {
                    "VI_OFFSETS": {
                        "AI16": 0,
                        "DI32": 256,
                        "SP32": 260
                    },
                    "VO_OFFSETS": {
                        "AO16": 0,
                        "DO32": 64
                    },
                    "VX_LEN": {
                        "VI": 320,
                        "VO": 68
                    }
                },
                {
                    "VI_OFFSETS": {
                        "AI32": 0,
                        "SP32": 192
                    },
                    "VO_OFFSETS": {},
                    "VX_LEN": {
                        "VI": 256,
                        "VO": 0
                    }
                }
            ]
        }
```

# More to come.. 
comments please to peter.milne@d-tacq.com





