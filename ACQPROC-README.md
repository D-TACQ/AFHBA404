# ACQPROC-README

## ACQPROC:
- a single compile-once executable supports any 4 UUT's (ACQ2106), any payload.
- the system is defined by a config file (json), and the system autoconfigures itself.
- we're using a SHM to gather all IO data from the UUT's in a single AI,DI,AO,DO vectors,
 it's assumed that the PCS is another process on other cores that interacts with the SHM.
- acqproc outputs the aggregate configuration as a config file (json), this publishes indices into shared memory vectors that we expect would be useful for the PCS and offsets of salient fields in the individual VI, VO vectors of each UUT, used by our post-shot analysis tools - the goal is that post-shot analysis is automatic, rather than depending on large numbers of command line constants. 
- logic to handle special cases - eg 
 - BOLO8 (data not in phase), 
 - WATCHDOG (a pulse output before trigger).

- For a NEW PCS:
 - Define the config file
 - Subclass the shared memory interface SystemInterface to connect to your system.

eg the Subclass could be a SIMULINK wrapper.


## Glossary:
- PCS: Plasma Control System
- SHM: Shared Memory
- UUT: Unit Under Test, in our case, an ACQ2106 DAQ Appliance with variable module payload.
- AFHBA404: Host Bus Adapter with 4 fiber link ports, supporting up to 4UUT's.
- acqproc: compile once, configuration driven data handling engine supporting 4UUT's as the IO processor for a PCS

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

## sample config file, pcs1.json
```json
{
    "AFHBA": {
        "UUT": [
...
           	{ 	"name" : "acq2106_555",  
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

# More to come.. comments please to peter.milne@d-tacq.com





