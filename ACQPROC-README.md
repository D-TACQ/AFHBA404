ACQPROC-REQADME

ACQPROC:
- a single compile-once executable supports any 4 UUT's (ACQ2106), any payload.
- the system is defined by a config file (json), and the system autoconfigures itself.
- we're using a SHM to gather all IO data from the UUT's in a single AI,DI,AO,DO vectors,
 it's assumed that the PCS is another process on other cores that interacts with the SHM.
- acqproc outputs the aggregate configuration as a config file (json), this publishes indices into shared memory vectors that we expect would be useful for the PCS and offsets of salient fields in the individual VI, VO vectors of each UUT, used by our post-shot analysis tools - the goal is that post-shot analysis is automatic, rather than depending on large numbers of command line constants. 

- For a NEW PCS:
-- Define the config file
-- Subclass the shared memory interface SystemInterface to connect to your system.

eg the Subclass could be a SIMULINK wrapper.


Glossary:
- PCS: Plasma Control System
- UUT: Unit Under Test, in our case, an ACQ2106 DAQ Appliance with variable module payload.
- AFHBA404: Host Bus Adapter with 4 fiber link ports, supporting up to 4UUT's.
- acqproc: compile once, configuration driven data handling engine supporting 4UUT's as the IO processor for a PCS.
- VI : Vector Input: UUT sends a sample every clock, this is the VI, comprised of AI16, AI32, DI32, SP32
- VO : Vector Output: UUT fetches an output sample every clock, this is the VO, comprising AO16, DO32, CP32
- AI16 : Analog Input, 16 bit.(eg ACQ424ELF-32)
- AI32 : Analog Input, 32 bit (eg BOLO8)
- DI32 : Digital Input, 32 bit (eg DIO432ELF)
- SP32 : Scratch Pad, 32 bit unsigned. SPAD[0] is the TLATCH or sample count.
- AO16 : Analog Output, 16 bit, (eg AO424ELF-32)
- DO32 : Digital Output, 32 bit, (eg DI423ELF)
- CP32 : Calc Value, 32 bit, an intermediate calc result from the PCS. This is NOT sent to the UUT, but is stored in the raw output.




