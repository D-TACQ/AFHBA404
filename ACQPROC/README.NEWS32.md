# NEWS32 : example 4 x ACQ2106+ACQ435, located N,E,W,S at distance.
 * units each have a free running clock
 * start continuous capture on AFHBA broadcast TRG
 * instrument system by outputting TRG on FP GPIO at end of each cycle.
 * ./ACQPROC/configs/news32.json defines the whole system
 * ./ACQPROC/configs/n32.json defines a single-box minimal system

# Example Operation with single box:

## First, run a control
```
./scripts/acqproc_config_freerunning_acq435 --acq435SR 49999 acq2106_130

```
 * View result in cs-studio:
![GitHub](DOC/CONFIG.png)

## Now run the embedded control program
```
cd ACQPROC
VERBOSE=0 RTPRIO=10 NTRIGGERS=1 HW=1 ./acqproc_broadcast_trigger configs/n32.json 1000000
```

 * View result in cs-studio ... now the system is capturing data.

![Github](DOC/RUN.png)

 * View Live result in scope: C3 is the output trigger. C2 is the actual signal, zero crossing detected at -820 us
 * GD=39 * 20us = 780 us
 * Overhead of teeing up and broadcasting the marker is O(10us)
 * => overhead of the HOST SW < 30usec
 * Assume 4 x free running boxes, all samples collected at the same time: skew = +/-1 sample (20usec)
 * Worst case latency: 810+20 = 830usec.

![Github](DOC.RESULT.png)

 * Measuring the broadcast trigger overhead ... scope shows NTRIGGERS=3 group of 3 at 10usec spacing..
```
VERBOSE=0 RTPRIO=10 NTRIGGERS=3 HW=1 ./acqproc_broadcast_trigger configs/n32.json 1000000
```
![Github](TRIGGEROVERHEAD.png)

# Operation with 4 boxes

## Check config with no hardware
```
RTPRIO=10 NTRIGGERS=1 HW=1 ./acqproc_broadcast_trigger configs/news32.json 1000000

```
## Run with hardware
```
RTPRIO=10 NTRIGGERS=1 HW=1 ./acqproc_broadcast_trigger configs/news32.json 1000000
``` 


Please try it. Send questions to peter dot milne@d-tacq.com
