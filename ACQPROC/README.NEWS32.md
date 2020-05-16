# NEWS32 : example 4 x ACQ2106+ACQ435, located N,E,W,S at distance.
 * units each have a free running clock
 * start continuous capture on AFHBA broadcast TRG
 * instrument system by outputting TRG on FP GPIO at end of each cycle.
 * ./ACQPROC/configs/news32.json defines the whole system
 * ./ACQPROC/configs/n32.json defines a single-box minimal system

# Example Operation:

## First, run a control
```
./scripts/acqproc_config_freerunning_acq435 --acq435SR 49999 acq2106_130

```
View result in cs-studio:
![GitHub](DOC/CONFIG.png)
 
