# text input to match afhba404-acq2106-llc-fat.pdf
some users report difficulty in cut and paste from pdf.

# 2 INSTALL

## Load Device Driver
[dt100@brotto AFHBA404]$ sudo ./scripts/loadNIRQ

## Check AFHBA404 serial number and Firmware version
[dt100@brotto AFHBA404]$ sudo ./scripts/afhba404-get-ident
brotto Device d1ac:4104 Device Serial Number af-ba-40-40-11-00-41-0b

## Check UUTS connected to the link.
[dt100@brotto AFHBA404]$ sudo ./scripts/get-ident-all
brotto 0 acq2106_110 A
brotto 1 acq2106_110 B
## In this case, our UUT is acq2106_110, and, for demonstration purposes we connected TWO fiber optic links:
DEVNUM=0, LPORT=A, UUT=ACQ2106_110, RPORT=A
DEVNUM=1, LPORT=B, UUT=ACQ2106_111, RPORT=B
Where:
DEVNUM : Linux device number 01,2,3 maps to LPORT : local PORT (SFP) A,B,C,D
UUT    : Unit Under Test: hostname, ideally, this is the network hostname as well
RPORT : remove PORT (SFP) A or B (C, D are NOT available for use at this time)
We demonstrate operation on either link. The scripting we demonstrate has been tested on
ALL of LPORT A,B,C,D and RPORT A,B


# 

