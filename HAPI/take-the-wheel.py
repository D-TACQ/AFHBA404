#!/usr/bin/env python3

"""
./HAPI/take-the-wheel

If all connected UUTS are on the same remote PORT, switch all distributors
to source from that port.

Use Case:
- 4UUTS, Port A -> HOSTA
- 4UUTS, Port B -> HOSTB

HOSTA runs "take-the-wheel" : HOSTA takes control of the output
HOSTB runs "take-the-wheel" : HOSTB takes control of the output

Typically HOSTA, HOSTB are shadow systems (Prod, Dev perhaps).
Typically HOSTA, HOSTB will receive identical sites, spad

A single common shot config script could run on either of HOSTA or HOSTB
(or indeed be embedded in the UUT power up), and then the only run time config iit to determine who "takes the wheel"

"""

import argparse
import acq400_hapi

import subprocess
from collections import namedtuple

def get_connections():
    conns = {}
    p = subprocess.Popen(["./scripts/get-ident-all", ""], \
             stdout=subprocess.PIPE, stderr=subprocess.PIPE, \
             universal_newlines=True)
    output, errors = p.communicate()
    fields = "host", "dev", "uut", "cx"
    HostComms = namedtuple('HostComms', " ".join(fields))
    for ii, ln in enumerate(output.split('\n')):
        lns = ln.split(' ')
        if len(lns) == 4:
            record = HostComms(**dict(zip(fields, ln.split(' '))))
            conns[ii] = record
    return conns

def take_the_wheel(uuts, cx):
    for uutname in uuts:
        uut = acq400_hapi.factory(uutname)
        uut.s0.distributor = "comms={}".format(cx)

def run_main(args):
    conns = get_connections()
    cx = None
    uuts = []

    for key, value in conns.items():
       print("{} {}".format(key, value))
       if not cx:
           cx = value.cx
       elif cx != value.cx:
           print("ERROR: mixed comms on this host")
           return -1
       uuts.append(value.uut)

    take_the_wheel(uuts, cx)
    return 0

   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="take-the-wheel")
    run_main(parser.parse_args())

