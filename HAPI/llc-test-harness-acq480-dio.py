#!/usr/bin/env python

""" 
./llc-test-harness-ai4-dio32.py UUT1 
"""



import argparse
import acq400_hapi
import time
import os


EXTCLKDIV = int(os.getenv("EXTCLKDIV", "10"))
SIMULATE = os.getenv("SIMULATE", "")
AISITES = os.getenv("AISITES", "1,2")
AOSITES = os.getenv("AOSITES", "")
DOSITES = os.getenv("DOSITES", "6")

def hit_resets(svc):    
    for knob in svc.help():
        if (knob.endswith('RESET')):
            svc.set_knob(knob, '1')            
            
def clear_counters(uuts):
    for uut in uuts:
        for cx in [ 'cA', 'cB']:
            hit_resets(uut.svc[cx])
            
def init_common(uut):
    uut.s1.CLKDIV = EXTCLKDIV
    uut.s1.clkdiv = EXTCLKDIV
    
def init_ai(uut):
    init_common(uut)
    
    for s in uut.modules:
        uut.modules[s].simulate = '1' if str(s) in SIMULATE else '0'
    uut.cA.aggregator = 'sites={}'.format(AISITES)
    uut.cB.aggregator = 'sites={}'.format(AISITES)

def init_ao(uut, slave=False):
    aom = "s{}".format(AOSITES.split(',')[0])
    if slave:
	uut.svc[aom].clk = '1,2,1'
    uut.svc[aom].trg = '1,1,1'
    uut.svc[aom].CLKDIV = 10
    uut.svc[aom].clkdiv = 10
    uut.svc[aom].lotide = 256
     
    npad = 0
    uut.s0.distributor = "sites={} comms=2 pad={} on".format(DOSITES, npad)
    print("init_ao() done {} {}".format(uut.uut, aom))

def init_do(uut, slave=False):
    for site in DOSITES:
	sn = "s{}".format(site)
        uut.svc[sn].clk = '1,2,1'
	uut.svc[sn].trg = '1,1,1'
#    uut.svc[sn].CLKDIV = 1
        uut.svc[sn].clkdiv = 10
        uut.svc[sn].lotide = 256
        uut.svc[sn].byte_is_output = '1,1,1,1'
        uut.svc[sn].mode = 1

    npad = 0
    uut.s0.distributor = "sites={} comms=2 pad={} on".format(DOSITES, npad)
    print("init_ao() done {} {}".format(uut.uut, DOSITES))
     
def run_main(args):
    uuts = [ acq400_hapi.Acq2106(addr) for addr in args.uuts ]
    uut = uuts[0]
   
    print("llc-test-harness-acq480-dio432.py uut {}".format(uut.uut))
    clear_counters(uuts)    
    init_ai(uut)
    init_do(uut)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="llc-test-harness-acq480-dio432.py")
    parser.add_argument("uuts", nargs=1, help="name the uut")
    run_main(parser.parse_args())

