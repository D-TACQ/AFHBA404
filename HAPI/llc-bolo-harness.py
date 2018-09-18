#!/usr/bin/env python

""" 
llc-bolo-harness.py UUT1 
"""



import argparse
import acq400_hapi
import time
import os


AISITES = os.getenv("AISITES", "1")


def hit_resets(svc):    
    for knob in svc.help():
        if (knob.endswith('RESET')):
            svc.set_knob(knob, '1')            
            
def clear_counters(uuts):
    for uut in uuts:
        for cx in [ 'cA', 'cB']:
            hit_resets(uut.svc[cx])
            
    
def init_ai(uut):
    bolo_chan_factor = 24
    spad_len = 8
    uut.s0.spad = '1,{},0'.format(spad_len)
    uut.cA.spad = '1'
    uut.cA.aggregator = 'sites={}'.format(AISITES)
    uut.cB.spad = '1'
    uut.cB.aggregator = 'sites={}'.format(AISITES)
    uut.s0.run0 = '{}'.format(AISITES)
    # we need to calculate a new nchan based on the number of bolo sites in
    # the uut. uut assumes bolo has 8 channels, but post dsp each site has 
    # 24 channels. 
    uut.s0.NCHAN = int(AISITES.split(",")[-1])*bolo_chan_factor+spad_len
    
def run_main(args):
    uuts = [ acq400_hapi.Acq2106(addr) for addr in args.uuts ]
   
    print("initialise {} uuts {}".format(len(uuts), args.uuts)) 
    clear_counters(uuts)    
    init_ai(uuts[0])
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="llc-bolo-harness.py")
    parser.add_argument("uuts", nargs='+', help="name the uuts")
    run_main(parser.parse_args())

