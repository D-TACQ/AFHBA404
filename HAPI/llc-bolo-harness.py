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
    uut.cA.aggregator = 'sites={}'.format(AISITES)    
    uut.cB.aggregator = 'sites={}'.format(AISITES)    

        
def run_main(args):
    uuts = [ acq400_hapi.Acq2106(addr) for addr in args.uuts ]
   
    print("initialise {} uuts {}".format(len(uuts), args.uuts)) 
    clear_counters(uuts)    
    init_ai(uuts[0])
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="llc-bolo-harness.py")
    parser.add_argument("uuts", nargs='+', help="name the uuts")
    run_main(parser.parse_args())

