#!/usr/bin/env python

""" 
acq2106_llc-run-full-auto-two.py UUT1 UUT2
"""



import argparse
import acq400_hapi
import time

def hit_resets(svc):    
    for knob in svc.help():
        if (knob.endswith('RESET')):
            svc.set_knob(knob, '1')            
            
def clear_counters(uuts):
    for uut in uuts:
        for cx in [ 'cA', 'cB']:
            hit_resets(uut.svc[cx])
            
        
def run_main(args):
    uuts = [ acq400_hapi.Acq2106(addr) for addr in args.uuts ]
    uut1 = uuts[0]
    uut2 = uuts[1]
    
    clear_counters(uuts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="aq2106_llc-run-full-auto-two.py")
    parser.add_argument("uuts", nargs=2, help="name the uuts")
    run_main(parser.parse_args())

