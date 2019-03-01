#!/usr/bin/env python

"""
acq2106_llc-run-full-auto-two.py UUT1 UUT2
"""


import argparse
import acq400_hapi
import os


EXTCLKDIV = int(os.getenv("EXTCLKDIV", "100"))
SIMULATE = os.getenv("SIMULATE", "")
AISITES = os.getenv("AISITES", "1,2,3,4,5,6")
AOSITES = os.getenv("AOSITES", "1,2")
DOSITES = os.getenv("DOSITES", "5")
XOCOMMS = os.getenv("XOCOMMS", "A")


def hit_resets(svc):
    for knob in svc.help():
        if (knob.endswith('RESET')):
            svc.set_knob(knob, '1')


def clear_counters(uuts):
    for uut in uuts:
        for cx in [ 'cA', 'cB']:
            hit_resets(uut.svc[cx])


def init_clks(uut):
    uut.s1.CLKDIV = EXTCLKDIV
    uut.s1.clkdiv = EXTCLKDIV
    uut.s0.SIG_SYNC_OUT_CLK_DX = "d2"


def init_spad_us(uut):
    trg = uut.s1.trg
    trg = trg[4:9]
    print "trg = ", trg
    uut.s0.spad1_us = trg


def init_ai(uut):
    #init_common(uut)
    init_spad_us(uut)
    for s in uut.modules:
        uut.modules[s].simulate = '1' if str(s) in SIMULATE else '0'
    uut.s0.spad = '1,16,0'
    uut.cA.spad = '1'
    uut.cA.aggregator = 'sites={}'.format(AISITES)
    uut.cB.spad = '1'
    uut.cB.aggregator = 'sites={}'.format(AISITES)


def run_main(args):
    uuts = [ acq400_hapi.Acq2106(addr) for addr in args.uuts ]

    print("initialise {} uuts {}".format(len(uuts), args.uuts))
    clear_counters(uuts)
    init_clks(uuts[0])
    
    for uut in uuts[0:]:
        init_ai(uut)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="aq2106_llc-run-full-auto-two.py")
    parser.add_argument("uuts", nargs='+', help="name the uuts")
    run_main(parser.parse_args())

