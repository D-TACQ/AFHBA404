#!/usr/bin/env python

"""
thomson_config.py
"""


import argparse
import acq400_hapi
import os


EXTCLKDIV = int(os.getenv("EXTCLKDIV", "100"))
SIMULATE = os.getenv("SIMULATE", "")
AISITES = os.getenv("AISITES", "1,2,3,4,5,6")
XOCOMMS = os.getenv("XOCOMMS", "A")


def hit_resets(svc):
    for knob in svc.help():
        if (knob.endswith('RESET')):
            svc.set_knob(knob, '1')


def clear_counters(uuts):
    uutn = 0
    for uut in uuts:
        for cx in [ 'cA', 'cB']:
            hit_resets(uut.svc[cx])
        uut.s0.spad4 = "{}{}{}{}{}".format('4444', uutn, uutn, uutn, uutn)
        uut.s0.spad5 = "{}{}{}{}{}".format('5555', uutn, uutn, uutn, uutn)
        uut.s0.spad6 = "{}{}{}{}{}".format('6666', uutn, uutn, uutn, uutn)
        uut.s0.spad7 = "{}{}{}{}{}".format('7777', uutn, uutn, uutn, uutn)
        uutn += 1


def init_clks(uut):
    uut.s1.CLKDIV = EXTCLKDIV
    uut.s1.clkdiv = EXTCLKDIV
# use raw clock, avoid sync with slow sample clock
#    uut.s0.SIG_SYNC_OUT_CLK_DX = "d2"


def init_spad_us(uut):
    uut.s0.spad1_us = '1,2,1' 			# start counting on derived trigger


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
    parser = argparse.ArgumentParser(description="thomson_config.py")
    parser.add_argument("uuts", nargs='+', help="name the uuts")
    run_main(parser.parse_args())

