#!/usr/bin/env python

"""
thomson_config.py UUT1 UUT2 UUT3 UUT4  
UUT1 is MASTER
"""


import argparse
import acq400_hapi
import os

# https://gist.github.com/endolith/114336/eff2dc13535f139d0d6a2db68597fad2826b53c3
def gcd(a,b):
    """Compute the greatest common divisor of a and b"""
    while b > 0:
        a, b = b, a % b
    return a
    
def lcm(a, b):
    """Compute the lowest common multiple of a and b"""
    return a * b / gcd(a, b)


EXTCLKDIV = int(os.getenv("EXTCLKDIV", "100"))
EXT_UCLK = int(os.getenv("EXT_UCLK", "0"))
SIMULATE = os.getenv("SIMULATE", "")
AISITES = os.getenv("AISITES", "1,2,3,4,5,6")
XOCOMMS = os.getenv("XOCOMMS", "A")
PULSE_EDGE = int(os.getenv("PULSE_EDGE", "0"))
TRG_EDGE   = int(os.getenv("TRG_EDGE", "0"))
# Default thomson buffer length is 4096. Now user configurable.
BUFLEN = int(os.getenv("BUFLEN", "4096"))

def hit_resets(svc):
    for knob in svc.help():
        if (knob.endswith('RESET')):
            svc.set_knob(knob, '1')


def clear_counters(uuts):
    for uut in uuts:
        print("clear_counters {}".format(uut.uut))
        for cx in [ 's0', 'cA', 'cB']:
            hit_resets(uut.svc[cx])


def init_clks(uuts):
    print("init_clks")
    role = "master"

    for uut in uuts:
        uut.s0.SIG_SRC_TRG_0 = 'EXT' if role=="master" else 'HDMI'
        print("shot trigger role:{} value:{}".format(role, uut.s0.SIG_SRC_TRG_0))
        uut.s0.SIG_SYNC_OUT_TRG = 'TRG'
        uut.s0.SIG_SYNC_OUT_TRG_DX = 'd0'
        uut.s0.spad1_us = '1,0,{}'.format(TRG_EDGE)

        if role=="master":
            uut.s0.SIG_SRC_TRG_1 = 'FP_SYNC'

        uut.s0.SIG_SRC_CLK_0 = 'GPG0' if role=="master" else 'HDMI'
        print("adc clk_src role:{} value:{}".format(role, uut.s0.SIG_SRC_CLK_0))

        uut.s0.SIG_SYNC_OUT_CLK = 'CLK'
        uut.s0.SIG_SYNC_OUT_CLK_DX = 'd0'

        role = "slave"

#    uut.s1.CLKDIV = EXTCLKDIV
#    uut.s1.clkdiv = EXTCLKDIV
#    uut.s0.SIG_SYNC_OUT_CLK_DX = "d2"

GPGDX = 1               # output d0
FLATTOP = 5             # pulse width

def set_delay(uut, args):
    uut.s0.GPG_ENABLE = '0'
    stl = ""
    stl = stl + "+{},{}\n".format(args.delay,GPGDX)
    stl = stl + "+{},{}\n".format(FLATTOP,0)

    uut.load_gpg(stl)
    uut.s0.gpg_trg = '1,1,{}'.format(PULSE_EDGE)
    uut.s0.gpg_clk = '1,1,1'
    uut.s0.SIG_CLK_MB_SET = '10000000'
    uut.s0.GPG_MODE = 'LOOPWAIT'
    uut.s0.GPG_ENABLE = '1'


def init_ai(uut):
    for s in uut.modules:
        uut.modules[s].simulate = '1' if str(s) in SIMULATE else '0'
    uut.s0.spad = '1,16,0'
    uut.cA.spad = '1'
    uut.cA.aggregator = 'sites={}'.format(AISITES)
    uut.cB.spad = '1'
    uut.cB.aggregator = 'sites={}'.format(AISITES)
    print "TRG set now Sample On Demand SOD"
    uut.s1.sod = '1'
    uut.s1.trg = '0,0,0'
    uut.s1.clk = '1,0,1'
    uut.s0.run0
    ssb = int(uut.s0.ssb)
    uut.s0.bufferlen = BUFLEN


def set_ext_uclk_counter(uuts, enable):
    """
    Configure system to use an external clock source
    as the usec counter.
    """
    for num, uut in enumerate(uuts):

        if num == 0:
            uut.s0.SIG_SRC_SYNC_0 = 'GPG0'    # delyed SOD clock source
            uut.s0.SIG_SRC_CLK_0 = 'HDMI' if enable else 'INT01M'
        
        uut.s0.spad1_us_clk_src = 1           # always source us_clk from CLK.d1
        
        uut.s1.sync = '1,0,1'                 # SOD clk setup, pull CLK from SYNC.
        uut.s0.SIG_SYNC_OUT_SYNC_DX = 'd0'
        uut.s1.clk_from_sync = 1

    return None


def run_main(args):
    uuts = [ acq400_hapi.Acq2106(addr) for addr in args.uuts ]
    uutm = uuts[0]

    print("initialise {} uuts {}".format(len(uuts), args.uuts))

    if args.clear_counters:
        clear_counters(uuts)
    else:
        print("clear_counters() commented out (slow)")
    init_clks(uuts)
    set_delay(uutm, args)

    for uut in uuts:
        init_ai(uut)

    set_ext_uclk_counter(uuts,EXT_UCLK)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="thomson_config.py")
    parser.add_argument("--delay", type=int, default=5, help="delay in usecs (min 5)")
    parser.add_argument("--clear_counters", type=int, default=0, help="clear counters before shot (slow)")
    parser.add_argument("--verbose", type=int, default=0, help="verbosity, default 0")
    parser.add_argument("uuts", nargs='+', help="name the uuts")
    run_main(parser.parse_args())

