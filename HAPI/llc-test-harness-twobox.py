#!/usr/bin/env python

"""
acq2106_llc-run-full-auto-two.py UUT1 UUT2
"""



import argparse
import acq400_hapi
import time
import os


EXTCLKDIV = int(os.getenv("EXTCLKDIV", "10"))
SIMULATE = os.getenv("SIMULATE", "")
AISITES = os.getenv("AISITES", "1,2,3,4,5,6")
AOSITES = os.getenv("AOSITES", "1,2")
# If you have two DIO cards then include them below
# even if one (or both) is a PWM.
DOSITES = os.getenv("DOSITES", "5,6")
XOCOMMS = os.getenv("XOCOMMS", "A")
PWMSITES = os.getenv("PWMSITES", "")

def hit_resets(svc):
    return None
    for knob in svc.help():
        if (knob.endswith('RESET')):
            svc.set_knob(knob, '1')

def clear_counters(uuts):
    return None # remove this to reset counters again
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
    uut.s0.spad = '1,16,0'
    uut.s0.aggregator = 'sites={}'.format(AISITES)
    uut.cA.spad = '1'
    uut.cA.aggregator = 'sites={}'.format(AISITES)
    uut.cB.spad = '1'
    uut.cB.aggregator = 'sites={}'.format(AISITES)

def init_ao(uut, slave=False):
    if len(AOSITES) == 0:
        print("AOSITES NOT INITIALIZED")
        # return
    aosites_list = AOSITES.split(',')
#    if len(aosites_list) == 0
    aom = "s{}".format(AOSITES.split(',')[0])
    if slave:
    	uut.svc[aom].clk = '1,1,1'
        uut.svc[aom].trg = '1,1,1'
    #    uut.svc[aom].CLKDIV = 1
        # uut.svc[aom].clkdiv = 1
        uut.svc[aom].lotide = 256
        # uut.s0.distributor = "sites={} comms=2 pad=0 on".format(AOSITES)
        uut.s0.distributor = "sites={} comms=A pad=0 on".format(AOSITES)
        print "DEBUG -- distributor"

    if PWMSITES != "":
        for pwmsite in PWMSITES.split(','):
            pwm = "s{}".format(pwmsite)
            uut.svc[pwm].pwm_clkdiv = "3e8" # 1000 in hex.

    if DOSITES != "":

        print "configuring for DO"
        for site in DOSITES.split(','):
            dom = "s{}".format(site)
            print "dom = ", dom
            # uut.svc[dom].clkdiv = 1
            uut.svc[dom].mode = "0"
            uut.svc[dom].lotide = "256"
            uut.svc[dom].byte_is_output = "1,1,1,1"
            uut.svc[dom].clk = "1,1,1"
            uut.svc[dom].trg = "1,0,1"
            #uut.svc[dom].trg = "0,0,0"
            #uut.svc[dom].clkdiv=67
            uut.svc[dom].mode = "1"
            #uut.svc[dom].mode = "2"

        XOSITES = "{},{}".format(AOSITES, DOSITES)
        #TCAN = str(16 - 1)
        TCAN = "15"
        # TCAN = str(0)
        print "XOSITES = {}, TCAN = {}, XOCOMMS = {}".format(XOSITES, TCAN, XOCOMMS)
        uut.s0.distributor = "sites={} pad={} comms={} on".format(XOSITES, TCAN, XOCOMMS)

    print("init_ao() done {} {}".format(uut.uut, aom))


def run_main(args):
    uuts = [ acq400_hapi.Acq2106(addr) for addr in args.uuts ]

    print("initialise {} uuts {}".format(len(uuts), args.uuts))
    clear_counters(uuts)
    init_ai(uuts[0])
    # if two boxes ASSUME second box AO
    if len(uuts) > 1:
        init_ao(uuts[1], slave=True)
    else:
     	print("init_ao {}".format(uuts[0].uut))
        init_ao(uuts[0], slave=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="aq2106_llc-run-full-auto-two.py")
    parser.add_argument("uuts", nargs='+', help="name the uuts")
    run_main(parser.parse_args())

