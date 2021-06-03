#!/usr/bin/env python

"""
A script that configures the aggregator and/or distributor depending on what
modules are inside the system.

Usage:
./llc-config-utility.py [uut name 1] [uut name 2] ... [uut name N]


Definitions:

VI : Vector Input [to HOST] :: AI + DI + SPAD     
VO : Vector Output [from HOST} :: AO + DO + TCAN

Where:
AI : Analog Input
DI : Digital Input
SPAD : Scratchpad (metadata), eg TLATCH at SPAD[0]

AO : Analog Output
DO : Digital Output
TCAN : Filler data to fill packet, discarded on ACQ2106

The DMA Engine DMAC lives on the ACQ2106
ACQ2106 PUSHES VI to the HOST
ACQ2106 PULLS VO from the HOST

On the ACQ2106 :

The AGGREGATOR collects a single sample VI comprising AI,DI,SPAD ("spad" control on the aggregator)
The DISTRIBUTOR farms out a single sample VO comprising AO,DO,TCAN ("pad" control on the distributor)

"""

from __future__ import print_function
import numpy
import acq400_hapi
import argparse
import json
import re


def get_devnum(args, uut):
    import subprocess
    hostname = uut.s0.HN

    try:
        pwd = subprocess.check_output(['pwd'])
        if pwd.decode("utf-8").split("/")[-1] == "AFHBA404\n":
            ident = subprocess.check_output(
                ['./scripts/get-ident-all']).decode("utf-8").split("\n")
            for item in ident:
                if hostname in item:
                    devnum = item.split(" ")[1]
                    break
                elif item == ident[-1]:
                    # if we have not matched by the last entry error out.
                    print(
                        "No AFHBA404 port populated by {}. Please check connections.".format(hostname))
                    exit(1)
        else:
            devnum = 0
    except Exception:
        print("Not in AFHBA404 directory. Defaulting to devnum = 0")
        devnum = 0
    return devnum


def _calculate_padding(vlen):
    number_of_bytes_in_a_long = 4
    return 16 - (vlen//number_of_bytes_in_a_long) % 16


def calculate_spad(xi_vector_length):
    return _calculate_padding(xi_vector_length)


def calculate_tcan(xo_vector_length):
    return _calculate_padding(xo_vector_length)


def calculate_vector_length(uut, SITES, DIOSITES, PWMSITES=None):
    vector_length = 0
    for site in SITES:
        nchan = int(uut.modules[site].get_knob('NCHAN'))
        data32 = int(uut.modules[site].get_knob('data32'))
        if data32:
            vector_length += (nchan * 4)
        else:
            vector_length += (nchan * 2)

    if DIOSITES:
        for site in DIOSITES:
            vector_length += 4

    if PWMSITES:
        for site in DIOSITES:
            vector_length += 64

    return vector_length


def config_sync_clk(uut):
    """
    Configures the MBCLK to be output on the front panel SYNC connector.
    """
    uut.s0.SIG_SRC_SYNC_0 = 'MBCLK'
    uut.s0.SIG_FP_SYNC = 'SYNC0'
    return None


def config_aggregator(args, uut, AISITES, DISITES, COMMS):
    # This function calculates the ai vector size from the number of channels
    # and word size of the AISITES argument provided to it and then sets the
    # spad, aggregator and NCHAN parameters accordingly.
    TOTAL_SITES = (AISITES + DISITES)
    TOTAL_SITES.sort()
    TOTAL_SITES = ','.join(map(str, TOTAL_SITES))
    print(TOTAL_SITES)

    ai_vector_length = calculate_vector_length(uut, AISITES, DISITES)

    # now check if we need a spad
    spad = calculate_spad(ai_vector_length)
    if spad != 0:
        uut.s0.spad = '1,{},0'.format(spad)
        uut.cA.spad = 1
        # uut.cB.spad = 1 commented out because this is NOT always true.

    print('Aggregator settings: sites={} spad={}'.format(TOTAL_SITES, spad))
    uut.s0.aggregator = 'sites={}'.format(TOTAL_SITES)
    uut.svc['c{}'.format(COMMS)].aggregator = 'sites={}'.format(TOTAL_SITES)    
    # uut.cB.aggregator = AISITES commented out because this is NOT always true.
    uut.s0.run0 = TOTAL_SITES
    return None


def config_distributor(args, uut, AOSITES, DOSITES, PWMSITES, COMMS):
    TOTAL_SITES = (AOSITES + DOSITES + PWMSITES)
    ao_vector = calculate_vector_length(uut, AOSITES, DOSITES, PWMSITES)
    TCAN = calculate_tcan(ao_vector)
    if TCAN == 16:
        # If the TCAN is 16 then we're just taking up space for no reason, so
        # set it to zero. The reason we don't need this for SPAD is that we
        # encode some useful information in there.
        TCAN = 0
    
    TOTAL_SITES.sort()
    TOTAL_SITES = ','.join(map(str, TOTAL_SITES))
    print(TOTAL_SITES)
    print('Distributor settings: sites={} pad={} comms={} on'.format(
        TOTAL_SITES, TCAN, COMMS))
    uut.s0.distributor = 'sites={} pad={} comms={} on'.format(
        TOTAL_SITES, TCAN, COMMS)

    return None


def config_VI(args, uut, AISITES, DISITES, sod=False, COMMS='A'):
    uut.s0.SIG_SYNC_OUT_CLK_DX = 'd2'
    if args.us == 1:
        trg = uut.s1.trg.split(" ")[0].split("=")[1]
        uut.s0.spad1_us = trg  # set the usec counter to the same as trg.
    if args.lat == 1:
        uut.s0.LLC_instrument_latency = 1
    if args.fp_sync_clk == 1:
        config_sync_clk(uut)
    if sod:
        for site in AISITES:
            uut.modules[site].sod = 1
    config_aggregator(args, uut, AISITES, DISITES, COMMS)


def config_VO(args, uut, AOSITES, DOSITES, PWMSITES, MSITES, COMMS='A'):
    if len(MSITES):
        signal = acq400_hapi.sigsel(site=int(MSITES[0]))
        signal2 = signal
    elif len(AOSITES):
        signal = acq400_hapi.sigsel()
        signal2 = acq400_hapi.sigsel(site=AOSITES[0])
    else:
        signal2 = signal = acq400_hapi.sigsel()

    for idx, site in enumerate(AOSITES):
        uut.modules[site].lotide = 256
        if idx ==0:
            if len(MSITES):
                uut.modules[site].clk = signal
                uut.modules[site].CLKDIV = 1
            
                
        signal = signal2

    for site in DOSITES:
        uut.modules[site].mode = '0'
        uut.modules[site].lotide = '256'
        uut.modules[site].byte_is_output = '1,1,0,0'  # @@todo should be json configured.
        uut.modules[site].mode = '1'
        signal = signal2

    for site in PWMSITES:
        uut.modules[site].pwm_clkdiv = '%x' % 1000

    config_distributor(args, uut, AOSITES, DOSITES, PWMSITES, COMMS)


def load_json(json_file):
    with open(json_file) as _json_file:
        json_data = json.load(_json_file)
    return json_data


# @@todo ... this is all backwards. Here we build the model from the ACTUAL UUT, but the goal is to build the model from the DATAFILE, then validate the actual HW.
# temp fix: assume ALL DIO32 are DI if DI32 specified, assume ALL DIO32 are DO if DO32 specified.
def enum_sites(uut, uut_json):
    AISITES = []
    DISITES = []
    AOSITES = []
    DOSITES = []
    PWMSITES = []
    for site in sorted(uut.modules):
        try:
            module_name = uut.modules[site].get_knob('module_name')
            if module_name.startswith('acq'):
                AISITES.append(site)
            elif module_name.startswith('ao'):
                AOSITES.append(site)
            elif module_name.startswith('dio'):

                if uut.modules[site].get_knob('module_type') == '61':
                    if 'DI32' in uut_json['VI'].keys() or args.include_dio_in_aggregator:
                        if not 'DI32' in uut_json['VI'].keys():
                            print("WARNING: deprecated DI32 requested but DI32 not in config file")
                        DISITES.append(site)
                    if 'DO32' in uut_json['VI'].keys():
                        DOSITES.append(site)
                if uut.modules[site].get_knob('module_type') == '6B':
                    module_variant = int(
                        uut.modules[site].get_knob('module_variant'))
                    if module_variant in [1, 2]:
                        PWMSITES.append(site)

                        if uut_json != None:
                            if not "PW32" in uut_json['VO'].keys():
                                print(
                                    "Warning: PWM site found, but no PWM specified in json.")
                    elif uut_json != None:
                        if "PW32" in uut_json['VO'].keys():
                            print(
                                "Warning: PWM included in json configuration but site is NOT a PWM.")
        except Exception as err:
            print("ERROR: ", err)
            continue
    return AISITES, DISITES, AOSITES, DOSITES, PWMSITES


def get_json_vo_len(uut_json):
    AO = 0
    DO = 0
    if 'AO16' in uut_json['VO'].keys():
        AO += uut_json['VO']['AO16'] * 2
    if 'DO32' in uut_json['VO'].keys():
        DO += uut_json['VO']['DO32'] * 4
    return AO + DO


def get_json_vi_len(uut_json):
    AI = 0
    DI = 0
    SP = 0
    if 'AI16' in uut_json['VI'].keys():
        AI += uut_json['VI']['AI16'] * 2
    if 'DI32' in uut_json['VI'].keys():
        DI += uut_json['VI']['DI32'] * 4
    if 'SP32' in uut_json['VI'].keys():
        SP += uut_json['VI']['SP32'] * 4
    return AI + DI + SP

def get_comms(uut_json):
    if 'COMMS' in uut_json.keys():
        return uut_json['COMMS']
    else:
        return 'A'

def verify_json_file(AISITES, AOSITES, DISITES, DOSITES, PWMSITES, uut_json, uut, uut_name):
    CRED = "\x1b[1;31m"
    CEND = "\33[0m"
    agg_vector = calculate_vector_length(uut, AISITES, DISITES, PWMSITES)
    spad = calculate_spad(agg_vector)
    total_agg_vector = agg_vector + (spad * 4)
    dist_vector = calculate_vector_length(uut, AOSITES, DOSITES, PWMSITES)

    json_agg_vector = get_json_vi_len(uut_json)
    json_dist_vector = get_json_vo_len(uut_json)

    if total_agg_vector != json_agg_vector:
        print(CRED, "Problem found: UUT: {} aggregator different from JSON. Please check JSON.".format(uut_name), CEND)
        print(CRED, "Aggregator vector: {}, json aggregator vector: {}".format(total_agg_vector, json_agg_vector), CEND)
    if dist_vector != json_dist_vector:
        print(CRED, "Problem found: UUT distributor different from JSON. Please check JSON.", CEND)
        print(CRED, "Distributor vector: {}, json distributor vector: {}".format(dist_vector, json_dist_vector), CEND)
    return None


def read_knob(k):
    with open(k) as fp:
        return fp.read().replace('\n', '')

def check_link(uut_def, dev_num):
    uut_name = uut_def['name']
    link_uut = read_knob("/dev/rtm-t.{}.ctrl/acq_ident".format(dev_num))
    if link_uut == uut_name:
        link_port = read_knob("/dev/rtm-t.{}.ctrl/acq_port".format(dev_num))
        if link_port != get_comms(uut_def):            
            print("WARNING: json specifies uut {} port {}, we have {}, going with it".
                                        format(uut_name, get_comms(uut_def), link_port))
        return link_port
    else:
        print("ERROR: json specifies uut {} but we have {}".format(uut_name, link_uut))
    
     
    sys.exit(1)      

def config_auto(args, uut_def, dev_num):
    comms = check_link(uut_def, dev_num)
    
    uut_name = uut_def['name']
    
    
    uut = acq400_hapi.Acq2106(uut_name)

    AISITES, DISITES, AOSITES, DOSITES, PWMSITES = enum_sites(uut, uut_def)
    sod = True if 'sod' in uut_def['type'] else False
    
    verify_json_file(AISITES, AOSITES, DISITES, DOSITES, PWMSITES, uut_def, uut, uut_name)

    if len(AISITES) != 0 or len(DISITES) != 0:
        config_VI(args, uut, AISITES, DISITES, sod, comms)

    if len(DOSITES) != 0 or len(AOSITES) != 0 or len(PWMSITES) != 0:
        config_VO(args, uut, AOSITES, DOSITES, PWMSITES, AISITES, comms)

    return None


def get_args():
    parser = argparse.ArgumentParser(description='Auto LLC config tool.')

    parser.add_argument('--include_dio_in_aggregator', type=int, default=0,
                        help='Since DIO cards can be used as input or output we can decide whether'
                        'or not to include them in the aggregator set.')

    parser.add_argument('--us', type=int, default=1,
                        help='Whether or not to set the microsecond counter.')

    parser.add_argument('--lat', type=int, default=1,
                        help='Whether or not to set the latency statistics on the acq2106.')

    parser.add_argument('--fp_sync_clk', type=int, default=0,
                        help="sync clock output to FP for monitoring")

    parser.add_argument('--json_file', type=str, default=None,
                        help='Where to load the json file from.')

    parser.add_argument('jsfile', help="configuraton file")

    args = parser.parse_args()
    if not args.json_file:
        args.json_file = args.jsfile
    
    if args.include_dio_in_aggregator:
        print("DEPRECATED: please define DI32 in config file instead")
        
    return args 

def update_dev_num(dev_num, uut_def): 
    try:
        dev_num = int(uut_def['DEVNUM'])
    except:
        pass

    return dev_num
      
def run_main():
    args = get_args()
        
    json = load_json(args.json_file)
    
    dev_num = update_dev_num(0, json)                   # maybe global dev_num, else 0
            
    uut_json = [uut for uut in json['AFHBA']['UUT']]
    
        #print("DEBUG: {}".format(uut_json))
    for uut in uut_json:
        dev_num = update_dev_num(dev_num, uut)          # maybe uut specific dev num, else current
        config_auto(args, uut, dev_num)
        dev_num += 1                                    # increment dev_num default
    return None


if __name__ == '__main__':
    run_main()
