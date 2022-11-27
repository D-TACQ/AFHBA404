#!/usr/bin/env python3

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


TCAN: because the UUT DISTRIBUTOR simply dumps the padding (needed for %64 byte alignment. 
NEW: pad data can become "HUDP RELAY" for onward transmission on UDP.

"""

from __future__ import print_function
import numpy
import acq400_hapi
import argparse
import json
import re
import sys


def _calculate_padding(vlen):
    number_of_bytes_in_a_long = 4
    return 16 - (vlen//number_of_bytes_in_a_long) % 16


def calculate_spad(xi_vector_length):
    return _calculate_padding(xi_vector_length)


def calculate_tcan(xo_vector_length):
    return _calculate_padding(xo_vector_length)


def calculate_vector_length(uut, ASITES=None, DSITES=None, PWMSITES=None):
    vector_length = 0
    if ASITES:
        for site in ASITES:
            nchan = int(uut.modules[site].get_knob('NCHAN'))
            data32 = int(uut.modules[site].get_knob('data32'))
            vector_length += (nchan * (4 if data32 else 2))        

    if DSITES:
        for site in DSITES:
            vector_length += 4

    if PWMSITES:
        for site in PWMSITES:
            vector_length += 64

    return vector_length


def config_sync_clk(uut):
    """
    Configures the MBCLK to be output on the front panel SYNC connector.
    """
    uut.s0.SIG_SRC_SYNC_0 = 'MBCLK'
    uut.s0.SIG_FP_SYNC = 'SYNC0'
    return None


def config_aggregator(args, uut, COMMS):
    # This function calculates the ai vector size from the number of channels
    # and word size of the AISITES argument provided to it and then sets the
    # spad, aggregator and NCHAN parameters accordingly.
    TOTAL_SITES = (uut.AISITES + uut.DISITES)
    TOTAL_SITES.sort()
    TOTAL_SITES = ','.join(map(str, TOTAL_SITES))
#    print("config_aggregator({}) sites={}".format(uut.uut, TOTAL_SITES))

    ai_vector_length = calculate_vector_length(uut, ASITES=uut.AISITES, DSITES=uut.DISITES)

    # now check if we need a spad
    spadlen = calculate_spad(ai_vector_length)
    spad_en = 1 if spadlen > 0 else 0

    print('{} Aggregator settings: sites={} (spad={}) (comms={})'.format(uut.uut, TOTAL_SITES, spadlen, COMMS))
    uut.s0.aggregator = 'sites={}'.format(TOTAL_SITES)
    uut.s0.spad = '{},{},0'.format(spad_en, spadlen)
    if spadlen > 6:
        uut.s0.spad6 = COMMS + '6666' + uut.uut.split('_')[1]
        uut.s0.spad7 = COMMS + '7777' + uut.uut.split('_')[1]

    uut.svc['c{}'.format(COMMS)].aggregator = 'sites={}'.format(TOTAL_SITES)    
    uut.svc['c{}'.format(COMMS)].spad = spad_en

    uut.s0.run0 = TOTAL_SITES
    return None


def config_distributor(args, uut, COMMS):
    TOTAL_SITES = (uut.AOSITES + uut.DOSITES + uut.PWMSITES)
    ao_vector = calculate_vector_length(uut, ASITES=uut.AOSITES, DSITES=uut.DOSITES, PWMSITES=uut.PWMSITES)
    TCAN = calculate_tcan(ao_vector) + uut.HP32
    if TCAN == 16 and uut.HP32 == 0:
        # If the TCAN is 16 then we're just taking up space for no reason, so
        # set it to zero. The reason we don't need this for SPAD is that we
        # encode some useful information in there.
        TCAN = 0

    TOTAL_SITES.sort()
    TOTAL_SITES = ','.join(map(str, TOTAL_SITES))
#    print("config_distributor({}) sites={}".format(uut.uut, TOTAL_SITES))
    print('{} Distributor settings: sites={} pad={} comms={} on'.format(
        uut.uut, TOTAL_SITES, TCAN, COMMS))
    uut.s0.distributor = 'sites={} pad={} comms={} on'.format(
        TOTAL_SITES, TCAN, COMMS)
    uut.s0.dssb = ao_vector + TCAN*4
    
    if uut.HP32:
        uut.s10.slice_off = ao_vector
        uut.s10.slice_len = uut.HP32*4

    return None


def config_VI(args, uut, sod=False, COMMS='A'):
    uut.s0.SIG_SYNC_OUT_CLK_DX = 'd2'
    vis = []
    vis.extend(uut.AISITES)
    vis.extend(uut.DISITES)
    uut.VISITES = sorted(vis)

    if args.us == 1:
        trg = uut.modules[uut.VISITES[0]].trg.split(" ")[0].split("=")[1]
        uut.s0.spad1_us = trg  # set the usec counter to the same as trg.
    if args.lat == 1:
        uut.s0.LLC_instrument_latency = 1
    if args.fp_sync_clk == 1:
        config_sync_clk(uut)
    if sod:
        for site in AISITES:
            uut.modules[site].sod = 1
    config_aggregator(args, uut, COMMS)


def config_VO(args, uut, MSITES, COMMS='A'):
    if len(MSITES):
        signal = acq400_hapi.sigsel(site=int(MSITES[0]))
        signal2 = signal
    elif len(uut.AOSITES):
        signal = acq400_hapi.sigsel()
        signal2 = acq400_hapi.sigsel(site=uut.AOSITES[0])
    else:
        signal2 = signal = acq400_hapi.sigsel()

    for idx, site in enumerate(uut.AOSITES):
        uut.modules[site].lotide = 256
        if idx ==0:
            if len(MSITES):
                uut.modules[site].clk = signal
                uut.modules[site].CLKDIV = 1


        signal = signal2

    for idx, site in enumerate(uut.DOSITES):
        uut.modules[site].mode = '0'
        uut.modules[site].lotide = '256'
        uut.modules[site].byte_is_output = uut.DO_BYTE_IS_OUTPUT
        uut.modules[site].mode = '1'
        signal = signal2

    for site in uut.PWMSITES:
        uut.modules[site].pwm_clkdiv = '%x' % 1000

    config_distributor(args, uut, COMMS)


def load_json(json_file):
    with open(json_file) as _json_file:
        json_data = json.load(_json_file)
    return json_data


# default DO output config 16DO, 16DI

DO_BYTE_IS_OUTPUT_DEFAULT = '1,1,0,0'
# @@todo ... this is all backwards. Here we build the model from the ACTUAL UUT, but the goal is to build the model from the DATAFILE, then validate the actual HW.
# temp fix: assume ALL DIO32 are DI if DI32 specified, assume ALL DIO32 are DO if DO32 specified.
def enum_sites(uut, uut_def):
    uut.AISITES = []
    uut.DISITES = []
    uut.AOSITES = []
    uut.DOSITES = []
    uut.DO_BYTE_IS_OUTPUT = []
    uut.PWMSITES = []
    for site in sorted(uut.modules):
        try:
            module_name = uut.modules[site].get_knob('module_name')
            if module_name.startswith('acq'):
                uut.AISITES.append(site)
            elif module_name.startswith('ao'):
                uut.AOSITES.append(site)
            elif module_name.startswith('dio'):

                if uut.modules[site].get_knob('module_type') == '61':
                    if 'DI32' in uut_def['VI'].keys() or args.include_dio_in_aggregator:
                        if not 'DI32' in uut_def['VI'].keys():
                            print("WARNING: deprecated DI32 requested but DI32 not in config file")
                        uut.DISITES.append(site)
                    if 'DO32' in uut_def['VO'].keys():
                        uut.DOSITES.append(site)
                        uut.DO_BYTE_IS_OUTPUT.append(DO_BYTE_IS_OUTPUT_DEFAULT)
                if uut.modules[site].get_knob('module_type') == '6B':
                    module_variant = int(
                        uut.modules[site].get_knob('module_variant'))
                    if module_variant in [1, 2]:
                        uut.PWMSITES.append(site)

                        if uut_def != None:
                            if not "PW32" in uut_def['VO'].keys():
                                print(
                                    "Warning: PWM site found, but no PWM specified in json.")
                    elif uut_def != None:
                        if "PW32" in uut_def['VO'].keys():
                            print(
                                "Warning: PWM included in json configuration but site is NOT a PWM.")
        except Exception as err:
            print("ERROR: ", err)
            continue 



json_word_sizes = {
    'AI16': 2, 'AO16': 2,
    'AI32': 4, 'AO20': 4,
    'DI32': 4, 'DO32': 4,
    'HP32': 4,
    'PWM' : 64, 
    'SP32': 4   
}
def get_json_len(uut_def, vx, mt):
    try:
        return uut_def[vx][mt] * json_word_sizes[mt]
    except:
        return 0

def get_json_vx_len(uut_def, vx):
    vx_len = 0
    for mt in list(json_word_sizes):        
        vx_len += get_json_len(uut_def, vx, mt)
    return vx_len

def get_json_sites(uut_def, vx, xsite):
    if xsite in uut_def[vx].keys():
        return uut_def[vx][xsite]
    else:
        return None




def get_comms(uut_def):
    if 'COMMS' in uut_def.keys():
        return uut_def['COMMS']
    else:
        return 'A'

CRED = "\x1b[1;31m"
CBLU = "\x1b[1;34m"
CMAG = "\x1b[1;35m"
CEND = "\33[0m"

def json_override_actual(uut_def, uut_name, sites, vx, st):
    print("json_override_actual( {} {} {} {})".format(uut_name, sites, vx, st))
    if len(sites) == 0:
        return

    
    jsites = get_json_sites(uut_def, vx, st)
    print("json_override_actual jsites:{}".format(jsites))

    if jsites:
        sj = set(jsites)
        su = set(sites)

        if sj == su:
            pass
        elif sj.issubset(su):
            sites.clear()
            sites.extend(jsites)
            print(CBLU, "INFO: UUT: {} using subset of available {} sites {}".format(uut_name, st, sites), CEND)
        else:
            print(CRED, "ERROR: UUT: {} JSON {} not in actual set.".format(uut_name, st), CEND)
            sys.exit(1)
    else:
        print(CRED, "ERROR: UUT: {} JSON {} lacks {} list.".format(uut_name, vx, st), CEND)
        sys.exit(1)

def customize_HP32(uut, uut_def):
    hp32 = 0
    if 'HP32' in uut_def['VO'].keys():
        hp32 = int(uut_def['VO']['HP32'])
    uut.HP32 = hp32
    
def customize_DO_BYTE_IS_OUTPUT(uut, uut_def):
    try:
        if do_dir_def in uut_def['VO']['DO_BYTE_IS_OUTPUT']:
            for idx, module_dir in enumerate(do_dir_def):
                uut.DO_BYTE_IS_OUTPUT[idx] = module_dir
    except NameError:
        pass

def matchup_json_file(uut, uut_def, uut_name):
    agg_vector = calculate_vector_length(uut, ASITES=uut.AISITES, DSITES=uut.DISITES)
    spad = calculate_spad(agg_vector)
    total_agg_vector = agg_vector + (spad * 4)
    dist_vector = calculate_vector_length(uut, ASITES=uut.AOSITES, DSITES=uut.DOSITES, PWMSITES=uut.PWMSITES)

    json_agg_vector = get_json_vx_len(uut_def, 'VI')
    json_dist_vector = get_json_vx_len(uut_def, 'VO')

    if json_agg_vector > total_agg_vector:
        print(CRED, "ERROR: UUT: {} JSON VI {} greater than actual possible len {}.".format(uut_name, json_agg_vector, total_agg_vector), CEND)
        sys.exit(1)
    if total_agg_vector != json_agg_vector:
        json_override_actual(uut_def, uut_name, uut.AISITES, 'VI', 'AISITES')       
        json_override_actual(uut_def, uut_name, uut.DISITES, 'VI', 'DISITES')         

    if dist_vector != json_dist_vector:
        json_override_actual(uut_def, uut_name, uut.AOSITES, 'VO', 'AOSITES')
        json_override_actual(uut_def, uut_name, uut.DOSITES, 'VO', 'DIOSITES')               

    customize_HP32(uut, uut_def)
    customize_DO_BYTE_IS_OUTPUT(uut, uut_def)
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
            print(CMAG, "WARNING: json specifies uut {} port {}, we have {}, going with it".
                                        format(uut_name, get_comms(uut_def), link_port), CEND)
        return link_port
    else:
        print(CRED, "ERROR: json specifies uut {} but we have {}".format(uut_name, link_uut), CEND)


    sys.exit(1)      

def config_auto(args, uut_def, dev_num):
    comms = check_link(uut_def, dev_num)

    uut_name = uut_def['name']
    uut = acq400_hapi.factory(uut_name)

    enum_sites(uut, uut_def)
    sod = True if 'sod' in uut_def['type'] else False

    matchup_json_file(uut, uut_def, uut_name)

    if len(uut.AISITES) != 0 or len(uut.DISITES) != 0:
        config_VI(args, uut, sod, comms)

    if len(uut.DOSITES) != 0 or len(uut.AOSITES) != 0 or len(uut.PWMSITES) != 0:
        config_VO(args, uut, uut.AISITES, comms)

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

    parser.add_argument('jsfile', help="configuration file")

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

    uut_def = [uut for uut in json['AFHBA']['UUT']]     # uut_def : uut top node in config tree

        #print("DEBUG: {}".format(uut_def))
    for uut in uut_def:
        dev_num = update_dev_num(dev_num, uut)          # maybe uut specific dev num, else current
        config_auto(args, uut, dev_num)
        dev_num += 1                                    # increment dev_num default
    return None


if __name__ == '__main__':
    run_main()
