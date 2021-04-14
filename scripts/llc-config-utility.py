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


def calculate_vector_length(uut, SITES, DIOSITES, PWMSITES):
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


def config_aggregator(args, uut, AISITES, DIOSITES, PWMSITES):
    # This function calculates the ai vector size from the number of channels
    # and word size of the AISITES argument provided to it and then sets the
    # spad, aggregator and NCHAN parameters accordingly.
    if args.include_dio_in_aggregator:
        TOTAL_SITES = (AISITES + DIOSITES)
    else:
        TOTAL_SITES = AISITES
        DIOSITES = None

    TOTAL_SITES.sort()
    TOTAL_SITES = ','.join(map(str, TOTAL_SITES))
    print(TOTAL_SITES)

    ai_vector_length = calculate_vector_length(uut, AISITES, DIOSITES, PWMSITES)

    # now check if we need a spad
    spad = calculate_spad(ai_vector_length)
    if spad != 0:
        uut.s0.spad = '1,{},0'.format(spad)
        uut.cA.spad = 1
        # uut.cB.spad = 1 commented out because this is NOT always true.

    print('Aggregator settings: sites={} spad={}'.format(TOTAL_SITES, spad))
    uut.s0.aggregator = 'sites={}'.format(TOTAL_SITES)
    uut.cA.aggregator = 'sites={}'.format(TOTAL_SITES)
    # uut.cB.aggregator = AISITES commented out because this is NOT always true.
    uut.s0.run0 = TOTAL_SITES
    return None


def config_distributor(args, uut, DIOSITES, AOSITES, AISITES, PWMSITES):
    TOTAL_SITES = (AOSITES + DIOSITES + PWMSITES)
    ao_vector = calculate_vector_length(uut, AOSITES, DIOSITES, PWMSITES)
    TCAN = calculate_tcan(ao_vector)
    if TCAN == 16:
        # If the TCAN is 16 then we're just taking up space for no reason, so
        # set it to zero. The reason we don't need this for SPAD is that we
        # encode some useful information in there.
        TCAN = 0
    XOCOMMS = 'A' if len(AISITES) == 0 else 'A'
    TOTAL_SITES.sort()
    TOTAL_SITES = ','.join(map(str, TOTAL_SITES))
    print(TOTAL_SITES)
    print('Distributor settings: sites={} pad={} comms={} on'.format(
        TOTAL_SITES, TCAN, XOCOMMS))
    uut.s0.distributor = 'sites={} pad={} comms={} on'.format(
        TOTAL_SITES, TCAN, XOCOMMS)

    return None


def config_VI(args, uut, AISITES, DIOSITES, PWMSITES, sod=False):
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
    config_aggregator(args, uut, AISITES, DIOSITES, PWMSITES)


def config_VO(args, uut, DIOSITES, AOSITES, AISITES, PWMSITES):
    if len(AISITES):
        signal = acq400_hapi.sigsel(site=AISITES[0])
        signal2 = signal
    elif len(AOSITES):
        signal = acq400_hapi.sigsel()
        signal2 = acq400_hapi.sigsel(site=AOSITES[0])
    else:
        signal2 = signal = acq400_hapi.sigsel()

    for site in AOSITES:
        uut.modules[site].lotide = 256
        signal = signal2

    for site in DIOSITES:
        uut.modules[site].mode = '0'
        uut.modules[site].lotide = '256'
        uut.modules[site].byte_is_output = '1,1,0,0'
        uut.modules[site].mode = '1'
        signal = signal2

    for site in PWMSITES:
        uut.modules[site].pwm_clkdiv = '%x' % 1000

    config_distributor(args, uut, DIOSITES, AOSITES, AISITES, PWMSITES)


def load_json(json_file):
    with open(json_file) as _json_file:
        json_data = json.load(_json_file)
    return json_data


def enum_sites(uut, uut_json=None):
    AISITES = []
    AOSITES = []
    DIOSITES = []
    PWMSITES = []
    for site in uut.modules:
        try:
            module_name = uut.modules[site].get_knob('module_name')
            if module_name.startswith('acq'):
                AISITES.append(site)
            elif module_name.startswith('ao'):
                AOSITES.append(site)
            elif module_name.startswith('dio'):

                if uut.modules[site].get_knob('module_type') == '61':
                    DIOSITES.append(site)
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
    return AISITES, AOSITES, DIOSITES, PWMSITES


def config_auto(args, uut, uut_json=None):

    uut = acq400_hapi.Acq2106(uut)

    AISITES, AOSITES, DIOSITES, PWMSITES = enum_sites(uut, uut_json)
    sod = True if 'sod' in uut_json['type'] else False

    if len(AISITES) != 0:
        config_VI(args, uut, AISITES, DIOSITES, PWMSITES, sod)

    if len(DIOSITES) != 0 or len(AOSITES) != 0:
        config_VO(args, uut, DIOSITES, AOSITES, AISITES, PWMSITES)

    return None


def run_main():
    parser = argparse.ArgumentParser(description='Auto LLC config tool.')

    parser.add_argument('--include_dio_in_aggregator', type=int, default=0,
                        help='Since DIO cards can be used as input or output we can decide whether'
                        'or not to include them in the aggregator set.')

    parser.add_argument('--auto', type=int, default=1,
                        help='Whether or not to automatically configure the UUTs.')

    parser.add_argument('--us', type=int, default=1,
                        help='Whether or not to set the microsecond counter.')

    parser.add_argument('--lat', type=int, default=1,
                        help='Whether or not to set the latency statistics on the acq2106.')

    parser.add_argument('--fp_sync_clk', type=int, default=0,
                        help="sync clock output to FP for monitoring")

    parser.add_argument('--json_file', type=str, default='./test.json',
                        help='Where to load the json file from.')

    parser.add_argument('uuts', nargs='+', help="uuts")

    args = parser.parse_args()
    if args.auto == 1:
        json = load_json(args.json_file)
        uut_json = [uut for uut in json['AFHBA']['UUT']]
        #print("DEBUG: {}".format(uut_json))
        for index, uut in enumerate(args.uuts):
            config_auto(args, uut, uut_json[index])
    return None


if __name__ == '__main__':
    run_main()
