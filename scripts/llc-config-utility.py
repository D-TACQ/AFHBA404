#!/usr/bin/env python

"""
A script that configures the aggregator and/or distributor depending on what
modules are inside the system.

Usage:
./llc-config-utility.py [uut name 1] [uut name 2] ... [uut name N]
"""

from __future__ import print_function
import numpy
import acq400_hapi
import argparse


def config_sync_clk(uut):
    """
    Configures the MBCLK to be output on the front panel SYNC connector.
    """
    uut.s0.SIG_SRC_SYNC_0 = 'MBCLK'
    uut.s0.SIG_FP_SYNC = 'SYNC0'
    return None


def get_devnum(args, uut):
    import subprocess
    hostname = uut.s0.HN

    try:
        pwd = subprocess.check_output(['pwd'])
        if pwd.decode("utf-8").split("/")[-1] == "AFHBA404\n":
            ident = subprocess.check_output(['./scripts/get-ident-all']).decode("utf-8").split("\n")
            for item in ident:
                if hostname in item:
                    devnum = item.split(" ")[1]
                    break
                elif item == ident[-1]:
                    # if we have not matched by the last entry error out.
                    print("No AFHBA404 port populated by {}. Please check connections.".format(hostname))
                    exit(1)
        else:
            devnum = 0
    except Exception:
        print("Not in AFHBA404 directory. Defaulting to devnum = 0")
        devnum = 0
    return devnum


def calculate_spad(ai_vector_length):
    number_of_bytes_in_a_long = 4
    # Modulo 16 because we need multiples of 16 long words (64 bytes).
    remainder = (ai_vector_length / number_of_bytes_in_a_long) % 16
    spad = 16 - remainder
    return spad


def calculate_vector_length(uut, SITES, DIOSITES):
    vector_length = 0
    for site in SITES:
        nchan = int(eval('uut.s{}.NCHAN'.format(site)))
        data32 = int(eval('uut.s{}.data32'.format(site)))
        if data32:
            vector_length += (nchan * 4)
        else:
            vector_length += (nchan * 2)

    if DIOSITES:
        for site in DIOSITES:
            vector_length += 4

    return vector_length


def config_aggregator(args, uut, AISITES, DIOSITES):
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

    ai_vector_length = calculate_vector_length(uut, AISITES, DIOSITES)

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


def config_distributor(args, uut, DIOSITES, AOSITES, AISITES):

    for site in AOSITES:
        aom = "s{}".format(site)
        uut.svc[aom].lotide = 256

    for site in DIOSITES:
        dio = "s{}".format(site)
        uut.svc[dio].mode = '0'
        uut.svc[dio].lotide = '256'
        uut.svc[dio].byte_is_output = '1,1,0,0'
        uut.svc[dio].clk = '1,1,1'
        uut.svc[dio].trg = '1,0,1'
        uut.svc[dio].mode = '1'

    TOTAL_SITES = (AOSITES + DIOSITES)
    ao_vector = calculate_vector_length(uut, AOSITES, DIOSITES)
    TCAN = calculate_spad(ao_vector)
    if TCAN == 16:
        # If the TCAN is 16 then we're just taking up space for no reason, so
        # set it to zero. The reason we don't need this for SPAD is that we
        # encode some useful information in there.
        TCAN = 0
    # If there are AISITES in the system then use port B for AO+DO. Else port A.
    XOCOMMS = 'A' if len(AISITES) == 0 else 'A'
    TOTAL_SITES.sort()
    TOTAL_SITES = ','.join(map(str, TOTAL_SITES))
    print(TOTAL_SITES)
    print('Distributor settings: sites={} pad={} comms={} on'.format(TOTAL_SITES, TCAN, XOCOMMS))
    uut.s0.distributor = 'sites={} pad={} comms={} on'.format(TOTAL_SITES, TCAN, XOCOMMS)

    return None


def config_auto(args, uut):
    # vector_len =
    AISITES = []
    AOSITES = []
    DIOSITES = []
    uut = acq400_hapi.Acq2106(uut)

    for site in [1,2,3,4,5,6]:
        try:
            module_name = eval('uut.s{}.module_name'.format(site))
            if module_name.startswith('acq'):
                AISITES.append(site)
            elif module_name.startswith('ao'):
                AOSITES.append(site)
            elif module_name.startswith('dio'):
                DIOSITES.append(site)
        except Exception:
            continue

    if len(AISITES) != 0:
        config_aggregator(args, uut, AISITES, DIOSITES)
    # if len(AOSITES) != 0:
        # config_distributor(args, uut, AOSITES, DIO)
    if len(DIOSITES) != 0 or len(AOSITES) != 0:
        config_distributor(args, uut, DIOSITES, AOSITES, AISITES)

    config_sync_clk(uut)

    if args.us == 1:
        trg = uut.s1.trg.split(" ")[0].split("=")[1]
        uut.s0.spad1_us = trg # set the usec counter to the same as trg.

    if args.lat == 1:
        uut.s0.LLC_instrument_latency = 1

    if args.cmd == 1:
        # Create and print a representitive cpucopy command.
        # We need to iterate over the sites as s0 NCHAN now includes the spad.
        TOTAL_SITES = AISITES + DIOSITES if args.include_dio_in_aggregator else AISITES
        print("DEBUG = ", TOTAL_SITES)
        aichan = sum([int(getattr(getattr(uut, "s{}".format(site)), "NCHAN")) for site in AISITES])
        aochan = sum([int(getattr(getattr(uut, "s{}".format(site)), "NCHAN")) for site in AOSITES])
        nshorts = (aichan) + (2 * len(DIOSITES))
        DO32 = 1 if DIOSITES else 0
        spad_longs = uut.s0.spad.split(",")[1]
        tcan_longs = uut.s0.distributor.split(" ")[3].split("=")[1]
        devnum = get_devnum(args, uut)

        command = "DUP1=0 NCHAN={} AOCHAN={} DO32={} SPADLONGS={} DEVNUM={}" \
        " LLCONTROL/afhba-llcontrol-cpucopy" \
        .format(nshorts, aochan, DO32, spad_longs, devnum)

        print("Outbound vector composition: {} short words of AI, "
        "{} longword(s) of DI, and {} longwords of SPAD.".format(aichan*2, DO32, spad_longs))

        print("Inbound vector composition: {} short words of AO, "
        "{} longword(s) of DO, and {} longwords of TCAN.".format(aochan*2, DO32, tcan_longs))

        print("The scratchpad will start at position {} in the vector. \n".format(int(nshorts/2)))

        # print("\n", command, sep="")
        print("\n", command)

    return None


def run_main():
    parser = argparse.ArgumentParser(description='Auto LLC config tool.')

    parser.add_argument('--include_dio_in_aggregator', type=int, default=0,
    help='Since DIO cards can be used as input or output we can decide whether' \
    'or not to include them in the aggregator set.')

    parser.add_argument('--auto', type=int, default=1,
    help='Whether or not to automatically configure the UUTs.')

    parser.add_argument('--cmd', type=int, default=1,
    help='Whether or not to include an example cpucopy command for the system' \
    ' being configured')

    parser.add_argument('--us', type=int, default=1,
    help='Whether or not to set the microsecond counter.')

    parser.add_argument('--lat', type=int, default=1,
    help='Whether or not to set the latency statistics on the acq2106.')

    parser.add_argument('uuts', nargs='+', help="uuts")

    args = parser.parse_args()
    if args.auto == 1:
        for uut in args.uuts:
            config_auto(args, uut)
    return None


if __name__ == '__main__':
    run_main()
