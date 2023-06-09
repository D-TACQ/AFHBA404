#!/usr/bin/env python3
'''
lsafhba : list all afhba connections
--save_config SYSNAME  :: save an auto-generator ACQPROC config file
Created on 18 Feb 2022

@author: pgm
'''

import argparse
import acq400_hapi
import json


def mtype(mod):
#    print("is_adc:{}".format(mod.is_adc))
    mt = "none"
    if mod.is_adc.split(" ")[0]=='1':
        mt = "AI"
    elif mod.MTYPE[0] == '4':
        mt = "AO"
    elif mod.MTYPE[0] == '6':
        mt = "DI"
    else:
        mt = "UN"
    return "{}{}".format(mt, 32 if mod.data32=='1' else 16)

def get_vi(uut, conn, args):
    NC = { 'AI16': 0, 'AI32': 0, 'DI32': 0, 'SP32': 0 }

    vi = {}

    for site in uut.sites:
        mod = uut.modules[site]

        mt = mod.MTYPE
        nchan = int(mod.NCHAN)
        d32 = mod.data32 == '1'
        is_adc = mod.is_adc.split(" ")[0] == '1'

        if is_adc:
            NC["AI{}".format(32 if d32 else 16)] += nchan
        elif mt[0] == '6':
            NC['DI32'] += 1

    len_vi = 0
    for key, value in NC.items():
        if value > 0:
            len_vi += value * (2 if key == "AI16" else 4)
            vi[key] = value

    vi['SP32'] = (16*4 - len_vi%64) // 4

    xi_sites = 0
    for site_cat in ('AISITES', 'DIOSITES'):
        sc = uut.get_site_types()[site_cat]
        if len(sc) > 0:
            vi[site_cat] = sc
            xi_sites += len(sc)
    vi['NXI'] = xi_sites

    return vi

def get_vo(uut, conn, args):
    NC = { 'AO16': 0, 'AO32': 0, 'DO32': 0 }

    vo = {}

    for site in uut.sites:
        mod = uut.modules[site]

        mt = mod.MTYPE
        nchan = int(mod.NCHAN)
        d32 = mod.data32 == '1'

        if mt[0] == '4':
            NC["AO{}".format(32 if d32 else 16)] += nchan
        elif mt[0] == '6':
            NC['DO32'] += 1

    len_vo = 0
    for key, value in NC.items():
        if value > 0:
            len_vo += value * (2 if key == "AO16" else 4)
            vo[key] = value

    xo_sites = 0
    for site_cat in ('AOSITES', 'DIOSITES'):
        sc = uut.get_site_types()[site_cat]
        if len(sc) > 0:
            vo[site_cat] = sc
            xo_sites += len(sc)
    vo['NXO'] = xo_sites

    return vo

def get_role(idx, values, args):
    if idx == 0:
        return 'master'
    return 'slave'

def run_main(args):
    conns = acq400_hapi.afhba404.get_connections()
    uuts = []

    for idx, value in conns.items():

        uut = acq400_hapi.factory(value.uut)

        print(f"{value.dev} {value}")
        if args.verbose > 0:
            sites = uut.sites
            print(f"\tpayload: {sites}")
            for site in sites:
                mod = uut.modules[int(site)]
                nchan = mod.NCHAN
                model = mod.MODEL.split(" ")[0]
                print(f"\t\tsite:{site} MODULE {mtype(mod)} {model} {nchan}")

        if args.save_config:
            new_dev = {}
            new_dev['DEVNUM'] = int(value.dev)
            new_dev['name'] = value.uut
            new_dev['type'] = 'pcs'
            new_dev['sync_role'] = get_role(idx, value, args)
            new_dev['COMMS'] = value.cx
            new_dev['VI'] = get_vi(uut, value, args)
            new_dev['VO'] = get_vo(uut, value, args)
            uuts.append(new_dev)

    if args.save_config:
        config = {}
        config['AFHBA'] = {}
        config['AFHBA']['UUT'] = uuts
        config = json.dumps(config, indent=4)
        if args.verbose > 0:
            print('Json:')
            print(config)
        with open(args.save_config, "w") as file:
            file.write(config)
            print(f"created {args.save_config}")

def get_parser():
    parser = argparse.ArgumentParser(description='list all attached acq2x06 devices')
    parser.add_argument('--save_config', default=None, help='save configuration skeleton')
    parser.add_argument('--verbose', default=0, type=int, help='increase verbosity')
    return parser

if __name__ == '__main__':
    run_main(get_parser().parse_args())
