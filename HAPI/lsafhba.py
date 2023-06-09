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


def get_VI(uut, conn, args):
    NC = { 'AI16': 0, 'AI32': 0, 'DI32': 0, 'SP32': 0 }

    VI_cfg = {}

    for site in uut.sites:
        mod = uut.modules[site]

        mt = mod.MTYPE
        nchan = int(mod.NCHAN)
        is_adc = mod.is_adc.split(" ")[0] == '1'

        if is_adc:
            NC["AI{}".format(32 if mod.data32 == '1' else 16)] += nchan
        elif mt[0] == '6':
            NC['DI32'] += 1

    len_VI = 0
    for key, value in NC.items():
        if value > 0:
            len_VI += value * (2 if key == "AI16" else 4)
            VI_cfg[key] = value

    VI_cfg['SP32'] = (16*4 - len_VI%64) // 4

    XI_sites = 0
    for site_cat in ('AISITES', 'DIOSITES'):
        sc = uut.get_site_types()[site_cat]
        if len(sc) > 0:
            VI_cfg[site_cat] = sc
            XI_sites += len(sc)
    VI_cfg['NXI'] = XI_sites

    return VI_cfg

def get_VO(uut, conn, args):
    NC = { 'AO16': 0, 'AO32': 0, 'DO32': 0 }

    VO_cfg = {}

    for site in uut.sites:
        mod = uut.modules[site]

        mt = mod.MTYPE
        nchan = int(mod.NCHAN)
     
        if mt[0] == '4':
            NC["AO{}".format(32 if mod.data32 == '1' else 16)] += nchan
        elif mt[0] == '6':
            NC['DO32'] += 1

    len_VO = 0
    for key, value in NC.items():
        if value > 0:
            len_VO += value * (2 if key == "AO16" else 4)
            VO_cfg[key] = value

    XO_sites = 0
    for site_cat in ('AOSITES', 'DIOSITES'):
        sc = uut.get_site_types()[site_cat]
        if len(sc) > 0:
            VO_cfg['DO_BYTE_IS_OUTPUT'] = args.b_output.split(' ')
            VO_cfg[site_cat] = sc
            XO_sites += len(sc)
    VO_cfg['NXO'] = XO_sites

    return VO_cfg

def get_role(idx, values, args):
    if args.master:
        if values.uut == args.master:
            return 'master'
        return 'slave'
    if idx == 0:
        return 'master'
    return 'slave'

def run_main(args):
    conns = acq400_hapi.afhba404.get_connections()
    uuts = []

    for idx, value in conns.items():
        if args.lports:
            if value.dev not in args.lports.split(','):
                continue

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
            new_dev['VI'] = get_VI(uut, value, args)
            new_dev['VO'] = get_VO(uut, value, args)
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
    parser.add_argument('--master', default=None, help='uut to use as master')
    parser.add_argument('--lports', default=None, help='local ports to use ie 1,2,3')
    parser.add_argument('--b_output', default="1,1,0,0", help='DO_BYTE_IS_OUTPUT values 1,0,0,0 0,0,1,0 1,1,1,1')
    return parser

if __name__ == '__main__':
    run_main(get_parser().parse_args())
