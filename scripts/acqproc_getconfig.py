#!/usr/bin/env python3

"""
Script parses ACQPROC config file to create environment variables UUT1 [UUT2...] UUTS

Usage: ./scripts/acqproc_getconfig.py llc_configs/config1.json

"""

import argparse
import json

def parse_config(args):
    with open(args.config) as f:
        jdata = json.load(f)

    uuts = []
    sync_roles = []
    imax = 1
    has_master = False

    for uut in jdata['AFHBA']['UUT']:

        name = uut['name']

        if 'sync_role' in uut:
            sync_role = uut['sync_role']
        else:
            sync_role = 'slave' if has_master else 'master'

        if 'master' in sync_role:
            uuts.insert(0, name)
            sync_roles.insert(0, sync_role)
            has_master = True
        else:
            uuts.append(name)
            sync_roles.append(sync_role)

        imax += 1

    with open(args.env, "w") as f:
        f.write(f"# parse_config {args.config}\n")
        for ii, name in enumerate(uuts):
            f.write(f'UUT{ii+1}={name}\n')
        uuts = " ".join(uuts)
        sync_roles = " ".join(sync_roles)
        f.write(f'SYNC_ROLES="{sync_roles}"\n')
        f.write(f'UUTS="{uuts}"\n')
        f.write(f'DEVMAX={imax}\n')

def run_main():
    parser = argparse.ArgumentParser(description='acqproc_getconfig')
    parser.add_argument('config', help="path to config file")
    parser.add_argument('--env', default='acqproc_multi.env', help="path to env file")
    parse_config(parser.parse_args())

if __name__ == '__main__':
    run_main()
