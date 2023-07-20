#!/usr/bin/env python3

import acq400_hapi
from acq400_hapi import afhba404
import argparse
import time

"""
usage: afhba_check_link.py [-h] [uutnames ...]

Afhba Link Checker

positional arguments:
  uutnames    uuts to check leave empty to check all connections

options:
  -h, --help  show this help message and exit
"""

def get_parser():
    parser = argparse.ArgumentParser(description='Afhba Link Checker')
    parser.add_argument('uutnames', nargs='*', help="uuts to check. Omit to check all connections")
    return parser

def run_main(args):
    check_lanes(args.uutnames)

def check_lanes(uuts):
    for conn in afhba404.get_connections().values():
        if conn.uut in uuts or len(uuts) == 0:
            check_lane_status(conn.uut, conn.dev, conn.cx)

def check_lane_status(uutname, lport, rport, uut=None, verbose=True):
    def output(msg):
        if verbose:
            print(msg)
    link_state = afhba404.get_link_state(lport)
    if link_state.LANE_UP and link_state.RPCIE_INIT:
        output(f"[{uutname}:{rport} -> afhba.{lport}] Link Good")
        return True
    if not uut:
        uut = acq400_hapi.factory(uutname)
    comm_api = getattr(uut, f'c{rport}')
    if not hasattr(comm_api, 'TX_DISABLE'):
        output(f"[{uutname}:{rport} -> afhba.{lport}] Link down: unable to fix (old firmware)")
        return False
    output(f"[{uutname}:{rport} -> afhba.{lport}] Link Down: attempting fix")
    attempt = 1
    while attempt <= 5:
        comm_api.TX_DISABLE = 1
        time.sleep(0.5)
        comm_api.TX_DISABLE = 0
        time.sleep(0.5)
        link_state = afhba404.get_link_state(lport)
        if link_state.RPCIE_INIT:
            output(f'[{uutname}:{rport} -> afhba.{lport}] Link Fixed')
            return True
        attempt += 1
    output(f'[{uutname}:{rport} -> afhba.{lport}] Link down: unable to fix')
    return False  

if __name__ == '__main__':
    run_main(get_parser().parse_args())