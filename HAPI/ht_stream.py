#!/usr/bin/env python3

"""High Throughput Stream from a UUT

    - data on local SFP/AFHBA
    - control on Ethernet
"""

import argparse
import os
from acq400_hapi import factory, afhba404

def run_main(args):

    lport, rport = get_ports(args)
    uut = factory(args.uut)
    outroot = os.path.join(f"/mnt/afhba.{lport}", args.uut)
    os.system(f"sudo mkdir -p {outroot} -m 0777")
    setup_aggregator(uut, rport)
    if args.nbuffers == 0: args.nbuffers = 9999999999

    env = f"RTM_DEVNUM={lport} NBUFS={args.nbuffers} CONCAT={args.concat} RECYCLE={args.recycle} OUTROOT={outroot}"
    os.system(f"{env} ./STREAM/rtm-t-stream-disk")

def get_ports(args):
    lport = None
    rport = None
    for idx, conn in afhba404.get_connections().items():
        if conn.uut == args.uut:
            if args.rport:
                if conn.cx == args.rport:
                    return (conn.dev, conn.cx)
            elif lport == None:
                lport = conn.dev
                rport = conn.cx
            else:
                exit('Error: multiple connections specify with --rport ')
    if not lport:
        exit('Error: Connection not found')
    return (lport, rport)

def setup_aggregator(uut, rport):
    sites = ','.join(uut.get_aggregator_sites())
    comm_site = getattr(uut, f'c{rport}')
    comm_site.aggregator = f"sites={sites} on"

def get_parser():
    parser = argparse.ArgumentParser(description='High Throughput Stream from a UUT')
    parser.add_argument('--concat', default=0, type=int, help="concatenate buffers zero indexed") 
    parser.add_argument('--recycle', default=1, type=int, help="overwite buffers, Warning disabling can exceed host memory") 
    parser.add_argument('--nbuffers', default=5000, type=int, help="Number of buffers, 0 for unlimited")    
    parser.add_argument('--rport', default=None, help="uut remote port")
    parser.add_argument('uut', help="uut hostname")
    return parser

if __name__ == '__main__':
    run_main(get_parser().parse_args())