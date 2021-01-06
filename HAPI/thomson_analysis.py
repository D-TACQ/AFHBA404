#!/usr/bin/env python


import numpy as np
#import matplotlib.pyplot as plt
import argparse
import json

"""
Avoid this error on piping to head
close failed in file object destructor:
sys.excepthook is missing
lost sys.stderr
"""
import sys
#sys.excepthook = lambda *args: None


def check_column(data, col, total_cols, total_systems):
    sample_count = [[] for i in range(total_systems)]
    for system in range(0,total_systems):
        for count in data[col + (system * total_cols)::total_cols * total_systems]:
            sample_count[system].append(count)
    return sample_count


def plot_chan(data, chan, total_systems):
    plt.plot(data[chan::224 * total_systems])
    plt.show()


def fix_args(args):
    args.longs = args.AI/2 + args.SP
    args.shorts = args.longs*2
    args.SPIX = args.AI/2
    args.sp_plot = set(('IX', 'SC', 'US', 'DUS') if args.spcols == "ALL" else tuple(args.spcols.split(",")))
    try:
        args.ai_plot = [ (c-1) for c in tuple(map(int, args.aicols.split(",")))]
    except ValueError:
        args.ai_plot = None
    return args


def run_analysis(args):
    if args.combine != None:
        combine(args)
    short_data = np.fromfile(args.src, dtype=np.int16)
    long_data = np.fromfile(args.src, dtype=np.int32)

    sample_count = check_column(long_data, args.SPIX+0, args.longs, args.nuuts)
    usec_count = check_column(long_data,   args.SPIX+1, args.longs, args.nuuts)

    with open(args.capstat, "r") as capstat:
        print(capstat.readlines()[0])

    if args.ai_plot != None:
        aicols = []
        for col in args.ai_plot:
            aicols.append(check_column(short_data, col, args.shorts, args.nuuts))

    FF = "{:>8}"					# Field Format, right justify 8 chars
    UUTS = range(0, args.nuuts) if args.uutcols == None else [ (c-1) for c in tuple(map(int, args.uutcols.split(',')))]

    for num, counter in enumerate(sample_count[0]):
	hc = []
	rc = []						# report columns 
        if 'IX' in args.sp_plot:
            if num == 0:
		hc.append(FF.format('IX'))
	    rc.append(FF.format(num))

	if 'SC' in args.sp_plot:
            for uut in UUTS:
                if num == 0:
                    hc.append(FF.format('SC.{}'.format(uut)))
	        rc.append(FF.format(sample_count[uut][num]))
        if 'US' in args.sp_plot:
            for uut in UUTS:
                if num == 0:
                    hc.append(FF.format('US.{}'.format(uut)))
                rc.append(FF.format(usec_count[uut][num]))
        if 'DUS' in args.sp_plot:
            for uut in UUTS:
                if num == 0:
                    hc.append(FF.format('DUS.{}'.format(uut)))
                rc.append(FF.format(0 if num < 2 else usec_count[uut][num]-usec_count[uut][num-1]))

        if args.ai_plot != None:
            for uut in UUTS:
	        for col, ch in enumerate(args.ai_plot):
                    if num == 0:
                        hc.append(FF.format('{}.CH{:03}'.format(uut, col+1)))
                    rc.append(FF.format(aicols[col][uut][num]))
        try:
            if num == 0:
                print(",".join(hc))
	    print(",".join(rc))
            sys.stdout.flush()
        except IOError:
            break

    return None


def get_uut_info(uut_json):
    uuts = []
    longwords = 0
    for uut in uut_json["AFHBA"]["UUT"]:
        uuts.append(uut["name"])
        longwords = int(uut["VI"]["SP32"]) + int((uut["VI"]["AI16"]) / 2)
    return longwords, uuts


def load_json(json_file):
    with open(json_file) as _json_file:
        json_data = json.load(_json_file)
    return json_data


def combine(args):

    uut_json = load_json(args.combine)
    longwords, uuts = get_uut_info(uut_json)

    data = []
    for uut in uuts:
        uut_data = np.fromfile("{}_VI.dat".format(
            uut), dtype=np.int32).reshape((-1, longwords))
        data.append(uut_data)

    total_data = np.concatenate(data, axis=1).flatten()
    total_data.tofile("LLCONTROL/afhba.0.log")
    return None


def run_main():
    parser = argparse.ArgumentParser(description='thomson analysis')
    parser.add_argument('--nuuts', type=int, default=4, help="number of uuts in set")
    parser.add_argument('--AI', type=int, default=192, help="number of AI channels (int16)")
    parser.add_argument('--SP', type=int, default=16, help="number of columns in scratchpad (int32)")
    parser.add_argument('--src', default="/home/dt100/PROJECTS/AFHBA404/LLCONTROL/afhba.0.log", help="source log file")
    parser.add_argument('--capstat', default="/home/dt100/PROJECTS/AFHBA404/LLCONTROL/llc-run-full-auto-thomson.txt", help="capture status log file")
    parser.add_argument('--spcols', default="ALL", help="SP cols to plot, default: ALL, opts: IX, SC, US, DUS")
    parser.add_argument('--aicols', default="", help="list of AI channels to dump index from 1")
    parser.add_argument('--uutcols', default=None, help="list uuts to include in --aicols dump [default:ALL]")
    parser.add_argument('--combine', default=None, help="Set this to a json config file path to combine ACQPROC data.")
    parser.add_argument('--verbose', type=int, default=0, help="verbose")
    run_analysis(fix_args(parser.parse_args()))


if __name__  == '__main__':
    run_main()


