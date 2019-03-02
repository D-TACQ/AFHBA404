#!/usr/bin/env python


import numpy as np
#import matplotlib.pyplot as plt
import argparse

def check_column(data, col, total_cols, total_systems):
    sample_count = [[],[],[],[]]
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
    args.ai_plot = [ (c-1) for c in tuple(map(int, args.aicols.split(",")))]
    return args


def run_analysis(args):
    short_data = np.fromfile(args.src, dtype=np.int16)
    long_data = np.fromfile(args.src, dtype=np.int32)

    sample_count = check_column(long_data, args.SPIX+0, args.longs, args.nuuts)
    usec_count = check_column(long_data,   args.SPIX+1, args.longs, args.nuuts)

    aicols = []
    for col in args.ai_plot:
        aicols.append(check_column(short_data, col, args.shorts, args.nuuts))

    FF = "{0:10}"					# Field Format
    UUTS = range(0, args.nuuts)

    for num, counter in enumerate(sample_count[0]):
	rc = []						# report columns 
	rc.append(FF.format(num))

	for uut in UUTS:
	    rc.append(FF.format(sample_count[uut][num]))
        for uut in UUTS:
            rc.append(FF.format(usec_count[uut][num]))
        for uut in UUTS:
            rc.append(FF.format(0 if num <= 2 else usec_count[uut][num]-usec_count[uut][num-1]))

        for uut in UUTS:
	    for col, ch in enumerate(args.ai_plot):
                rc.append(FF.format(aicols[col][uut][num]))

	print(",".join(rc))


    #print "\n \n"

    if sample_count[0] == sample_count[1] and sample_count[1] == sample_count[2] and sample_count[2] == sample_count[3]:
        print "Sample numbers identical"


    #plot_chan(short_data, 1, 4)
    return None


def run_main():
    parser = argparse.ArgumentParser(description='thomson analysis')
    parser.add_argument('--nuuts', type=int, default=4, help="number of uuts in set")
    parser.add_argument('--AI', type=int, default=192, help="number of AI channels (int16)")
    parser.add_argument('--SP', type=int, default=16, help="number of columns in scratchpad (int32)")
    parser.add_argument('--src', default="../LLCONTROL/afhba.0.log", help="source log file")
    parser.add_argument('--aicols', default="", help="list of AI channels to dump index from 1")
    parser.add_argument('--verbose', type=int, default=0, help="verbose")
    run_analysis(fix_args(parser.parse_args()))


if __name__  == '__main__':
    run_main()


