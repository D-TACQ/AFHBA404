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


def run_analysis(args):
    short_data = np.fromfile("../LLCONTROL/afhba.0.log", dtype=np.int16)
    long_data = np.fromfile("../LLCONTROL/afhba.0.log", dtype=np.int32)

    sample_count = check_column(long_data, 96, 112, 4)
    usec_count = check_column(long_data, 97, 112, 4)

    for num, counter in enumerate(sample_count[0]):
#        print "{0:10} {0:10} {0:10} {0:10} {0:10} {0:10} {0:10} {0:10}".format("","","","","","","","")
        print "{0:10}  ".format(sample_count[0][num]), \
        "{0:10}  ".format(sample_count[1][num]), \
        "{0:10}  ".format(sample_count[2][num]), \
        "{0:10}  ".format(sample_count[3][num]), \
        "{0:10}  ".format(usec_count[0][num]), \
        "{0:10}  ".format(usec_count[1][num]), \
        "{0:10}  ".format(usec_count[2][num]), \
        "{0:10}  ".format(usec_count[3][num])


    #print "\n \n"

    if sample_count[0] == sample_count[1] and sample_count[1] == sample_count[2] and sample_count[2] == sample_count[3]:
        print "Sample numbers identical"


    #plot_chan(short_data, 1, 4)
    return None


def run_main():
    parser = argparse.ArgumentParser(description='thomson analysis')
    parser.add_argument('--verbose', type=int, default=0, help="verbose")
    run_analysis(parser.parse_args())


if __name__  == '__main__':
    run_main()


