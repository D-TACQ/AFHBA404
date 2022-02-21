#!/usr/bin/env python

"""
This is a script intended to pull off the LLC T_LATCH data, calculate the delta times and plot them in a histogram.

Usage:

Usage for 1AI BOLO with spad length of 8:
python t_latch_histogram.py --nchan=48 --spad_len=8

Usage for 4 x acq424, 2 x ao424
python t_latch_histogram.py

"""


import argparse
import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
import os
import threading
import concurrent.futures
import json

import matplotlib

matplotlib.use("TKAgg")

def plot_histogram(histo, args):
    plt.bar(histo.keys(), histo.values(), 1)
    plt.title("Histogram of T_LATCH values. N > 1 means N-1 samples were missed.")
    plt.ylabel("Number of occurrences on a log scale.")
    plt.xlabel("T_LATCH differences.")
    plt.yscale("log")
    plt.show()
    return None


def collect_dtimes(t_latch):
    histo = {1: 0, 2: 0, 3: 0}
    ideal = np.arange(t_latch[0], t_latch.shape[-1]+t_latch[0])
    if np.array_equal(t_latch, ideal):
        histo[1] += len(t_latch)
    else:
        pos = 0
        while True:
            t_latch_test = np.subtract(ideal, t_latch)
            first_nonzero = (t_latch_test != 0).argmax(axis=0)
            if first_nonzero == 0:
                break
            pos = first_nonzero + 1
            diff = t_latch[first_nonzero] - ideal[first_nonzero]
            if diff in histo:
                histo[diff] += 1
            else:
                histo[diff] = 1

            t_latch = t_latch[pos:]
            try:
                ideal = np.arange(t_latch[0], t_latch.shape[-1]+t_latch[0])
                if t_latch.shape[-1] == 0:
                    break
            except:
                break

    return histo


def collect_tlatch(args):
    if args.src == "default":
        home = expanduser("~")
        data = np.fromfile(home + "/" + "PROJECTS/AFHBA404/" + args.name + "_VI.dat", dtype=np.int32)
    else:
        data = np.fromfile(args.src + "/" + args.name + "_VI.dat", dtype=np.int32)

    t_latch = data[int(args.nchan/2)::int(args.nchan/2+args.spad_len)]
    print("Finished collecting data")
    return t_latch, data


def run_spad_analysis(args, data):

    data = np.frombuffer(data.tobytes(), dtype=np.uint16)
    print(data.shape)
    nchan = args.nchan

    data_end = nchan - 1
    print("data end: {}".format(data_end))

    latest_spad_offset = 9
    mean_spad_offset = 10
    min_spad_offset = 11
    max_spad_offset = 12
    diffs_offset = 13
    print("spad_len: ", args.spad_len)
    print(nchan + 2 * args.spad_len)
    diffs = data[data_end + diffs_offset::nchan + 2 * args.spad_len]
    diff_locations = []

    prev = diffs[0]
    for pos, item in enumerate(diffs):
        if item != prev:
            prev = item
            diff_locations.append(pos)

    latest_data = data[data_end+latest_spad_offset::nchan + 2 * args.spad_len]
    latest_data = np.take(latest_data, diff_locations)

    mean_data = data[data_end+mean_spad_offset::nchan + 2 * args.spad_len]
    mean_data = np.take(mean_data, diff_locations)
    min_data = data[data_end+min_spad_offset::nchan + 2 * args.spad_len]
    min_data = np.take(min_data, diff_locations)
    max_data = data[data_end+max_spad_offset::nchan + 2 * args.spad_len]
    max_data = np.take(max_data, diff_locations)

    latest_data = latest_data.astype(float) * 15 /1000
    mean_data = mean_data.astype(float) * 15 /1000
    mean_data = np.round(mean_data, decimals=2)
    min_data = min_data.astype(float) * 15 /1000
    max_data = max_data.astype(float) * 15 /1000

    fig, axs = plt.subplots(1, 1, sharey=False, sharex=False, tight_layout=True)
    print("len diffs = {}".format(len(diff_locations)))
    print("max data = {}".format(max_data))
    hist_bins = np.histogram_bin_edges(max_data, bins='doane')
    axs.hist(max_data, bins=hist_bins, label="A histogram of the maximum latency data")

    axs.title.set_text("histogram of max data on a log scale (n = {})".format(len(max_data)))
    axs.set_xlabel('Time in microseconds.')
    axs.set_ylabel('Frequency')
    axs.set_yscale('log', nonposy='clip')
    axs.text(.98, .95, "Mean: {}".format(np.round(np.mean(max_data), 3)), transform=axs.transAxes, ha="right", va="top")
    axs.text(.98, .9, "Min: {}".format(np.min(max_data)), transform=axs.transAxes, ha="right", va="top")
    axs.text(.98, .85, "Max: {}".format(np.max(max_data)), transform=axs.transAxes, ha="right", va="top")
    axs.text(.98, .8, "Standard Deviation: {}".format(np.around(np.std(max_data),3)), transform=axs.transAxes, ha="right", va="top")
    axs.axvline(x = (np.percentile(max_data, 99)), color="r", linestyle=":", label="A line representing the 99th percentile")
    # axs.text(.9, .8, "Standard Deviation: {}".format(np.std(max_data)), transform=axs.transAxes, ha="right", va="top")
    fig.set_figheight(7)
    fig.set_figwidth(9.5)
    axs.legend(loc=9)

    plt.show()
    return None


def get_json(path):
    with open(path) as f:
        jdata = json.load(f)
    return jdata

def show_hexdump(uut):
    name = uut['name']
    if 'VI' in uut.keys():
        command = "hexdump -ve \'"
        try:
            command += '{}/4 "%08x," '.format(uut['VI']['AI32'])
        except:
            pass
        try:
            command += '{}/2 "%04x," '.format(uut['VI']['AI16'])
        except:
            pass
        try:
            command += '{}/4 "%08x," '.format(uut['VI']['DI32'])
        except:
            pass
        try:
            command += '{}/4 "%08x," '.format(uut['VI']['SP32'])
        except:
            pass
        command += '"\\n"\' {}_VI.dat '.format(name)
        print(command)


def run_analysis(args):

    if args.json == 1:
        if args.json_src == "default":
            home = expanduser("~")
            jdata = get_json(home + '/PROJECTS/AFHBA404/runtime.json')

        else:
            jdata = get_json(args.json_src)

        for uut in jdata['AFHBA']['UUT']:
            args.name = uut['name']
            if "VI" not in uut.keys():
                continue
            print("Running analysis for UUT: {}".format(args.name))
            show_hexdump(uut)
            try:
                args.dix_len = uut['VI']['DI32']
            except KeyError:
                args.dix_len = 0
            args.nchan = uut['VI']['AI16'] + 2 * args.dix_len
            args.spad_len = uut['VI']['SP32']
            try:
                args.ao_len = uut['VO']['AO16']
            except KeyError:
                args.ao_len = 0

            tlatch, data = collect_tlatch(args)
            t_latch_split = np.array_split(tlatch, 8)
            histo = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(collect_dtimes, t_latch_split)
                for result in results:
                    for key in result:
                        if key in histo:
                            histo[key] += result[key]
                        else:
                            histo[key] = result[key]

            for key in histo:
                print("T_LATCH differences: ", key, ", happened: ", histo[key], " times")
            plot_histogram(histo, args)

            if args.ao_len > 0:
                run_spad_analysis(args, data)
            else:
                print("Cannot run latency analysis for this UUT.")


def run_main():
    parser = argparse.ArgumentParser(description='acqproc_analysis')
    parser.add_argument('--ones', default=0, type=int, help="The ones argument allows the user to plot the instances "
                                                            "where the calculated t_latch difference is equal to one. "
                                                            "This is the default case and so this will dwarf the other "
                                                            "numbers in the histogram.")
    parser.add_argument('--src', default="default", type=str, help="Location to pull data from for analysis. Note that this is just the path to AFHBA404.")
    parser.add_argument('--nchan', default=128, type=int, help="How many physical channels are contained in the data EXCLUDING SCRATCHPAD.")
    parser.add_argument('--spad_len', default=16, type=int, help="How long the scratchpad is. Default is 16 long words")
    parser.add_argument('--verbose', default=0, type=int, help='Prints status messages as the stream is running')
    parser.add_argument('--json', default=0, type=int, help='If True load config from json file.')
    parser.add_argument('--json_src', default="default", type=str, help="Location to read json from.")
    # parser.add_argument('uuts', nargs='+', help="uuts")
    run_analysis(parser.parse_args())


if __name__ == '__main__':
    run_main()
