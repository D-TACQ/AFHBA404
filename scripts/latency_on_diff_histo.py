#!/usr/bin/env python


import acq400_hapi
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():

    parser = argparse.ArgumentParser(description='llc latency histogram test')
    parser.add_argument('--file', default="./afhba.0.log", type=str,
    help='Which data file to analyse')
    parser.add_argument('uut', nargs=1, help="uut ")
    args = parser.parse_args()

    uut = acq400_hapi.Acq400(args.uut[0])
    print("Reading data")
    data = np.fromfile(args.file, dtype=np.uint16)
    print("Received data")

    data = np.frombuffer(data.tobytes(), dtype=np.uint16)
    nchan = uut.nchan()

    # get data_end by taking nchan and subtracting the spad, then subtracting 1.
    data_end = nchan - (2 * int(uut.s0.spad.split(",")[1])) - 1
    print("data end = {}".format(data_end))


    latest_spad_offset = 9
    mean_spad_offset = 10
    min_spad_offset = 11
    max_spad_offset = 12
    diffs_offset = 13

    diffs = data[data_end+diffs_offset::nchan]
    diff_locations = []

    prev = diffs[0]
    for pos, item in enumerate(diffs):
        if item != prev:
            prev = item
            diff_locations.append(pos)

    latest_data = data[data_end+latest_spad_offset::nchan]
    latest_data = np.take(latest_data, diff_locations)

    mean_data = data[data_end+mean_spad_offset::nchan]
    mean_data = np.take(mean_data, diff_locations)
    min_data = data[data_end+min_spad_offset::nchan]
    min_data = np.take(min_data, diff_locations)
    max_data = data[data_end+max_spad_offset::nchan]
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


if __name__ == '__main__':
    main()

