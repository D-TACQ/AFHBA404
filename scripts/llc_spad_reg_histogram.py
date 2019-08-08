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

    fig, axs = plt.subplots(1, 5, sharey=False, sharex=False, tight_layout=True)

    num_bins = np.arange(min(latest_data), max(latest_data), (max(latest_data) - min(latest_data))/np.sqrt(len(latest_data)))
    axs[0].hist(latest_data, bins=num_bins)
    axs[0].title.set_text("histogram of latest data")

    num_bins = np.arange(min(mean_data), max(mean_data), (max(mean_data) - min(mean_data))/np.sqrt(len(mean_data)))
    axs[1].hist(mean_data, bins=num_bins)
    axs[1].title.set_text("histogram of average data")

    num_bins = np.arange(min(min_data), max(min_data), (max(min_data) - min(min_data))/np.sqrt(len(min_data)))
    axs[2].hist(min_data, bins=num_bins)
    axs[2].title.set_text("histogram of min data")

    num_bins = np.arange(min(max_data), max(max_data), (max(max_data) - min(max_data))/np.sqrt(len(max_data)))
    axs[3].hist(max_data, bins=num_bins)
    axs[3].title.set_text("histogram of max data")

    num_bins = np.arange(min(latest_data), max(latest_data), (max(latest_data)-min(latest_data))/np.sqrt(len(latest_data)))
    axs[4].hist(latest_data, bins=num_bins)
    axs[4].title.set_text('histogram of latest data on a log scale')
    axs[4].set_yscale('log', nonposy='clip')

    plt.show()
    return None


if __name__ == '__main__':
    main()

