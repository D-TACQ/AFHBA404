#!/usr/bin/env python3


import acq400_hapi
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
# from collections import Counter

def main(args):

    uut = acq400_hapi.Acq400(args.uutname)
    print("Reading data")
    data = np.fromfile(args.file, dtype=np.uint16)
    print("Data read in")

    data = np.frombuffer(data.tobytes(), dtype=np.uint16)

    data_size = 4 if uut.s0.data32 == '1' else 2
    ssb = int(uut.s0.ssb)
    spadstart = int(uut.s0.spadstart)
    spadlen = (ssb - spadstart) // 4
    nchan = ssb // data_size
    nchan_log = (ssb - spadstart) // data_size + spadlen

    print(f"[{args.uutname}] adc_chans={spadstart // data_size} spad_chans={spadlen} nchan={nchan} nchan_log={nchan_log} ssb={ssb} data_size={data_size}")

    # get data_end by taking nchan and subtracting the spad, then subtracting 1.
    data_end = nchan - (2 * spadlen) - 1

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

    print("Overall Mean    = {:.2f}us" .format(np.mean(mean_data)))
    print("Maximum Latency = {:.2f}us" .format(np.max(max_data)))
    print("Minimum Latency = {:.2f}us" .format(np.min(min_data)))

    fig, axs = plt.subplots(1, 1, sharey=False, sharex=False, tight_layout=True, figsize=(11.7, 8.27))
    num_bins = np.arange(min(latest_data), max(latest_data), (max(latest_data) - min(latest_data))/np.sqrt(len(latest_data)))
    axs.hist(latest_data, bins=num_bins)
    axs.title.set_text("D-TACQ LLC Latency Histogram, AO Fetch to Update")

    axs.text(.98, .9, "Min: {:.2f}us".format(np.min(min_data)), transform=axs.transAxes, ha="right", va="top")
    axs.text(.98, .85, "Max: {:.2f}us".format(np.max(max_data)), transform=axs.transAxes, ha="right", va="top")
    axs.text(.98, .80, "Mean: {:.2f}us".format(np.mean(mean_data)), transform=axs.transAxes, ha="right", va="top")

    plt.xlabel('Time(us)')
    plt.ylabel('Frequency')

    axs.set_xlabel('Latency (us)')
    axs.set_ylabel('Count')

    filename = f"{args.uutname}_llc_histogram.png"
    filepath = os.path.join(args.root, filename)
    plt.savefig(filepath, format='png', dpi=300, bbox_inches="tight")
    print(f"Plot saved to {filepath}")

    plt.show()
    return None


def get_parser():
    parser = argparse.ArgumentParser(description='llc latency histogram test')

    parser.add_argument('--file', default="./afhba.0.log", type=str, help='Which data file to analyse')
    parser.add_argument('--root', default="./", type=str, help='dir to save image')

    parser.add_argument('uutname', help="uut hostname")
    return parser

if __name__ == '__main__':
    main(get_parser().parse_args())