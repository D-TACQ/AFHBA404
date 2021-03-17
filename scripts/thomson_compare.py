#!/usr/bin/python3


import matplotlib.pyplot as plt
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str,
                        default="./acq2106_119_11/acq2106_119_CH00", help='ARM file name.')
    parser.add_argument('--file2', type=str,
                        default="./acq2106_119_VI.dat", help='Host file name.')
    args = parser.parse_args()
    return args


def load_data(file1, file2):
    arm_data = np.fromfile(file1, dtype=np.uint32)
    host_data = np.fromfile(file2, dtype=np.uint32)[112:]
    return arm_data, host_data


def check_col_counters(arm_data, host_data, column, nchan, test="sample counts"):
    arm_column = arm_data[column::nchan]
    host_column = host_data[column::nchan]
    if arm_column != host_column:
        print("ARM and HOST {} are not identical. This is okay, doing more tests to confirm...".format(test))

    return None


def check_data(arm_data, host_data):
    CGREEN = "\x1b[1;32m"
    CEND = "\33[0m"
    nchan = 112
    SPAD0 = 96

    arm_data = arm_data.reshape((-1, nchan))
    host_data = host_data.reshape((-1, nchan))

    counter = 0
    indices = []

    print("\nLooking for missing samples in HOST data now.")
    print("-"*50)
    for num, sample in enumerate(arm_data):
        try:
            if sample[SPAD0] != host_data[counter][SPAD0]:
                print("ARM sample: {} not in host_data.".format(sample[SPAD0]))
                print(sample[SPAD0], host_data[counter][SPAD0])
                indices.append(num)
            else:
                counter += 1
        except Exception:
            # If exception then index is out of bounds. Add index to list so it
            # can be removed later.
            indices.append(num)
            continue
    print("-"*50)
    # Remove row indices from arm_data so we only compare data that exists in both data sets.
    arm_data = np.delete(arm_data, indices, axis=0)

    # Check new ARM data is the same as HOST data
    if not np.array_equal(arm_data, host_data):
        for num, sample in enumerate(arm_data):
            if not np.array_equal(arm_data[num], host_data[num]):
                print("Samples not equal: ", num)
        print("\nARM data does not match HOST data after removing missed samples. Something has gone wrong. Please inspect data.")
    else:
        print(CGREEN, "\nARM data matches HOST data once missing samples have been accounted for. Test successful.", CEND)
    return None


def main():
    args = get_args()
    arm_data, host_data = load_data(args.file1, args.file2)

    # Check if arrays are equal to avoid doing heavier processing if it isn't needed
    if np.array_equal(arm_data, host_data):
        print("ARM and HOST files are identical.")
    else:
        check_col_counters(arm_data, host_data, 112, 96, "sample counters")
        check_col_counters(arm_data, host_data, 112, 96, "microsecond counters")
        check_data(arm_data, host_data)
    return None

if __name__ == '__main__':
    main()
