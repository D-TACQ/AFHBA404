#!/bin/python3


import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser(description='multi-uut channel test.')

    parser.add_argument('--file0', default='./uut0_data.dat', type=str,
    help='Which file to load for UUT0.')

    parser.add_argument('--file1', default='./uut1_data.dat', type=str,
    help='Which file to load for UUT1.')

    parser.add_argument('--file2', default='./uut2_data.dat', type=str,
    help='Which file to load for UUT2.')

    parser.add_argument('--nshorts', default=160, type=int,
    help='The number of longs in each UUT. Each UUT must be symmetrical.')

    args = parser.parse_args()

    uut0_data = np.fromfile(args.file0, dtype=np.int16).reshape((-1, 160))
    uut1_data = np.fromfile(args.file1, dtype=np.int16).reshape((-1, 160))
    uut2_data = np.fromfile(args.file2, dtype=np.int16).reshape((-1, 160))

    plt.plot(uut0_data[0:,0], 'b', label='Signal generator input')
    plt.plot(uut0_data[1:,1], 'g', label='AO1 loopback')
    plt.plot(uut0_data[0:,64], 'c', label='AO2 loopback')
    plt.plot(uut1_data[0:,0], 'k', label='AO3 loopback')
    plt.legend()
    plt.show()

    return None


if __name__ == '__main__':
    main()

