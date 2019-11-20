#!/bin/python3


import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser(description='multi-uut spad test.')

    parser.add_argument('--file', default='./afhba.2.log', type=str,
    help='Which file to split into 3 for easier analysis.')

    parser.add_argument('--nuuts', default=3, type=int,
    help='How many UUTs were used to capture the data stored in the file arg.')

    parser.add_argument('--nlongs', default=80, type=int,
    help='The number of longs in each UUT. Each UUT must be symmetrical.')

    args = parser.parse_args()

    data = np.fromfile(args.file, dtype=np.int32)
    data = np.reshape(data, (-1, args.nlongs*args.nuuts))
    
    for num, uut in enumerate(range(args.nuuts)):

        uut_data = data[0:,num*args.nlongs:num*args.nlongs+args.nlongs]
        uut_data = np.reshape(uut_data, (1, -1))
        uut_data.tofile("uut{}_data.dat".format(uut))

    return None


if __name__ == '__main__':
    main()

