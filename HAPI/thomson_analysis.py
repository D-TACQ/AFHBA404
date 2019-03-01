#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt


def check_column(data, col, total_cols, total_systems):
    sample_count = [[],[],[],[]]
    for system in range(0,total_systems):
        for count in data[col + (system * total_cols)::total_cols * total_systems]:
            sample_count[system].append(count)
    return sample_count


def plot_chan(data, chan, total_systems):
    plt.plot(data[chan::224 * total_systems])
    plt.show()


#short_data = np.fromfile("LLCONTROL/afhba.0.log", dtype=np.int16)
#long_data = np.fromfile("LLCONTROL/afhba.0.log", dtype=np.int32)
short_data = np.fromfile("afhba.0.log", dtype=np.int16)
long_data = np.fromfile("afhba.0.log", dtype=np.int32)


sample_count = check_column(long_data, 96, 112, 4)

for num, counter in enumerate(sample_count[0]):
    print "1:  {0:10}  ".format(sample_count[0][num]), "2:  {0:10}  ".format(sample_count[1][num]), "3:  {0:10}  ".format(sample_count[2][num]), "4:  {0:10}  ".format(sample_count[3][num])

print "\n \n"

usec_count = check_column(long_data, 97, 112, 4)

for num, counter in enumerate(usec_count[0]):
    print "1:  {0:10}  ".format(usec_count[0][num]), "2:  {0:10}  ".format(usec_count[1][num]), "3:  {0:10}  ".format(usec_count[2][num]), "4:  {0:10}  ".format(usec_count[3][num])


if sample_count[0] == sample_count[1] and sample_count[1] == sample_count[2] and sample_count[2] == sample_count[3]:
    print "Sample numbers identical"


#for item in sample_count:
#    print item

plot_chan(short_data, 1, 4)

#data = short_data[0:191]

#for channel in short_data[]
#print "Average data value"


