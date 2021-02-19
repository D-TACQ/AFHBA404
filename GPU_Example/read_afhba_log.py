
import struct
from matplotlib import pyplot as plt
import numpy
import sys

fname = 'acq2106_gpu.log'
#fname = 'sim_in.log'

chan_plot = []
if len(sys.argv)>1: #only plot  specific channels
	for c in sys.argv[1:]:
		chan_plot.append(int(c))
else:
	for c in range(16):
		chan_plot.append(c)

nChan = 16
data = []
f = open(fname,'rb')
byte = f.read(2)
ix = 0
while len(byte) == 2:
	dat = struct.unpack('h',byte)
	data.append(dat[0])
	#print dat[0],byte
	byte = f.read(2)


#with open(fname,'rb') as f:
#	byte = f.read(2)
#	while byte != "":
#		dat = struct.unpack('h',byte)
#		data.append(dat[0])
#		#print dat[0],byte
#		byte = f.read(2)

nSamp = int(len(data) / nChan)
print(nSamp)
sampleRate = 1e7
dt = 1./sampleRate # Timestep
#time = numpy.arange(0,nSamp*dt,dt)
time = numpy.arange(0,nSamp)*dt
# Now need to sort data, there are 16 channels:
dat = numpy.zeros([nChan,nSamp])
print(nSamp*dt)

time = time - 5e-3

# Save data in a numpy text file, if you want to export specific channels
dat_out = []
#dat_out.append(time)
for it in range(len(time)):
	tt = [time[it]]
	for chan in chan_plot:
		tt.append(data[chan+it*nChan])
	dat_out.append(tt)
numpy.savetxt('acq2106_reduced.txt',dat_out)

#for chan in range(9,14):
#for chan in range(nChan):
for chan in chan_plot:
	#print(chan)
	dat[chan,:] = data[chan:len(data):nChan]
	plt.plot(time,dat[chan,:],'.-')
#	plt.show()
#	for c in range(chan):
#		plt.plot(time,dat[c,:])
	#plt.plot(dat[chan,:])
plt.legend(['0','1','2','3','4','5','6','7','8','9'])
#plt.legend(['I_cap','I_coil','{flux}','V_cap','V_spa'])
plt.rcParams.update({'font.size':22})
plt.rcParams.update({'axes.titlesize':20})
plt.rcParams.update({'axes.labelsize':16})
plt.title('ACQ482 data on GPU')
#plt.title('Simulation Data on GPU')
plt.ylabel('Counts')
plt.xlabel('Time (seconds)')
plt.show()
