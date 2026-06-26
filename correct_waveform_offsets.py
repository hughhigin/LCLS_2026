import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt


#usage: correct_waveform_offsets.py <reborn h5 file of a run with /waveforms dataset>

#the waveforms dataset, from psana, is something like (nevents, 4, 1000) in shape

chidx = 2 #index of channel with signal

h5 = h5py.File(sys.argv[1])
wf = h5['/waveforms'][:]

#there is a repeating pattern of 16 in the diode signal. 
#i.e., plotting every 16th point shows a nice high signal for 16 plots, but
#plotting every point looks like noise. Every "unit" of every-16th-point
#is offset from the others. We need to figure out that offset,
#and correct it. Note that depending on the beamline setup or psana
#setup or something, the waveform does not always have the same number
#of points. When we first noticed this it had 1200 points, so there was
#a nice integer division of 1200 points / 16 units = 75 points per unit.
#but the next time a year or so later there was only 1000 points, which does
#not divide evenly by 16. I'm not sure why. For now it seems okay to just truncate
#the number of points such that it's evenly tiled to the nearest integer.

#to figure out the offsets, first generate the 16 units by averaging
#over the run and saving them in an array to generate the highest signal
#to determine the offsets

wfmean = wf[:,chidx,:].mean(0)

nunits = 16
npts = len(wfmean)
npts_per_unit = npts//nunits #integer division
npts_remaining = npts%nunits

wfunits = np.zeros((nunits, npts_per_unit)) #some of these will have a point missing at the end
for i in range(nunits):
    wfunits[i] = wfmean[i::nunits][:npts_per_unit]

#calculate offsets for each unit by least squares
#assume we are scaling to the 0th unit
offsets = np.zeros(nunits)
nmin = 0
nmax = 10 #for scaling look at the region with highest signal
for i in range(1,nunits):
    o = np.mean(wfunits[0] - wfunits[i]) #just take the mean difference, its easy and robust 
    offsets[i] = o

#     plt.plot(wfunits[i]+o,'.-')

# plt.show()

#now these offsets need to be tiled to match the number of points in the signal
tiled_offsets = np.tile(offsets,npts_per_unit+1)[:npts]
print(tiled_offsets.shape)

#plot a random shot and the mean, now with the offsets
f,ax = plt.subplots(1,2)
ax[0].plot(wf[10000,chidx,:],'.',label='Single shot')
ax[0].plot(wfmean,'k.',label='Mean over run')
ax[0].set_title("Without Offsets")
ax[1].plot(wf[10000,chidx,:]+tiled_offsets,'.',label='Single shot')
ax[1].plot(wfmean+tiled_offsets,'k.',label='Mean over run')
ax[1].set_title("With Offsets")

plt.legend()
plt.show()

np.save('waveform_offsets.npy',tiled_offsets)