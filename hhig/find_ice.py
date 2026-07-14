#!/usr/bin/env python3

# %% Script from Kaz to load data
#
from psana import *
import matplotlib.pyplot as plt
import sys,os
from array import array
import numpy as np

# Hugh imports
import h5py
import matplotlib.pyplot as plt

# %% Set params

experiment= 'cxi101672626'
# run = [98]
run = [95]
#########################################
looplength = len(run)

counter = 0

##range of output over events, and pixels
##
# Gain: 0 is high gain (signal?) 1 is low gain (flare)
# Mask variable
epix_shape = (352,384)
any_gain = np.zeros(epix_shape, dtype=bool)
sum_gain = []

g0cut_epix10k = 1<<14
for r in range(0,looplength):

    text = "exp=%s:run=%s:smd" % (experiment, run[r])
    ds = DataSource(text)
    det2 = Detector('epix10ka_0')

    for nevent,evt in enumerate(ds.events()):


        raw_data = det2.raw_data(evt)
        data_adu = (raw_data & ((g0cut_epix10k)-1))
        gainbit = ((raw_data>g0cut_epix10k)*1)[0]

        # any_gain = np.logical_or(any_gain,gainbit)
        sum_gain.append(np.sum(gainbit.flatten()))

        if(nevent%100==0):
            print(nevent," Events processed \n",)
            sys.stdout.flush()

    # Save in reborn ish format?

trip_inds = [i for i,val in enumerate(sum_gain) if val > 50]
print(trip_inds)

plt.plot(sum_gain); plt.savefig("sum_gain.png")
plt.title(f"run number {run[r]}")
plt.show()

print(su)
