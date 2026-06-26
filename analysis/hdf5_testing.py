#!/usr/bin/env python3

# %% Imports

import sys, os
import numpy as np
import h5py
import matplotlib.pyplot as plt


# %% Load h5

ref_dir = "results/ref_h5s/"
ref_name = "cxil1005322-r0007_0.cxi"
file = ref_dir + ref_name

h5om = h5py.File(file)
q = h5om['data/q'][:]
q = q.mean(0)

radials = h5om['data/radial'][:]
print(h5om['LCLS'].keys())

om_fid = h5om['LCLS/fiducial'][:]
psi = h5om['LCLS/post_sample_intensity'][:]

for ii, (fid,ps) in enumerate(zip(om_fid,psi)):
    if ii % 1000 == 0:
        print(str(ii) + ": " + str(fid) + ", " + str(ps))
