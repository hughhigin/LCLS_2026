#!/usr/bin/env python3


import psana
from psana import *
import sys, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

# Scipy
from scipy.fft import fft,rfft,rfftfreq
from scipy.signal import detrend,welch,savgol_filter

exp_id = "cxi101672626"
reb_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/reb_hdf5/"
# reb_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/reb_hdf5_noMasks/"
#
os.chdir(reb_dir)

for run in range(35,41):
    h5e_name = f"{exp_id}_r00{run}_epix_step1.h5"
    h5j_name = f"{exp_id}_r00{run}_jungfrau_step1.h5"

    # h5e = h5py.File(h5e_name, 'r+')
    # h5j = h5py.File(h5j_name, 'r+')

    with h5py.File(h5e_name, 'r+') as h5e:

        if "/digitizer" in h5e:
            del h5e['/digitizer']
        if "/digi_wftimes" in h5e:
            del h5e['/digi_wftimes']
        if "/sum" in h5e:
            del h5e['/sum']
        if "/sum2" in h5e:
            del h5e['/sum2']
        if "/wsum" in h5e:
            del h5e['/wsum']
        if "/sdev" in h5e:
            del h5e['/sdev']

    with h5py.File(h5j_name, 'r+') as h5j:
        if "/digitizer" in h5j:
            del h5j['/digitizer']
        if "/digi_wftimes" in h5j:
            del h5j['/digi_wftimes']
        if "/sum" in h5j:
            del h5j['/sum']
        if "/sum2" in h5j:
            del h5j['/sum2']
        if "/wsum" in h5j:
            del h5j['/wsum']
        if "/wsum" in h5j:
            del h5j['/sdev']
