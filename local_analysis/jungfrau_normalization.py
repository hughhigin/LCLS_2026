#!/usr/bin/env python3


# %% Imports, filepaths

# IO
from pathlib import Path
import pickle
import h5py

# Fitting
import numpy as np
from numpy.polynomial.polynomial import polyfit
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

# Filepaths
exp = "cxi101672626"
data_dir = Path("/home/hugh/beamtime_processing/LCLS_20260629/deIced_20260722/")

# Buffer data
buff_run = 83

# buff_h5e_path = data_dir / f"{exp}_r{buff_run:04d}_epix.h5"
buff_h5j_path = data_dir / f"{exp}_r{buff_run:04d}_jungfrau.h5"

# buff_dg2 = np.load(data_dir / f"r{buff_run:04d}_dg2_totalIntensity.npy")
# buff_hfx = np.load(data_dir / f"r{buff_run:04d}_hfx_totalIntensity.npy")
# buff_em = np.load(data_dir / f"r{buff_run:04d}_em_totalIntensity.npy")
# buff_fee = np.load(data_dir / f"r{buff_run:04d}_fee.npy")
# buff_wave8 = np.load(data_dir / f"r{buff_run:04d}_wave8_waveform.npy")
# buff_qad = np.load(data_dir / f"r{buff_run:04d}_qadc01_waveform.npy")

# AT25 data
# at_run = 63

# at_h5e_path = data_dir / f"{exp}_r{at_run:04d}_epix.h5"
# at_h5j_path = data_dir / f"{exp}_r{at_run:04d}_jungfrau.h5"

# # at_dg2 = np.load(data_dir / f"r{at_run:04d}_dg2_totalIntensity.npy")
# # at_hfx = np.load(data_dir / f"r{at_run:04d}_hfx_totalIntensity.npy")
# # at_em = np.load(data_dir / f"r{at_run:04d}_em_totalIntensity.npy")
# # at_fee = np.load(data_dir / f"r{at_run:04d}_fee.npy")
# # at_wave8 = np.load(data_dir / f"r{at_run:04d}_wave8_waveform.npy")
# # at_qad = np.load(data_dir / f"r{at_run:04d}_qadc01_waveform.npy")

# %% check hdf5

with h5py.File(buff_h5j_path, "r") as h5:
    for key in h5.keys():
        print(key)

    qad = h5["/qad_svd1"][:]
