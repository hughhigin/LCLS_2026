#!/usr/bin/env python3

import numpy as np
import pyqtgraph as pg
import joblib
from reborn.dataframe import DataFrame
from reborn.external.pyqtgraph import imview
from reborn.external.lcls import LCLSFrameGetter
from reborn import analysis
from reborn.viewers.qtviews import PADView
from reborn.analysis.parallel import ParallelAnalyzer
from reborn.analysis.saxs import RadialProfiler
# from reborn import detector

from psana import *
import config
import runstats
import reborn
import os,sys
import psana
from scipy import ndimage
import h5py
import matplotlib.pyplot as plt
import argparse
import shutil

exp_id = "cxi101672626"

default_config = config.default_config()
memory = joblib.Memory(default_config["joblib_directory"])

# Hardcoded numbers
run_number =  104
detector="jungfrau"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=int, default=None, help="Run number")
    args = parser.parse_args()
    run_number = args.run

# %% running
run_str = f"r{run_number:04d}"

max_events = 1e7

reb_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/reb_hdf5/"
h5j_name = reb_dir + f"{exp_id}_{run_str}_jungfrau_step1.h5"
newj_name = reb_dir + f"{exp_id}_{run_str}_jungfrau_step1_noIceJ.h5"

h5e_name = reb_dir + f"{exp_id}_{run_str}_epix_step1.h5"
newe_name = reb_dir + f"{exp_id}_{run_str}_epix_step1_noIceJ.h5"

shutil.copy(h5e_name, newe_name)
shutil.copy(h5j_name, newj_name)

thresh = 20000
im_dir =  "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/analysis/"
npy_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/npy_files/"

stats = runstats.get_runstats(run_number=run_number, det=detector, maxes=True)
maxes = stats["maxes"]

pinds = np.where(maxes >= thresh)[0]
print(pinds + 1)

with h5py.File(newj_name, 'a') as h5re:
    fids = h5re['/frame_id']
    num_frames = len(fids)
    for key in h5re.keys():
        # print(key)
        val = h5re[key]
        val_sh = val.shape
        # print(val_sh)
        if (val_sh and (val_sh[0] == num_frames)):
            val = np.delete(val[:], pinds, axis=0)
            # print(val.shape)
            del h5re[key]
            h5re.create_dataset('/' + key, data=val)

with h5py.File(newe_name, 'a') as h5re:
    fids = h5re['/frame_id']
    num_frames = len(fids)
    for key in h5re.keys():
        # print(key)
        val = h5re[key]
        val_sh = val.shape
        # print(val_sh)
        if (val_sh and (val_sh[0] == num_frames)):
            val = np.delete(val[:], pinds, axis=0)
            # print(val.shape)
            del h5re[key]
            h5re.create_dataset('/' + key, data=val)


# # %% Test

# with h5py.File(newe_name, 'r') as h5re:
#     fids = h5re['/frame_id']
#     for key in h5re.keys():
#         print(key)
#         val = h5re[key]
#         val_sh = val.shape
#         print(val_sh)

# with h5py.File(newj_name, 'r') as h5re:
#     fids = h5re['/frame_id']
#     for key in h5re.keys():
#         print(key)
#         val = h5re[key]
#         val_sh = val.shape
#         print(val_sh)
