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
#
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

exp_id = "cxi101672626"

default_config = config.default_config()
memory = joblib.Memory(default_config["joblib_directory"])

# Hardcoded numbers
run_number =  104
detector="jungfrau"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=int, default=None, help="Run number")
    parser.add_argument("-d", "--detector", type=str, default='jungfrau', help="Detector to analyze (default=jungfrau, can also be epix.)")
    args = parser.parse_args()
    detector = args.detector
    run_number = args.run

run_str = f"r{run_number:04d}"

max_events = 1e7

npy_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/npy_files/"
plot_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/analysis/plots/"

reb_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/reb_hdf5/"
jungfrau_h5_name = f"{exp_id}_{run_str}_jungfrau_step1.h5"

thresh = 20000
im_dir =  "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/analysis/"
npy_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/npy_files/"
run_folder = im_dir + f"{detector}_r{run_number}_maxJung{thresh}_screenshots/"
os.makedirs(run_folder, exist_ok=True)

stats = runstats.get_runstats(run_number=run_number, det=detector, maxes=True)
maxes = stats["maxes"]

pinds = np.where(maxes >= thresh)[0]
print(pinds + 1)
