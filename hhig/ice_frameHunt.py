#!/usr/bin/env python3

# import argparse
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
import reborn
import os,sys
import psana
from scipy import ndimage
import h5py
import matplotlib.pyplot as plt

exp_id = "cxi101672626"

default_config = config.default_config()
memory = joblib.Memory(default_config["joblib_directory"])

# Hardcoded numbers
run_number =  104
if __name__ == "__main__":
    run_number = int(sys.argv[1])
run_str = f"r{run_number:04d}"

max_events = 1e7
detector="jungfrau"

npy_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/npy_files/"
plot_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/analysis/plots/"

reb_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/reb_hdf5/"
jungfrau_h5_name = f"{exp_id}_{run_str}_jungfrau_step1.h5"

gain_filename = f"epix_r{run_number:04d}_gain_bitSums.npy"

# %% Pull mean radials and epix gain bits

# Jungfrau radials:

j_radials = []
with h5py.File(reb_dir + jungfrau_h5_name) as h5j:
    j_radials = h5j["/mean"][:]
    j_sdev = h5j["/sdev"][:]
    dg2 = h5j["/dg2"][:]

# Epix gain:
if os.path.isfile(npy_dir + gain_filename):
    sum_gain = np.load(npy_dir + gain_filename)
else:
    g0cut_epix10k = 1<<14
    text = "exp=%s:run=%s:smd" % (exp_id, run_number)
    ds = DataSource(text)
    det2 = Detector('epix10ka_0')

    sum_gain = []
    for nevent,evt in enumerate(ds.events()):

        raw_data = det2.raw_data(evt)
        data_adu = (raw_data & ((g0cut_epix10k)-1))
        gainbit = ((raw_data>g0cut_epix10k)*1)[0]

        # any_gain = np.logical_or(any_gain,gainbit)
        sum_gain.append(np.sum(gainbit.flatten()))

        if(nevent%100==0):
            print(nevent," Events processed \n",)
            sys.stdout.flush()

    np.save(npy_dir + gain_filename, sum_gain)


# %% Reborn SNR filter

# Can we use # of pixels above SNR = snr_thresh?

# Loop through framegetter

conf = config.get_config(run_number, detector)
detectors = conf["pad_detectors"]

framegetter = LCLSFrameGetter(
        run_number=run_number,
        max_events=max_events,
        experiment_id=conf["experiment_id"],
        pad_detectors=detectors,
        cachedir=conf["cachedir"],
        postprocessors=None,
        photon_wavelength_pv=conf["photon_wavelength_pv"]
)
print("finished framegetter set up")

df0 = framegetter.get_first_frame()
geom = df0.get_pad_geometry()

bright_thresh = 20000 # Ice monsters
bright_sum = []
print(f"Framegetter, loop through run {run_number} frames:")
for nevent,df in enumerate(framegetter):
    data = df.raw_data
    bright_sum.append(sum(data >= bright_thresh))

    if (nevent%1000 == 0):
        print(f"Event {nevent}")

out_name = f"jungfrau_{run_str}_brightSum_over{bright_thresh}.npy"
np.save(out_name, bright_sum)

# %% Try runstats version

# %% PADVIEW to save images

# conf = config.get_config(run_number, detector)
# detectors = conf["pad_detectors"]

# framegetter = LCLSFrameGetter(
#         run_number=run_number,
#         max_events=max_events,
#         experiment_id=conf["experiment_id"],
#         pad_detectors=detectors,
#         cachedir=conf["cachedir"],
#         postprocessors=None,
#         photon_wavelength_pv=conf["photon_wavelength_pv"]
# )
# print("finished framegetter set up")

# pv = PADView(frame_getter=framegetter)
# pv.start()

# for frame_ind in trip_inds:
#     pv.show_frame(frame_number=frame_ind)
#     pv.update_display()
#     pv.save_screenshot(run_folder + f'matlab_frame{frame_ind+1}_screenshot.jpg')
