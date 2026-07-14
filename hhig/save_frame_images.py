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

exp_id = "cxi101672626"

default_config = config.default_config()
memory = joblib.Memory(default_config["joblib_directory"])

# Hardcoded numbers
run_number =  104

if __name__ == "__main__":
    run_number = int(sys.argv[1])

max_events = 1e7
detector="jungfrau"

thresh = 20 # Epix thresh
epix_shape = (352,384)
any_gain = np.zeros(epix_shape, dtype=bool)
sum_gain = []

im_dir =  "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/analysis/"
npy_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/npy_files/"
run_folder = im_dir + f"{detector}_r{run_number}_epixTrip{thresh}_screenshots/"
os.makedirs(run_folder, exist_ok=True)

gain_filename = f"epix_r{run_number:04d}_gain_bitSums.npy"

if os.path.isfile(npy_dir + gain_filename):
    sum_gain = np.load(npy_dir + gain_filename)
else:
    g0cut_epix10k = 1<<14
    text = "exp=%s:run=%s:smd" % (exp_id, run_number)
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

    np.save(npy_dir + gain_filename, sum_gain)

trip_inds = np.array([i for i,val in enumerate(sum_gain) if val > thresh])
print(f"Run {run_number} Tripped indices, at least {thresh} epix gain flips:")
print(trip_inds)

# Make padview

# frame_inds = [
#     1986,
#     2735,
#     3600,
#     5165,
#     6279,
#     6383,
#     7222,
#     8176,
#     9424,
#     13182,
#     13360,
#     14588,
#     15440,
#     17797,
#     21117,
#     21235,
#     21447,
#     21668,
#     23361,
#     24220,
# ]

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

pv = PADView(frame_getter=framegetter)
pv.start()

for frame_ind in trip_inds:
    pv.show_frame(frame_number=frame_ind)
    pv.update_display()
    pv.save_screenshot(run_folder + f'matlab_frame{frame_ind+1}_screenshot.jpg')
