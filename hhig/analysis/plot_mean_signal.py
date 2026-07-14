#!/usr/bin/env python3

# %% Imports, params

import sys, os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt

exp_id = "cxi101672626"
# reb_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/reb_hdf5/"
reb_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/reb_hdf5_noMasks/"
out_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/analysis/plots/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=int, default=None, help="Run number")
    parser.add_argument("--start", type=int, default=0, help="start frame number")
    parser.add_argument("--stop",type=int, default=None, help="stop frame number")
    parser.add_argument("--step",type=int, default=1, help="step number")
    parser.add_argument("-d", "--detector", type=str, default='jungfrau', help="Detector to analyze (default=jungfrau, can also be epix.)")
    args = parser.parse_args()

    run = args.run
    det = args.detector
    basename = f"{exp_id}_r{run:04d}_{det}_step{args.step}"
    fname = f"{exp_id}_r{run:04d}_{det}_step{args.step}.h5"

    h5 = h5py.File(reb_dir + fname)
    q = h5['q_bins'][:]
    radials = h5['mean'][:]
    det_mean = np.mean(radials, axis=1)

    r_step = 1000
    # for i in range(0,radials.shape[0],r_step):
    #     plt.plot(q,radials[i]) # No normalization
    plt.plot(det_mean / np.mean(det_mean), label=f"{det} mean signal")
    # plt.plot(det_mean), label=f"{det} mean signal")

    plt.legend()
    plt.xlabel('Event index')
    plt.ylabel('mean intensity')
    plt.title(basename)
    plt.savefig(out_dir + basename+'_mean_det_normalized.png',dpi=150)
    # plt.savefig(out_dir + basename+'_mean_det.png',dpi=150)
    plt.show()
