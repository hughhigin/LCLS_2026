#!/usr/bin/env python3

# %% Imports, params

import sys, os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt

exp_id = "cxi101672626"
home_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/post_processing/"
reb_dir = home_dir + "results/reb_hdf5/"
out_dir = home_dir + "analysis/plots/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=int, default=None, help="Run number")
    parser.add_argument("--start", type=int, default=0, help="start frame number")
    parser.add_argument("--step", type=int, default=1, help="step number")
    parser.add_argument("--stop",type=int, default=None, help="stop frame number")
    parser.add_argument("-d", "--detector", type=str, default='jungfrau', help="Detector to analyze (default=jungfrau, can also be epix.)")
    args = parser.parse_args()

    step = args.step
    run = args.run
    det = args.detector
    basename = f"{exp_id}_r{run:04d}_{det}_step{step}.h5"
    fname = f"{exp_id}_r{run:04d}_{det}_step{step}.h5"

    h5 = h5py.File(reb_dir + fname)
    q = h5['q_bins'][:] / 1e10 # Inverse angstroms
    radials = h5['mean'][:]
    dg2 = h5['dg2'][:]

    r_step = 1000
    for i in range(0,radials.shape[0],r_step):
        plt.plot(q,radials[i]) # No normalization

    mean_rad = radials.mean(0)
    plt.plot(q,mean_rad,'k-',lw=3,label='mean')
    plt.legend()
    plt.xlabel('q')
    plt.ylabel('I(q)')
    plt.title(basename)
    plt.savefig(out_dir + basename+'_radials.png',dpi=150)
    plt.show()
