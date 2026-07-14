#!/usr/bin/env python3

# TODO: Plot Fourier spectrum (Welch average)

#
import psana
from psana import *
import sys, os
from glob import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

# Scipy
from scipy.fft import fft,rfft,rfftfreq
from scipy.signal import detrend,welch,savgol_filter


exp_id = "cxi101672626"
reb_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/reb_hdf5_noMasks/"
out_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/analysis/plots/"

# Parameters for fft averaging
f_sample = 120 # 120 HZ sampling
nseg = 512

os.chdir(reb_dir)

# Default welch: nperseg=256, window=Hann,etc

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
    basename = f"{exp_id}_r{run:04d}_{det}_step{step}"
    fname = f"{exp_id}_r{run:04d}_{det}_step{step}.h5"

    h5 = h5py.File(reb_dir + fname)
    radials = h5['mean'][:]
    det_mean = np.mean(radials, axis=1)

    ffig,fax = plt.subplots((3,1))
    # fax[0]: dg2
    plt.plot(det_mean, label=f"{det} mean signal")

    # fax[]
    mean_rad = radials.mean(0)
    plt.plot(edw[0], edw[1], label="Epix / DG2")
    plt.legend()
    plt.xlabel('q')
    plt.ylabel('I(q)')
    plt.title(basename)
    plt.savefig(out_dir + basename+'_spectrum_comparison.png',dpi=150)
    plt.show()

    fax.plot(edw[0], edw[1], label="Epix / DG2")
    fax.plot(jdw[0], jdw[1], label="Jungfrau / DG2")
    # fax.plot(efw[0], efw[1], label="Epix / f_11")
    # fax.plot(jfw[0], jfw[1], label="Jungfrau / f_11")
    # fax.plot(1 + (daX[1:]/max(daX[1:])), label='DG2 dar')
    #
    fax.set(xlabel="frequency (Hz)", ylabel="Amplitude")
    fax.set_title(f"Run {run} frequency comp")
    ffig.legend()
    # ffig.savefig(
    #     out_dir + f"{fname.split('.')[0]}_savgol{savgol_win}_welch{nseg}_spectrum_full.png"
    # )
    # fax.set_xlim((0,10))
    # ffig.savefig(
    #     out_dir + f"{fname.split('.')[0]}_savgol{savgol_win}_welch{nseg}_spectrum_zoomed.png"
    # )
    ffig.canvas.draw()
    ffig.canvas.flush_events()
