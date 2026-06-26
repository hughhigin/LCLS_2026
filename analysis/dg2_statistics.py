#!/usr/bin/env python3

# %% Imports and setup
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
from scipy.signal import detrend,welch

# %% params

exp_id = "cxil1005322"
reb_dir = "/sdf/data/lcls/ds/cxi/cxil1005322/results/hhig/practice_0603/results/reb_hdf5/"
com_dir = "/sdf/data/lcls/ds/cxi/cxi101620426/results/hhig/results/reb_hdf5/"

# Parameters for fft averaging
f_sample = 120 # 120 HZ sampling
nseg = 2048

# Default welch: nperseg=256, window=Hann,etc

# %% Main loop: epix only version

# %% Main loop: get data

os.chdir(reb_dir)
epix_files = glob(f"{exp_id}_*epix_bigMask.h5")
jungfrau_files = glob(f"{exp_id}_*jungfrau_oldMasks.h5")
epix_files.sort()
jungfrau_files.sort()

e_dg2_list = []
e_normDiff_list = []
e_mean_list = []
j_dg2_list = []
j_normDiff_list = []
j_mean_list = []
for efile,jfile in zip(epix_files,jungfrau_files):
    # Check number?
    # Load data

    with h5py.File(efile, 'a') as h5epix:
        psi = np.array(h5epix['/post_sample_intensity'])
        mean_vals = np.mean(h5epix['mean'],axis=1) # Mean over all detector
        sub_vals = (mean_vals - np.mean(mean_vals))/np.std(mean_vals) - (psi - np.mean(psi))/np.std(psi)
        # Address drift?
        e_dg2_list.append(psi)
        e_mean_list.append(mean_vals)
        # e_normDiff_list.append(sub_vals)

    with h5py.File(jfile, 'a') as h5jpix:
        psi = np.array(h5jpix['/post_sample_intensity'])
        mean_vals = np.mean(h5jpix['mean'],axis=1) # Mean over all detector
        sub_vals = (mean_vals - np.mean(mean_vals))/np.std(mean_vals) - (psi - np.mean(psi))/np.std(psi)
        j_dg2_list.append(psi)
        # j_normDiff_list.append(sub_vals)
        j_mean_list.append(mean_vals)

# Save aggregate data files
np.save('epix_dg2_list.npy',e_dg2_list)
np.save('epix_mean_list.npy',e_mean_list)
np.save('jungfrau_dg2_list.npy',j_dg2_list)
np.save('jungfrau_mean_list.npy',j_mean_list)

# %% VISUAL RUNNING
# TODO: Visual inspect in 3x1 plot: data, diode fft, normDiff fft
tfig,tax = plt.subplots()
ffig,fax = plt.subplots()
for ii,(dg2,mean_vals) in enumerate(
        zip(epix_dg2_mat[start_runIdx:], epix_mean_mat[start_runIdx:])):
    # Remove nans
    dg2 = dg2[~np.isnan(dg2)]
    mean_vals = mean_vals[~np.isnan(mean_vals)]

    # Filter, dg2 > (drops to 0)
    dg_inds = dg2 > 5e3
    dg2 = dg2[dg_inds]
    mean_vals = mean_vals[dg_inds]

    # Normalized and then smoothed ratios
    norm_vals = mean_vals / dg2
    if len(norm_vals) >= savgol_win:
        smooth_vals = savgol_filter(norm_vals, savgol_win, savgol_order)
        smooth_vals = norm_vals - smooth_vals

        d_norm = (dg2 - np.mean(dg2))/(np.std(dg2))
        fname = epix_files[ii + start_runIdx]
        tax.clear()
        tax.plot((mean_vals - np.mean(mean_vals))/np.std(mean_vals),label="Detector")
        tax.plot(dvert + d_norm, label="DG2")
        # tax.plot((norm_vals - np.mean(norm_vals))/np.std(norm_vals),label="Detector / DG2")
        tax.plot((smooth_vals - np.mean(smooth_vals))/np.std(smooth_vals),label="Smoothed det/DG2")
        # tax.plot((sub_vals - np.mean(sub_vals))/np.std(sub_vals),label="Detector - DG2")
        # tax.plot(2*dvert + (dpsi - np.mean(dpsi))/(np.std(dpsi)), label="DG2 from dark run")

        tax.set_xlabel("Frame number")
        tax.set_ylabel("Intensity")
        tax.set_title(f"ii: {ii}, {fname.split('.')[0]}")
        tax.set_xlim((1000,1500))
        tax.set_ylim((-1*max(d_norm),dvert + max(d_norm)))
        tax.legend()
        tfig.canvas.draw()
        tfig.canvas.flush_events()


        fax.clear()
        N = len(norm_vals)
        fs = 120
        T = 1/fs
        t = np.arange(N)

        # dpsi_f = rfftfreq(N,T)
        psi_f = rfftfreq(N,T)
        # aX = abs(rfft(dpsi)/N)
        # daX = abs(rfft(dpsi)/N)
        naX = abs(rfft(smooth_vals)/N)

        # fax.plot(dpsi_f,aX)
        # fax.semilogy(daX, label='DG2 dar')
        # fax.semilogy(naX, label="259 Detector / DG2")
        fax.plot(psi_f[1:], (naX[1:]/max(naX[1:])), label="Detector / DG2")
        # fax.plot(1 + (daX[1:]/max(daX[1:])), label='DG2 dar')
        #
        fax.set(xlabel="frequency (AU)", ylabel="Amplitude")
        fax.set_title(fname.split(".")[0])
        ffig.legend()
        ffig.canvas.draw()
        ffig.canvas.flush_events()

        input("cont?")


# %% Load aggregate data files
epix_dg2_mat = np.load('epix_dg2_list.npy')

epix_mean_mat = np.load('epix_mean_list.npy')
jungfrau_mean_mat = np.load('jungfrau_mean_list.npy')
#
# %% Average frequency
#
## %% Straight to welch

start_runIdx = 12
nseg = 2048
savgol_win = 3000
savgol_order = 2

ef_list = []
ew_list = []
e_fnames = []
for ii,(dg2,mean_vals) in enumerate(
        zip(epix_dg2_mat[start_runIdx:], epix_mean_mat[start_runIdx:])
):
    fname = epix_files[start_runIdx + ii]
    # Remove nans
    dg2 = dg2[~np.isnan(dg2)]
    mean_vals = mean_vals[~np.isnan(mean_vals)]

    # Filter, dg2 > (drops to 0)
    dg_inds = dg2 > 5e3
    dg2 = dg2[dg_inds]
    mean_vals = mean_vals[dg_inds]

    # Normalized and then smoothed ratios
    norm_vals = mean_vals / dg2

    if (len(norm_vals) >= nseg) and (len(norm_vals) >= savgol_win):
        smooth_vals = savgol_filter(norm_vals, savgol_win, savgol_order)
        smooth_vals = norm_vals - smooth_vals
        ew = welch(smooth_vals, fs=f_sample, nperseg=nseg, detrend='linear')
        ef_list.append(ew[0])
        ew_list.append(ew[1])
        e_fnames.append(fname.split('.')[0])

# Convert to arrays
ef_arr = np.array(ef_list)
ew_arr = np.array(ew_list)

# Plot average

ew_avg = np.mean(ew_arr, axis=0)
efig,eax = plt.subplots()
eax.plot(ef_arr[0,:], ew_avg)
eax.set(title=f"Welch averaged spectrum, nseg={nseg}",
        xlabel="Frequency (Hz)", ylabel="power (AU)"
        )

# Calculate fourier transforms
# 2D FFT plot?
# Welch method to combine
# Save output
