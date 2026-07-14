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
from scipy.signal import detrend,welch,savgol_filter

# %% params

exp_id = "cxil1005322"
reb_dir = "/sdf/data/lcls/ds/cxi/cxil1005322/results/hhig/practice_0603/results/reb_hdf5/"
com_dir = "/sdf/data/lcls/ds/cxi/cxi101620426/results/hhig/results/reb_hdf5/"
out_dir = "/sdf/data/lcls/ds/cxi/cxil1005322/results/hhig/practice_0603/analysis/dg2_plots/"

# Parameters for fft averaging
f_sample = 120 # 120 HZ sampling
nseg = 2048

# Default welch: nperseg=256, window=Hann,etc

# %% Main loop: epix only version

# %% Main loop: get data
# NOTE: Adding mean waveform channel 2
#
os.chdir(reb_dir)
epix_files = glob(f"{exp_id}_*epix_bigMask.h5")
jungfrau_files = glob(f"{exp_id}_*jungfrau_oldMasks.h5")
epix_files.sort()
jungfrau_files.sort()

# e_dg2_list = []
# e_normDiff_list = []
# e_mean_list = []
e_waveMean_list = []
# j_dg2_list = []
# j_normDiff_list = []
# j_mean_list = []
j_waveMean_list = []
for efile,jfile in zip(epix_files,jungfrau_files):
    # Check number?
    # Load data

    with h5py.File(efile, 'a') as h5epix:
        # psi = np.array(h5epix['/post_sample_intensity'])
        # mean_vals = np.mean(h5epix['mean'],axis=1) # Mean over all detector
        wmean_vals = np.mean(h5epix['waveforms'][:,2,300:400],axis=1) # Mean over all detector
        # sub_vals = (mean_vals - np.mean(mean_vals))/np.std(mean_vals) - (psi - np.mean(psi))/np.std(psi)
        # Address drift?
        # e_dg2_list.append(psi)
        # e_mean_list.append(mean_vals)
        e_waveMean_list.append(wmean_vals)
        # e_normDiff_list.append(sub_vals)

    with h5py.File(jfile, 'a') as h5jpix:
        # psi = np.array(h5jpix['/post_sample_intensity'])
        # mean_vals = np.mean(h5jpix['mean'],axis=1) # Mean over all detector
        wmean_vals = np.mean(h5jpix['waveforms'][:,2,300:400],axis=1) # Mean over all detector
        # sub_vals = (mean_vals - np.mean(mean_vals))/np.std(mean_vals) - (psi - np.mean(psi))/np.std(psi)
        # j_dg2_list.append(psi)
        # j_normDiff_list.append(sub_vals)
        # j_mean_list.append(mean_vals)
        j_waveMean_list.append(wmean_vals)

    print("Done with " + efile + " " + jfile)

# Convert to arrays
elens = [len(elist) for elist in e_waveMean_list]
max_len = max(elens)
# e_dg2_mat = np.full((len(e_waveMean_list), max_len), np.nan)
# j_dg2_mat = np.full((len(e_waveMean_list), max_len), np.nan)
# e_mean_mat = np.full((len(e_waveMean_list), max_len), np.nan)
# j_mean_mat = np.full((len(e_waveMean_list), max_len), np.nan)

e_waveMean_mat = np.full((len(e_waveMean_list), max_len), np.nan)
j_waveMean_mat = np.full((len(e_waveMean_list), max_len), np.nan)

for ii,elen in enumerate(elens):
    # e_dg2_mat[ii,:elen] = e_dg2_list[ii]
    # j_dg2_mat[ii,:elen] = j_dg2_list[ii]
    # e_mean_mat[ii,:elen] = e_mean_list[ii]
    # j_mean_mat[ii,:elen] = j_mean_list[ii]
    e_waveMean_mat[ii,:elen] = e_waveMean_list[ii]
    j_waveMean_mat[ii,:elen] = j_waveMean_list[ii]

# Save aggregate data files
# np.save('epix_dg2_mat.npy',e_dg2_mat)
# np.save('epix_mean_mat.npy',e_mean_mat)
# np.save('jungfrau_dg2_mat.npy',j_dg2_mat)
# np.save('jungfrau_mean_mat.npy',j_mean_mat)

np.save('epix_waveMean300to400_mat.npy',e_waveMean_mat)
np.save('jungfrau_waveMean300to400_mat.npy',j_waveMean_mat)

# %% Load
os.chdir(reb_dir)
epix_dg2_mat = np.load('epix_dg2_mat.npy')
jungfrau_dg2_mat = np.load('jungfrau_dg2_mat.npy')
epix_mean_mat = np.load('epix_mean_mat.npy')
jungfrau_mean_mat = np.load('jungfrau_mean_mat.npy')
# epix_waveMean_mat = np.load('epix_waveMean_mat.npy')
# jungfrau_waveMean_mat = np.load('jungfrau_waveMean_mat.npy')
epix_waveMean_mat = np.load('epix_waveMean300to400_mat.npy')
jungfrau_waveMean_mat = np.load('jungfrau_waveMean300to400_mat.npy')

# %% VISUAL RUNNING: DG2
# TODO: Visual inspect in 3x1 plot: data, diode fft, normDiff fft
dvert = 3
# start_runIdx = 10
start_runIdx = 165
savgol_win = 500
savgol_order = 2
nseg = 512

tfig,tax = plt.subplots()
ffig,fax = plt.subplots()
for ii,(dg2,mean_vals) in enumerate(
        zip(jungfrau_dg2_mat[start_runIdx:], jungfrau_mean_mat[start_runIdx:])):
    # Remove nans
    dg2 = dg2[~np.isnan(dg2)]
    mean_vals = mean_vals[~np.isnan(mean_vals)]

    # Filter, dg2 > (drops to 0)
    dg_inds = dg2 > 5e3
    dg2 = dg2[dg_inds]
    mean_vals = mean_vals[dg_inds]

    # fname = epix_files[ii + start_runIdx]
    fname = jungfrau_files[ii + start_runIdx]
    if (ii + start_runIdx) < 200:
        print(fname)
        print(len(mean_vals))

    # Normalized and then smoothed ratios
    norm_vals = mean_vals / dg2
    if len(norm_vals) >= savgol_win:
        smooth_vals = savgol_filter(norm_vals, savgol_win, savgol_order)
        smooth_vals = norm_vals - smooth_vals

        ew = welch(smooth_vals, fs=f_sample, nperseg=nseg, detrend='linear')

        d_norm = (dg2 - np.mean(dg2))/(np.std(dg2))
        # fname = epix_files[ii + start_runIdx]
        fname = jungfrau_files[ii + start_runIdx]
        tax.clear()
        tax.plot(2*dvert + (mean_vals - np.mean(mean_vals))/np.std(mean_vals),label="Detector")
        tax.plot(dvert + d_norm, label="DG2")
        # tax.plot((norm_vals - np.mean(norm_vals))/np.std(norm_vals),label="Detector / DG2")
        tax.plot((smooth_vals - np.mean(smooth_vals))/np.std(smooth_vals),label="Smoothed det/DG2")
        # tax.plot((sub_vals - np.mean(sub_vals))/np.std(sub_vals),label="Detector - DG2")
        # tax.plot(2*dvert + (dpsi - np.mean(dpsi))/(np.std(dpsi)), label="DG2 from dark run")

        tax.set_xlabel("Frame number")
        tax.set_ylabel("Intensity")
        tax.set_title(f"ii: {ii}, {fname.split('.')[0]}")
        tax.set_ylim((-1*max(d_norm),2.5*dvert + max(d_norm)))
        tax.legend()
        # tfig.savefig(out_dir + f"{fname.split('.')[0]}_savgol{savgol_win}_signal_full.png")
        tax.set_xlim((1000,1500))
        tfig.savefig(out_dir + f"{fname.split('.')[0]}_savgol{savgol_win}_signal_zoomed.png")
        # tfig.canvas.draw()
        # tfig.canvas.flush_events()

        fax.clear()
        N = len(norm_vals)
        fs = 120
        T = 1/fs
        t = np.arange(N)

            # dpsi_f = rfftfreq(N,T)
        # psi_f = rfftfreq(N,T)
            # aX = abs(rfft(dpsi)/N)
            # daX = abs(rfft(dpsi)/N)
        # naX = abs(rfft(smooth_vals)/N)

        # fax.plot(dpsi_f,aX)
        # fax.semilogy(daX, label='DG2 dar')
        # fax.semilogy(naX, label="259 Detector / DG2")
        # fax.plot(psi_f[1:], (naX[1:]/max(naX[1:])), label="Detector / DG2")
        fax.plot(ew[0], ew[1], label="Detector / DG2")
        # fax.plot(1 + (daX[1:]/max(daX[1:])), label='DG2 dar')
        #
        fax.set(xlabel="frequency (Hz)", ylabel="Amplitude")
        fax.set_title(fname.split(".")[0])
        ffig.legend()
        # ffig.savefig(
        #     out_dir + f"{fname.split('.')[0]}_savgol{savgol_win}_welch{nseg}_spectrum_full.png"
        # )
        fax.set_xlim((0,10))
        ffig.savefig(
            out_dir + f"{fname.split('.')[0]}_savgol{savgol_win}_welch{nseg}_spectrum_zoomed.png"
        )
        # ffig.canvas.draw()
        # ffig.canvas.flush_events()

        # input("cont?")

# %% VISUAL RUNNING: WAVEFORMS
# TODO: Visual inspect in 3x1 plot: data, diode fft, normDiff fft
dvert = 4
start_runIdx = 212
savgol_win = 1000
savgol_order = 2

wfig,wax = plt.subplots()
ffig,fax = plt.subplots()
cfig,cax = plt.subplots(1,2)
for ii,(dg2,wave,mean_vals) in enumerate(
        zip(epix_dg2_mat[start_runIdx:], epix_waveMean_mat[start_runIdx:], epix_mean_mat[start_runIdx:])):
# for ii,(dg2,wave,mean_vals) in enumerate(
#         zip(jungfrau_dg2_mat[start_runIdx:], jungfrau_waveMean_mat[start_runIdx:], jungfrau_mean_mat[start_runIdx:])):
    # Remove nans
    dg2 = dg2[~np.isnan(dg2)]
    wave = wave[~np.isnan(wave)]
    mean_vals = mean_vals[~np.isnan(mean_vals)]

    # Filter, dg2 > (drops to 0)
    # wave_inds = wave > 5e3
    # wave = wave[wave_inds]
    # mean_vals = mean_vals[wave_inds]
    dg_inds = dg2 > 5e3
    dg2 = dg2[dg_inds]
    mean_vals = mean_vals[dg_inds]
    wave = wave[dg_inds]


    # Normalized and then smoothed ratios
    norm_vals = mean_vals / wave
    if len(norm_vals) >= savgol_win:
        smooth_vals = savgol_filter(norm_vals, savgol_win, savgol_order)
        smooth_vals = norm_vals - smooth_vals

        ew = welch(smooth_vals, fs=f_sample, nperseg=nseg, detrend='linear')

        d_norm = (dg2 - np.mean(dg2))/(np.std(dg2))
        wave_norm = (wave - np.mean(wave))/(np.std(wave))
        fname = epix_files[ii + start_runIdx]
        # fname = jungfrau_files[ii + start_runIdx]

        # Correlations
        for ax in cax:
            ax.clear()
        cfig.suptitle("Correlations, " + fname.split(".")[0])
        cax[0].scatter(dg2, mean_vals, label = "DG2")
        cax[1].scatter(wave, mean_vals, label = "Wave[:,2,:]")
        cax[0].set(xlabel='DG2', ylabel='Mean Epix')
        cax[1].set(xlabel='Wave Ch 2')
        cfig.canvas.draw()
        cfig.canvas.flush_events()

        wax.clear()
        wax.plot(3*dvert + d_norm, label="DG2")
        wax.plot(2*dvert + (mean_vals - np.mean(mean_vals))/np.std(mean_vals),label="Detector")
        wax.plot(dvert + wave_norm, label="wave")
        wax.plot((smooth_vals - np.mean(smooth_vals))/np.std(smooth_vals),label="Smoothed det/DG2")

        wax.set_xlabel("Frame number")
        wax.set_ylabel("Intensity")
        wax.set_title(f"ii: {ii}, {fname.split('.')[0]}")
        wax.set_ylim((-1*max(wave_norm),4*dvert + max(wave_norm)))
        wax.legend()
        wax.set_xlim((1000,1500))
        wfig.savefig(out_dir + f"{fname.split('.')[0]}_savgol{savgol_win}_signal_zoomed.png")
        wfig.canvas.draw()
        wfig.canvas.flush_events()

        fax.clear()
        N = len(norm_vals)
        fs = 120
        T = 1/fs
        t = np.arange(N)

            # dpsi_f = rfftfreq(N,T)
        # psi_f = rfftfreq(N,T)
            # aX = abs(rfft(dpsi)/N)
            # daX = abs(rfft(dpsi)/N)
        # naX = abs(rfft(smooth_vals)/N)

        # fax.plot(dpsi_f,aX)
        # fax.semilogy(daX, label='DG2 dar')
        # fax.semilogy(naX, label="259 Detector / DG2")
        # fax.plot(psi_f[1:], (naX[1:]/max(naX[1:])), label="Detector / DG2")
        fax.plot(ew[0], ew[1], label="Detector / wave ch3")
        # fax.plot(1 + (daX[1:]/max(daX[1:])), label='DG2 dar')
        #
        fax.set(xlabel="frequency (Hz)", ylabel="Amplitude")
        fax.set_title(fname.split(".")[0])
        ffig.legend()
        ffig.savefig(
            out_dir + f"{fname.split('.')[0]}_waveNorm_savgol{savgol_win}_welch{nseg}_spectrum_full.png"
        )
        fax.set_xlim((0,10))
        ffig.savefig(
            out_dir + f"{fname.split('.')[0]}_waveNorm_savgol{savgol_win}_welch{nseg}_spectrum_zoomed.png"
        )
        ffig.canvas.draw()
        ffig.canvas.flush_events()

        input("cont?")


# %% VISUAL RUNNING: UPSTREAM DIODE
# TODO: Rewrite for upstream diode correlations
dvert = 4
start_runIdx = 212
savgol_win = 1000
savgol_order = 2

wfig,wax = plt.subplots()
ffig,fax = plt.subplots()
cfig,cax = plt.subplots(1,2)
for ii,(dg2,wave,mean_vals) in enumerate(
        zip(epix_dg2_mat[start_runIdx:], epix_waveMean_mat[start_runIdx:], epix_mean_mat[start_runIdx:])):
# for ii,(dg2,wave,mean_vals) in enumerate(
#         zip(jungfrau_dg2_mat[start_runIdx:], jungfrau_waveMean_mat[start_runIdx:], jungfrau_mean_mat[start_runIdx:])):
    # Remove nans
    dg2 = dg2[~np.isnan(dg2)]
    wave = wave[~np.isnan(wave)]
    mean_vals = mean_vals[~np.isnan(mean_vals)]

    # Filter, dg2 > (drops to 0)
    # wave_inds = wave > 5e3
    # wave = wave[wave_inds]
    # mean_vals = mean_vals[wave_inds]
    dg_inds = dg2 > 5e3
    dg2 = dg2[dg_inds]
    mean_vals = mean_vals[dg_inds]
    wave = wave[dg_inds]


    # Normalized and then smoothed ratios
    norm_vals = mean_vals / wave
    if len(norm_vals) >= savgol_win:
        smooth_vals = savgol_filter(norm_vals, savgol_win, savgol_order)
        smooth_vals = norm_vals - smooth_vals

        ew = welch(smooth_vals, fs=f_sample, nperseg=nseg, detrend='linear')

        d_norm = (dg2 - np.mean(dg2))/(np.std(dg2))
        wave_norm = (wave - np.mean(wave))/(np.std(wave))
        fname = epix_files[ii + start_runIdx]
        # fname = jungfrau_files[ii + start_runIdx]

        # Correlations
        for ax in cax:
            ax.clear()
        cfig.suptitle("Correlations, " + fname.split(".")[0])
        cax[0].scatter(dg2, mean_vals, label = "DG2")
        cax[1].scatter(wave, mean_vals, label = "Wave[:,2,:]")
        cax[0].set(xlabel='DG2', ylabel='Mean Epix')
        cax[1].set(xlabel='Wave Ch 2')
        cfig.canvas.draw()
        cfig.canvas.flush_events()

        wax.clear()
        wax.plot(3*dvert + d_norm, label="DG2")
        wax.plot(2*dvert + (mean_vals - np.mean(mean_vals))/np.std(mean_vals),label="Detector")
        wax.plot(dvert + wave_norm, label="wave")
        wax.plot((smooth_vals - np.mean(smooth_vals))/np.std(smooth_vals),label="Smoothed det/DG2")

        wax.set_xlabel("Frame number")
        wax.set_ylabel("Intensity")
        wax.set_title(f"ii: {ii}, {fname.split('.')[0]}")
        wax.set_ylim((-1*max(wave_norm),4*dvert + max(wave_norm)))
        wax.legend()
        wax.set_xlim((1000,1500))
        wfig.savefig(out_dir + f"{fname.split('.')[0]}_savgol{savgol_win}_signal_zoomed.png")
        wfig.canvas.draw()
        wfig.canvas.flush_events()

        fax.clear()
        N = len(norm_vals)
        fs = 120
        T = 1/fs
        t = np.arange(N)

            # dpsi_f = rfftfreq(N,T)
        # psi_f = rfftfreq(N,T)
            # aX = abs(rfft(dpsi)/N)
            # daX = abs(rfft(dpsi)/N)
        # naX = abs(rfft(smooth_vals)/N)

        # fax.plot(dpsi_f,aX)
        # fax.semilogy(daX, label='DG2 dar')
        # fax.semilogy(naX, label="259 Detector / DG2")
        # fax.plot(psi_f[1:], (naX[1:]/max(naX[1:])), label="Detector / DG2")
        fax.plot(ew[0], ew[1], label="Detector / wave ch3")
        # fax.plot(1 + (daX[1:]/max(daX[1:])), label='DG2 dar')
        #
        fax.set(xlabel="frequency (Hz)", ylabel="Amplitude")
        fax.set_title(fname.split(".")[0])
        ffig.legend()
        ffig.savefig(
            out_dir + f"{fname.split('.')[0]}_waveNorm_savgol{savgol_win}_welch{nseg}_spectrum_full.png"
        )
        fax.set_xlim((0,10))
        ffig.savefig(
            out_dir + f"{fname.split('.')[0]}_waveNorm_savgol{savgol_win}_welch{nseg}_spectrum_zoomed.png"
        )
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
        xlabel="Frequency (Hz)", ylabel="power (Hz)"
        )

# Calculate fourier transforms
# 2D FFT plot?
# Welch method to combine
# Save output
