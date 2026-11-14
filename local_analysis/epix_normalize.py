#!/usr/bin/env python3

# Streamlined rewrite of "epix_masking.py" to focus on the normalization
#
# %% Imports, filepaths

# IO
import os
from pathlib import Path
import pickle
import h5py
import pandas as pd

# Fitting
import numpy as np
from numpy.polynomial.polynomial import polyfit

# from sklearn.decomposition import PCA, TruncatedSVD
from scipy.optimize import curve_fit
from scipy import interpolate, signal
from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.pyplot as plt
import bioxtasraw.RAWAPI as raw
import swaxs_functions.processing as pf

# General plotting settings
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams.update({"font.size": 14})

# Filepaths
exp = "cxi101672626"
# data_dir = Path("/home/hugh/beamtime_processing/LCLS_20260629/epix_reMask/")
# data_dir = Path("/home/hugh/beamtime_processing/LCLS_20260629/epix_tweaks/")
data_dir = Path("/home/hugh/beamtime_processing/LCLS_20260629/epix_adjustXY/")
# data_dir = Path("/home/hugh/beamtime_processing/LCLS_20260629/epix_reMask_v1/")
ref_dir = Path("/home/hugh/beamtime_processing/LCLS_20260629/ref_profiles/")
det_means_thresh = 1e7
qad_thresh = 1.5
qad_m_avg = 5.5e-06
qad_b_avg = 0.3
slope_thresh = 0.8

dg2_thresh = 2e5
dg2_bin_width = 5000
dg2_bin_edges = np.arange(0, dg2_thresh + dg2_bin_width, dg2_bin_width)
dg2_bin_centers = dg2_bin_edges[:-1] + (dg2_bin_edges[1:] - dg2_bin_edges[:-1]) / 2

# %% Try fresh

set_name = "AT25 100uM"
noJet_run = 60
buff_run = 61
sample_run = 63
# set_name = "AT48 250uM"
# noJet_run = 66
# buff_run = 72
# sample_run = 74

noJet_h5e_path = data_dir / f"{exp}_r{noJet_run:04d}_epix.h5"
buff_h5e_path = data_dir / f"{exp}_r{buff_run:04d}_epix.h5"
sample_h5e_path = data_dir / f"{exp}_r{sample_run:04d}_epix.h5"

with h5py.File(noJet_h5e_path, "r") as h5e:
    noJet_radials = h5e["/mean"][:]
    noJet_dg2 = h5e["/dg2"][:]
    noJet_qad_svd1 = h5e["/qad_svd1"][:]
    noJet_q = h5e["/q_bins"][:]
with h5py.File(buff_h5e_path, "r") as h5e:
    buff_radials = h5e["/mean"][:]
    buff_dg2 = h5e["/dg2"][:]
    buff_qad_svd1 = h5e["/qad_svd1"][:]
    buff_q = h5e["/q_bins"][:]
with h5py.File(sample_h5e_path, "r") as h5e:
    sample_radials = h5e["/mean"][:]
    sample_dg2 = h5e["/dg2"][:]
    sample_qad_svd1 = h5e["/qad_svd1"][:]
    sample_q = h5e["/q_bins"][:]

# Get filter bools for noJet, buffer, and sample files
noq_inds = sample_radials.sum(axis=0) == 0
q_inds = ~noq_inds

noJet_bools = noJet_dg2 > dg2_thresh

buff_qad_maxes = [
    np.max(buff_qad_svd1[(buff_dg2 > ij) & (buff_dg2 < ik)])
    for ij, ik in zip(dg2_bin_edges[:-1], dg2_bin_edges[1:])
]
buff_max_b, buff_max_m = polyfit(dg2_bin_centers, buff_qad_maxes, 1)
buff_bools = (
    (buff_dg2 > dg2_thresh)
    & (buff_qad_svd1 < buff_dg2 * buff_max_m + buff_max_b)
    & (buff_qad_svd1 > slope_thresh * (buff_dg2 * buff_max_m + buff_max_b))
)

sample_qad_maxes = [
    np.max(sample_qad_svd1[(sample_dg2 > ij) & (sample_dg2 < ik)])
    for ij, ik in zip(dg2_bin_edges[:-1], dg2_bin_edges[1:])
]
sample_max_b, sample_max_m = polyfit(dg2_bin_centers, sample_qad_maxes, 1)
sample_bools = (
    (sample_dg2 > dg2_thresh)
    & (sample_qad_svd1 < sample_dg2 * sample_max_m + sample_max_b)
    & (sample_qad_svd1 > slope_thresh * (sample_dg2 * sample_max_m + sample_max_b))
)

noJet_dg2 = noJet_dg2 - dg2_thresh
buff_dg2 = buff_dg2 - dg2_thresh
sample_dg2 = sample_dg2 - dg2_thresh

# Get noJet slope
noJet_detMeans = noJet_radials.mean(axis=1)
ransac = RANSACRegressor()
ransac.fit(
    noJet_dg2[noJet_bools].reshape(-1, 1), noJet_detMeans[noJet_bools].reshape(-1, 1)
)
est = ransac.estimator_
noJet_dg2_m = est.coef_[0][0]
noJet_dg2_b = est.intercept_[0]

buff_radials = buff_radials - noJet_dg2_b
sample_radials = sample_radials - noJet_dg2_b

# Normalize by spb
buff_ind_file = f"Hugh{buff_run}.txt"
buff_df = pd.read_csv(data_dir / buff_ind_file, header=None, sep=" ")
buff_inds = buff_df.iloc[:, 0].to_numpy() - 1
buff_spb = buff_df.iloc[:, 2].to_numpy()

sample_ind_file = f"Hugh{sample_run}.txt"
sample_df = pd.read_csv(data_dir / sample_ind_file, header=None, sep=" ")
sample_inds = sample_df.iloc[:, 0].to_numpy() - 1
sample_spb = sample_df.iloc[:, 2].to_numpy()

buff_mean = np.nanmean(
    np.array([buff_radials[idx] / buff_spb[ii] for ii, idx in enumerate(buff_inds)]),
    axis=0,
)

sample_mean = np.nanmean(
    np.array(
        [sample_radials[idx] / sample_spb[ii] for ii, idx in enumerate(sample_inds)]
    ),
    axis=0,
)

q = sample_q[q_inds]
buff_mean = buff_mean[q_inds]
sample_mean = sample_mean[q_inds]


# %% DG2 noJet scaling

# TBD: Implement offset from noJet background
i_scale = 1e-10

# labels = ["AT25_100uM", "AT25_1mM", "AT25_1mM_tsub", "AT48_250uM", "SAMII_static"]
# noJet_runs = [60, 66, 66, 66, 80]
# buff_runs = [61, 67, 70, 72, 101]
# sample_runs = [63, 69, 69, 74, 99]
# buff_scales = [0.994, 0.9985, 0.9985, 0.994, 0.994]
# cvals = ["tab:blue", "tab:orange", "tab:orange", "tab:green", "tab:red"]
# ref_names = [
#     "AT25-100uM_LCLS_NaCl-150mM_MOPS-10mM_SAXS_HFLUX.dat",
#     "AT25-1mM_LCLS_NaCl-150mM_MOPS-10mM_SAXS_HFLUX_scale1mM.dat",
#     "AT25-1mM_LCLS_NaCl-150mM_MOPS-10mM_SAXS_HFLUX_scale100uM.dat",
#     "dna_lp_03B_S017_B016_sub_scale.dat",
#     "SAXS_200uMSAM2_100mMNa_1mMMg_Ligand.dat",
# ]
# ref_cvals = ["darkblue", "darkorange", "darkorange", "darkgreen", "darkred"]

labels = ["AT25_100uM", "AT25_1mM", "AT48_250uM", "SAMII_static"]
noJet_runs = [60, 66, 66, 80]
buff_runs = [61, 67, 72, 101]
sample_runs = [63, 69, 74, 99]
# buff_scales = [0.994, 0.9985, 0.994, 0.994] # Lois numbers?
buff_scales = [1.1451, 1.0485, 0.9415, 0.994]  # Manual check numbers
fudges = [0.5, 1.04, 0.92, 1]
# fudges = [20, 38, 32, 13]  # QAD fudges
# buff_scales = [0.994, 1.5, 0.9415, 0.994] # Manual fudge numbers
# fudges= [1, 1.82, 0.92, 1]
cvals = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
ref_names = [
    "AT25-100uM_LCLS_NaCl-150mM_MOPS-10mM_SAXS_HFLUX.dat",
    "AT25-1mM_LCLS_NaCl-150mM_MOPS-10mM_SAXS_HFLUX_scale1mM.dat",
    "dna_lp_03B_S017_B016_sub_scale.dat",
    "SAXS_200uMSAM2_100mMNa_1mMMg_Ligand.dat",
]

ref_cvals = ["darkblue", "sienna", "darkgreen", "darkred"]
# jfig, jaxs = plt.subplots(len(labels), figsize=(5, 10), layout="constrained")
# sfig, saxs = plt.subplots(len(labels), figsize=(5, 10), layout="constrained")
approx_noJet_mean = 5e7

jfig, jaxs = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
sfig, saxs = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
rfig, raxs = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")

for lnum in range(len(labels)):
    jax = jaxs.ravel()[lnum]
    jax.set_title(f"{labels[lnum]} subtraction")
    sax = saxs.ravel()[lnum]
    sax.set_title(f"{labels[lnum]} SPB normalized traces")
    rax = raxs.ravel()[lnum]
    rax.set_title(f"{labels[lnum]} SPB normalized residuals")
    noJet_run = noJet_runs[lnum]
    buff_run = buff_runs[lnum]
    sample_run = sample_runs[lnum]
    buff_scale = buff_scales[lnum]
    # buff_scale = 1
    ref_path = ref_dir / ref_names[lnum]
    ref_prof = raw.load_profiles(ref_path)[0]

    noJet_h5e_path = data_dir / f"{exp}_r{noJet_run:04d}_epix.h5"
    buff_h5e_path = data_dir / f"{exp}_r{buff_run:04d}_epix.h5"
    sample_h5e_path = data_dir / f"{exp}_r{sample_run:04d}_epix.h5"

    # Load mean detector values from radials

    with h5py.File(noJet_h5e_path, "r") as h5e:
        noJet_fids = h5e["/frame_id"][:]
        noJet_radials = h5e["/mean"][:]
        noJet_dg2 = h5e["/dg2"][:]
        noJet_qad_svd1 = h5e["/qad_svd1"][:]
        noJet_radial_offset = h5e["/radial_offset"]
        noJet_qad_offset = h5e["/qad_offset"]
        q = h5e["/q_bins"][:]

    with h5py.File(buff_h5e_path, "r") as h5e:
        buff_fids = h5e["/frame_id"][:]
        buff_radials = h5e["/mean"][:]
        buff_dg2 = h5e["/dg2"][:]
        buff_qad_svd1 = h5e["/qad_svd1"][:]
        buff_radial_offset = h5e["/radial_offset"]
        buff_qad_offset = h5e["/qad_offset"]
        q = h5e["/q_bins"][:]

    with h5py.File(sample_h5e_path, "r") as h5e:
        sample_fids = h5e["/frame_id"][:]
        sample_radials = h5e["/mean"][:]
        sample_dg2 = h5e["/dg2"][:]
        sample_qad_svd1 = h5e["/qad_svd1"][:]
        sample_radial_offset = h5e["/radial_offset"]
        sample_qad_offset = h5e["/qad_offset"]

    # noJet offsets
    # Filter with same params
    noJet_qad_mean = noJet_qad_svd1.mean()
    noJet_qad_min = noJet_qad_svd1.min()
    print(f"Mean noJet qad: {noJet_qad_mean:1.3f}")
    print(f"Min noJet qad: {noJet_qad_min:1.3f}")

    noJet_detMeans = noJet_radials.mean(axis=0)
    noJet_detMean_mean = noJet_detMeans.mean()
    noJet_detMean_min = noJet_detMeans.min()
    print(f"Mean noJet detMean: {noJet_detMean}")
    print(f"Min noJet detMean: {noJet_detMean}")

    # Pick q bounds from mask
    noq_inds = sample_radials.sum(axis=0) == 0
    q_inds = ~noq_inds

    buff_radials = buff_radials - approx_noJet_mean
    sample_radials = sample_radials - approx_noJet_mean

    buff_qad = buff_qad_svd1 - noJet_qad_mean
    sample_qad = sample_qad_svd1 - noJet_qad_mean

    # %% sum_peak_buff filtering list from lois
    # Use filtered traces to average, try sum_peak_buff normalization
    # buff_radials = buff_deletes
    # sample_radials = sample_deletes

    # open: Hugh{buff_run}
    buff_ind_file = f"Hugh{buff_run}.txt"
    buff_df = pd.read_csv(data_dir / buff_ind_file, header=None, sep=" ")
    buff_inds = buff_df.iloc[:, 0].to_numpy() - 1
    buff_spb = buff_df.iloc[:, 2].to_numpy()

    sample_ind_file = f"Hugh{sample_run}.txt"
    sample_df = pd.read_csv(data_dir / sample_ind_file, header=None, sep=" ")
    sample_inds = sample_df.iloc[:, 0].to_numpy() - 1
    sample_spb = sample_df.iloc[:, 2].to_numpy()

    # qad_thresh = 0.6  # Sets minimum backscatter (I0 * jet width)
    # dg2_thresh = 0
    r_step = 10

    # for i in range(0, buff_radials.shape[0], r_step):
    #     if i in buff_inds:
    #         plt.plot(q, (buff_radials[i]) / buff_spb[i])

    # for ii in range(len(buff_inds)):
    #     idx = buff_inds[ii]
    #     plt.plot(q, (buff_radials[idx]) / buff_spb[ii])

    # Try QAD
    # buff_spb = buff_qad[buff_inds]
    # sample_spb = sample_qad[sample_inds]

    buff_mean = np.nanmean(
        np.array(
            [buff_radials[idx] / buff_spb[ii] for ii, idx in enumerate(buff_inds)]
        ),
        axis=0,
    )

    # for ii in range(len(sample_inds)):
    #     idx = buff_inds[ii]
    #     plt.plot(q, (sample_radials[idx]) / sample_spb[ii])

    sample_mean = np.nanmean(
        np.array(
            [sample_radials[idx] / sample_spb[ii] for ii, idx in enumerate(sample_inds)]
        ),
        axis=0,
    )

    q = q[q_inds]
    buff_mean = buff_mean[q_inds]
    sample_mean = sample_mean[q_inds]
    # %% No Jet background

    # coefs = polyfit(buff_qad_svd1[qad_inds], rmeds[qad_inds], 1)
    qad_thresh = 0.8
    # noJet_mean = np.nanmean(
    #     np.array(
    #         [
    #             radial / noJet_qad
    #             for radial, noJet_qad in zip(noJet_radials, noJet_qad_svd1)
    #         ]
    #     ),
    #     axis=0,
    # )
    # noJet_mean = noJet_mean[q_inds]

    # Simple scaling
    # noJet_buff_scaled = noJet_mean / np.mean(noJet_mean) * np.mean(buff_mean)
    # noJet_buff_scaled = noJet_buff_scaled - np.max(noJet_buff_scaled - buff_mean)

    # plt.plot(q / 1e10, noJet_buff_scaled)
    # plt.plot(q / 1e10, buff_mean)

    # noJet_sample_scaled = noJet_mean / np.mean(noJet_mean) * np.mean(sample_mean)
    # noJet_sample_scaled = noJet_sample_scaled - np.max(
    #     noJet_sample_scaled - sample_mean
    # )

    # plt.plot(q / 1e10, noJet_sample_scaled)
    # plt.plot(q / 1e10, sample_mean)

    # Project along SVD direction: response function over course of noJet
    # noJet_nframes = len(noJet_radials)
    # jet_svd1 = np.zeros((noJet_nframes, 1))

    # svd1 = TruncatedSVD(n_components=1)
    # jet_svd = svd1.fit(noJet_radials[:, q_inds])

    # print("Performing SVD transform")

    # j = 0
    # for ii in range(noJet_nframes):
    #     jet_svd1[ii] = jet_svd.transform(noJet_radials[ii, q_inds].reshape(1, -1))[0]

    # buff_svd1 = jet_svd.transform(buff_mean.reshape(1, -1))
    # buff_svd1_comp = jet_svd.inverse_transform(buff_svd1)
    # buff_svd1_comp = buff_svd1_comp.flatten()

    # sample_svd1 = jet_svd.transform(sample_mean.reshape(1, -1))
    # sample_svd1_comp = jet_svd.inverse_transform(sample_svd1)
    # sample_svd1_comp = sample_svd1_comp.flatten()

    # buff_noSVD = buff_mean - buff_svd1_comp + np.mean(buff_svd1_comp)
    # sample_noSVD = sample_mean - sample_svd1_comp + np.mean(sample_svd1_comp)
    # sax.plot(sample_noSVD)
    # sax.plot(buff_noSVD)

    # jet_sub = sample_noSVD - buff_scale * buff_noSVD
    # jet_sub_csv_name = f"{labels[lnum]}_jet_sub_spb.csv"
    # jet_sub_df = pd.DataFrame({"q": q / 1e10, "I": jet_sub * i_scale})
    # jet_sub_df.to_csv(data_dir / jet_sub_csv_name, index=False)

    # %% Subtraction

    sax.plot(sample_mean)
    sax.plot(buff_mean)

    # Take away unclear q values
    qmin_c = 0.023
    qmax_c = 0.04
    qinds_c = (q >= 1e10 * qmin_c) & (q <= 1e10 * qmax_c)
    sub = sample_mean - (buff_scale) * buff_mean
    sub = fudges[lnum] * sub

    jax.plot(q / 1e10, sub * i_scale, color=cvals[lnum], alpha=0.3)
    jax.plot(q[qinds_c] / 1e10, sub[qinds_c] * i_scale, color=cvals[lnum])
    jax.plot(ref_prof.getQ(), ref_prof.getI(), color=ref_cvals[lnum], lw=2)

    jax.set_xlim([min(q / 1e10), max(q / 1e10)])
    jax.set_ylim([i_scale * min(sub), i_scale * max(sub)])

    # i_scale = 1e-10
    # Look at residual plots
    qscale = q / 1e10
    qref_all = ref_prof.getQ()
    qref_trim_inds = (qref_all >= np.min(qscale)) & (qref_all <= np.max(qscale))
    qref_trim = qref_all[qref_trim_inds]
    iref_trim = ref_prof.getI()[qref_trim_inds]
    qref_mat = np.repeat(qref_trim[:, np.newaxis], len(qscale), axis=1)
    q_diff2 = (qscale - qref_mat) ** 2
    qbin_val_inds = q_diff2 == np.min(q_diff2, axis=1)[:, np.newaxis]
    qbin_group_inds = q_diff2 == np.min(q_diff2, axis=0)

    qbin_vals = np.zeros(qref_trim.shape)
    ibin_vals = np.zeros(qref_trim.shape)
    for qct, qval_ind in enumerate(qbin_val_inds):
        qbin_vals[qct] = qscale[qval_ind][0]
        ibin_vals[qct] = i_scale * sub[qbin_group_inds[qct]].mean()

    rax.plot(qbin_vals, iref_trim - ibin_vals, color=cvals[lnum])

    # save
    inds = ~np.isinf(sub) & (sub > 0)
    sub_csv_name = f"{labels[lnum]}_sub_spb.csv"
    sub_df = pd.DataFrame({"q": q[inds] / 1e10, "I": sub[inds] * i_scale})
    sub_df.to_csv(data_dir / sub_csv_name, index=False)

    inds = ~np.isinf(sample_mean) & (sample_mean > 0)
    sample_csv_name = f"{labels[lnum]}_spb.csv"
    sample_df = pd.DataFrame({"q": q[inds] / 1e10, "I": sample_mean[inds] * i_scale})
    sample_df.to_csv(data_dir / sample_csv_name, index=False)

    inds = ~np.isinf(buff_mean) & (buff_mean > 0)
    buff_csv_name = f"{labels[lnum]}_buff_spb.csv"
    buff_df = pd.DataFrame({"q": q[inds] / 1e10, "I": buff_mean[inds] * i_scale})
    buff_df.to_csv(data_dir / buff_csv_name, index=False)

    jfig.show()
    # sfig.show()
    # rfig.show()
    # break


# %% Rewrite as loop

# TBD: Implement offset from noJet background
i_scale = 1e-10

# labels = ["AT25_100uM", "AT25_1mM", "AT25_1mM_tsub", "AT48_250uM", "SAMII_static"]
# noJet_runs = [60, 66, 66, 66, 80]
# buff_runs = [61, 67, 70, 72, 101]
# sample_runs = [63, 69, 69, 74, 99]
# buff_scales = [0.994, 0.9985, 0.9985, 0.994, 0.994]
# cvals = ["tab:blue", "tab:orange", "tab:orange", "tab:green", "tab:red"]
# ref_names = [
#     "AT25-100uM_LCLS_NaCl-150mM_MOPS-10mM_SAXS_HFLUX.dat",
#     "AT25-1mM_LCLS_NaCl-150mM_MOPS-10mM_SAXS_HFLUX_scale1mM.dat",
#     "AT25-1mM_LCLS_NaCl-150mM_MOPS-10mM_SAXS_HFLUX_scale100uM.dat",
#     "dna_lp_03B_S017_B016_sub_scale.dat",
#     "SAXS_200uMSAM2_100mMNa_1mMMg_Ligand.dat",
# ]
# ref_cvals = ["darkblue", "darkorange", "darkorange", "darkgreen", "darkred"]

labels = ["AT25_100uM", "AT25_1mM", "AT48_250uM", "SAMII_static"]
noJet_runs = [60, 66, 66, 80]
buff_runs = [61, 67, 72, 101]
sample_runs = [63, 69, 74, 99]
# buff_scales = [0.994, 0.9985, 0.994, 0.994] # Lois numbers?
buff_scales = [1.1451, 1.0485, 0.9415, 0.994]  # Manual check numbers
fudges = [0.5, 1.04, 0.92, 1]
# fudges = [20, 38, 32, 13]  # QAD fudges
# buff_scales = [0.994, 1.5, 0.9415, 0.994] # Manual fudge numbers
# fudges= [1, 1.82, 0.92, 1]
cvals = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
ref_names = [
    "AT25-100uM_LCLS_NaCl-150mM_MOPS-10mM_SAXS_HFLUX.dat",
    "AT25-1mM_LCLS_NaCl-150mM_MOPS-10mM_SAXS_HFLUX_scale1mM.dat",
    "dna_lp_03B_S017_B016_sub_scale.dat",
    "SAXS_200uMSAM2_100mMNa_1mMMg_Ligand.dat",
]

ref_cvals = ["darkblue", "sienna", "darkgreen", "darkred"]
# jfig, jaxs = plt.subplots(len(labels), figsize=(5, 10), layout="constrained")
# sfig, saxs = plt.subplots(len(labels), figsize=(5, 10), layout="constrained")
approx_noJet_mean = 5e7

jfig, jaxs = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
sfig, saxs = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
rfig, raxs = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")

for lnum in range(len(labels)):
    jax = jaxs.ravel()[lnum]
    jax.set_title(f"{labels[lnum]} subtraction")
    sax = saxs.ravel()[lnum]
    sax.set_title(f"{labels[lnum]} SPB normalized traces")
    rax = raxs.ravel()[lnum]
    rax.set_title(f"{labels[lnum]} SPB normalized residuals")
    noJet_run = noJet_runs[lnum]
    buff_run = buff_runs[lnum]
    sample_run = sample_runs[lnum]
    buff_scale = buff_scales[lnum]
    # buff_scale = 1
    ref_path = ref_dir / ref_names[lnum]
    ref_prof = raw.load_profiles(ref_path)[0]

    noJet_h5e_path = data_dir / f"{exp}_r{noJet_run:04d}_epix.h5"
    buff_h5e_path = data_dir / f"{exp}_r{buff_run:04d}_epix.h5"
    sample_h5e_path = data_dir / f"{exp}_r{sample_run:04d}_epix.h5"

    # Load mean detector values from radials

    with h5py.File(noJet_h5e_path, "r") as h5e:
        noJet_fids = h5e["/frame_id"][:]
        noJet_radials = h5e["/mean"][:]
        noJet_dg2 = h5e["/dg2"][:]
        noJet_qad_svd1 = h5e["/qad_svd1"][:]
        noJet_radial_offset = h5e["/radial_offset"]
        noJet_qad_offset = h5e["/qad_offset"]
        q = h5e["/q_bins"][:]

    with h5py.File(buff_h5e_path, "r") as h5e:
        buff_fids = h5e["/frame_id"][:]
        buff_radials = h5e["/mean"][:]
        buff_dg2 = h5e["/dg2"][:]
        buff_qad_svd1 = h5e["/qad_svd1"][:]
        buff_radial_offset = h5e["/radial_offset"]
        buff_qad_offset = h5e["/qad_offset"]
        q = h5e["/q_bins"][:]

    with h5py.File(sample_h5e_path, "r") as h5e:
        sample_fids = h5e["/frame_id"][:]
        sample_radials = h5e["/mean"][:]
        sample_dg2 = h5e["/dg2"][:]
        sample_qad_svd1 = h5e["/qad_svd1"][:]
        sample_radial_offset = h5e["/radial_offset"]
        sample_qad_offset = h5e["/qad_offset"]

    # noJet offsets
    # Filter with same params
    noJet_qad_mean = noJet_qad_svd1.mean()
    noJet_qad_min = noJet_qad_svd1.min()
    print(f"Mean noJet qad: {noJet_qad_mean:1.3f}")
    print(f"Min noJet qad: {noJet_qad_min:1.3f}")

    noJet_detMeans = noJet_radials.mean(axis=0)
    noJet_detMean_mean = noJet_detMeans.mean()
    noJet_detMean_min = noJet_detMeans.min()
    print(f"Mean noJet detMean: {noJet_detMean_mean}")
    print(f"Min noJet detMean: {noJet_detMean_min}")

    # Pick q bounds from mask
    noq_inds = sample_radials.sum(axis=0) == 0
    q_inds = ~noq_inds

    buff_radials = buff_radials - approx_noJet_mean
    sample_radials = sample_radials - approx_noJet_mean

    buff_qad = buff_qad_svd1 - noJet_qad_mean
    sample_qad = sample_qad_svd1 - noJet_qad_mean

    # %% sum_peak_buff filtering list from lois
    # Use filtered traces to average, try sum_peak_buff normalization
    # buff_radials = buff_deletes
    # sample_radials = sample_deletes

    # open: Hugh{buff_run}
    buff_ind_file = f"Hugh{buff_run}.txt"
    buff_df = pd.read_csv(data_dir / buff_ind_file, header=None, sep=" ")
    buff_inds = buff_df.iloc[:, 0].to_numpy() - 1
    buff_spb = buff_df.iloc[:, 2].to_numpy()

    sample_ind_file = f"Hugh{sample_run}.txt"
    sample_df = pd.read_csv(data_dir / sample_ind_file, header=None, sep=" ")
    sample_inds = sample_df.iloc[:, 0].to_numpy() - 1
    sample_spb = sample_df.iloc[:, 2].to_numpy()

    # qad_thresh = 0.6  # Sets minimum backscatter (I0 * jet width)
    # dg2_thresh = 0
    r_step = 10

    # for i in range(0, buff_radials.shape[0], r_step):
    #     if i in buff_inds:
    #         plt.plot(q, (buff_radials[i]) / buff_spb[i])

    # for ii in range(len(buff_inds)):
    #     idx = buff_inds[ii]
    #     plt.plot(q, (buff_radials[idx]) / buff_spb[ii])

    # Try QAD
    # buff_spb = buff_qad[buff_inds]
    # sample_spb = sample_qad[sample_inds]

    buff_mean = np.nanmean(
        np.array(
            [buff_radials[idx] / buff_spb[ii] for ii, idx in enumerate(buff_inds)]
        ),
        axis=0,
    )

    # for ii in range(len(sample_inds)):
    #     idx = buff_inds[ii]
    #     plt.plot(q, (sample_radials[idx]) / sample_spb[ii])

    sample_mean = np.nanmean(
        np.array(
            [sample_radials[idx] / sample_spb[ii] for ii, idx in enumerate(sample_inds)]
        ),
        axis=0,
    )

    q = q[q_inds]
    buff_mean = buff_mean[q_inds]
    sample_mean = sample_mean[q_inds]
    # %% No Jet background

    # coefs = polyfit(buff_qad_svd1[qad_inds], rmeds[qad_inds], 1)
    qad_thresh = 0.8
    # noJet_mean = np.nanmean(
    #     np.array(
    #         [
    #             radial / noJet_qad
    #             for radial, noJet_qad in zip(noJet_radials, noJet_qad_svd1)
    #         ]
    #     ),
    #     axis=0,
    # )
    # noJet_mean = noJet_mean[q_inds]

    # Simple scaling
    # noJet_buff_scaled = noJet_mean / np.mean(noJet_mean) * np.mean(buff_mean)
    # noJet_buff_scaled = noJet_buff_scaled - np.max(noJet_buff_scaled - buff_mean)

    # plt.plot(q / 1e10, noJet_buff_scaled)
    # plt.plot(q / 1e10, buff_mean)

    # noJet_sample_scaled = noJet_mean / np.mean(noJet_mean) * np.mean(sample_mean)
    # noJet_sample_scaled = noJet_sample_scaled - np.max(
    #     noJet_sample_scaled - sample_mean
    # )

    # plt.plot(q / 1e10, noJet_sample_scaled)
    # plt.plot(q / 1e10, sample_mean)

    # Project along SVD direction: response function over course of noJet
    # noJet_nframes = len(noJet_radials)
    # jet_svd1 = np.zeros((noJet_nframes, 1))

    # svd1 = TruncatedSVD(n_components=1)
    # jet_svd = svd1.fit(noJet_radials[:, q_inds])

    # print("Performing SVD transform")

    # j = 0
    # for ii in range(noJet_nframes):
    #     jet_svd1[ii] = jet_svd.transform(noJet_radials[ii, q_inds].reshape(1, -1))[0]

    # buff_svd1 = jet_svd.transform(buff_mean.reshape(1, -1))
    # buff_svd1_comp = jet_svd.inverse_transform(buff_svd1)
    # buff_svd1_comp = buff_svd1_comp.flatten()

    # sample_svd1 = jet_svd.transform(sample_mean.reshape(1, -1))
    # sample_svd1_comp = jet_svd.inverse_transform(sample_svd1)
    # sample_svd1_comp = sample_svd1_comp.flatten()

    # buff_noSVD = buff_mean - buff_svd1_comp + np.mean(buff_svd1_comp)
    # sample_noSVD = sample_mean - sample_svd1_comp + np.mean(sample_svd1_comp)
    # sax.plot(sample_noSVD)
    # sax.plot(buff_noSVD)

    # jet_sub = sample_noSVD - buff_scale * buff_noSVD
    # jet_sub_csv_name = f"{labels[lnum]}_jet_sub_spb.csv"
    # jet_sub_df = pd.DataFrame({"q": q / 1e10, "I": jet_sub * i_scale})
    # jet_sub_df.to_csv(data_dir / jet_sub_csv_name, index=False)

    # %% Subtraction

    sax.plot(sample_mean)
    sax.plot(buff_mean)

    # Take away unclear q values
    qmin_c = 0.023
    qmax_c = 0.04
    qinds_c = (q >= 1e10 * qmin_c) & (q <= 1e10 * qmax_c)
    sub = sample_mean - (buff_scale) * buff_mean
    sub = fudges[lnum] * sub

    jax.plot(q / 1e10, sub * i_scale, color=cvals[lnum], alpha=0.3)
    jax.plot(q[qinds_c] / 1e10, sub[qinds_c] * i_scale, color=cvals[lnum])
    jax.plot(ref_prof.getQ(), ref_prof.getI(), color=ref_cvals[lnum], lw=2)

    jax.set_xlim([min(q / 1e10), max(q / 1e10)])
    jax.set_ylim([i_scale * min(sub), i_scale * max(sub)])

    # i_scale = 1e-10
    # Look at residual plots
    qscale = q / 1e10
    qref_all = ref_prof.getQ()
    qref_trim_inds = (qref_all >= np.min(qscale)) & (qref_all <= np.max(qscale))
    qref_trim = qref_all[qref_trim_inds]
    iref_trim = ref_prof.getI()[qref_trim_inds]
    qref_mat = np.repeat(qref_trim[:, np.newaxis], len(qscale), axis=1)
    q_diff2 = (qscale - qref_mat) ** 2
    qbin_val_inds = q_diff2 == np.min(q_diff2, axis=1)[:, np.newaxis]
    qbin_group_inds = q_diff2 == np.min(q_diff2, axis=0)

    qbin_vals = np.zeros(qref_trim.shape)
    ibin_vals = np.zeros(qref_trim.shape)
    for qct, qval_ind in enumerate(qbin_val_inds):
        qbin_vals[qct] = qscale[qval_ind][0]
        ibin_vals[qct] = i_scale * sub[qbin_group_inds[qct]].mean()

    rax.plot(qbin_vals, iref_trim - ibin_vals, color=cvals[lnum])

    # save
    inds = ~np.isinf(sub) & (sub > 0)
    sub_csv_name = f"{labels[lnum]}_sub_spb.csv"
    sub_df = pd.DataFrame({"q": q[inds] / 1e10, "I": sub[inds] * i_scale})
    sub_df.to_csv(data_dir / sub_csv_name, index=False)

    inds = ~np.isinf(sample_mean) & (sample_mean > 0)
    sample_csv_name = f"{labels[lnum]}_spb.csv"
    sample_df = pd.DataFrame({"q": q[inds] / 1e10, "I": sample_mean[inds] * i_scale})
    sample_df.to_csv(data_dir / sample_csv_name, index=False)

    inds = ~np.isinf(buff_mean) & (buff_mean > 0)
    buff_csv_name = f"{labels[lnum]}_buff_spb.csv"
    buff_df = pd.DataFrame({"q": q[inds] / 1e10, "I": buff_mean[inds] * i_scale})
    buff_df.to_csv(data_dir / buff_csv_name, index=False)

    jfig.show()
    # sfig.show()
    # rfig.show()
    # break

# %% Tweak calibration
profile_dir = data_dir / "profiles/"

profile_names = [
    "dna_lp_03B_S017_B016_sub.dat",
    "AT48_jet_sub_spb.dat",
    "AT48_sub_spb.dat",
    # "AT25_100uM_sub_spb.dat",
    # "AT25_jet_sub_spb.dat",
]

profile_paths = [str(profile_dir) + "/" + name for name in profile_names]
profiles = raw.load_profiles(profile_paths)

ssrl_prof = profiles[0]
jet_prof = profiles[1]
spb_prof = profiles[2]

# Smooth, calculate peaks
ssrl_win = 25
epix_win = 150  # 495 points total

ssrl_q = ssrl_prof.getQ()
ssrl_inds = ssrl_q > 0.01

ssrl_smooth = signal.savgol_filter(ssrl_prof.getI(), ssrl_win, 2)
ssrl_smooth_trunc = ssrl_smooth[ssrl_inds]
jet_smooth = signal.savgol_filter(jet_prof.getI(), epix_win, 2)

# plt.plot(ssrl_q[ssrl_inds], ssrl_prof.getI()[ssrl_inds])
# plt.plot(ssrl_q[ssrl_inds], ssrl_smooth_trunc)

# plt.plot(jet_prof.getQ(), jet_prof.getI())
# plt.plot(jet_prof.getQ(), jet_smooth)

ssrl_qPeak = ssrl_qTrunc[ssrl_smooth_trunc == np.max(ssrl_smooth_trunc)]
jet_qPeak = jet_prof.getQ()[jet_smooth == np.max(jet_smooth)]

q_offset = jet_qPeak - ssrl_qPeak
i_scale = np.max(jet_smooth) / np.max(ssrl_smooth_trunc)

plt.plot(ssrl_q[ssrl_inds], ssrl_smooth_trunc * i_scale)
plt.plot(jet_prof.getQ() - 1.5 * q_offset, jet_smooth)

# Restrict DG2 for curves

# Current params: qmin ~ 0.017, qmax ~ 0.049

# %% Subtraction: AT25 1 mM

sub = sample_mean - (0.998) * buff_mean
plt.plot(q / 1e10, sub)
i_scale = 1e-10

# save
inds = ~np.isinf(sub) & (sub > 0)
sub_csv_name = "AT25_1mM_sub_spb.csv"
sub_df = pd.DataFrame({"q": q[inds] / 1e10, "I": sub[inds] * i_scale})
sub_df.to_csv(data_dir / sub_csv_name, index=False)

inds = ~np.isinf(sample_mean) & (sample_mean > 0)
sample_csv_name = "AT25_1mM_spb.csv"
sample_df = pd.DataFrame({"q": q[inds] / 1e10, "I": sample_mean[inds] * i_scale})
sample_df.to_csv(data_dir / sample_csv_name, index=False)

inds = ~np.isinf(buff_mean) & (buff_mean > 0)
buff_csv_name = "AT25_1mM_buff_spb.csv"
buff_df = pd.DataFrame({"q": q[inds] / 1e10, "I": buff_mean[inds] * i_scale})
buff_df.to_csv(data_dir / buff_csv_name, index=False)

# %% Section Name
