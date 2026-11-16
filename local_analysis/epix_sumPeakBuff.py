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
from sklearn.linear_model import RANSACRegressor


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

# %% Simplify noJet incorporation

# Filter parameters, run selection
# set_name = "AT25 100uM"
# run_numbers = [60, 61, 63]
set_name = "AT25 1mM"
run_numbers = [66, 67, 69]
# set_name = "AT48 250uM"
# run_numbers = [66, 72, 74]
det_means_thresh = 1e7
qad_thresh = 1.5
qad_m_avg = 5.5e-06
qad_b_avg = 0.3
slope_thresh = 0.8

# dg2_thresh = 100
# dg2_thresh = 1e5
dg2_thresh = 2e5
dg2_bin_width = 5000
dg2_bin_edges = np.arange(0, dg2_thresh + dg2_bin_width, dg2_bin_width)
dg2_bin_centers = dg2_bin_edges[:-1] + (dg2_bin_edges[1:] - dg2_bin_edges[:-1]) / 2

noJet_h5e_path = data_dir / f"{exp}_r{run_numbers[0]:04d}_epix.h5"
buff_h5e_path = data_dir / f"{exp}_r{run_numbers[1]:04d}_epix.h5"
sample_h5e_path = data_dir / f"{exp}_r{run_numbers[2]:04d}_epix.h5"

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
noJet_bools = noJet_dg2 > dg2_thresh
# buff_bools = buff_dg2 > dg2_thresh
# sample_bools = sample_dg2 > dg2_thresh

# buff_qad_maxes = [
#     np.max(buff_qad_svd1[(buff_dg2 > ij) & (buff_dg2 < ik)])
#     for ij, ik in zip(dg2_bin_edges[:-1], dg2_bin_edges[1:])
# ]
# buff_max_b, buff_max_m = polyfit(dg2_bin_centers, buff_qad_maxes, 1)
# buff_bools = (
#     (buff_dg2 > dg2_thresh)
#     & (buff_qad_svd1 < buff_dg2 * buff_max_m + buff_max_b)
#     & (buff_qad_svd1 > slope_thresh * (buff_dg2 * buff_max_m + buff_max_b))
# )

# sample_qad_maxes = [
#     np.max(sample_qad_svd1[(sample_dg2 > ij) & (sample_dg2 < ik)])
#     for ij, ik in zip(dg2_bin_edges[:-1], dg2_bin_edges[1:])
# ]
# sample_max_b, sample_max_m = polyfit(dg2_bin_centers, sample_qad_maxes, 1)
# sample_bools = (
#     (sample_dg2 > dg2_thresh)
#     & (sample_qad_svd1 < sample_dg2 * sample_max_m + sample_max_b)
#     & (sample_qad_svd1 > slope_thresh * (sample_dg2 * sample_max_m + sample_max_b))
# )
#
# Normalize by spb
buff_ind_file = f"Hugh{run_numbers[1]}.txt"
buff_df = pd.read_csv(data_dir / buff_ind_file, header=None, sep=" ")
buff_inds = buff_df.iloc[:, 0].to_numpy() - 1
buff_spb = buff_df.iloc[:, 2].to_numpy()
buff_bools = np.zeros(buff_dg2.shape, dtype="bool")
buff_bools[buff_inds] = True

sample_ind_file = f"Hugh{run_numbers[2]}.txt"
sample_df = pd.read_csv(data_dir / sample_ind_file, header=None, sep=" ")
sample_inds = sample_df.iloc[:, 0].to_numpy() - 1
sample_spb = sample_df.iloc[:, 2].to_numpy()
sample_bools = np.zeros(sample_dg2.shape, dtype="bool")
sample_bools[sample_inds] = True


# Offset by threshold
# noJet_dg2 = noJet_dg2 - dg2_thresh
# buff_dg2 = buff_dg2 - dg2_thresh
# sample_dg2 = sample_dg2 - dg2_thresh

# Get noJet slope
noJet_detMeans = noJet_radials.mean(axis=1)
ransac = RANSACRegressor()
ransac.fit(
    noJet_dg2[noJet_bools].reshape(-1, 1), noJet_detMeans[noJet_bools].reshape(-1, 1)
)
est = ransac.estimator_
noJet_dg2_m = est.coef_[0][0]
noJet_dg2_b = est.intercept_[0]

print(f"DG2 m: {noJet_dg2_m}, DG2 b: {noJet_dg2_b}")

noJet_qadMean = noJet_qad_svd1[noJet_bools].mean()

buff_detMeans = buff_radials.mean(axis=1)
sample_detMeans = sample_radials.mean(axis=1)

# Plot check: DG2
afig, aaxs = plt.subplots(1, 2, layout="constrained")
rlabels = [
    "noJet",
    "buffer",
    "sample",
]
aax = aaxs[0]
aax.scatter(
    noJet_detMeans[noJet_bools],
    noJet_dg2[noJet_bools],
    alpha=0.05,
    label=rlabels[0],
)
aax.scatter(
    buff_detMeans[buff_bools],
    buff_dg2[buff_bools],
    alpha=0.05,
    label=rlabels[1],
)
aax.scatter(
    sample_detMeans[sample_bools],
    sample_dg2[sample_bools],
    alpha=0.05,
    label=rlabels[2],
)
leg = aax.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
aax.set_xlim(0, 2e8)

aax = aaxs[1]
aax.scatter(
    noJet_detMeans[noJet_bools] - noJet_dg2_b,
    noJet_dg2[noJet_bools],
    alpha=0.05,
    label=rlabels[0],
)
aax.scatter(
    buff_detMeans[buff_bools] - noJet_dg2_b,
    buff_dg2[buff_bools],
    alpha=0.05,
    label=rlabels[1],
)
aax.scatter(
    sample_detMeans[sample_bools] - noJet_dg2_b,
    sample_dg2[sample_bools],
    alpha=0.05,
    label=rlabels[2],
)
aax.set_xlim(0, 2e8)
leg = aax.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
afig.suptitle(f"{set_name} DG2 linear correction (RANSAC)")

# Plot check: QAD
bfig, baxs = plt.subplots(1, 2, layout="constrained")
aax = baxs[0]
aax.scatter(
    noJet_detMeans[noJet_bools],
    noJet_qad_svd1[noJet_bools],
    alpha=0.05,
    label=rlabels[0],
)
aax.scatter(
    buff_detMeans[buff_bools],
    buff_qad_svd1[buff_bools],
    alpha=0.05,
    label=rlabels[1],
)
aax.scatter(
    sample_detMeans[sample_bools],
    sample_qad_svd1[sample_bools],
    alpha=0.05,
    label=rlabels[2],
)
aax.set_xlim(0, 2e8)
leg = aax.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)

aax = baxs[1]
aax.scatter(
    noJet_detMeans[noJet_bools] - noJet_dg2_b,
    noJet_qad_svd1[noJet_bools],
    alpha=0.05,
    label=rlabels[0],
)
aax.scatter(
    buff_detMeans[buff_bools] - noJet_dg2_b,
    buff_qad_svd1[buff_bools],
    alpha=0.05,
    label=rlabels[1],
)
aax.scatter(
    sample_detMeans[sample_bools] - noJet_dg2_b,
    sample_qad_svd1[sample_bools],
    alpha=0.05,
    label=rlabels[2],
)
aax.set_xlim(0, 2e8)
leg = aax.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
bfig.suptitle(f"{set_name} QAD after DG2 offset (RANSAC)")

# Plot check: QAD norm double scale
cfig, caxs = plt.subplots(1, 2, layout="constrained")
aax = caxs[0]
aax.scatter(
    noJet_detMeans[noJet_bools],
    noJet_qad_svd1[noJet_bools],
    alpha=0.05,
    label=rlabels[0],
)
aax.scatter(
    buff_detMeans[buff_bools],
    buff_qad_svd1[buff_bools],
    alpha=0.05,
    label=rlabels[1],
)
aax.scatter(
    sample_detMeans[sample_bools],
    sample_qad_svd1[sample_bools],
    alpha=0.05,
    label=rlabels[2],
)
aax.set_xlim(0, 2e8)
leg = aax.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)

aax = caxs[1]
aax.scatter(
    (noJet_detMeans[noJet_bools] - noJet_dg2_b) / noJet_dg2_m,
    noJet_qad_svd1[noJet_bools],
    alpha=0.05,
    label=rlabels[0],
)
aax.scatter(
    (buff_detMeans[buff_bools] - noJet_dg2_b)
    / noJet_dg2_m
    / (buff_qad_svd1[buff_bools] - noJet_qadMean),
    buff_qad_svd1[buff_bools],
    alpha=0.05,
    label=rlabels[1],
)
aax.scatter(
    (sample_detMeans[sample_bools] - noJet_dg2_b)
    / noJet_dg2_m
    / (sample_qad_svd1[sample_bools] - noJet_qadMean),
    sample_qad_svd1[sample_bools],
    alpha=0.05,
    label=rlabels[2],
)
# aax.set_xlim(0, 2e8)
leg = aax.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
cfig.suptitle(f"{set_name} QAD after DG2 offset, QAD scale")

# Plot check: QAD norm single scale
dfig, daxs = plt.subplots(1, 2, layout="constrained")
aax = daxs[0]
aax.scatter(
    noJet_detMeans[noJet_bools],
    noJet_qad_svd1[noJet_bools],
    alpha=0.05,
    label=rlabels[0],
)
aax.scatter(
    buff_detMeans[buff_bools],
    buff_qad_svd1[buff_bools],
    alpha=0.05,
    label=rlabels[1],
)
aax.scatter(
    sample_detMeans[sample_bools],
    sample_qad_svd1[sample_bools],
    alpha=0.05,
    label=rlabels[2],
)
aax.set_xlim(0, 2e8)
leg = aax.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)

aax = daxs[1]
aax.scatter(
    (noJet_detMeans[noJet_bools] - noJet_dg2_b) / noJet_dg2_m,
    noJet_qad_svd1[noJet_bools],
    alpha=0.05,
    label=rlabels[0],
)
aax.scatter(
    ((buff_detMeans[buff_bools] - noJet_dg2_b) / noJet_dg2_m)
    / (buff_qad_svd1[buff_bools] - noJet_qadMean),
    buff_qad_svd1[buff_bools],
    alpha=0.05,
    label=rlabels[1],
)
aax.scatter(
    ((sample_detMeans[sample_bools] - noJet_dg2_b) / noJet_dg2_m)
    / (sample_qad_svd1[sample_bools] - noJet_qadMean),
    sample_qad_svd1[sample_bools],
    alpha=0.05,
    label=rlabels[2],
)
# aax.set_xlim(0, 2e8)
leg = aax.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)
dfig.suptitle(f"{set_name} QAD after DG2 offset, QAD scale")

# %% Check qad normalization with noJet: naive normalization

set_name = "AT25 100uM"
run_numbers = [60, 61, 63]
set_name = "AT25 1mM"
run_numbers = [66, 67, 69]
# set_name = "AT48 250uM"
# run_numbers = [66, 72, 74]
dg2_thresh = 2e5
det_means_thresh = 1e7
qad_thresh = 1.5
qad_m_avg = 5.5e-06
qad_b_avg = 0.3
slope_thresh = 0.8

all_qad = []
all_dg2 = []
all_det_means = []

uall_qad = []
uall_dg2 = []
uall_det_means = []

qfig, qax = plt.subplots(1, 3, layout="constrained")  # QAD detector correlations
dfig, dax = plt.subplots(1, 3, layout="constrained")  # DG2 detector correlations
afig, aax = plt.subplots(1, 2, layout="constrained")  # all filtered correlations
ufig, uax = plt.subplots(1, 2, layout="constrained")  # all unfiltered correlations
for ii, run in enumerate(run_numbers):
    h5e_path = data_dir / f"{exp}_r{run:04d}_epix.h5"
    with h5py.File(h5e_path, "r") as h5e:
        fids = h5e["/frame_id"][:]
        radials = h5e["/mean"][:]
        dg2 = h5e["/dg2"][:]
        qad_svd1 = h5e["/qad_svd1"][:]
        radial_offset = h5e["/radial_offset"]
        qad_offset = h5e["/qad_offset"]
        q = h5e["/q_bins"][:]

    det_means = np.mean(radials, axis=1)

    # Estimate line from dg2, qad below thresh for filtering
    if ii == 0:
        bools = dg2 > dg2_thresh
    else:
        dg2_bin_width = 5000
        dg2_bin_edges = np.arange(0, dg2_thresh + dg2_bin_width, dg2_bin_width)
        dg2_bin_centers = (
            dg2_bin_edges[:-1] + (dg2_bin_edges[1:] - dg2_bin_edges[:-1]) / 2
        )
        qad_maxes = [
            np.max(qad_svd1[(dg2 > ij) & (dg2 < ik)])
            for ij, ik in zip(dg2_bin_edges[:-1], dg2_bin_edges[1:])
        ]
        max_b, max_m = polyfit(dg2_bin_centers, qad_maxes, 1)
        bools = (
            (dg2 > dg2_thresh)
            & (qad_svd1 < dg2 * max_m + max_b)
            & (qad_svd1 > slope_thresh * (dg2 * max_m + max_b))
        )

    # Offset by threshold
    # dg2 = dg2 - dg2_thresh

    all_det_means.append(det_means[bools])
    all_qad.append(qad_svd1[bools])
    all_dg2.append(dg2[bools])

    # DG2
    ax = dax.ravel()[ii]
    # dg2_b, dg2_m = polyfit(det_means[bools], dg2[bools], 1)
    ransac = RANSACRegressor()
    ransac.fit(det_means[bools].reshape(-1, 1), dg2[bools].reshape(-1, 1))
    dg2_ransac = ransac.predict(det_means[bools].reshape(-1, 1))
    ax.scatter(det_means[bools], dg2[bools], alpha=0.08)
    # ax.plot(det_means[bools], det_means[bools] * dg2_m + dg2_b, "k", lw=2)
    # ax.plot(det_means[bools], det_means[bools] * dg2_m + dg2_b, "k", lw=2)
    ax.plot(det_means[bools], dg2_ransac, "k", lw=2)
    est = ransac.estimator_
    dg2_m = est.coef_[0]
    dg2_b = est.intercept_[0]
    ax.plot(det_means[bools], det_means[bools] * dg2_m + dg2_b, "r", lw=2)
    ax.set_title(
        f"Run {run}, corr = {np.corrcoef(det_means[bools], dg2[bools])[0,1]:1.2f}"
    )
    # ax.set_ylim(0, 400000)
    # dg2_b, dg2_m = polyfit(det_means, dg2, 1)
    # ax.scatter(det_means, dg2, alpha=0.08)
    # ax.plot(det_means[bools], det_means[bools] * dg2_m + dg2_b, "k", lw=2)
    # ax.set_title(f"Run {run}, corr = {np.corrcoef(det_means, dg2)[0,1]:1.2f}")

    # ax.set_xlabel("det_means")
    ax.set_ylabel("dg2")
    dfig.suptitle(f"{set_name} det_mean, dg2 correlation (filtered)")
    # QAD
    ax = qax.ravel()[ii]
    ransac = RANSACRegressor()
    ransac.fit(det_means[bools].reshape(-1, 1), qad_svd1[bools].reshape(-1, 1))
    qad_ransac = ransac.predict(det_means[bools].reshape(-1, 1))
    qad_b, qad_m = polyfit(det_means[bools], qad_svd1[bools], 1)
    ax.scatter(det_means[bools], qad_svd1[bools], alpha=0.08)
    ax.plot(det_means[bools], qad_ransac, "k", lw=2)
    # ax.plot(det_means[bools], det_means[bools] * qad_m + qad_b, "k", lw=2)
    ax.set_title(
        f"Run {run}, corr = {np.corrcoef(det_means[bools], qad_svd1[bools])[0,1]:1.2f}"
    )
    # ax.set_ylim(1, 3)
    #
    # Robustly fit linear model with RANSAC algorithm

    # qad_b, qad_m = polyfit(det_means, qad_svd1, 1)
    #

    # ax.scatter(det_means, qad_svd1, alpha=0.08)
    # # ax.plot(det_means, det_means * qad_m + qad_b, "k", lw=2)
    # ax.set_title(f"Run {run}, corr = {np.corrcoef(det_means, qad_svd1)[0,1]:1.2f}")

    # ax.set_xlabel("det_means")
    ax.set_ylabel("qad_svd1")
    qfig.suptitle(f"{set_name} det_mean, qad correlation (unfiltered)")


# aax[0].scatter(all_det_means.flatten(), all_dg2.flatten(), alpha=0.05)
# aax[1].scatter(all_det_means.flatten(), all_qad.flatten(), alpha=0.05)
rlabels = [
    "noJet",
    "buffer",
    "sample",
]
for rnum in range(len(all_dg2)):
    aax[0].scatter(all_det_means[rnum], all_dg2[rnum], alpha=0.05, label=rlabels[rnum])
    aax[1].scatter(all_det_means[rnum], all_qad[rnum], alpha=0.05)

afig.suptitle(f"{set_name} NoJet, buffer, and sample filtered")
aax[0].set_xlabel("Detector means")
aax[1].set_xlabel("Detector means")
aax[0].set_ylabel("DG2 means")
aax[1].set_ylabel("QAD means")
aax[0].set_xlim(0, 2e8)
aax[1].set_xlim(0, 2e8)
# aax[0].set_xlim(0, 1.5e8)
# aax[1].set_xlim(0, 1.5e8)
leg = aax[0].legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)

# uax[0].scatter(uall_det_means.flatten(), uall_dg2.flatten(), alpha=0.05)
# uax[1].scatter(uall_det_means.flatten(), uall_qad.flatten(), alpha=0.05)
for rnum in range(len(uall_dg2)):
    uax[0].scatter(uall_det_means[rnum], uall_dg2[rnum], alpha=0.05)
    uax[1].scatter(uall_det_means[rnum], uall_qad[rnum], alpha=0.05)
ufig.suptitle(f"{set_name} NoJet, buffer, and sample unfiltered")
uax[0].set_xlabel("Detector means")
uax[1].set_xlabel("Detector means")
uax[0].set_ylabel("DG2 means")
uax[1].set_ylabel("QAD means")
uax[0].set_xlim(0, 2e8)
uax[1].set_xlim(0, 2e8)


# %% Check qad normalization with noJet

set_name = "AT25 100uM"
run_numbers = [60, 61, 63]
# set_name = "AT25 1mM"
# run_numbers = [66, 67, 69]
# set_name = "AT48 250uM"
# run_numbers = [66, 72, 74]
dg2_thresh = 2e5
det_means_thresh = 1e7
qad_thresh = 1.5
qad_m_avg = 5.5e-06
qad_b_avg = 0.3
slope_thresh = 0.8

all_qad = []
all_dg2 = []
all_det_means = []

uall_qad = []
uall_dg2 = []
uall_det_means = []

qfig, qax = plt.subplots(1, 3, layout="constrained")  # QAD detector correlations
dfig, dax = plt.subplots(1, 3, layout="constrained")  # DG2 detector correlations
afig, aax = plt.subplots(1, 2, layout="constrained")  # all filtered correlations
ufig, uax = plt.subplots(1, 2, layout="constrained")  # all unfiltered correlations
for ii, run in enumerate(run_numbers):
    h5e_path = data_dir / f"{exp}_r{run:04d}_epix.h5"
    with h5py.File(h5e_path, "r") as h5e:
        fids = h5e["/frame_id"][:]
        radials = h5e["/mean"][:]
        dg2 = h5e["/dg2"][:]
        qad_svd1 = h5e["/qad_svd1"][:]
        radial_offset = h5e["/radial_offset"]
        qad_offset = h5e["/qad_offset"]
        q = h5e["/q_bins"][:]

    det_means = np.mean(radials, axis=1)

    # Estimate line from dg2, qad below thresh for filtering
    if ii == 0:
        bools = dg2 > dg2_thresh
    else:
        dg2_bin_width = 5000
        dg2_bin_edges = np.arange(0, dg2_thresh + dg2_bin_width, dg2_bin_width)
        dg2_bin_centers = (
            dg2_bin_edges[:-1] + (dg2_bin_edges[1:] - dg2_bin_edges[:-1]) / 2
        )
        qad_maxes = [
            np.max(qad_svd1[(dg2 > ij) & (dg2 < ik)])
            for ij, ik in zip(dg2_bin_edges[:-1], dg2_bin_edges[1:])
        ]
        max_b, max_m = polyfit(dg2_bin_centers, qad_maxes, 1)
        bools = (
            (dg2 > dg2_thresh)
            & (qad_svd1 < dg2 * max_m + max_b)
            & (qad_svd1 > slope_thresh * (dg2 * max_m + max_b))
        )

    # Offset by threshold
    dg2 = dg2 - dg2_thresh

    uall_det_means.append(det_means)
    all_det_means.append(det_means[bools])
    uall_qad.append(qad_svd1)
    all_qad.append(qad_svd1[bools])
    uall_dg2.append(dg2)
    all_dg2.append(dg2[bools])

    # QAD
    ax = qax.ravel()[ii]
    ransac = RANSACRegressor()
    ransac.fit(det_means[bools].reshape(-1, 1), qad_svd1[bools].reshape(-1, 1))
    qad_ransac = ransac.predict(det_means[bools].reshape(-1, 1))
    qad_b, qad_m = polyfit(det_means[bools], qad_svd1[bools], 1)
    ax.scatter(det_means[bools], qad_svd1[bools], alpha=0.08)
    ax.plot(det_means[bools], qad_ransac, "k", lw=2)
    # ax.plot(det_means[bools], det_means[bools] * qad_m + qad_b, "k", lw=2)
    ax.set_title(
        f"Run {run}, corr = {np.corrcoef(det_means[bools], qad_svd1[bools])[0,1]:1.2f}"
    )
    # ax.set_ylim(1, 3)
    #
    # Robustly fit linear model with RANSAC algorithm

    # qad_b, qad_m = polyfit(det_means, qad_svd1, 1)
    #

    # ax.scatter(det_means, qad_svd1, alpha=0.08)
    # # ax.plot(det_means, det_means * qad_m + qad_b, "k", lw=2)
    # ax.set_title(f"Run {run}, corr = {np.corrcoef(det_means, qad_svd1)[0,1]:1.2f}")

    # ax.set_xlabel("det_means")
    ax.set_ylabel("qad_svd1")
    qfig.suptitle(f"{set_name} det_mean, qad correlation (unfiltered)")

    # DG2
    ax = dax.ravel()[ii]
    # dg2_b, dg2_m = polyfit(det_means[bools], dg2[bools], 1)
    ransac = RANSACRegressor()
    ransac.fit(det_means[bools].reshape(-1, 1), dg2[bools].reshape(-1, 1))
    dg2_ransac = ransac.predict(det_means[bools].reshape(-1, 1))
    ax.scatter(det_means[bools], dg2[bools], alpha=0.08)
    # ax.plot(det_means[bools], det_means[bools] * dg2_m + dg2_b, "k", lw=2)
    # ax.plot(det_means[bools], det_means[bools] * dg2_m + dg2_b, "k", lw=2)
    ax.plot(det_means[bools], dg2_ransac, "k", lw=2)
    est = ransac.estimator_
    dg2_m = est.coef_[0]
    dg2_b = est.intercept_[0]
    ax.plot(det_means[bools], det_means[bools] * dg2_m + dg2_b, "r", lw=2)
    ax.set_title(
        f"Run {run}, corr = {np.corrcoef(det_means[bools], dg2[bools])[0,1]:1.2f}"
    )
    # ax.set_ylim(0, 400000)
    # dg2_b, dg2_m = polyfit(det_means, dg2, 1)
    # ax.scatter(det_means, dg2, alpha=0.08)
    # ax.plot(det_means[bools], det_means[bools] * dg2_m + dg2_b, "k", lw=2)
    # ax.set_title(f"Run {run}, corr = {np.corrcoef(det_means, dg2)[0,1]:1.2f}")

    # ax.set_xlabel("det_means")
    ax.set_ylabel("dg2")
    dfig.suptitle(f"{set_name} det_mean, dg2 correlation (unfiltered)")


# aax[0].scatter(all_det_means.flatten(), all_dg2.flatten(), alpha=0.05)
# aax[1].scatter(all_det_means.flatten(), all_qad.flatten(), alpha=0.05)
rlabels = [
    "noJet",
    "buffer",
    "sample",
]
for rnum in range(len(all_dg2)):
    aax[0].scatter(all_det_means[rnum], all_dg2[rnum], alpha=0.05, label=rlabels[rnum])
    aax[1].scatter(all_det_means[rnum], all_qad[rnum], alpha=0.05)

afig.suptitle(f"{set_name} NoJet, buffer, and sample filtered")
aax[0].set_xlabel("Detector means")
aax[1].set_xlabel("Detector means")
aax[0].set_ylabel("DG2 means")
aax[1].set_ylabel("QAD means")
aax[0].set_xlim(0, 2e8)
aax[1].set_xlim(0, 2e8)
# aax[0].set_xlim(0, 1.5e8)
# aax[1].set_xlim(0, 1.5e8)
leg = aax[0].legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)

# uax[0].scatter(uall_det_means.flatten(), uall_dg2.flatten(), alpha=0.05)
# uax[1].scatter(uall_det_means.flatten(), uall_qad.flatten(), alpha=0.05)
for rnum in range(len(uall_dg2)):
    uax[0].scatter(uall_det_means[rnum], uall_dg2[rnum], alpha=0.05)
    uax[1].scatter(uall_det_means[rnum], uall_qad[rnum], alpha=0.05)
ufig.suptitle(f"{set_name} NoJet, buffer, and sample unfiltered")
uax[0].set_xlabel("Detector means")
uax[1].set_xlabel("Detector means")
uax[0].set_ylabel("DG2 means")
uax[1].set_ylabel("QAD means")
uax[0].set_xlim(0, 2e8)
uax[1].set_xlim(0, 2e8)

# %% pick values

for rnum in range(len(all_dg2)):
    # plt.scatter(all_det_means[rnum], all_dg2[rnum], alpha=0.05, label=rlabels[rnum])
    # plt.hist(all_det_means[rnum], bins=50, label=rlabels[rnum])
    # plt.hist(
    #     all_qad[rnum],
    #     bins=50,
    #     label=f"{rlabels[rnum]}, mean {all_qad[rnum].mean():1.2f}, min {all_qad[rnum].min():1.2f}",
    # )
    plt.hist(
        all_det_means[rnum],
        bins=50,
        alpha=0.8,
        # label=f"{rlabels[rnum]}, mean {all_det_means[rnum].mean():1.2f}, min {all_det_means[rnum].min():1.2f}",
        label=f"{rlabels[rnum]}",
    )
leg = plt.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)

# %% Redo subtraction using SPB approximations
# Subtraction, scale parameters
dg2_thresh = 2e5
det_means_thresh = 1e7
qad_thresh = 1.5
qad_m_avg = 5.5e-06
qad_b_avg = 0.3
slope_thresh = 0.8
i_scale = 1e-9

# Estimate line from dg2, qad below thresh
dg2_bin_width = 5000
dg2_bin_edges = np.arange(0, dg2_thresh + dg2_bin_width, dg2_bin_width)
dg2_bin_centers = dg2_bin_edges[:-1] + (dg2_bin_edges[1:] - dg2_bin_edges[:-1]) / 2

# Loop through traces
labels = ["AT25_100uM", "AT25_1mM", "AT25_1mM_tsub", "AT48_250uM"]
# labels = ["AT25_100uM", "AT25_1mM", "AT25_1mM_tsub", "AT48_250uM", "SAMII_static"]
noJet_runs = [60, 66, 66, 66]
buff_runs = [61, 67, 70, 72]
sample_runs = [63, 69, 69, 74]
buff_scales = [0.994, 0.9985, 0.9985, 0.994]

cfig, caxs = plt.subplots(len(labels), figsize=(7, 10), layout="constrained")
pfig, paxs = plt.subplots(len(labels), figsize=(7, 10), layout="constrained")
for lnum in range(len(labels)):
    cax = caxs.ravel()[lnum]
    cax.set_title(f"{labels[lnum]} subtraction")
    pax = paxs.ravel()[lnum]
    pax.set_title(f"{labels[lnum]} SPB normalized traces")
    noJet_run = noJet_runs[lnum]
    buff_run = buff_runs[lnum]
    sample_run = sample_runs[lnum]
    buff_scale = buff_scales[lnum]

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

    # Normalization

    buff_qad_maxes = [
        np.max(buff_qad_svd1[(buff_dg2 > ij) & (buff_dg2 < ik)])
        for ij, ik in zip(dg2_bin_edges[:-1], dg2_bin_edges[1:])
    ]
    buff_max_b, buff_max_m = polyfit(dg2_bin_centers, buff_qad_maxes, 1)

    buff_inds = (
        (buff_dg2 > dg2_thresh)
        & (buff_qad_svd1 < buff_dg2 * buff_max_m + buff_max_b)
        & (buff_qad_svd1 > slope_thresh * (buff_dg2 * buff_max_m + buff_max_b))
    )
    buff_noq_inds = np.mean(buff_radials, 0) == 0
    buff_q_inds = ~buff_noq_inds

    buff_mean = np.nanmean(
        buff_radials[buff_inds] / buff_qad_svd1[buff_inds, np.newaxis], axis=0
    )
    buff_mean = buff_mean[buff_q_inds]

    sample_qad_maxes = [
        np.max(sample_qad_svd1[(sample_dg2 > ij) & (sample_dg2 < ik)])
        for ij, ik in zip(dg2_bin_edges[:-1], dg2_bin_edges[1:])
    ]
    sample_max_b, sample_max_m = polyfit(dg2_bin_centers, sample_qad_maxes, 1)

    sample_inds = (
        (sample_dg2 > dg2_thresh)
        & (sample_qad_svd1 < sample_dg2 * sample_max_m + sample_max_b)
        & (sample_qad_svd1 > slope_thresh * (sample_dg2 * sample_max_m + sample_max_b))
    )
    sample_noq_inds = np.mean(sample_radials, 0) == 0
    sample_q_inds = ~sample_noq_inds
    sample_mean = np.nanmean(
        sample_radials[sample_inds] / sample_qad_svd1[sample_inds, np.newaxis],
        axis=0,
    )
    sample_mean = sample_mean[sample_q_inds]

    pax.plot(sample_mean)
    pax.plot(buff_mean)

    # %% Subtraction

    sub = sample_mean - (buff_scale) * buff_mean
    cax.plot(q[sample_q_inds] / 1e10, sub)

    # save
    inds = ~np.isinf(sub) & (sub > 0)
    sub_csv_name = f"{labels[lnum]}_sub_qad.csv"
    sub_df = pd.DataFrame({"q": q[sample_q_inds] / 1e10, "I": sub * i_scale})
    sub_df.to_csv(data_dir / sub_csv_name, index=False)

    inds = ~np.isinf(sample_mean) & (sample_mean > 0)
    sample_csv_name = f"{labels[lnum]}_qad.csv"
    sample_df = pd.DataFrame({"q": q[sample_q_inds] / 1e10, "I": sample_mean * i_scale})
    sample_df.to_csv(data_dir / sample_csv_name, index=False)

    inds = ~np.isinf(buff_mean) & (buff_mean > 0)
    buff_csv_name = f"{labels[lnum]}_buff_qad.csv"
    buff_df = pd.DataFrame({"q": q[sample_q_inds] / 1e10, "I": buff_mean * i_scale})
    buff_df.to_csv(data_dir / buff_csv_name, index=False)

# %% Check SPB approximation using Hugh files

# run_numbers = [72, 74]
run_numbers = [61, 63]
dg2_thresh = 2e5
det_means_thresh = 1e7
qad_thresh = 1.5
qad_m_avg = 5.5e-06
qad_b_avg = 0.3
slope_thresh = 0.8

all_qad = []
all_dg2 = []
all_bools = []
all_spb = []
cfig, cax = plt.subplots(1, 2, layout="constrained")  # Look at scatter plots
pfig, pax = plt.subplots(1, 2, layout="constrained")  #
for ii, run in enumerate(run_numbers):
    h5e_path = data_dir / f"{exp}_r{run:04d}_epix.h5"
    with h5py.File(h5e_path, "r") as h5e:
        fids = h5e["/frame_id"][:]
        radials = h5e["/mean"][:]
        dg2 = h5e["/dg2"][:]
        qad_svd1 = h5e["/qad_svd1"][:]
        radial_offset = h5e["/radial_offset"]
        qad_offset = h5e["/qad_offset"]
        q = h5e["/q_bins"][:]

    ind_file = f"Hugh{run}.txt"
    df = pd.read_csv(data_dir / ind_file, header=None, sep=" ")
    inds = df.iloc[:, 0].to_numpy() - 1
    bools = np.full(dg2.shape, False)
    bools[inds] = True
    spb = df.iloc[:, 2].to_numpy()

    det_means = np.mean(radials, axis=1)

    # Try to calculate lines of best fit to compare
    qad_b, qad_m = polyfit(dg2[bools], qad_svd1[bools], 1)
    print(f"Intercept: {qad_b}, slope: {qad_m}")

    # Estimate line from dg2, qad below thresh
    dg2_bin_width = 5000
    dg2_bin_edges = np.arange(0, dg2_thresh + dg2_bin_width, dg2_bin_width)
    dg2_bin_centers = dg2_bin_edges[:-1] + (dg2_bin_edges[1:] - dg2_bin_edges[:-1]) / 2

    qad_maxes = [
        np.max(qad_svd1[(dg2 > ij) & (dg2 < ik)])
        for ij, ik in zip(dg2_bin_edges[:-1], dg2_bin_edges[1:])
    ]
    max_b, max_m = polyfit(dg2_bin_centers, qad_maxes, 1)

    cfig.suptitle("Diode, sum_peak_buff correlation")
    ax = cax.ravel()[ii]

    test_inds = (
        (dg2 > dg2_thresh)
        & (qad_svd1 < dg2 * max_m + max_b)
        & (qad_svd1 > slope_thresh * (dg2 * max_m + max_b))
    )
    # ax.scatter(dg2[~obools], qad_svd1[~obools], alpha=1, label="rejected")
    # ax.scatter(dg2[obools], qad_svd1[obools], alpha=1, label="Open filters")
    # ax.scatter(dg2[bools], qad_svd1[bools], alpha=1, label="tight SPB")
    # ax.scatter(dg2[test_inds], qad_svd1[test_inds], alpha=0.1)

    # Compare spb values
    # Estimate slope
    # spb_b, spb_m = polyfit(qad_svd1[obools], ospb, 1)
    spb_b, spb_m = polyfit(qad_svd1[bools], spb, 1)

    # ax.scatter(qad_svd1[obools], ospb, alpha=0.08)
    # ax.plot(qad_svd1[obools], qad_svd1[obools] * spb_m + spb_b, "k", lw=2)
    # ax.set_title(f"Run {run}, corr = {np.corrcoef(qad_svd1[obools], ospb)[0,1]:1.2f}")

    # ax.scatter(qad_svd1[bools], spb, alpha=0.08)
    # ax.plot(qad_svd1[bools], qad_svd1[bools] * spb_m + spb_b, "k", lw=2)
    # ax.set_title(f"Run {run}, corr = {np.corrcoef(qad_svd1[bools], spb)[0,1]:1.2f}")
    ax.scatter(qad_svd1[bools], spb, alpha=0.08)
    ax.plot(qad_svd1[bools], qad_svd1[bools] * spb_m + spb_b, "k", lw=2)
    ax.set_title(f"Run {run}, corr = {np.corrcoef(qad_svd1[bools], spb)[0,1]:1.2f}")

    ax.set_xlabel("qad_svd1")
    ax.set_ylabel("sum_peak_buff")

    noq_inds = np.mean(radials, 0) == 0
    q_inds = ~noq_inds

    tmeans_v0 = np.average(radials[test_inds, :], axis=0)
    tmeans = np.average(radials[test_inds, :], axis=0, weights=qad_svd1[test_inds])
    smeans = np.average(radials[bools, :], axis=0, weights=spb)
    smeans_v0 = np.average(radials[bools, :], axis=0)
    # smeans = np.mean(radials[bools, :], 0)
    ax = pax.ravel()[ii]
    # ax.set_sup
    ax.set_title(f"Run {run}")
    ax.set_xlabel("q ($\\AA^{-1}$)")
    # ax.plot(q[q_inds], smeans[q_inds])
    ax.plot(q[q_inds], tmeans[q_inds])
    # ax.plot(q[q_inds], smeans_v0[q_inds])
    ax.plot(q[q_inds], tmeans_v0[q_inds])

# cax.ravel()[-1].legend()

# %% Check SPB approximation using closed files

run_numbers = [67, 69]
# run_numbers = [72, 74]
dg2_thresh = 2e5
det_means_thresh = 1e7
qad_thresh = 1.5
qad_m_avg = 5.5e-06
qad_b_avg = 0.3
slope_thresh = 0.8

all_qad = []
all_dg2 = []
all_bools = []
all_spb = []
cfig, cax = plt.subplots(1, 2, layout="constrained")  # Look at scatter plots
# pfig, pax = plt.subplots(1, 2, layout="constrained")  #
for ii, run in enumerate(run_numbers):
    h5e_path = data_dir / f"{exp}_r{run:04d}_epix.h5"
    with h5py.File(h5e_path, "r") as h5e:
        fids = h5e["/frame_id"][:]
        radials = h5e["/mean"][:]
        dg2 = h5e["/dg2"][:]
        qad_svd1 = h5e["/qad_svd1"][:]
        radial_offset = h5e["/radial_offset"]
        qad_offset = h5e["/qad_offset"]
        q = h5e["/q_bins"][:]

    ind_file = f"Hugh{run}.txt"
    df = pd.read_csv(data_dir / ind_file, header=None, sep=" ")
    inds = df.iloc[:, 0].to_numpy() - 1
    bools = np.full(dg2.shape, False)
    bools[inds] = True
    spb = df.iloc[:, 2].to_numpy()

    det_means = np.mean(radials, axis=1)

    # Try to calculate lines of best fit to compare
    # qad_b, qad_m = polyfit(dg2[bools], qad_svd1[bools], 1)
    # print(f"Intercept: {qad_b}, slope: {qad_m}")

    det_b, det_m = polyfit(det_means[bools], qad_svd1[bools], 1)
    print(f"Intercept: {det_b}, slope: {det_m}")

    # Estimate line from dg2, qad below thresh
    dg2_bin_width = 5000
    dg2_bin_edges = np.arange(0, dg2_thresh + dg2_bin_width, dg2_bin_width)
    dg2_bin_centers = dg2_bin_edges[:-1] + (dg2_bin_edges[1:] - dg2_bin_edges[:-1]) / 2

    qad_maxes = [
        np.max(qad_svd1[(dg2 > ij) & (dg2 < ik)])
        for ij, ik in zip(dg2_bin_edges[:-1], dg2_bin_edges[1:])
    ]
    max_b, max_m = polyfit(dg2_bin_centers, qad_maxes, 1)

    cfig.suptitle("det_means, qad_svd1 correlation (good traces)")
    ax = cax.ravel()[ii]

    test_inds = (
        (dg2 > dg2_thresh)
        & (qad_svd1 < dg2 * max_m + max_b)
        & (qad_svd1 > slope_thresh * (dg2 * max_m + max_b))
    )
    spb_b, spb_m = polyfit(det_means[bools], spb, 1)

    ax.scatter(det_means[bools], qad_svd1[bools], alpha=0.08)
    ax.plot(det_means[bools], det_means[bools] * det_m + det_b, "k", lw=2)
    ax.set_title(
        f"Run {run}, corr = {np.corrcoef(det_means[bools], qad_svd1[bools])[0,1]:1.2f}"
    )

    # ax.set_xlim([0, 2e8])
    ax.set_xlabel("Mean intensity")
    ax.set_ylabel("sum_peak_buff")

    noq_inds = np.mean(radials, 0) == 0
    q_inds = ~noq_inds

    tmeans_v0 = np.average(radials[test_inds, :], axis=0)
    tmeans = np.average(radials[test_inds, :], axis=0, weights=qad_svd1[test_inds])
    smeans = np.average(radials[bools, :], axis=0, weights=spb)
    smeans_v0 = np.average(radials[bools, :], axis=0)
    # smeans = np.mean(radials[bools, :], 0)
    # ax = pax.ravel()[ii]
    # ax.set_sup
    # ax.set_title(f"Run {run}")
    # ax.set_xlabel("q ($\\AA^{-1}$)")
    # # ax.plot(q[q_inds], smeans[q_inds])
    # ax.plot(q[q_inds], tmeans[q_inds])
    # # ax.plot(q[q_inds], smeans_v0[q_inds])
    # ax.plot(q[q_inds], tmeans_v0[q_inds])

# cax.ravel()[-1].legend()

# %% Check SPB approximation using open files

run_numbers = [61, 63]
dg2_thresh = 2e5
det_means_thresh = 1e7
qad_thresh = 1.5
qad_m_avg = 5.5e-06
qad_b_avg = 0.3
slope_thresh = 0.8

all_qad = []
all_dg2 = []
all_bools = []
all_spb = []
cfig, cax = plt.subplots(1, 2, layout="constrained")  # Look at scatter plots
pfig, pax = plt.subplots(1, 2, layout="constrained")  #
for ii, run in enumerate(run_numbers):
    h5e_path = data_dir / f"{exp}_r{run:04d}_epix.h5"
    with h5py.File(h5e_path, "r") as h5e:
        fids = h5e["/frame_id"][:]
        radials = h5e["/mean"][:]
        dg2 = h5e["/dg2"][:]
        qad_svd1 = h5e["/qad_svd1"][:]
        radial_offset = h5e["/radial_offset"]
        qad_offset = h5e["/qad_offset"]
        q = h5e["/q_bins"][:]

    ind_file = f"Hugh{run}.txt"
    df = pd.read_csv(data_dir / ind_file, header=None, sep=" ")
    inds = df.iloc[:, 0].to_numpy() - 1
    bools = np.full(dg2.shape, False)
    bools[inds] = True
    spb = df.iloc[:, 2].to_numpy()

    oind_file = f"Hugh{run}open.txt"
    odf = pd.read_csv(data_dir / oind_file, header=None, sep=" ")
    oinds = odf.iloc[:, 0].to_numpy() - 1
    obools = np.full(dg2.shape, False)
    obools[oinds] = True
    ospb = odf.iloc[:, 2].to_numpy()

    det_means = np.mean(radials, axis=1)

    # Try to calculate lines of best fit to compare
    qad_b, qad_m = polyfit(dg2[bools], qad_svd1[bools], 1)
    print(f"Intercept: {qad_b}, slope: {qad_m}")

    # Estimate line from dg2, qad below thresh
    dg2_bin_width = 5000
    dg2_bin_edges = np.arange(0, dg2_thresh + dg2_bin_width, dg2_bin_width)
    dg2_bin_centers = dg2_bin_edges[:-1] + (dg2_bin_edges[1:] - dg2_bin_edges[:-1]) / 2

    qad_maxes = [
        np.max(qad_svd1[(dg2 > ij) & (dg2 < ik)])
        for ij, ik in zip(dg2_bin_edges[:-1], dg2_bin_edges[1:])
    ]
    max_b, max_m = polyfit(dg2_bin_centers, qad_maxes, 1)

    cfig.suptitle("Diode, sum_peak_buff correlation")
    ax = cax.ravel()[ii]

    test_inds = (
        (dg2 > dg2_thresh)
        & (qad_svd1 < dg2 * max_m + max_b)
        & (qad_svd1 > slope_thresh * (dg2 * max_m + max_b))
    )
    # ax.scatter(dg2[~obools], qad_svd1[~obools], alpha=1, label="rejected")
    # ax.scatter(dg2[obools], qad_svd1[obools], alpha=1, label="Open filters")
    # ax.scatter(dg2[bools], qad_svd1[bools], alpha=1, label="tight SPB")
    # ax.scatter(dg2[test_inds], qad_svd1[test_inds], alpha=0.1)

    # Compare spb values
    # Estimate slope
    # spb_b, spb_m = polyfit(qad_svd1[obools], ospb, 1)
    spb_b, spb_m = polyfit(det_means[obools], ospb, 1)

    # ax.scatter(qad_svd1[obools], ospb, alpha=0.08)
    # ax.plot(qad_svd1[obools], qad_svd1[obools] * spb_m + spb_b, "k", lw=2)
    # ax.set_title(f"Run {run}, corr = {np.corrcoef(qad_svd1[obools], ospb)[0,1]:1.2f}")
    # ax.scatter(det_means[obools], ospb, alpha=0.08)
    # ax.plot(det_means[obools], qad_svd1[obools] * spb_m + spb_b, "k", lw=2)
    # ax.set_title(f"Run {run}, corr = {np.corrcoef(qad_svd1[obools], ospb)[0,1]:1.2f}")
    ax.scatter(det_means[obools], qad_svd1[obools], alpha=0.08)
    # ax.plot(det_means[obools], qad_svd1[obools] * spb_m + spb_b, "k", lw=2)
    ax.set_title(
        f"Run {run}, corr = {np.corrcoef(det_means[bools], qad_svd1[bools])[0,1]:1.2f}"
    )

    ax.set_xlabel("qad_svd1")
    ax.set_ylabel("sum_peak_buff")

    noq_inds = np.mean(radials, 0) == 0
    q_inds = ~noq_inds

    tmeans_v0 = np.average(radials[test_inds, :], axis=0)
    tmeans = np.average(radials[test_inds, :], axis=0, weights=qad_svd1[test_inds])
    smeans = np.average(radials[bools, :], axis=0, weights=spb)
    smeans_v0 = np.average(radials[bools, :], axis=0)
    # smeans = np.mean(radials[bools, :], 0)
    ax = pax.ravel()[ii]
    # ax.set_sup
    ax.set_title(f"Run {run}")
    ax.set_xlabel("q ($\\AA^{-1}$)")
    # ax.plot(q[q_inds], smeans[q_inds])
    ax.plot(q[q_inds], tmeans[q_inds])
    # ax.plot(q[q_inds], smeans_v0[q_inds])
    ax.plot(q[q_inds], tmeans_v0[q_inds])

# cax.ravel()[-1].legend()

# %% Check correlations with SPB and diode: blinded by run numbers
# run_numbers = [63, 67, 69]  # AT25 only
run_numbers = [61, 63, 67, 69]  # AT25 only
# run_numbers = [61, 63, 67, 69]  # AT25 only
run_numbers = [61, 63, 67, 69, 72, 74]  # All runs
# run_numbers = [74]  # All runs

dg2_thresh = 2e5
det_means_thresh = 1e7
qad_thresh = 1.5
qad_m_avg = 5.5e-06
qad_b_avg = 0.3
slope_thresh = 0.8

all_qad = []
all_dg2 = []
all_bools = []
all_spb = []
cfig, cax = plt.subplots(2, 3, layout="constrained")  # Look at scatter plots
pfig, pax = plt.subplots(2, 3, layout="constrained")  #
for ii, run in enumerate(run_numbers):
    h5e_path = data_dir / f"{exp}_r{run:04d}_epix.h5"
    with h5py.File(h5e_path, "r") as h5e:
        fids = h5e["/frame_id"][:]
        radials = h5e["/mean"][:]
        dg2 = h5e["/dg2"][:]
        qad_svd1 = h5e["/qad_svd1"][:]
        radial_offset = h5e["/radial_offset"]
        qad_offset = h5e["/qad_offset"]
        q = h5e["/q_bins"][:]

    ind_file = f"Hugh{run}.txt"
    df = pd.read_csv(data_dir / ind_file, header=None, sep=" ")
    inds = df.iloc[:, 0].to_numpy() - 1
    bools = np.full(dg2.shape, False)
    bools[inds] = True
    spb = df.iloc[:, 2].to_numpy()

    det_means = np.mean(radials, axis=1)

    # Try to calculate lines of best fit to compare
    qad_b, qad_m = polyfit(dg2[bools], qad_svd1[bools], 1)
    print(f"Intercept: {qad_b}, slope: {qad_m}")

    # Estimate line from dg2, qad below thresh
    dg2_bin_width = 5000
    dg2_bin_edges = np.arange(0, dg2_thresh + dg2_bin_width, dg2_bin_width)
    dg2_bin_centers = dg2_bin_edges[:-1] + (dg2_bin_edges[1:] - dg2_bin_edges[:-1]) / 2

    qad_maxes = [
        np.max(qad_svd1[(dg2 > ij) & (dg2 < ik)])
        for ij, ik in zip(dg2_bin_edges[:-1], dg2_bin_edges[1:])
    ]
    max_b, max_m = polyfit(dg2_bin_centers, qad_maxes, 1)

    ax = cax.ravel()[ii]

    test_inds = (
        (dg2 > dg2_thresh)
        & (qad_svd1 < dg2 * max_m + max_b)
        & (qad_svd1 > slope_thresh * (dg2 * max_m + max_b))
    )
    ax.scatter(dg2[~bools], qad_svd1[~bools], alpha=0.1)
    ax.scatter(dg2[bools], qad_svd1[bools], alpha=0.1)
    # ax.scatter(dg2[test_inds], qad_svd1[test_inds], alpha=0.1)
    # ax.plot(dg2, dg2 * qad_m + qad_b, "k", lw=2)
    # # ax.plot(dg2, dg2 * qad_m_avg + qad_b_avg, "r", lw=2)
    # ax.plot(dg2, dg2 * max_m + max_b, "r", lw=2)
    # ax.plot(dg2, dg2 * (max_m * 0.7) + max_b, "r", lw=2)
    # ax.plot(dg2, dg2 * (qad_m_avg - qad_m_avg / 5) + qad_b_avg, "r", lw=2)
    # ax.plot(dg2, dg2 * (qad_m_avg + qad_m_avg / 5) + qad_b_avg, "r", lw=2)
    # ax.plot(dg2, dg2 * qad_m_avg + (qad_b_avg + qad_b_avg / 2), "r", lw=2)
    # ax.plot(dg2, dg2 * qad_m_avg + (qad_b_avg - qad_b_avg / 2), "r", lw=2)

    # Compare spb values
    # ax.scatter(det_means[bools], spb)

    # test_inds = (det_means > det_means_thresh) & (qad_svd1 > qad_thresh)
    # plt.scatter(det_means[~inds], qad_svd1[~inds])
    # plt.scatter(det_means[test_inds], qad_svd1[test_inds])
    # plt.scatter(det_means[inds], qad_svd1[inds])
    # break

    noq_inds = np.mean(radials, 0) == 0
    q_inds = ~noq_inds

    tmeans = np.mean(radials[test_inds, :], 0)
    smeans = np.mean(radials[bools, :], 0)
    ax = pax.ravel()[ii]
    ax.plot(q[q_inds], smeans[q_inds])
    ax.plot(q[q_inds], tmeans[q_inds])


# %% Double check run 74
#
run = 74
h5e_path = data_dir / f"{exp}_r{run:04d}_epix.h5"
with h5py.File(h5e_path, "r") as h5e:
    fids = h5e["/frame_id"][:]
    radials = h5e["/mean"][:]
    dg2 = h5e["/dg2"][:]
    qad_svd1 = h5e["/qad_svd1"][:]
    radial_offset = h5e["/radial_offset"]
    qad_offset = h5e["/qad_offset"]
    q = h5e["/q_bins"][:]

ind_file = f"Hugh{run}.txt"
df = pd.read_csv(data_dir / ind_file, header=None, sep=" ")
inds = df.iloc[:, 0].to_numpy() - 1
spb = df.iloc[:, 2].to_numpy()


# %% Load data for testing normalization schemes

# AT25 100 uM
noJet_run = 60
buff_run = 61
sample_run = 63
# sample_run = 67

# AT25 1 mM
noJet_run = 66
buff_run = 67
sample_run = 69

# AT48 200 uM no
# noJet_run = 66
# buff_run = 72
# sample_run = 74

noJet_h5e_path = data_dir / f"{exp}_r{noJet_run:04d}_epix.h5"
buff_h5e_path = data_dir / f"{exp}_r{buff_run:04d}_epix.h5"
sample_h5e_path = data_dir / f"{exp}_r{sample_run:04d}_epix.h5"

i_scale = 1e-10

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

buff_mean = np.nanmean(
    np.array([buff_radials[idx] / buff_spb[ii] for ii, idx in enumerate(buff_inds)]),
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

noq_inds = (sample_mean == 0) & (buff_mean == 0)
q_inds = ~noq_inds
q = q[q_inds]
buff_mean = buff_mean[q_inds]
sample_mean = sample_mean[q_inds]
# %% No Jet background

# coefs = polyfit(buff_qad_svd1[qad_inds], rmeds[qad_inds], 1)
qad_thresh = 0.8
noJet_mean = np.nanmean(
    np.array(
        [radial / noJet_qad for radial, noJet_qad in zip(noJet_radials, noJet_qad_svd1)]
    ),
    axis=0,
)
noJet_mean = noJet_mean[q_inds]

# Simple scaling
noJet_buff_scaled = noJet_mean / np.mean(noJet_mean) * np.mean(buff_mean)
noJet_buff_scaled = noJet_buff_scaled - np.max(noJet_buff_scaled - buff_mean)

# plt.plot(q / 1e10, noJet_buff_scaled)
# plt.plot(q / 1e10, buff_mean)

noJet_sample_scaled = noJet_mean / np.mean(noJet_mean) * np.mean(sample_mean)
noJet_sample_scaled = noJet_sample_scaled - np.max(noJet_sample_scaled - sample_mean)

# plt.plot(q / 1e10, noJet_sample_scaled)
# plt.plot(q / 1e10, sample_mean)

# Project along SVD direction: response function over course of noJet
noJet_nframes = len(noJet_radials)
jet_svd1 = np.zeros((noJet_nframes, 1))

svd1 = TruncatedSVD(n_components=1)
jet_svd = svd1.fit(noJet_radials[:, q_inds])

print("Performing SVD transform")

j = 0
for ii in range(noJet_nframes):
    jet_svd1[ii] = jet_svd.transform(noJet_radials[ii, q_inds].reshape(1, -1))[0]


buff_svd1 = jet_svd.transform(buff_mean.reshape(1, -1))
buff_svd1_comp = jet_svd.inverse_transform(buff_svd1)
buff_svd1_comp = buff_svd1_comp.flatten()

sample_svd1 = jet_svd.transform(sample_mean.reshape(1, -1))
sample_svd1_comp = jet_svd.inverse_transform(sample_svd1)
sample_svd1_comp = sample_svd1_comp.flatten()

buff_noSVD = buff_mean - buff_svd1_comp + np.mean(buff_svd1_comp)
sample_noSVD = sample_mean - sample_svd1_comp + np.mean(sample_svd1_comp)
plt.plot(sample_noSVD)
plt.plot(buff_noSVD)


jet_sub = sample_noSVD - 0.994 * buff_noSVD
# jet_sub_csv_name = "AT48_jet_sub_spb.csv"
# jet_sub_csv_name = "AT25_1mM_jet_sub_spb.csv"
jet_sub_csv_name = "AT25_jet_sub_spb.csv"
jet_sub_df = pd.DataFrame({"q": q / 1e10, "I": jet_sub * i_scale})
# jet_sub_df.to_csv(data_dir / jet_sub_csv_name, index=False)

# %% Subtraction: AT48

sub = sample_mean - (0.994) * buff_mean
plt.plot(q / 1e10, sub)
i_scale = 1e-10

# save
inds = ~np.isinf(sub) & (sub > 0)
sub_csv_name = "AT48_sub_spb.csv"
sub_df = pd.DataFrame({"q": q[inds] / 1e10, "I": sub[inds] * i_scale})
sub_df.to_csv(data_dir / sub_csv_name, index=False)

inds = ~np.isinf(sample_mean) & (sample_mean > 0)
sample_csv_name = "AT48_spb.csv"
sample_df = pd.DataFrame({"q": q[inds] / 1e10, "I": sample_mean[inds] * i_scale})
sample_df.to_csv(data_dir / sample_csv_name, index=False)

inds = ~np.isinf(buff_mean) & (buff_mean > 0)
buff_csv_name = "AT48_buff_spb.csv"
buff_df = pd.DataFrame({"q": q[inds] / 1e10, "I": buff_mean[inds] * i_scale})
buff_df.to_csv(data_dir / buff_csv_name, index=False)

# %% Subtraction: AT25

sub = sample_mean - (0.994) * buff_mean
plt.plot(q / 1e10, sub)
i_scale = 1e-10

# save
inds = ~np.isinf(sub) & (sub > 0)
sub_csv_name = "AT25_100uM_sub_spb.csv"
sub_df = pd.DataFrame({"q": q[inds] / 1e10, "I": sub[inds] * i_scale})
sub_df.to_csv(data_dir / sub_csv_name, index=False)

inds = ~np.isinf(sample_mean) & (sample_mean > 0)
sample_csv_name = "AT25_100uM_spb.csv"
sample_df = pd.DataFrame({"q": q[inds] / 1e10, "I": sample_mean[inds] * i_scale})
sample_df.to_csv(data_dir / sample_csv_name, index=False)

inds = ~np.isinf(buff_mean) & (buff_mean > 0)
buff_csv_name = "AT25_100uM_buff_spb.csv"
buff_df = pd.DataFrame({"q": q[inds] / 1e10, "I": buff_mean[inds] * i_scale})
buff_df.to_csv(data_dir / buff_csv_name, index=False)

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

# %% Diode comparison
