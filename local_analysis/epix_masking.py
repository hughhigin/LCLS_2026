#!/usr/bin/env python3

# %% Imports, filepaths

# IO
from pathlib import Path
import pickle
import h5py

# Fitting
import numpy as np
from numpy.polynomial.polynomial import polyfit
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

# Filepaths
exp = "cxi101672626"
data_dir = Path("/home/hugh/beamtime_processing/LCLS_20260629/epix_reMask/")

buff_run = 61
sample_run = 63

buff_run = 67
sample_run = 69

# buff_run = 72
# sample_run = 74

buff_h5e_path = data_dir / f"{exp}_r{buff_run:04d}_epix.h5"
sample_h5e_path = data_dir / f"{exp}_r{sample_run:04d}_epix.h5"

# %% General plotting settings

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams.update({"font.size": 14})

# %% Load mean detector values from radials

with h5py.File(buff_h5e_path, "r") as h5e:
    buff_radials = h5e["/mean"][:]
    buff_dg2 = h5e["/dg2"][:]
    buff_qad_svd1 = h5e["/qad_svd1"][:]
    buff_radial_offset = h5e["/radial_offset"]
    buff_qad_offset = h5e["/qad_offset"]
    q = h5e["/q_bins"][:]

with h5py.File(sample_h5e_path, "r") as h5e:
    at_radials = h5e["/mean"][:]
    at_dg2 = h5e["/dg2"][:]
    at_qad_svd1 = h5e["/qad_svd1"][:]
    at_radial_offset = h5e["/radial_offset"]
    at_qad_offset = h5e["/qad_offset"]


# %% Check diode correlations: buffer

# 4 panel figure of correlations: Dg2
buffE_tmeans = buff_radials.mean(axis=1)
atE_tmeans = at_radials.mean(axis=1)

cfig, caxs = plt.subplots(2, 2, layout="constrained")
caxs[0, 0].scatter(buff_dg2, buffE_tmeans)
caxs[0, 0].set_ylabel("dg2")
caxs[0, 0].set_ylim([-0.5 * 1e8, 2 * 1e8])

caxs[1, 0].scatter(buff_qad_svd1, buffE_tmeans)
caxs[1, 0].set_ylabel("qadc01")
caxs[1, 0].set_xlabel(f"Epix, run {buff_run}")
caxs[1, 0].set_ylim([-0.5 * 1e8, 2 * 1e8])

caxs[0, 1].scatter(at_dg2, atE_tmeans)
caxs[0, 1].set_ylim([-0.5 * 1e8, 3 * 1e8])
caxs[1, 1].scatter(at_qad_svd1, atE_tmeans)
caxs[1, 1].set_ylim([-0.5 * 1e8, 3 * 1e8])
caxs[1, 1].set_xlabel(f"Epix, run {sample_run}")

# Correlation coefficients

buff_dg2_corr = np.corrcoef(buff_dg2, buffE_tmeans)
buff_qad_corr = np.corrcoef(buff_qad_svd1, buffE_tmeans)
at_dg2_corr = np.corrcoef(at_dg2, atE_tmeans)
at_qad_corr = np.corrcoef(at_qad_svd1, atE_tmeans)

print(buff_dg2_corr)
print(buff_qad_corr)
print(at_dg2_corr)
print(at_qad_corr)

# %% Strongest diode correlation above QAD=0.8?

qad_inds = buff_qad_svd1 > 0.8
# Calculate correlations

# Plot as a function of Q
bcorrmat = np.corrcoef(np.transpose(buff_radials), buff_qad_svd1)
scorrmat = np.corrcoef(np.transpose(at_radials), at_qad_svd1)
bdcorrmat = np.corrcoef(np.transpose(buff_radials), buff_dg2)
sdcorrmat = np.corrcoef(np.transpose(at_radials), at_dg2)
plt.plot(q * 1e-10, bcorrmat[:-1, -1], lw=2, label=f"buff ({buff_run}) qad")
plt.plot(q * 1e-10, scorrmat[:-1, -1], lw=2, label=f"sample ({sample_run}) qad")
plt.plot(q * 1e-10, bdcorrmat[:-1, -1], lw=2, label=f"buff ({buff_run}) dg2")
plt.plot(q * 1e-10, sdcorrmat[:-1, -1], lw=2, label=f"sample ({sample_run}) dg2")
plt.xlabel("q (inverse Angstroms)")
plt.ylabel("Pearson correlation with QAD SVD1")
# plt.ylim([0.9, 1])
plt.title("Epix intensity backscattering correlation")
plt.legend()

# %% Scatter different metrics for fitting

# plt.scatter(nz_dg2, nz_rmean / nz_dg2, label = "Epix / dg2")
plt.scatter(nz_vnum, nz_rmean, label="Epix averaged signal")
# plt.scatter(nz_vnum, nz_rmean / nz_vnum, label = "Epix average / diode")
# plt.scatter(nz_dg2, nz_rmax / nz_dg2, label = "Epix / dg2")
plt.scatter(nz_vnum, nz_rmed, label="Epix averaged signal")
# plt.scatter(nz_vnum, (nz_rmed - coefs[0]) / nz_vnum, label = "Epix average / diode")
# plt.scatter(nz_vnum, nz_rmed / (nz_vnum - rcoefs[0]), label = "Epix average / diode")
# plt.scatter(nz_vnum, nz_rmean / nz_vnum**(5/3), label = "Epix average / diode")
# plt.scatter(nz_dg2, nz_rmean / nz_dg2)
# plt.xlim(0.1,3.5)
# plt.ylim(-1e7, 1e8)
plt.legend()
plt.xlabel("Diode values (first principal component)")
plt.ylabel("Average detector intensity")
plt.show()

# %% Median intensity

# Truncate fit at set value
buff_qad_svd1 = buff_qad_svd1.squeeze()
qad_thresh = 0.8
qad_inds = buff_qad_svd1 > qad_thresh

rmeds = np.median(buff_radials, axis=1)
coefs = polyfit(buff_qad_svd1[qad_inds], rmeds[qad_inds], 1)
rcoefs = polyfit(rmeds[qad_inds], buff_qad_svd1[qad_inds], 1)

plt.scatter(buff_qad_svd1, rmeds)
plt.scatter(buff_qad_svd1, rmeds / buff_qad_svd1)
plt.scatter(buff_qad_svd1, (rmeds - coefs[0]) / buff_qad_svd1)
plt.scatter(
    buff_qad_svd1[qad_inds], rmeds[qad_inds] / (buff_qad_svd1[qad_inds] - rcoefs[0])
)
plt.plot(buff_qad_svd1, buff_qad_svd1 * coefs[1] + coefs[0], "k", lw=2)


# %% Section Name


# Fit to two lines
def twoLine(x, x0, y0, k1, k2):
    return np.piecewise(
        x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0]
    )


p, e = curve_fit(twoLine, buff_qad_svd1, rmeds, p0=[0.7, 0.5, 1e7, 2e7])
vd = np.linspace(0, max(buff_qad_svd1), 1000)

radial_offset = p[1] - p[3] * p[0]
qad_offset = p[0] - p[1] / p[3]
plt.plot(buff_qad_svd1, rmeds, "o")
plt.plot(0, radial_offset, "r*", ms=4)
plt.plot(qad_offset, 0, "r*", ms=4)
plt.plot(vd, twoLine(vd, *p))

# %% Radials

r_step = 100
for i in range(0, buff_radials.shape[0], r_step):
    if buff_qad_svd1[i] > qad_thresh:
        # plt.plot(q, (buff_radials[i] - coefs[0]) / buff_qad_svd1[i])  # No normalization
        plt.plot(q, (buff_radials[i] - coefs[0]) / buff_dg2[i])  # No normalization
        # plt.plot(q, radials[i]/(buff_qad_svd1[i] - rcoefs[0]))  # No normalization
        # plt.plot(q, (radials[i])/buff_qad_svd1[i])  # No normalization
        # plt.plot(q, (radials[i]))  # No normalization

buff_mean = np.nanmean(
    np.array(
        [
            (radial - coefs[0]) / buff_qad
            for radial, buff_qad in zip(buff_radials, buff_qad_svd1)
            if (buff_qad > qad_thresh)
        ]
    ),
    axis=0,
)

# %% Radials
qad_thresh = 0.6
r_step = 10
for i in range(0, at_radials.shape[0], r_step):
    if at_qad_svd1[i] > qad_thresh:
        plt.plot(
            q * 1e-10, (at_radials[i] - coefs[0]) / at_qad_svd1[i]
        )  # No normalization
        # plt.plot(q, (at_radials[i] - coefs[0]) / at_dg2[i])  # No normalization
        # plt.plot(q, radials[i]/(buff_qad_svd1[i] - rcoefs[0]))  # No normalization
        # plt.plot(q, (radials[i])/buff_qad_svd1[i])  # No normalization
        # plt.plot(q, (radials[i]))  # No normalization

sample_mean = np.nanmean(
    np.array(
        [
            (radial - coefs[0]) / at_qad
            for radial, at_qad in zip(at_radials, at_qad_svd1)
            if (at_qad > qad_thresh)
        ]
    ),
    axis=0,
)
# %% Buffer subtraction attempts

plt.plot(buff_mean)
plt.plot(sample_mean)

# %% sub

plt.title(f"Naive subtraction, run {sample_run}")
plt.plot(q / 1e10, sample_mean - buff_mean)
plt.xlabel(r"q $\AA$$^{-1}$")
