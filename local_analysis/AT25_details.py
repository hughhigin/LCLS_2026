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
data_dir = Path("/home/hugh/beamtime_processing/LCLS_20260629/")

# Buffer data
buff_run = 61

buff_h5e_path = data_dir / f"{exp}_r{buff_run:04d}_epix.h5"
buff_h5j_path = data_dir / f"{exp}_r{buff_run:04d}_jungfrau.h5"

# buff_dg2 = np.load(data_dir / f"r{buff_run:04d}_dg2_totalIntensity.npy")
# buff_hfx = np.load(data_dir / f"r{buff_run:04d}_hfx_totalIntensity.npy")
# buff_em = np.load(data_dir / f"r{buff_run:04d}_em_totalIntensity.npy")
# buff_fee = np.load(data_dir / f"r{buff_run:04d}_fee.npy")
# buff_wave8 = np.load(data_dir / f"r{buff_run:04d}_wave8_waveform.npy")
# buff_qad = np.load(data_dir / f"r{buff_run:04d}_qadc01_waveform.npy")

# AT25 data
at_run = 63

at_h5e_path = data_dir / f"{exp}_r{at_run:04d}_epix.h5"
at_h5j_path = data_dir / f"{exp}_r{at_run:04d}_jungfrau.h5"

# at_dg2 = np.load(data_dir / f"r{at_run:04d}_dg2_totalIntensity.npy")
# at_hfx = np.load(data_dir / f"r{at_run:04d}_hfx_totalIntensity.npy")
# at_em = np.load(data_dir / f"r{at_run:04d}_em_totalIntensity.npy")
# at_fee = np.load(data_dir / f"r{at_run:04d}_fee.npy")
# at_wave8 = np.load(data_dir / f"r{at_run:04d}_wave8_waveform.npy")
# at_qad = np.load(data_dir / f"r{at_run:04d}_qadc01_waveform.npy")

# %% General plotting settings

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams.update({"font.size": 14})

# %% Load mean detector values from radials

# Buffer
with h5py.File(buff_h5e_path, "r") as h5e:  # pyright: ignore[reportArgumentType]
    buffE_tmeans = np.mean(
        h5e["mean"], axis=1
    )  # pyright: ignore[reportCallIssue, reportArgumentType]
with h5py.File(buff_h5j_path, "r") as h5j:
    buffJ_tmeans = np.mean(h5j["mean"], axis=1)

# AT25 sample
with h5py.File(at_h5e_path, "r") as h5e:
    atE_tmeans = np.mean(h5e["mean"], axis=1)
with h5py.File(at_h5j_path, "r") as h5j:
    atJ_tmeans = np.mean(h5j["mean"], axis=1)


# %% Check diode correlations: buffer
#
# qad: first take most variable region (5000:5500)
wmin = 5000
wmax = 5500
buff_qad_smean = buff_qad[:, wmin:wmax].mean(axis=1)

qfig, qax = plt.subplots()
qax.plot(buff_qad.mean(axis=0), label="Diode waveform (backscatter)")
qax.plot(range(wmin, wmax), buff_qad.mean(axis=0)[wmin:wmax], label="Selected for mean")
qax.set_title("qadc01 - simple mean")
qax.legend(loc="best")

# 4 panel figure of correlations: Dg2

cfig, caxs = plt.subplots(2, 2, layout="constrained")
cfig.suptitle(f"Run {buff_run}, AT25 buffer")
caxs[0, 0].scatter(buff_dg2, buffE_tmeans)
caxs[0, 0].set_ylabel("dg2")
caxs[0, 0].set_ylim([-0.5 * 1e8, 2 * 1e8])

caxs[1, 0].scatter(buff_qad_smean, buffE_tmeans)
caxs[1, 0].set_ylabel("Mean qadc01")
caxs[1, 0].set_xlabel("Epix")
caxs[1, 0].set_ylim([-0.5 * 1e8, 2 * 1e8])

caxs[0, 1].scatter(buff_dg2, buffJ_tmeans)

caxs[1, 1].scatter(buff_qad_smean, buffJ_tmeans)
caxs[1, 1].set_xlabel("Jungfrau")


# %% Check diode correlations: AT data
#
# qad: first take most variable region (5000:5500)
wmin = 5000
wmax = 5500
at_qad_smean = at_qad[:, wmin:wmax].mean(axis=1)

# 4 panel figure of correlations: Dg2

cfig, caxs = plt.subplots(2, 2, layout="constrained")
cfig.suptitle(f"Run {at_run}, AT25 buffer")
caxs[0, 0].scatter(at_dg2, atE_tmeans)
caxs[0, 0].set_ylabel("dg2")
caxs[0, 0].set_ylim([-0.5 * 1e8, 2 * 1e8])

caxs[1, 0].scatter(at_qad_smean, atE_tmeans)
caxs[1, 0].set_ylabel("Mean qadc01")
caxs[1, 0].set_xlabel("Epix")
caxs[1, 0].set_ylim([-0.5 * 1e8, 2 * 1e8])

caxs[0, 1].scatter(at_dg2, atJ_tmeans)

caxs[1, 1].scatter(at_qad_smean, atJ_tmeans)
caxs[1, 1].set_xlabel("Jungfrau")


# %% qad principal components

pca1 = PCA(n_components=1, whiten=True)
qad_pca = pca1.fit(buff_qad)
qad_num = qad_pca.transform(buff_qad).squeeze()
plt.scatter(buff_qad_smean, qad_num)
plt.show()
print(qad_pca.singular_values_)


sample_num = qad_pca.transform(at_qad).squeeze()

# %% Check what waveforms look like at different points
min_waveform = buff_qad[qad_num == np.min(qad_num), :].squeeze()
z_waveform = buff_qad[np.abs(qad_num) == np.min(np.abs(qad_num)), :].squeeze()

min_waveform = at_qad[sample_num == np.min(sample_num), :].squeeze()
z_waveform = at_qad[np.abs(sample_num) == np.min(np.abs(sample_num)), :].squeeze()

# %% Truncated svd
# Using to avoid PCA centering issue for negative values

svd1 = TruncatedSVD(n_components=1)
qad_svd = svd1.fit(buff_qad)

buff_qad = qad_svd.transform(buff_qad)
vsample_num = qad_svd.transform(at_qad)

plt.hist(vsample_num)
plt.hist(buff_qad)
plt.show()

# %% Save SVD

with open("qad_svd.pickle", "wb") as f:
    pickle.dump(qad_svd, f, pickle.HIGHEST_PROTOCOL)
    # del h5re['/waveforms'] #delete it if it previously existed

# %% check

with open("qad_svd.pickle", "rb") as f:
    qad_svd = pickle.load(f)

buff_dg2 = np.load(data_dir / f"r{buff_run:04d}_dg2_totalIntensity.npy")
# buff_hfx = np.load(data_dir / f"r{buff_run:04d}_hfx_totalIntensity.npy")
# buff_em = np.load(data_dir / f"r{buff_run:04d}_em_totalIntensity.npy")
# buff_fee = np.load(data_dir / f"r{buff_run:04d}_fee.npy")
# buff_wave8 = np.load(data_dir / f"r{buff_run:04d}_wave8_waveform.npy")
buff_qad = np.load(data_dir / f"r{buff_run:04d}_qadc01_waveform.npy")

# AT25 data
at_run = 63

at_h5e_path = data_dir / f"{exp}_r{at_run:04d}_epix.h5"
at_h5j_path = data_dir / f"{exp}_r{at_run:04d}_jungfrau.h5"

at_dg2 = np.load(data_dir / f"r{at_run:04d}_dg2_totalIntensity.npy")
# at_hfx = np.load(data_dir / f"r{at_run:04d}_hfx_totalIntensity.npy")
# at_em = np.load(data_dir / f"r{at_run:04d}_em_totalIntensity.npy")
# at_fee = np.load(data_dir / f"r{at_run:04d}_fee.npy")
# at_wave8 = np.load(data_dir / f"r{at_run:04d}_wave8_waveform.npy")
at_qad = np.load(data_dir / f"r{at_run:04d}_qadc01_waveform.npy")

buff_qad = qad_svd.transform(buff_qad)
at_qad = qad_svd.transform(at_qad)


# %% Check normalization

vs_2 = np.zeros(at_qad.shape)
for ii in range(len(vs_2)):
    vs_2[ii] = qad_svd.transform(buff_qad[[ii], :])[0]

# %% Section Name

with h5py.File(buff_h5j_path, "r") as h5j:
    buff_dg2 = h5j["dg2"][:]
    buff_radials = h5j["/mean"][:]
    q = h5j["q_bins"][:]
    for key in h5j.keys():
        print(key)

with h5py.File(at_h5j_path, "r") as h5j:
    at_dg2 = h5j["dg2"][:]
    at_radials = h5j["/mean"][:]
    q = h5j["q_bins"][:]
    for key in h5j.keys():
        print(key)

nzd_inds = dg2 > 100
nz_rad = at_radials[nzd_inds, :]
nz_rmean = nz_rad.mean(axis=1)
nz_rmed = np.median(nz_rad, axis=1)
nz_vnum = at_qad[nzd_inds].squeeze()
nz_dg2 = dg2[nzd_inds] * max(nz_vnum) / max(dg2)

# plt.scatter(nz_dg2, nz_rmean / nz_dg2, label = "Jungfrau / dg2")
plt.scatter(nz_vnum, nz_rmean, label="Jungfrau mean signal")
# plt.scatter(nz_vnum, nz_rmean / nz_vnum, label = "Jungfrau average / diode")
# plt.scatter(nz_dg2, nz_rmax / nz_dg2, label = "Jungfrau / dg2")
plt.scatter(nz_vnum, nz_rmed, label="Jungfrau median signal")
# plt.scatter(nz_vnum, (nz_rmed - coefs[0]) / nz_vnum, label = "Jungfrau average / diode")
# plt.scatter(nz_vnum, nz_rmed / (nz_vnum - rcoefs[0]), label = "Jungfrau average / diode")
# plt.scatter(nz_vnum, nz_rmean / nz_vnum**(5/3), label = "Jungfrau average / diode")
# plt.scatter(nz_dg2, nz_rmean / nz_dg2)
# plt.xlim(0.1,3.5)
# plt.ylim(-1e7, 1e8)
plt.legend()
plt.xlabel("Diode values (first principal component)")
plt.ylabel("Average detector intensity")
plt.show()

# %% Median intensity

# Truncate fit at set value
at_qad = at_qad.squeeze()
qad_thresh = 0.8
qad_inds = at_qad > qad_thresh

rmeds = np.median(at_radials, axis=1)
rmeans = np.mean(at_radials, axis=1)
coefs = polyfit(at_qad[qad_inds], rmeds[qad_inds], 1)
rcoefs = polyfit(rmeds[qad_inds], at_qad[qad_inds], 1)

plt.scatter(at_qad, rmeds / at_qad, label="Median / qad svd1")
plt.scatter(at_qad, (rmeds - coefs[0]) / at_qad, label="(Median - offset) / qad svd1")
plt.scatter(at_qad, rmeds, label="Median signal")
# plt.scatter(
#     at_qad[qad_inds], rmeds[qad_inds] / (at_qad[qad_inds] - rcoefs[0])
# )
plt.plot(at_qad, at_qad * coefs[1] + coefs[0], "k", lw=2)
plt.title("Run 63 qad normalization")
plt.xlabel("qadc01 svd1")
plt.ylabel("Intensity")
plt.legend()

# %% Strongest diode correlation above QAD=0.8?

qad_inds = at_qad.squeeze() > 0.8
# Calculate correlations
med_corr = np.corrcoef(at_qad[qad_inds], rmeds[qad_inds])
mean_corr = np.corrcoef(at_qad[qad_inds], rmeans[qad_inds])
print(f"Median corr: {med_corr[0,1]}")
print(f"Mean corr: {mean_corr[0,1]}")

# Plot as a function of Q
xcorrmat = np.corrcoef(np.transpose(at_radials), at_qad)
plt.plot(q * 1e-10, xcorrmat[:-1, -1], lw=2)
plt.xlabel("q (inverse Angstroms)")
plt.ylabel("Pearson correlation with QAD SVD1")
# plt.ylim([0.9, 1])
plt.title("Jungfrau intensity backscattering correlation")

# %% Strongest diode correlation above QAD=0.8?

# Calculate correlations

# Plot as a function of Q
bcorrmat = np.corrcoef(np.transpose(at_radials), at_qad.squeeze())
bdcorrmat = np.corrcoef(np.transpose(at_radials), dg2)
plt.plot(q * 1e-10, bcorrmat[:-1, -1], lw=2, label=f"buff ({buff_run}) qad")
plt.plot(q * 1e-10, bdcorrmat[:-1, -1], lw=2, label=f"buff ({buff_run}) dg2")
plt.xlabel("q (inverse Angstroms)")
plt.ylabel("Pearson correlation with QAD SVD1")
# plt.ylim([0.9, 1])
plt.title("Jungfrau intensity backscattering correlation")
plt.legend()

# %% Fit to two lines


def twoLine(x, x0, y0, k1, k2):
    return np.piecewise(
        x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0]
    )


# p, e = curve_fit(twoLine, at_qad, rmeds, p0=[0.5, 0.5e7, 1e7, 2e7])
p, e = curve_fit(twoLine, at_qad, rmeans, p0=[0.5, 0.5e7, 1e7, 2e7])
vd = np.linspace(0, max(at_qad), 1000)

radial_offset = p[1] - p[3] * p[0]
qad_offset = p[0] - p[1] / p[3]
# plt.plot(at_qad, rmeds, "o")
plt.plot(at_qad, rmeans, "o")
plt.plot(0, radial_offset, "r*", ms=4)
plt.plot(qad_offset, 0, "r*", ms=4)
plt.plot(vd, twoLine(vd, *p))

# %% Check correlation

qad_inds = at_qad > p[0]
rcoefs = polyfit(at_qad[qad_inds], rmeans[qad_inds], 1)
qcoefs = polyfit(rmeans[qad_inds], at_qad[qad_inds], 1)

plt.scatter(at_qad, rmeans)
plt.scatter(at_qad, rmeans / at_qad)
plt.scatter(at_qad, (rmeans - rcoefs[0]) / at_qad)
# plt.scatter(
#     at_qad[qad_inds], rmeans[qad_inds] / (at_qad[qad_inds] - qcoefs[0])
# )
plt.plot(at_qad, at_qad * rcoefs[1] + rcoefs[0], "k", lw=2)
plt.title(f"Run {run} Jungfrau QAD diode correction")
plt.xlabel("QAD backscattering")
plt.ylabel("Mean detector")
plt.legend()

# %% Radials
qad_thresh = 1
r_step = 100
for i in range(0, buff_radials.shape[0], r_step):
    if buff_qad[i] > qad_thresh:
        plt.plot(q, (buff_radials[i] - coefs[0]) / buff_qad[i])  # No normalization
        # plt.plot(q, (buff_radials[i] - coefs[0]) / buff_dg2[i])  # No normalization
        # plt.plot(q, radials[i]/(buff_qad_svd1[i] - rcoefs[0]))  # No normalization
        # plt.plot(q, (radials[i]) / buff_qad[i])  # No normalization
        # plt.plot(q, (radials[i]))  # No normalization

buff_mean = np.nanmean(
    np.array(
        [
            (radial - coefs[0]) / buff_qad_val
            # (radial) / buff_qad_val
            for radial, buff_qad_val in zip(buff_radials, buff_qad)
            if (buff_qad_val > qad_thresh)
        ]
    ),
    axis=0,
)

# %% Radials
r_step = 10
for i in range(0, at_radials.shape[0], r_step):
    if at_qad[i] > qad_thresh:
        plt.plot(q * 1e-10, (at_radials[i] - coefs[0]) / at_qad[i])  # No normalization
        # plt.plot(q, (at_radials[i] - coefs[0]) / at_dg2[i])  # No normalization
        # plt.plot(q, radials[i]/(buff_qad_svd1[i] - rcoefs[0]))  # No normalization
        # plt.plot(q, (radials[i]) / at_qad[i])  # No normalization
        # plt.plot(q, (radials[i]))  # No normalization

sample_mean = np.nanmean(
    np.array(
        [
            (radial - coefs[0]) / at_qad_val
            # (radial) / at_qad_val
            for radial, at_qad_val in zip(at_radials, at_qad)
            if (at_qad_val > qad_thresh)
        ]
    ),
    axis=0,
)
# %% Buffer subtraction attempts

plt.plot(buff_mean)
plt.plot(sample_mean)

# %% sub

plt.title(f"Naive subtraction, run {at_run}")
plt.plot(q / 1e10, sample_mean - buff_mean)
plt.xlabel(r"q $\AA$$^{-1}$")
