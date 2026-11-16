# %% Imports, filepaths

# IO
from pathlib import Path
import pickle
import h5py
import pandas as pd

# Fitting
import numpy as np
from numpy.polynomial.polynomial import polyfit
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import bioxtasraw.RAWAPI as raw
import swaxs_functions.processing as pf

# Filepaths
exp = "cxi101672626"
# data_dir = Path("/home/hugh/beamtime_processing/LCLS_20260629/deIced_20260722/")
old_dir = Path("/home/hugh/beamtime_processing/LCLS_20260629/deIced_20260722/")
data_dir = Path("/home/hugh/beamtime_processing/LCLS_20260629/deIced_20261111/")
work_dir = Path("/home/hugh/beamtime_processing/LCLS_20260629/jungfrau_spb/")
ref_dir = Path("/home/hugh/beamtime_processing/LCLS_20260629/ref_profiles/")

# Buffer data
buffer_labels = [
    "NaCl-150mM_MOPS-10mM",
    "NaCl-150mM_MOPS-10mM",
    "MOPS-1mM",
]
ref_names = [
    "S_A_dAdT_700uM_WAXS_bmatch-q2d9.dat",
    "S_A_dAdT_700uM_WAXS_bmatch-q2d9.dat",
    "S_A_AT48_800uM_WAXS_001_data_000001_00001.dat",
]
set_names = [
    "AT25 100 uM",
    "AT25 1 mM",
    "AT48 250 uM",
]
noJet_runs = [
    60,
    66,
    66,
]
buff_runs = [
    61,
    67,
    72,
]
sample_runs = [
    63,
    69,
    74,
]

# %% Get info to save profiles
rnum = 0
set_name = set_names[rnum]
buff_run = buff_runs[rnum]
sample_run = sample_runs[rnum]

endlim_qind = 100

# Scale to max of peak empirically, compare SPB

# buff_h5e_path = data_dir / f"{exp}_r{buff_run:04d}_epix.h5"
# buff_h5j_path = data_dir / f"{exp}_r{buff_run:04d}_jungfrau.h5"
buff_h5j_path = old_dir / f"{exp}_r{buff_run:04d}_jungfrau.h5"

buff_ind_file = f"Hugh{buff_run}.txt"
buff_df = pd.read_csv(work_dir / buff_ind_file, header=None, sep=" ")
buff_inds = buff_df.iloc[:, 0].to_numpy() - 1
buff_spb = buff_df.iloc[:, 2].to_numpy()

with h5py.File(buff_h5j_path, "r") as h5:
    # for key in h5.keys():
    #     print(key)
    buff_qad = h5["/qad_svd1"][:]
    buff_dg2 = h5["/dg2"][:]
    buff_means = h5["/mean"][:]
    buff_q = h5["/q_bins"][:] / 1e10

sample_h5j_path = old_dir / f"{exp}_r{sample_run:04d}_jungfrau.h5"

sample_ind_file = f"Hugh{sample_run}.txt"
sample_df = pd.read_csv(work_dir / sample_ind_file, header=None, sep=" ")
sample_inds = sample_df.iloc[:, 0].to_numpy() - 1
sample_spb = sample_df.iloc[:, 2].to_numpy()

with h5py.File(sample_h5j_path, "r") as h5:
    # for key in h5.keys():
    #     print(key)
    sample_qad = h5["/qad_svd1"][:]
    sample_dg2 = h5["/dg2"][:]
    sample_means = h5["/mean"][:]
    sample_q = h5["/q_bins"][:] / 1e10


# buff_avg = buff_means.mean(axis=0)
# buff_spb_avg = (buff_means[buff_inds, :]).mean(axis=0)
buff_peakVals = buff_means[buff_inds, :-endlim_qind].max(axis=1)[:, np.newaxis]
buff_spb_avg = (buff_means[buff_inds, :] / buff_spb[:, np.newaxis]).mean(axis=0)
buff_peak_radials = buff_means[buff_inds, :] / buff_peakVals
buff_peak_avg = buff_peak_radials.mean(axis=0)

sample_peakVals = sample_means[sample_inds, :-endlim_qind].max(axis=1)[:, np.newaxis]
sample_spb_avg = (sample_means[sample_inds, :] / sample_spb[:, np.newaxis]).mean(axis=0)
sample_peak_radials = sample_means[sample_inds, :] / sample_peakVals
sample_peak_avg = sample_peak_radials.mean(axis=0)

# spb_scale = 0.994
sub_spb_avg = sample_spb_avg - buff_spb_avg
sub_peak_avg = sample_peak_avg - buff_peak_avg

# Save to new text file

new_buff_ind_file = f"empirical_r{buff_run:04d}.txt"
buff_df["empirical_peak"] = buff_peakVals
buff_df.to_csv(work_dir / new_buff_ind_file, sep=" ", header=False, index=False)

new_sample_ind_file = f"empirical_r{sample_run:04d}.txt"
sample_df["empirical_peak"] = sample_peakVals
sample_df.to_csv(work_dir / new_sample_ind_file, sep=" ", header=False, index=False)

# Plot

# Parameters for AT25 100 mM
ref_name = ref_names[rnum]
ref_prof = raw.load_profiles([ref_dir / ref_name])[0]

bscale = 1
sub_spb_avg = sample_spb_avg - bscale * buff_spb_avg
sub_peak_avg = sample_peak_avg - bscale * buff_peak_avg

print(f"Qmin: {buff_q.min()}, Qmax: {buff_q.max()}")

ref_inds = ref_prof.getQ() > 0.1736
ref_q = ref_prof.getQ()[ref_inds]
# q_scale = 1.04
q_scale = 1.05
ref_i = ref_prof.getI()[ref_inds]
ref_scale = np.mean(sub_peak_avg) / np.mean(ref_i)
fudge = 0.68

# plt.plot(sample_q, sample_spb_scale_avg)
# plt.plot(buff_q, bscale * buff_spb_scale_avg)
plt.title("AT25 100uM Jungfrau reference")
plt.plot(
    buff_q / q_scale,
    sub_peak_avg,
    label=f"peak-match, buff scaled {bscale:.4f}, q_scaled {q_scale}",
)
plt.plot(ref_q, fudge * ref_scale * ref_i, label=f"{set_name} ref data")
# plt.semilogy(sub_q / q_scale, sub, label=f"LCLS subtraction, q_scaled {q_scale}")
# plt.semilogy(ref_q, fudge * ref_scale * ref_i, label="AT48 reference data")
plt.legend()
plt.plot(ref_q, np.zeros(ref_q.shape), "k")

# %% Old version: compare det. distances

buff_h5j_path = data_dir / f"{exp}_r{buff_run:04d}_jungfrau.h5"

buff_ind_file = f"Hugh{buff_run}.txt"
buff_df = pd.read_csv(work_dir / buff_ind_file, header=None, sep=" ")
buff_inds = buff_df.iloc[:, 0].to_numpy() - 1
buff_spb = buff_df.iloc[:, 2].to_numpy()

with h5py.File(buff_h5j_path, "r") as h5:
    # for key in h5.keys():
    #     print(key)
    buff_qad = h5["/qad_svd1"][:]
    buff_dg2 = h5["/dg2"][:]
    buff_means = h5["/mean"][:]
    buff_q = h5["/q_bins"][:] / 1e10

sample_h5j_path = data_dir / f"{exp}_r{sample_run:04d}_jungfrau.h5"

sample_ind_file = f"Hugh{sample_run}.txt"
sample_df = pd.read_csv(work_dir / sample_ind_file, header=None, sep=" ")
sample_inds = sample_df.iloc[:, 0].to_numpy() - 1
sample_spb = sample_df.iloc[:, 2].to_numpy()

with h5py.File(sample_h5j_path, "r") as h5:
    # for key in h5.keys():
    #     print(key)
    sample_qad = h5["/qad_svd1"][:]
    sample_dg2 = h5["/dg2"][:]
    sample_means = h5["/mean"][:]
    sample_q = h5["/q_bins"][:] / 1e10

# Repeat for old data
old_buff_h5j_path = old_dir / f"{exp}_r{buff_run:04d}_jungfrau.h5"

with h5py.File(old_buff_h5j_path, "r") as h5:
    # for key in h5.keys():
    #     print(key)
    old_buff_qad = h5["/qad_svd1"][:]
    old_buff_dg2 = h5["/dg2"][:]
    old_buff_means = h5["/mean"][:]
    old_buff_q = h5["/q_bins"][:] / 1e10

old_sample_h5j_path = old_dir / f"{exp}_r{sample_run:04d}_jungfrau.h5"

with h5py.File(old_sample_h5j_path, "r") as h5:
    # for key in h5.keys():
    #     print(key)
    old_sample_qad = h5["/qad_svd1"][:]
    old_sample_dg2 = h5["/dg2"][:]
    old_sample_means = h5["/mean"][:]
    old_sample_q = h5["/q_bins"][:] / 1e10

old_buff_spb_scale_avg = (
    old_buff_means[buff_inds, :] / old_buff_means[buff_inds, :-endlim_qind].max()
).mean(axis=0)
old_buff_spb_scale_std = (
    old_buff_means[buff_inds, :] / old_buff_means[buff_inds, :-endlim_qind].max()
).std(axis=0)

old_sample_avg = old_sample_means.mean(axis=0)
old_sample_spb_avg = (old_sample_means[sample_inds, :]).mean(axis=0)
old_sample_spb_scale_avg = (
    old_sample_means[sample_inds, :] / sample_spb[:, np.newaxis]
).mean(axis=0)
old_sample_spb_scale_std = (
    old_sample_means[sample_inds, :] / sample_spb[:, np.newaxis]
).std(axis=0)

old_buff_norm = old_buff_spb_scale_avg / old_buff_spb_scale_avg.mean()
old_buff_std = old_buff_spb_scale_std / old_buff_spb_scale_avg.mean()

# Plot against each other
# plt.plot(old_buff_q, old_buff_norm, "k", lw=2)
# plt.plot(old_buff_q, old_buff_norm + old_buff_std, "b")
# plt.plot(old_buff_q, old_buff_norm - old_buff_std, "b")
# plt.plot(old_buff_q, old_buff_norm + old_buff_std, "r")
# plt.plot(old_buff_q, old_buff_norm - old_buff_std, "r")

step = 100
for rad in (
    old_buff_means[buff_inds[::step], :]
    / old_buff_means[buff_inds[::step], :-endlim_qind].max(axis=1)[:, np.newaxis]
):
    plt.plot(rad)
# plt.plot(buff_q, buff_avg)

# %% Parameters for AT25 100 mM
ref_name = "S_A_dAdT_700uM_WAXS_bmatch-q2d9.dat"
ref_prof = raw.load_profiles([ref_dir / ref_name])[0]

# bscale = 1.1451
bscale = 1
# bscale = 0.9993
sub_spb_avg = sample_spb_avg - bscale * buff_spb_avg
sub_peak_avg = sample_peak_avg - bscale * buff_peak_avg

print(f"Qmin: {buff_q.min()}, Qmax: {buff_q.max()}")

ref_inds = ref_prof.getQ() > 0.1736
ref_q = ref_prof.getQ()[ref_inds]
# q_scale = 1.04
q_scale = 1.05
ref_i = ref_prof.getI()[ref_inds]
ref_scale = np.mean(sub_peak_avg) / np.mean(ref_i)
fudge = 0.68

# plt.plot(sample_q, sample_spb_scale_avg)
# plt.plot(buff_q, bscale * buff_spb_scale_avg)
plt.title("AT25 100uM Jungfrau reference")
plt.plot(
    buff_q / q_scale,
    sub_peak_avg,
    label=f"LCLS subtraction, buff scaled {bscale:.4f}, q_scaled {q_scale}",
)
plt.plot(ref_q, fudge * ref_scale * ref_i, label="AT25 reference data")
# plt.semilogy(sub_q / q_scale, sub, label=f"LCLS subtraction, q_scaled {q_scale}")
# plt.semilogy(ref_q, fudge * ref_scale * ref_i, label="AT48 reference data")
plt.legend()
plt.plot(ref_q, np.zeros(ref_q.shape), "k")

# %% Parameters for AT25 1 mM
#
ref_name = "S_A_dAdT_700uM_WAXS_bmatch-q2d9.dat"
ref_prof = raw.load_profiles([ref_dir / ref_name])[0]

# bscale = 1.1451
bscale = 1
# bscale = 0.9993
sub_spb_avg = sample_spb_avg - bscale * buff_spb_avg
sub_peak_avg = sample_peak_avg - bscale * buff_peak_avg

print(f"Qmin: {buff_q.min()}, Qmax: {buff_q.max()}")

ref_inds = ref_prof.getQ() > 0.1736
ref_q = ref_prof.getQ()[ref_inds]
# q_scale = 1.04
q_scale = 1.05
ref_i = ref_prof.getI()[ref_inds]
ref_scale = np.mean(sub_peak_avg) / np.mean(ref_i)
fudge = 0.68


# plt.plot(sample_q, sample_spb_scale_avg)
# plt.plot(buff_q, bscale * buff_spb_scale_avg)
plt.title("AT25 1mM Jungfrau reference")
plt.plot(ref_q, fudge * ref_scale * ref_i, label="AT25 500uM reference data")
plt.plot(
    buff_q / q_scale,
    sub_peak_avg,
    label=f"LCLS d=88mm, buff scaled {bscale:.4f}, q_scaled {q_scale}",
)
# plt.semilogy(sub_q / q_scale, sub, label=f"LCLS subtraction, q_scaled {q_scale}")
# plt.semilogy(ref_q, fudge * ref_scale * ref_i, label="AT48 reference data")
plt.legend()
plt.plot(ref_q, np.zeros(ref_q.shape), "k")

# %% Parameters for AT48
#
ref_name = "S_A_AT48_800uM_WAXS_001_data_000001_00001.dat"
ref_prof = raw.load_profiles([ref_dir / ref_name])[0]

# bscale = 1.1451
# bscale = 1
bscale = 0.9985
sub_spb_avg = sample_spb_avg - bscale * buff_spb_avg
sub_peak_avg = sample_peak_avg - bscale * buff_peak_avg

print(f"Qmin: {buff_q.min()}, Qmax: {buff_q.max()}")

ref_inds = ref_prof.getQ() > 0.1736
ref_q = ref_prof.getQ()[ref_inds]
# q_scale = 1.04
q_scale = 1.05
ref_i = ref_prof.getI()[ref_inds]
ref_scale = np.mean(sub_peak_avg) / np.mean(ref_i)
fudge = 0.75

# plt.plot(sample_q, sample_spb_scale_avg)
# plt.plot(buff_q, bscale * buff_spb_scale_avg)
plt.title("AT48 Jungfrau reference")
plt.plot(ref_q, fudge * ref_scale * ref_i, label="AT48 800uM reference data")
plt.plot(
    buff_q / q_scale,
    sub_peak_avg,
    label=f"Peak-match d=88mm, buff scaled {bscale:.4f}, q_scaled {q_scale}",
)
plt.plot(
    buff_q / q_scale,
    sub_spb_avg * sub_peak_avg.mean() / sub_spb_avg.mean(),
    label=f"SPB d=88mm, buff scaled {bscale:.4f}, q_scaled {q_scale}",
)
# plt.semilogy(sub_q / q_scale, sub, label=f"LCLS subtraction, q_scaled {q_scale}")
# plt.semilogy(ref_q, fudge * ref_scale * ref_i, label="AT48 reference data")
plt.legend()
plt.plot(ref_q, np.zeros(ref_q.shape), "k")

# %% Residual


# %% Parameters for AT48
#
ref_name = "S_A_AT48_800uM_WAXS_001_data_000001_00001.dat"
ref_prof = raw.load_profiles([ref_dir / ref_name])[0]

bscale = 0.9415
sub = sample_spb_scale_avg - bscale * buff_spb_scale_avg
lcls_inds = (sample_spb_scale_avg != 0) & (buff_spb_scale_avg != 0)
sub = sub[lcls_inds]
sub_q = sample_q[lcls_inds]

ref_inds = ref_prof.getQ() > 0.1736
ref_q = ref_prof.getQ()[ref_inds]
# q_scale = 1.05
q_scale = 1
ref_i = ref_prof.getI()[ref_inds]
ref_scale = np.mean(sub) / np.mean(ref_i)
fudge = 0.74

# plt.plot(sample_q, sample_spb_scale_avg)
# plt.plot(buff_q, bscale * buff_spb_scale_avg)
plt.title("AT48 Jungfrau reference")
plt.plot(
    sub_q / q_scale,
    sub,
    label=f"LCLS subtraction, buff scaled {bscale:.4f}, q_scaled {q_scale}",
)
plt.plot(ref_q, fudge * ref_scale * ref_i, label="AT48 reference data")
# plt.semilogy(sub_q / q_scale, sub, label=f"LCLS subtraction, q_scaled {q_scale}")
# plt.semilogy(ref_q, fudge * ref_scale * ref_i, label="AT48 reference data")
plt.legend()

# %% Save minimized hdf5s for Vivian

# %% Jungfrau Geometry checking

jung_t_vec = np.array(
    [0.00630122277293104, -0.03812006960322477, 0.08799999999999913], np.newaxis
)
jung_t_vec_old = np.array(
    [0.00630122277293104, -0.03812006960322477, 0.08899999999999913], np.newaxis
)
epix_fixZ_t_vec = np.array(
    [0.007699999999999993, -0.04638000000000004, 2.214], np.newaxis
)
b_hat = np.array([0, 0, 1], np.newaxis)

wavelength = 0.20568734029081948e-9

dq_dmm = 2 * np.pi
jung_pixlen = 7.5e-5  # 75 microns
epix_pixlen = 0.0001  # 100 microns


jvo = jung_t_vec_old
vhat = jvo / np.sqrt(np.dot(jvo, jvo))
qmag_old = 2e-10 * np.pi * np.linalg.norm(vhat - b_hat) / wavelength

jv = jung_t_vec
vhat = jv / np.sqrt(np.dot(jv, jv))
qmag = 2e-10 * np.pi * np.linalg.norm(vhat - b_hat) / wavelength
