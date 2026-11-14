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

# Old params: Qmin: 0.17364015067037117, Qmax: 2.663372772590155
# Adjusted: Qmin: 0.17364015067037117, Qmax: 2.663372772590155
#
# buff_dg2 = np.load(data_dir / f"r{buff_run:04d}_dg2_totalIntensity.npy")
# buff_hfx = np.load(data_dir / f"r{buff_run:04d}_hfx_totalIntensity.npy")
# buff_em = np.load(data_dir / f"r{buff_run:04d}_em_totalIntensity.npy")
# buff_fee = np.load(data_dir / f"r{buff_run:04d}_fee.npy")
# buff_wave8 = np.load(data_dir / f"r{buff_run:04d}_wave8_waveform.npy")
# buff_qad = np.load(data_dir / f"r{buff_run:04d}_qadc01_waveform.npy")

# AT25 data
# at_run = 63

# at_h5e_path = data_dir / f"{exp}_r{at_run:04d}_epix.h5"
# at_h5j_path = data_dir / f"{exp}_r{at_run:04d}_jungfrau.h5"

# # at_dg2 = np.load(data_dir / f"r{at_run:04d}_dg2_totalIntensity.npy")
# # at_hfx = np.load(data_dir / f"r{at_run:04d}_hfx_totalIntensity.npy")
# # at_em = np.load(data_dir / f"r{at_run:04d}_em_totalIntensity.npy")
# # at_fee = np.load(data_dir / f"r{at_run:04d}_fee.npy")
# # at_wave8 = np.load(data_dir / f"r{at_run:04d}_wave8_waveform.npy")
# # at_qad = np.load(data_dir / f"r{at_run:04d}_qadc01_waveform.npy")

# %% Geometry checking


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

# %% Get info to save profiles
#
buff_run = 61
sample_run = 63
# buff_run = 67
# sample_run = 69
# buff_run = 72
# sample_run = 74

# buff_h5e_path = data_dir / f"{exp}_r{buff_run:04d}_epix.h5"
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

# buff_h5e_path = data_dir / f"{exp}_r{buff_run:04d}_epix.h5"
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


buff_avg = buff_means.mean(axis=0)
buff_spb_avg = (buff_means[buff_inds, :]).mean(axis=0)
buff_spb_scale_avg = (buff_means[buff_inds, :] * buff_spb[:, np.newaxis]).mean(axis=0)

sample_avg = sample_means.mean(axis=0)
sample_spb_avg = (sample_means[sample_inds, :]).mean(axis=0)
sample_spb_scale_avg = (sample_means[sample_inds, :] * sample_spb[:, np.newaxis]).mean(
    axis=0
)

# Repeat for old data
old_buff_h5j_path = old_dir / f"{exp}_r{buff_run:04d}_jungfrau.h5"

with h5py.File(old_buff_h5j_path, "r") as h5:
    # for key in h5.keys():
    #     print(key)
    old_buff_qad = h5["/qad_svd1"][:]
    old_buff_dg2 = h5["/dg2"][:]
    old_buff_means = h5["/mean"][:]
    old_buff_q = h5["/q_bins"][:] / 1e10

# buff_h5e_path = data_dir / f"{exp}_r{buff_run:04d}_epix.h5"
old_sample_h5j_path = old_dir / f"{exp}_r{sample_run:04d}_jungfrau.h5"

with h5py.File(old_sample_h5j_path, "r") as h5:
    # for key in h5.keys():
    #     print(key)
    old_sample_qad = h5["/qad_svd1"][:]
    old_sample_dg2 = h5["/dg2"][:]
    old_sample_means = h5["/mean"][:]
    old_sample_q = h5["/q_bins"][:] / 1e10

old_buff_avg = old_buff_means.mean(axis=0)
old_buff_spb_avg = (old_buff_means[buff_inds, :]).mean(axis=0)
old_buff_spb_scale_avg = (old_buff_means[buff_inds, :] * buff_spb[:, np.newaxis]).mean(
    axis=0
)

old_sample_avg = old_sample_means.mean(axis=0)
old_sample_spb_avg = (old_sample_means[sample_inds, :]).mean(axis=0)
old_sample_spb_scale_avg = (
    old_sample_means[sample_inds, :] * sample_spb[:, np.newaxis]
).mean(axis=0)


# Plot against each other
plt.plot(old_buff_q, old_buff_avg)
plt.plot(buff_q, buff_avg)

# %% Parameters for AT25 100 mM
ref_name = "S_A_dAdT25_500uM_WAXS.dat"
ref_prof = raw.load_profiles([ref_dir / ref_name])[0]

bscale = 1.14
sub = sample_spb_scale_avg - bscale * buff_spb_scale_avg
lcls_inds = (sample_spb_scale_avg != 0) & (buff_spb_scale_avg != 0)
sub = sub[lcls_inds]
sub_q = sample_q[lcls_inds]

print(f"Qmin: {sub_q.min()}, Qmax: {sub_q.max()}")

ref_inds = ref_prof.getQ() > 0.1736
ref_q = ref_prof.getQ()[ref_inds]
# q_scale = 1.04
q_scale = 1
ref_i = ref_prof.getI()[ref_inds]
ref_scale = np.mean(sub) / np.mean(ref_i)
fudge = 0.68

# plt.plot(sample_q, sample_spb_scale_avg)
# plt.plot(buff_q, bscale * buff_spb_scale_avg)
plt.title("AT25 100uM Jungfrau reference")
plt.plot(
    sub_q / q_scale,
    sub,
    label=f"LCLS subtraction, buff scaled {bscale:.4f}, q_scaled {q_scale}",
)
plt.plot(ref_q, fudge * ref_scale * ref_i, label="AT25 reference data")
# plt.semilogy(sub_q / q_scale, sub, label=f"LCLS subtraction, q_scaled {q_scale}")
# plt.semilogy(ref_q, fudge * ref_scale * ref_i, label="AT48 reference data")
plt.legend()

# %% Parameters for AT25 1 mM
#
ref_name = "S_A_dAdT25_500uM_WAXS.dat"
ref_prof = raw.load_profiles([ref_dir / ref_name])[0]

bscale = 1.0472
sub = sample_spb_scale_avg - bscale * buff_spb_scale_avg
lcls_inds = (sample_spb_scale_avg != 0) & (buff_spb_scale_avg != 0)
sub = sub[lcls_inds]
sub_q = sample_q[lcls_inds]
q_scale = 1.01

old_bscale = 1.0472
old_sub = old_sample_spb_scale_avg - old_bscale * old_buff_spb_scale_avg
old_lcls_inds = (old_sample_spb_scale_avg != 0) & (old_buff_spb_scale_avg != 0)
old_sub = old_sub[old_lcls_inds]
old_sub_q = old_sample_q[old_lcls_inds]
old_q_scale = 1.04

print(f"Qmin: {sub_q.min()}, Qmax: {sub_q.max()}")

ref_inds = ref_prof.getQ() > 0.1736
ref_q = ref_prof.getQ()[ref_inds]
# q_scale = 1.04
ref_i = ref_prof.getI()[ref_inds]
ref_scale = np.mean(sub) / np.mean(ref_i)
fudge = 0.68

# plt.plot(sample_q, sample_spb_scale_avg)
# plt.plot(buff_q, bscale * buff_spb_scale_avg)
plt.title("AT25 1mM Jungfrau reference")
plt.plot(ref_q, fudge * ref_scale * ref_i, label="AT25 500uM reference data")
plt.plot(
    old_sub_q / old_q_scale,
    old_sub,
    label=f"LCLS d=88mm, buff scaled {old_bscale:.4f}, q_scaled {old_q_scale}",
)
plt.plot(
    sub_q / q_scale,
    sub,
    label=f"LCLS d=93mm, buff scaled {bscale:.4f}, q_scaled {q_scale}",
)
# plt.semilogy(sub_q / q_scale, sub, label=f"LCLS subtraction, q_scaled {q_scale}")
# plt.semilogy(ref_q, fudge * ref_scale * ref_i, label="AT48 reference data")
plt.legend()

# %% Parameters for AT48
#
ref_name = "S_A_AT48_800uM_WAXS_001_data_000001_00001.dat"
ref_prof = raw.load_profiles([ref_dir / ref_name])[0]

bscale = 0.9417
sub = sample_spb_scale_avg - bscale * buff_spb_scale_avg
lcls_inds = (sample_spb_scale_avg != 0) & (buff_spb_scale_avg != 0)
sub = sub[lcls_inds]
sub_q = sample_q[lcls_inds]
q_scale = 1.015

old_bscale = 0.9417
old_sub = old_sample_spb_scale_avg - old_bscale * old_buff_spb_scale_avg
old_lcls_inds = (old_sample_spb_scale_avg != 0) & (old_buff_spb_scale_avg != 0)
old_sub = old_sub[old_lcls_inds]
old_sub_q = old_sample_q[old_lcls_inds]
old_q_scale = 1.05

print(f"Qmin: {sub_q.min()}, Qmax: {sub_q.max()}")

ref_inds = ref_prof.getQ() > 0.1736
ref_q = ref_prof.getQ()[ref_inds]
# q_scale = 1.04
ref_i = ref_prof.getI()[ref_inds]
ref_scale = np.mean(sub) / np.mean(ref_i)
fudge = 0.67

# plt.plot(sample_q, sample_spb_scale_avg)
# plt.plot(buff_q, bscale * buff_spb_scale_avg)
plt.title("AT48 Jungfrau reference")
plt.plot(ref_q, fudge * ref_scale * ref_i, label="AT48 800uM reference data")
plt.plot(
    old_sub_q / old_q_scale,
    old_sub,
    label=f"LCLS d=88mm, buff scaled {old_bscale:.4f}, q_scaled {old_q_scale}",
)
plt.plot(
    sub_q / q_scale,
    sub,
    label=f"LCLS d=93mm, buff scaled {bscale:.4f}, q_scaled {q_scale}",
)
# plt.semilogy(sub_q / q_scale, sub, label=f"LCLS subtraction, q_scaled {q_scale}")
# plt.semilogy(ref_q, fudge * ref_scale * ref_i, label="AT48 reference data")
plt.legend()

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
