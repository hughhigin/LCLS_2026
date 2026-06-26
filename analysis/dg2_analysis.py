# NOTE: convert Q to inverse angstroms !!
# Process runs 224, 225
# Investigate DG2 values across runs
# Run 80
# Pretty: Runs 228, 232 (DNA day 5)

import psana
from psana import *
import sys, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

# Scipy
from scipy.fft import fft,rfft,rfftfreq

exp_id = "cxil1005322"
reb_dir = "/sdf/data/lcls/ds/cxi/cxil1005322/results/hhig/practice_0603/results/reb_hdf5/"
com_dir = "/sdf/data/lcls/ds/cxi/cxi101620426/results/hhig/results/reb_hdf5/"

# %% Psana direct examination

run = 226

# %% Compare figure: diode counts of dark vs. detector / psi signal
# Maxima: 500 / 7
# epix
# dark_run = 258
dark_run = 224
drun_str = f"r{dark_run:04d}"
# ref_name = "cxil1005322_" + run_str + "_epix_bigMask_backup2.h5"
dref_name = f"{exp_id}_{drun_str}_epix_bigMask.h5"
reb_depix_file = reb_dir + dref_name
h5depix = h5py.File(reb_depix_file)

run = 260
run_str = f"r{run:04d}"
ref_name = f"{exp_id}_{run_str}_epix_bigMask.h5"
reb_epix_file = reb_dir + ref_name
h5epix = h5py.File(reb_epix_file)

# crun_str = f"r{crun:04d}"
# cref_name = f"{exp_id}_{crun_str}_jungfrau_bigMask.h5"
# creb_epix_file = com_dir + cref_name
# ch5epix = h5py.File(creb_epix_file)

xmin = 1000
xmax = 1500
dvert = 5

# dpsi = np.array(h5depix['/post_sample_intensity'])[xmin:xmax]
dpsi = np.array(h5depix['/post_sample_intensity'])
psi = np.array(h5epix['/post_sample_intensity'])
mean_vals = np.mean(h5epix['mean'],axis=1) # Mean over all detector
# norm_vals = (mean_vals / psi)[xmin:xmax]
norm_vals = (mean_vals / psi)
sub_vals = (mean_vals - np.mean(mean_vals))/np.std(mean_vals) - (psi - np.mean(psi))/np.std(psi)

# Select only frames that have reasonable diode count
good_frames = np.abs(norm_vals) < 1e4
psi = psi[good_frames]
mean_vals = mean_vals[good_frames]
norm_vals = norm_vals[good_frames]
sub_vals = sub_vals[good_frames]

tfig, tax = plt.subplots()
tax.plot((mean_vals - np.mean(mean_vals))/np.std(mean_vals),label="Detector")
tax.plot((psi - np.mean(psi))/(np.std(psi)), label="DG2")
tax.plot((norm_vals - np.mean(norm_vals))/np.std(norm_vals),label="Detector / DG2")
tax.plot((sub_vals - np.mean(sub_vals))/np.std(sub_vals),label="Detector - DG2")
# tax.plot(2*dvert + (dpsi - np.mean(dpsi))/(np.std(dpsi)), label="DG2 from dark run")

tax.set_xlabel("Frame number")
tax.set_ylabel("Intensity")
# tax.set_xlim(1000,1500)
# tax.set_ylim(-3, 10)
tax.legend()
tfig.savefig('mean_test.png')
# tfig.show()

# %% FFT to look at frequency info
N = len(norm_vals)
fs = 120
T = 1/fs
t = np.arange(N)

# dpsi_f = rfftfreq(N,T)
psi_f = rfftfreq(N,T)
# aX = abs(rfft(dpsi)/N)
daX = abs(rfft(dpsi)/N)
naX = abs(rfft(norm_vals)/N)

ffig,fax = plt.subplots()
# fax.plot(dpsi_f,aX)
# fax.semilogy(daX, label='DG2 dar')
# fax.semilogy(naX, label="259 Detector / DG2")
fax.plot(psi_f[1:], (naX[1:]/max(naX[1:])), label="259 Detector / DG2")
# fax.plot(1 + (daX[1:]/max(daX[1:])), label='DG2 dar')
fax.set(xlabel="frequency (AU)", ylabel="Amplitude")
# fax.set_xlim(0,20)
ffig.legend()
# ffig.savefig('freq_test2_zoom.png')
ffig.savefig('freq_test2.png')

# DG2 dark - Freq of 7?
# Signal - Freq of 5?
#
# Invert FFT cutting off low frequency
# dg2_inverse = ifft(daX[:])
# plt.plot(dg2_inverse)

# %% Check psana directly

ds = DataSource('exp=' + exp_id + ':run=' + str(dark_run))
dist = Detector("CXI:DS1:MMS:06.RBV") # event code
det = psana.Detector('Acqiris')
psi_det = Detector("CXI-DG2-BMMON") # event code

re_fid = h5depix["/frame_id"][:]
re_fid_dict = {tuple(value): idx for idx, value in enumerate(re_fid)}
seconds = re_fid[:,0]
nanoseconds = re_fid[:,1]
fiducials = re_fid[:,2]

nevents = len(seconds)

#create waveforms array of length nevents from reborn.
waveforms = np.zeros((nevents, 4, 1000)) #waveform array is (4,1000) shape
wftimes = np.zeros((nevents, 4, 1000))
#create array for post_sample_intensity
psi = np.zeros((nevents))
j = 0

for nevent,evt in enumerate(ds.events()):
    evtId = evt.get(psana.EventId)
    sec = evtId.time()[0]
    ns = evtId.time()[1]
    fid = evtId.fiducials()
    #grab the reborn index associated with that timestamp
    idx = re_fid_dict[tuple((sec,ns,fid))]
    sys.stdout.write('\r%d/%d   %d'%(j,nevents, idx))
    sys.stdout.flush()
    # waveforms are in Volts, times are in Seconds
    waveforms[idx] = det.waveform(evt)
    wftimes[idx] = det.wftime(evt)
    try:
        psi[idx] = psi_det.get(evt).TotalIntensity()
    except:
        psi[idx] = -1
    j+=1
    # if j>100: break

waveforms = np.array(waveforms)
wftimes = np.array(wftimes)

# %% Check waveforms.dat

offsets = np.load('waveform_offsets.npy')


#%%
# Load h5 from OM, 2024

om_home = "/sdf/data/lcls/ds/cxi/cxil1005322/results/tgrant/om/hdf5/"
om_dir = om_home + "epix_" + run_str + "/"
ref_name = "cxil1005322-" + run_str + "_0.cxi"
om_file = om_dir + ref_name

h5om = h5py.File(om_file)
q = h5om['data/q'][:]
q = q.mean(0)

radials = h5om['data/radial'][:]
print(h5om['LCLS'].keys())

om_fid = h5om['LCLS/fiducial'][:]
psi = h5om['LCLS/post_sample_intensity'][:]

om_fid2 = pd.Series(om_fid).apply(lambda x:
                                    np.array(x.decode("utf-8").split("-"), dtype=int))
om_fid2 = np.stack(om_fid2.values)

# this line makes the om frame IDs a dictionary for easy look up
ps_fid_dict = {tuple(value): idx for idx, value in enumerate(ps_fid)}

idx = [ps_fid_dict[tuple(value)]
        for value in om_fid2 if tuple(value) in ps_fid_dict]

print("Matching idx values:" + str(len(idx)))

nps_fid_dict = {tuple(value): idx for idx, value in enumerate(no_psi_fids)}
nidx = [nps_fid_dict[tuple(value)]
        for value in om_fid2 if tuple(value) in nps_fid_dict]
print("Matching idx values:" + str(len(nidx)))

# for ii, (fid,ps) in enumerate(zip(om_fid2,psi)):
#     print(str(ii) + ": " + str(fid) + ", " + str(ps))
#     # if ii > 200:
#     #     break
