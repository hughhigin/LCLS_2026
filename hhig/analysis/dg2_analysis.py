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
from scipy.signal import detrend,welch,savgol_filter

exp_id = "cxi101672626"
# reb_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/reb_hdf5/"
reb_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/reb_hdf5_noMasks/"

# %% fee diode, signal correlations

os.chdir(reb_dir)

# TODO: turn into loop
run = 10
run_str = f"r{run:04d}"

reb_epix_file = f"{exp_id}_{run_str}_epix.h5"
reb_jungfrau_file = f"{exp_id}_{run_str}_jungfrau.h5"

h5e = h5py.File(reb_epix_file, 'r')
h5j = h5py.File(reb_epix_file, 'r')
re_fid = np.array(h5e["frame_id"])
re_fid_dict = {tuple(value): idx for idx, value in enumerate(re_fid)}
seconds = re_fid[:,0]
nevents = len(seconds)

# Loop through psana
ds = psana.DataSource('exp=' + exp_id + ':run=' + str(run))
dg2_det = psana.Detector("CXI-DG2-BMMON") # event code
fee_det = psana.Detector("FEEGasDetEnergy") # event code

dg2 = np.zeros(nevents)
# digitzer = np.zeros(nevents)
fee = np.zeros((6,nevents)) # FEEGasDetEnergy, 6 locations

j=0
for nevent,evt in enumerate(ds.events()):
    evtId = evt.get(psana.EventId)
    sec = evtId.time()[0]
    ns = evtId.time()[1]
    fid = evtId.fiducials()
    #grab the reborn index associated with that timestamp
    idx = re_fid_dict[tuple((sec,ns,fid))]

    try:
        dg2[idx] = dg2_det.get(evt).TotalIntensity()
    except:
        dg2[idx] = -1
    try:
        fee[0,idx] = fee_det.get(evt).f_11_ENRC()
        fee[1,idx] = fee_det.get(evt).f_12_ENRC()
        fee[2,idx] = fee_det.get(evt).f_21_ENRC()
        fee[3,idx] = fee_det.get(evt).f_22_ENRC()
        fee[4,idx] = fee_det.get(evt).f_63_ENRC()
        fee[5,idx] = fee_det.get(evt).f_64_ENRC()
    except:
        fee[0,idx] = -1
        fee[1,idx] = -1
        fee[2,idx] = -1
        fee[3,idx] = -1
        fee[4,idx] = -1
        fee[5,idx] = -1
    j+=1
# detector mean values
# look at correlations

emean = np.mean(h5e["mean"],axis=1)
jmean = np.mean(h5j["mean"],axis=1)

cfig,cax = plt.subplots(3,6)
for ii in range(fee.shape[0]):
    cax[0,ii].scatter(fee[ii,:],dg2/1e5)
    cax[1,ii].scatter(fee[ii,:],emean)
    cax[2,ii].scatter(fee[ii,:],jmean)

cax[0,0].set_ylabel("DG2")
cax[1,0].set_ylabel("Epix mean")
cax[2,0].set_ylabel("Jungfaru mean")

cax[2,0].set_xlabel("f_11_ENRC")
cax[2,1].set_xlabel("f_12_ENRC")
cax[2,2].set_xlabel("f_21_ENRC")
cax[2,3].set_xlabel("f_22_ENRC")
cax[2,4].set_xlabel("f_63_ENRC")
cax[2,5].set_xlabel("f_64_ENRC")

cfig.suptitle("FEEGasDetEnergy Signal correlations")

# %% check DG2 only
ccfig,ccax = plt.subplots(2,1)
ccax[0].scatter(dg2, emean)
ccax[1].scatter(dg2, jmean)
ccax[0].set_ylabel("Epix mean")
ccax[1].set_ylabel("Jungfrau mean")
ccax[1].set_xlabel("DG2")

# %% Compare smoothed, normalized signal
dvert = 3
savgol_win = 2000
savgol_order = 2
nseg = 2048
f_sample = 120 # 120 HZ sampling

dg_inds = dg2 > 5e3
f_inds = fee_trim[0,:] > 5
dg2_trim = dg2[dg_inds]
emean_trim = emean[dg_inds]
jmean_trim = jmean[dg_inds]

fee_trim = fee[:,dg_inds]

# Normalized and then smoothed ratios
e_dnorm = emean_trim / dg2_trim
j_dnorm = jmean_trim / dg2_trim
e_fnorm = emean_trim / fee_trim[0,:]
j_fnorm = jmean_trim / fee_trim[0,:]

tfig,tax = plt.subplots()
ffig,fax = plt.subplots()
if len(e_dnorm) >= savgol_win:
    # Calculate 4 smooth residuals
    # e_dsmooth = savgol_filter(e_dnorm, savgol_win, savgol_order)
    # e_dsmooth = e_dnorm - e_dsmooth
    e_dsmooth = e_dnorm
    edw = welch(e_dsmooth, fs=f_sample, nperseg=nseg, detrend='linear')

    # j_dsmooth = savgol_filter(j_dnorm, savgol_win, savgol_order)
    # j_dsmooth = j_dnorm - j_dsmooth
    j_dsmooth = j_dnorm
    jdw = welch(j_dsmooth, fs=f_sample, nperseg=nseg, detrend='linear')

    # e_fsmooth = savgol_filter(e_fnorm, savgol_win, savgol_order)
    # e_fsmooth = e_fnorm - e_fsmooth
    e_fsmooth = e_fnorm
    efw = welch(e_fsmooth, fs=f_sample, nperseg=nseg, detrend='linear')

    # j_fsmooth = savgol_filter(j_fnorm, savgol_win, savgol_order)
    # j_fsmooth = j_fnorm - j_fsmooth
    j_fsmooth = j_fnorm
    jfw = welch(j_fsmooth, fs=f_sample, nperseg=nseg, detrend='linear')

    # d_norm = (dg2 - np.mean(dg2))/(np.std(dg2))

    tax.clear()
    # tax.plot(3*dvert + (emean_vals - np.mean(emean_vals))/np.std(emean_vals),label="Epix")
    # tax.plot(2*dvert + (jmean_vals - np.mean(jmean_vals))/np.std(jmean_vals),label="Jungfrau")
    # tax.plot(dvert + d_norm, label="DG2")
    #
    # Plot all smoothed normalized curves
    tax.plot(3*dvert + (e_dsmooth - np.mean(e_dsmooth))/np.std(e_dsmooth),
             label="Smoothed Epix/DG2")
    tax.plot(2*dvert + (j_dsmooth - np.mean(j_dsmooth))/np.std(j_dsmooth),
             label="Smoothed Jungfrau/DG2")
    tax.plot(dvert + (e_fsmooth - np.mean(e_fsmooth))/np.std(e_fsmooth),
             label="Smoothed Epix/f_11")
    tax.plot((j_fsmooth - np.mean(j_fsmooth))/np.std(j_fsmooth),
             label="Smoothed Jungfrau/f_11")

    tax.set_xlabel("Frame number")
    tax.set_ylabel("Intensity")
    # tax.set_title(f"ii: {ii}, {fname.split('.')[0]}")
    tax.set_title(f"Run {run} normalizations")
    # tax.set_ylim((-1*max(d_norm),2.5*dvert + max(d_norm)))
    tax.legend()
    # tfig.savefig(out_dir + f"{fname.split('.')[0]}_savgol{savgol_win}_signal_full.png")
    #
    tax.set_xlim((1000,1500))

    # tfig.savefig(out_dir + f"{fname.split('.')[0]}_savgol{savgol_win}_signal_zoomed.png")
    # tfig.canvas.draw()
    # tfig.canvas.flush_events()

    fax.clear()
    N = len(dg2)
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



# %% Close

h5e.close()
h5j.close()

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
