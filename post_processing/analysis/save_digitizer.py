#!/usr/bin/env python3

import argparse
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
home_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/post_processing/"
reb_dir = f"{home_dir}results/reb_hdf5/"
npy_dir = f"{home_dir}results/npy_files/"
deIced_dir = f"{home_dir}results/deIced_hdf5/"

run = 61
max_events = 1e7

# %% Arguments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=int, default=None, help="Run number")
    args = parser.parse_args()
    run = args.run

# %% Manual save dg2, digitizer for different runs

run_str = f"r{run:04d}"

# Load hdf5 to check event stamps
h5e_name = deIced_dir + f"{exp_id}_{run_str}_epix.h5"
h5e = h5py.File(h5e_name, 'r')

fids = h5e['/frame_id'][:]
nevents = len(fids)
seconds = fids[:,0]
nanoseconds = fids[:,1]
fiducials = fids[:,2]
nevents = len(seconds)

re_fid_dict = {tuple(value): idx for idx, value in enumerate(fids)}

ds = psana.DataSource('exp=' + exp_id + ':run=' + str(run))
qad_det = psana.Detector("qadc1")
dg2_det = psana.Detector("CXI-DG2-BMMON") # event code
wave8_det = psana.Detector("CXI-DG2-BMMON-WF")
hfx_det = psana.Detector("HFX-DG2-BMMON")
em_det = psana.Detector("EM3L0-BMMON")
fee_det = psana.Detector("FEEGasDetEnergy") # event code

# Waveforms
qad = np.zeros((nevents, 32000))
qad_times = np.zeros((nevents, 32000))
wave8 = np.zeros((nevents, 8, 4096)) # Use 32 bit values
wave8_times = np.zeros((nevents, 8, 4096)) # Use 32 bit values

# Diode intensities
dg2 = np.zeros(nevents)
hfx = np.zeros(nevents)
em = np.zeros(nevents)
fee = np.zeros((nevents,6))

j = 0
for nevent,evt in enumerate(ds.events()):
    evtId = evt.get(psana.EventId)
    sec = evtId.time()[0]
    ns = evtId.time()[1]
    fid = evtId.fiducials()

    if tuple((sec,ns,fid)) in re_fid_dict:
        idx = re_fid_dict[tuple((sec,ns,fid))]
        sys.stdout.write('\r%d/%d   %d'%(j,nevents, idx))
        sys.stdout.flush()

        qad[idx,:] = qad_det.raw(evt)[0]
        qad_times[idx,:] = qad_det.raw(evt)[0]

        for ww in range(8):
            wave8[idx,ww,:] = wave8_det.raw(evt)[ww + 8]
            wave8_times[idx,ww,:] = wave8_det.raw(evt)[ww + 8]
        
        dg2[idx] = dg2_det.get(evt).TotalIntensity()
        hfx[idx] = hfx_det.get(evt).TotalIntensity()
        em[idx] = em_det.get(evt).TotalIntensity()

        try:
            fee[idx,0] = fee_det.get(evt).f_11_ENRC()
            fee[idx,1] = fee_det.get(evt).f_12_ENRC()
            fee[idx,2] = fee_det.get(evt).f_21_ENRC()
            fee[idx,3] = fee_det.get(evt).f_22_ENRC()
            fee[idx,4] = fee_det.get(evt).f_63_ENRC()
            fee[idx,5] = fee_det.get(evt).f_64_ENRC()
        except:
            fee[idx,0] = -1
            fee[idx,1] = -1
            fee[idx,2] = -1
            fee[idx,3] = -1
            fee[idx,4] = -1
            fee[idx,5] = -1

        dg2[idx] = dg2_det.get(evt).TotalIntensity()

        # try:
        #     dg2[idx] = dg2_det.get(evt).TotalIntensity()
        # except:
        #     dg2[idx] = -1

        j+=1

    # if nevent >= 100:
    #     break

# %% Saving

h5e.close()

np.save(npy_dir + f"r{run:04d}_dg2_totalIntensity.npy", dg2)
np.save(npy_dir + f"r{run:04d}_hfx_totalIntensity.npy", hfx)
np.save(npy_dir + f"r{run:04d}_em_totalIntensity.npy", em)
np.save(npy_dir + f"r{run:04d}_fee.npy", fee)

np.save(npy_dir + f"r{run:04d}_qadc01_waveform.npy", qad)
np.save(npy_dir + f"r{run:04d}_qadc01_wftimes.npy", qad_times)
np.save(npy_dir + f"r{run:04d}_wave8_waveform.npy", wave8)
np.save(npy_dir + f"r{run:04d}_wave8_wftimes.npy", wave8_times)

# %% Figure

# mean_val = np.mean(qad_arr,axis=0)
# xvals = np.arange(len(mean_val))
# fig,ax = plt.subplots()
# ax.plot(xvals, mean_val, label="Full waveform")
# ax.plot(xvals[4800:5500], mean_val[4800:5500], label="cropped region")
# ax.set_xlabel("waveform index")
# ax.set_ylabel("amplitude")
# ax.legend()
# fig.savefig("digitizer_averaged_region_dark.png")



