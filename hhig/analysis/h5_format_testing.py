import psana
from psana import *
import sys, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

exp_id = "cxil1005322"
run = 226
# CXI:DS1:MMS:06.RBV
ds = DataSource('exp=' + exp_id + ':run=' + str(run))
dist = Detector("CXI:DS1:MMS:06.RBV") # event code
psi_det = Detector("CXI-DG2-BMMON") # event code

# NOTE: convert Q to inverse angstroms !!
# Process runs 224, 225
# Investigate DG2 values across runs
# Run 80
# Pretty: Runs 228, 232 (DNA day 5)

# %% Load 2024 reborn h5, jung
# Maxima: 500 / 7
# epix
run = 258
run_str = f"r{run:04d}"
reb_dir = "results/reb_hdf5/"
# ref_name = "cxil1005322_" + run_str + "_epix_bigMask_backup2.h5"
ref_name = "cxil1005322_" + run_str + "_epix_bigMask.h5"
reb_epix_file = reb_dir + ref_name
h5repix = h5py.File(reb_epix_file)

run = 258
run_str = f"r{run:04d}"
reb_dir = "results/reb_hdf5/"
# ref_name = "cxil1005322_" + run_str + "_epix_bigMask_backup2.h5"
ref_name = "cxil1005322_" + run_str + "_epix_bigMask.h5"
# ref_name = "cxil1005322_" + run_str + "_jungfrau_oldMasks.h5"
reb_epix_file = reb_dir + ref_name
h5repix2 = h5py.File(reb_epix_file)


mean_vals = h5repix2['mean']; mean_vals.shape
psi = np.array(h5repix['/post_sample_intensity'])
psi2 = np.array(h5repix2['/post_sample_intensity'])

tfig, tax = plt.subplots()
# tax.plot(np.mean(mean_vals, axis=1)/psi2)
tax.plot(np.mean(mean_vals, axis=1))
# tax.plot((psi-np.mean(psi))/np.std(psi) + 3)
# tax.plot((psi2-np.mean(psi2))/np.std(psi2))
# tax.plot(np.mean(mean_vals, axis=1) / psi, 'o')
tax.set_xlim(1000,1500)
# tax.set_ylim(-5, 10)
tfig.show()

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
