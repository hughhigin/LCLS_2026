import psana
from psana import *
import sys, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

exp_id = "cxil1005322"
run = 224
# CXI:DS1:MMS:06.RBV
ds = DataSource('exp=' + exp_id + ':run=' + str(run))
dist = Detector("CXI:DS1:MMS:06.RBV") # event code
psi_det = Detector("CXI-DG2-BMMON") # event code

# NOTE: This section would only work if run is restored
# # Save timestamps
# times = []
# fids = []
# tIs = []
# ps_fid = []
# psi = []
# no_psi_fids = []
# for nevent, evt in enumerate(ds.events()):
#     evtId = evt.get(psana.EventId)
#     tval = evtId.time()
#     fid = evtId.fiducials()
#     ps_id = [tval[0], tval[1], fid]
#     ps_fid.append(ps_id)

#     # print("Time: " + str(evtId.time()))
#     # print("Fid:" + str(evtId.fiducials()))
#     try:
#         psi_val = psi_det.get(evt).TotalIntensity()
#     except:
#         no_psi_fids.append(ps_id)
#         psi_val = 1

#     psi.append(psi_val)
#     # print("\tTotal intensity: " + str(psi_det.get(evt).TotalIntensity()))

#     if nevent % 1e3 == 0:
#         print("nevent:" + str(nevent) + ", ID: " + str(ps_id))
#         print("Detector distance: " + str(dist))

# ps_fid = np.array(ps_fid)
# psi = np.array(psi)
# print(no_psi_fids)

# %% Load 2024 reborn h5, jung

run_str = f"r{run:04d}"
reb_dir = "/sdf/data/lcls/ds/cxi/cxil1005322/results/cxil1005322/hdf5/"
ref_name = "cxil1005322_r" + str(run) + "_jungfrau.h5"
reb_file = reb_dir + ref_name

# Load h5 from OM, 2024

om_home = "/sdf/data/lcls/ds/cxi/cxil1005322/results/tgrant/om/hdf5/"
om_dir = om_home + "epix_" + run_str + "/"
ref_name = "cxil1005322-" + run_str + "_0.cxi"
om_file = om_dir + ref_name

h5om = h5py.File(om_file)
h5re = h5py.File(reb_file)
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
