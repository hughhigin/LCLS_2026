import psana
from psana import *
import sys, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

# %% Load jungfrau, epix h5s from Reborn

# epix_name = "cxil1005322_r0236_epix_oldMasks.h5"
# jung_name = "cxil1005322_r0236_jungfrau_oldMasks.h5"
#
reb_dir = "results/reb_hdf5/"
os.chdir(reb_dir)

epix_suffix = "epix_bigMask.h5"
epix_names = glob("*" + epix_suffix)

jung_suffix = "jungfrau_oldMasks.h5"
jung_names = glob("*" + jung_suffix)

for epix_name in epix_names:
    pref = epix_name[:-len(epix_suffix)]
    jung_name = pref + jung_suffix
    if jung_name in jung_names:
        h5e = h5py.File(epix_name)
        h5j = h5py.File(jung_name)

        e_fid = h5e["/frame_id"][:]
        j_fid = h5j["/frame_id"][:]

        print(epix_name + " len: " + str(len(e_fid.flatten())))
        print("\t" + jung_name + " len: " + str(len(e_fid.flatten())))
        print("\tdifferent values: " + str(sum((e_fid - j_fid).flatten())))

        h5e.close()
        h5j.close()

# plt.plot(e_fid[3,:], j_fid[3,:])
# # plt.legend()
# plt.xlabel('e fids')
# plt.ylabel('j fids')
# plt.title("test")
# plt.savefig('r235_fid.png',dpi=150)
# plt.show()
