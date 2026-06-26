import sys
import numpy as np
import h5py
import shutil
import pandas as pd

# Use python args
#usage: copy_OM_metadata_to_reborn_h5.py <run_number>
# Redo arguments

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run", type=int, default=154, help="Run number")
parser.add_argument("-d", "--detector", type=str, default="jungfrau", help="Detector name")
args = parser.parse_args()

# r = args.run
# detector = args.detector
# detector = 'jungfrau'
detector = 'epix'
# r = 235
r = 7
run = f'{r:04d}'

results_dir = f'/sdf/data/lcls/ds/cxi/cxil1005322/results/hhig/practice_0603/results/'
om_h5dir = results_dir + f'/om_hdf5/'
re_h5dir = results_dir + f'/reb_hdf5/'

#OM h5 file
if detector == 'jungfrau':
    om_path = om_h5dir + f'jungfrau_r{run}/cxil1005322-r{run}-master.cxi'
elif detector == 'epix':
    om_path = om_h5dir + f'epix_r{run}/cxil1005322-r{run}-master.cxi'
else:
    print("wrong detector name (jungfrau or epix)")
    exit()

#reborn h5 file
if detector == 'jungfrau':
    re_path = re_h5dir + f'cxil1005322_r{run}_jungfrau.h5'
elif detector == 'epix':
    re_path = re_h5dir + f'cxil1005322_r{run}_epix.h5'
else:
    print("wrong detector name (jungfrau or epix)")
    exit()

# Backup the reborn HDF5 file to be safe
if detector == 'jungfrau':
    backup_re_path = re_h5dir + f'cxil1005322_r{run}_jungfrau_backup.h5'
elif detector == 'epix':
    backup_re_path = re_h5dir + f'cxil1005322_r{run}_epix_backup.h5'
else:
    print("wrong detector name (jungfrau or epix)")
    exit()
# shutil.copy(re_path, backup_re_path)

with h5py.File(om_path) as h5om, h5py.File(re_path, 'a') as h5re:

    print(h5om['LCLS'].keys())
    print(h5re.keys())

    om_fid = h5om['LCLS/fiducial'][:]
    re_fid = h5re["/frame_id"][:]

    # grab the indices of the OM fiducials that match the frameid from reborn

    # these lines converts the binary string of om frame ID to integer 3 element array
    # matching reborn format
    om_fid2 = pd.Series(om_fid).apply(lambda x:
                                      np.array(x.decode("utf-8").split("-"), dtype=int))
    om_fid2 = np.stack(om_fid2.values)

    # this line makes the om frame IDs a dictionary for easy look up
    om_fid2_dict = {tuple(value): idx for idx, value in enumerate(om_fid2)}

    # this line checks each frameid in the reborn events and finds which om event
    # matches using all three ID elements (sec, ns, fid)
    idx = [om_fid2_dict[tuple(value)]
           for value in re_fid if tuple(value) in om_fid2_dict]

    # double check that the reordered OM arrays match the reborn arrays for frame id
    # print("OM: ", om_fid[idx][10])
    # print("RE: ", re_fid[10])

    # grab the datasets from OM we want
    psi = h5om['LCLS/post_sample_intensity'][:]

    # write the reordered array to the reborn file
    if '/post_sample_intensity' in h5re:
        del h5re['/post_sample_intensity'] #delete it if it previously existed
    h5re.create_dataset('/post_sample_intensity', data=psi[idx])

    # the h5re file is automatically closed and written using the with command
