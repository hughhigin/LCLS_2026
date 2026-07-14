# Add event-id list from file for sub plotting


import argparse
import numpy as np
import pyqtgraph as pg
import joblib
from reborn.dataframe import DataFrame
from reborn.external.pyqtgraph import imview
from reborn.external.lcls import LCLSFrameGetter
from reborn import analysis
from reborn.fileio import misc
from reborn.viewers.qtviews import PADView
from reborn.analysis.parallel import ParallelAnalyzer
from reborn.analysis.saxs import RadialProfiler

import config
import os
import psana
from scipy import ndimage
import h5py

default_config = config.default_config()
memory = joblib.Memory(default_config["joblib_directory"])

# Hardcoded parameters to start
run_number = 7
max_events = 1e7
detector="epix"

filter_gain = True
gthresh = 20

filter_max = False
mthresh = 20000

# Filepaths
home_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/post_processing/"
reb_dir = home_dir + "results/reb_hdf5/"
npy_dir = home_dir + "results/npy_files/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=int, default=None, help="Run number")
    args = parser.parse_args()
    run_number = args.run

conf = config.get_config(run_number, detector)
detectors = conf["pad_detectors"]

# %% Load event_IDs

excludes = []

if filter_gain:
    gain_filename = f"{detector}_r{run_number:04d}_gain_bitSums.npy"
    print(f"Filtering gain, thresh: {gthresh}")
    # Load gain values
    sum_gain = np.load(npy_dir + gain_filename)
    excludes += list(np.where(sum_gain >= gthresh)[0])

if filter_max:
    print(f"Filtering max pixel, thresh: {mthresh}")
    # read from h5
    
    h5_name = reb_dir + f"{exp_id}_{run_str}_{detector}.h5"
    with h5py.File(h5_name, 'r') as h5re:
        maxes = h5re['/frame_maxes']
        num_frames = len(fids)
        excludes.append(np.where(maxes >= mthresh)[0])
 

# event_ids, requires cached info

cachedir = 'cache'
cachefile = f"{cachedir}/event_ids_{run_number:04d}.pkl"
if os.path.exists(cachefile):
    debug_message("Loading cached event Ids:", cachefile)
    event_ids = misc.load_pickle(cachefile)
else:
    data_string = "exp=%s:run=%s:smd" % (exp_id, run_number)
    data_source = psana.DataSource(data_string)
    for evt in data_source.events():
        evtId = evt.get(psana.EventId)
        event_ids.append(evtId.time() + (evtId.fiducials(),))
        debug_message("Caching event IDs:", cachefile)
        misc.save_pickle(event_ids, cachefile)

keep_ids = [eid for ii,eid in enumerate(event_ids) if (ii not in excludes)]
print(f"Keeping {len(keep_ids)} / {len(event_ids)} frames")

framegetter = LCLSFrameGetter(
        run_number=run_number,
        max_events=max_events,
        experiment_id=conf["experiment_id"],
        event_ids=keep_ids,
        pad_detectors=detectors,
        cachedir=conf["cachedir"],
        postprocessors=None,
        photon_wavelength_pv=conf["photon_wavelength_pv"]
)
print("finished framegetter set up")

pv = PADView(frame_getter=framegetter)
pv.start()

