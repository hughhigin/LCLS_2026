import argparse
import numpy as np
import pyqtgraph as pg
import joblib
from reborn.dataframe import DataFrame
from reborn.external.pyqtgraph import imview
from reborn.external.lcls import LCLSFrameGetter
from reborn import analysis
from reborn.viewers.qtviews import PADView
from reborn.analysis.parallel import ParallelAnalyzer
from reborn.analysis.saxs import RadialProfiler
# from reborn import detector
import config
# import config_noMasks as config
import reborn
import os
import psana
from scipy import ndimage
import h5py

default_config = config.default_config()
memory = joblib.Memory(default_config["joblib_directory"])

# Hardcoded parameters to start
run_number = 7
max_events = 1e7
detector="jungfrau"
# detector="epix"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=int, default=None, help="Run number")
    args = parser.parse_args()
    run_number = args.run

conf = config.get_config(run_number, detector)
detectors = conf["pad_detectors"]

# for d in detectors:
#     d["mask"] = None
framegetter = LCLSFrameGetter(
        run_number=run_number,
        max_events=max_events,
        experiment_id=conf["experiment_id"],
        pad_detectors=detectors,
        cachedir=conf["cachedir"],
        postprocessors=None,
        photon_wavelength_pv=conf["photon_wavelength_pv"]
)
print("finished framegetter set up")

# q = geom.q_mags(beam)
# pv = PADView(frame_getter=framegetter, mask=mask)
pv = PADView(frame_getter=framegetter)
pv.start()


