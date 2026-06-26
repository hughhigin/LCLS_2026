#!/usr/bin/env python3

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
from write_radials_to_h5_wave8 import MyParallelRadialProfiler,MyLCLSFrameGetter
# from reborn import detector
import config_hh as config # TEST CONFIG FOR METADATA
import reborn
import os
import psana
from scipy import ndimage
import h5py

default_config = config.default_config()
memory = joblib.Memory(default_config["joblib_directory"])

# %% Compare wave8 values between reborn (config_hh) and 2024 OM
