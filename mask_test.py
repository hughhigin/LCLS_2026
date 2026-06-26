#!/usr/bin/env python3

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
import reborn
from reborn import detector
# from reborn.external. import LCLSFrameGetter
from reborn.external.hdf5 import load_pad_data_from_h5, load_dataframe_from_h5
from reborn.dataframe import DataFrame

# import config
# import logging
# import os
# import inspect
# import pickle
# from typing import Optional, Dict, Any, List
# import numpy as np
# from ..shims import h5py
# from ..detector import PADGeometry, PADGeometryList
# from ..source import Beam
# from ..dataframe import DataFrame


mask_path = "masks4OM/epix_edges_innercircle_outercircle.h5"
h5 = h5py.File(mask_path, 'a')
m = load_pad_data_from_h5(mask_path, "data")

dataframe = DataFrame()
dataframe.set_mask(m)

# df = load_dataframe_from_h5(mask_path)

# m = reborn.detector.load_pad_data_from_h5(h5_file_path, "mask")
