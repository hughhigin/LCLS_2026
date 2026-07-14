#!/usr/bin/env python3

# %% Import

import config as config
import runstats as runstats

run_number = 95
detector = "epix"

conf = config.get_config(run_number, detector)

runstats_conf = conf["runstats"]
runstats_conf["stop"] = 1e7
runstats_conf["step"] = 1
runstats_conf["start"] = 0
runstats_conf["n_processes"] = 12

del runstats_conf["histogram_params"]


# FULLCOMMAND="python write_radials_to_h5.py -r ${RUN} -j ${CORES} -d ${DETECTOR}"

# stats = runstats.get_runstats(run_number=run_number, det=detector, maxes=True)
stats = runstats.get_runstats(
    run_number=run_number,
    det=detector,
    n_processes=12,
    max_frames=1e7,
)

# %% Load hdf5

exp_id = "cxi101672626"
run_number =  95
run_str = f"r{run_number:04d}"
reb_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/post_processing/results/reb_hdf5/"
h5e_name = reb_dir + f"{exp_id}_{run_str}_epix.h5"
newe_name = reb_dir + f"{exp_id}_{run_str}_epix_noIceJ.h5"
