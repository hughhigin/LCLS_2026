#!/usr/bin/env sh
#
# Script to blindly process a bunch of runs the same way
# by sending a series of SLURM jobs

MIN_RUN=$1
MAX_RUN=$2

for ((i=MIN_RUN; i<=MAX_RUN; i++)); do
    # REBORN RUNSTATS
    # ./batch_runstats_epix.sh "$i"
    # ./batch_runstats_jungfrau.sh "$i"
    # REBORN PROFILES
    ./reb_epix_slurm.sh "$i"
    ./reb_jungfrau_slurm.sh "$i"
    # OM PROFILES
    ./om_epix_slurm.sh "$i"
    ./om_jungfrau_slurm.sh "$i"
done
