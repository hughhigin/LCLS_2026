#!/usr/bin/env sh
#
# Script to blindly process a bunch of runs the same way
# by sending a series of SLURM jobs

MIN_RUN=$1
MAX_RUN=$2

# REBORN PROFILES

for ((i=MIN_RUN; i<=MAX_RUN; i++)); do
    ./reb_jungfrau_redo.sh "$i"
done
