#!/usr/bin/env sh
#
# Script to blindly process a bunch of runs the same way
# by sending a series of SLURM jobs

MIN_RUN=$1
MAX_RUN=$2

for ((i=MIN_RUN; i<=MAX_RUN; i++)); do
    ./egain_slurm.sh "$i"
done
