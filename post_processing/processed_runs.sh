#!/bin/bash
ls /sdf/data/lcls/ds/cxi/cxi101672626/results/post_processing/results/reb_hdf5 | grep 'cxi1016.*step1\.' | cut -d '_' -f 2,3
