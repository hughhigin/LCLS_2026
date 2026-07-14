#!/bin/bash
ls /sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/reb_hdf5 | grep 'cxi1016.*step1_noIceJ\.' | cut -d '_' -f 2,3
