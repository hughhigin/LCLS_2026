#!/bin/bash
ls /sdf/data/lcls/ds/cxi/cxi101672626/xtc | grep s00-c00 | cut -d '-' -f 2 | sed 's/r//g'
