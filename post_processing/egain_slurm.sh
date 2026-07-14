#!/usr/bin/env sh

RUN=$1
RUN_STR=$(printf "%04d" $RUN)
expt='cxi101672626'
DETECTOR='epix'
QUEUE='milano'
CORES=1

source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh

### New SLURM stuff
# Batch queue stuff

# FULLCOMMAND="$(which mpirun) -np ${CORES} $(pwd)/${mw}"
FULLCOMMAND="python save_eGain.py -r ${RUN}"
echo $FULLCOMMAND

#Submit to SLURM
sbatch << EOF
#!/bin/bash

#SBATCH -p ${QUEUE}
#SBATCH -t 10:00:00
#SBATCH --job-name eg_${RUN_STR}
#SBATCH --output results/egain_${RUN_STR}.out
#SBATCH --ntasks=$CORES
#SBATCH --account lcls:${expt}
$FULLCOMMAND
EOF

echo "Job sent to queue"


