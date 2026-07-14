#usage: from reborn directory with setup.sh script in it:
# ./run_reborn_slurm.sh 17

RUN=$1
RUN_STR=$(printf "%04d" $RUN)
expt='cxi101672626'
DETECTOR='jungfrau'
QUEUE='milano'
# CORES=120
CORES=240
# step=$2

source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh

### New SLURM stuff
#
# Batch queue stuff
#

# FULLCOMMAND="$(which mpirun) -np ${CORES} $(pwd)/${mw}"
FULLCOMMAND="python write_radials_to_h5.py -r ${RUN} -j ${CORES} -d ${DETECTOR}"
echo $FULLCOMMAND

#Submit to SLURM
sbatch << EOF
#!/bin/bash

#SBATCH -p ${QUEUE}
#SBATCH -t 10:00:00
#SBATCH --exclusive
#SBATCH --job-name rjung${RUN_STR}
#SBATCH --output results/reb_hdf5/bsub_r${RUN_STR}_${DETECTOR}.out
#SBATCH --ntasks=$CORES
#SBATCH --account lcls:${expt}
$FULLCOMMAND
EOF

echo "Job sent to queue"

