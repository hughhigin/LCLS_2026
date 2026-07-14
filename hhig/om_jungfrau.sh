# Fresh rewrite for Hugh's practice, 2026-05-xx

RUN=$1
RUN_STR=$(printf "%04d" $RUN)

# Running parameters
# maskfn=r0007_jungfrau_combined_mask.h5
# geomfn=r0007_jungfrau_padGeom.geom
maskfn=old_jungfrau_mask.h5
geomfn=old_jungfrau.geom
dir=results/om_hdf5/jungfrau_online_r$RUN_STR
expt='cxi101672626'
QUEUE='psanaq'
CORES=8

# Check for new directory
if [[ ! -e $dir ]]; then
    mkdir -p $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory" 1>&2
fi

# Copy .yaml, geom, and mask files
cd $dir
# cp ../../template_jungfrau.yaml monitor.yaml
cp ../../../template_jungfrau.yaml monitor.yaml
cp ../../../masks4OM/${geomfn} .
cp ../../../masks4OM/${maskfn} .

# sed commands: add run specific files for monitor.yaml
sed -i "s/RUN/$RUN_STR/" monitor.yaml
sed -i "s/MASKFILE/$maskfn/" monitor.yaml
#sed -i "s/DARKCALFILE/$darkcalfn/" monitor.yaml # *Dark file needed?
sed -i "s/GEOMFILE/$geomfn/" monitor.yaml
sed -i "s/EXP/$expt/" monitor.yaml

# Environment variables
mw=monitor_wrapper.sh
DATASOURCE="exp=$expt:run=$RUN "
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
source /sdf/group/lcls/ds/tools/om/setup/setup-psana1.sh

# Create and run monitor.sh
echo Creating and Running $(pwd)/${mw}
echo '#!/bin/bash' > $(pwd)/${mw}
echo '# File automatically created by the'  >> $(pwd)/${mw}
echo '# run_om.sh script' >> $(pwd)/${mw}
echo 'source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh' >> $(pwd)/${mw}
echo 'source /sdf/group/lcls/ds/tools/om/setup/setup-psana1.sh' >> $(pwd)/${mw}
echo "om_monitor ${DATASOURCE}" >> $(pwd)/${mw}
chmod +x $(pwd)/${mw}

# Run!
mpirun -n ${CORES} $(pwd)/${mw}

