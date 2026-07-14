# Hugh's edited version for practice

# In the last line, replace **X** with the number of OM nodes to run on each
# machine and **Y** with a comma-separated list of hostnames corresponding to the
# machines on which OnDA should run.

RUN=$1
RUN_STR=$(printf "%04d" $RUN)
maskfn=r0022_combined_mask.h5
# r0022_combined_mask.h5
#darkcalfn=cxilx5920_r6_imagesum.h5
# geomfn=r0046_epix_padGeom.geom
geomfn=r0035_epix_padGeom.geom
# geomfn=epix.geom
dir=results/om_hdf5/epix_online_r$RUN_STR
expt='cxi101672626'
QUEUE='psanaq'
CORES=8


if [[ ! -e $dir ]]; then
    mkdir -p $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory" 1>&2
fi

cd $dir

# cp ../../../template_epix.yaml monitor.yaml
cp ../../../template_epix.yaml monitor.yaml
cp ../../../masks4OM/${geomfn} .
cp ../../../masks4OM/${maskfn} .
#cp ../../../${darkcalfn} .

sed -i "s/RUN/$RUN_STR/" monitor.yaml
sed -i "s/MASKFILE/$maskfn/" monitor.yaml
#sed -i "s/DARKCALFILE/$darkcalfn/" monitor.yaml
sed -i "s/GEOMFILE/$geomfn/" monitor.yaml
sed -i "s/EXP/$expt/" monitor.yaml

mw=monitor_wrapper.sh

DATASOURCE="exp=$expt:run=$RUN "
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
source /sdf/group/lcls/ds/tools/om/setup/setup-psana1.sh

echo Creating and Running $(pwd)/${mw}
echo '#!/bin/bash' > $(pwd)/${mw}
echo '# File automatically created by the'  >> $(pwd)/${mw}
echo '# run_om.sh script' >> $(pwd)/${mw}
echo 'source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh' >> $(pwd)/${mw}
echo 'source /sdf/group/lcls/ds/tools/om/setup/setup-psana1.sh' >> $(pwd)/${mw}
echo "om_monitor ${DATASOURCE}" >> $(pwd)/${mw}
chmod +x $(pwd)/${mw}

mpirun -n ${CORES} $(pwd)/${mw}
