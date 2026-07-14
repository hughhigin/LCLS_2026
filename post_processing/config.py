# Draft config file for 2024 data - Hugh

# Marked parameter locations with asterisks

from reborn import detector, source
from reborn.detector import epix100_pad_geometry_list, PADGeometryList
from reborn.external.crystfel import geometry_file_to_pad_geometry_list
import numpy as np

# ** CHANGE FILEPATHS **
home_dir = "/sdf/data/lcls/ds/cxi/cxi101672626/results/post_processing/"

# default method from 2024, called by get_config to be modified
def default_config(detector='jungfrau'):
    r""" Create the default configurations.  You should not use this directly; instead use
    get_config which should provide run-specific parameters.
    detector = jungfrau by default, alternatively epix.
    """
    # general configurations
    # required keys: experiment_id
    # possible keys: results_directory, cachedir
    config = dict(experiment_id='cxi101672626', #*
                  results_directory='results',
                  hdf5_directory=home_dir + "results/reb_hdf5/", #*
                  cachedir='cache/',
                  debug=1,
                  joblib_directory="results/joblib", #*
                  photon_wavelength_pv='SIOC:SYS0:ML00:AO192') #*
    # detector configurations (we make a dictionary for every available PAD detector)
    # required keys: pad_id, geometry
    # possible keys: mask, motions
    # NOTES -- geometry: can be path to geom file or a pad_geometry_list_object
    #              mask: list of paths to masks (you can use multiple masks to take care of one particular feature)
    #                    example: ['badrows.mask', 'edges.mask', 'spots.mask', 'threshold.mask']
    #           motions: dictionary
    #                    example: {'epics_pv':'CXI:DS1:MMS:06.RBV', 'vector':[0, 0, 1e-3]}

    # Default mask and geometry files (Reborn format)

    # Epix
    epix_masks = [
        "./geometry/epix_edges.mask", # Mask out middle panel cross and 3 pixel border
        # "./geometry/epix_llPanel.mask", # Remove lower left panel due to gain change
    ]
    epix_geometry_file = './geometry/epix_recenter_run74_xyManual_up200um.json'
    epix10ka_0 = dict(
        pad_id='epix10ka_0',
        geometry=PADGeometryList(filepath=epix_geometry_file),
        data_type='calib',
        mask=epix_masks,
    )

    # Jungfrau
    jungfrau_geometry_file = './geometry/jungfrau_run46_AgBeh_powderFit.json' #*
    # jungfrau_geometry_file = './geometry/jungfrau_run18_xAdjust.json' #*
    jungfrau_masks = [
        "./geometry/jungfrau_edges.mask",
        "./geometry/jungfrau_center.mask",
        "./geometry/jungfrau_highQ.mask", # Mask a couple outer edge places
        "./geometry/jungfrau_pad02corner.mask", # Mask a couple outer edge places
        "./geometry/jungfrau_pad40corner.mask", # Mask a couple outer edge places
        "./geometry/jungfrau_pad44corner.mask", # Mask a couple outer edge places
        "./geometry/jungfrau_pad47corner.mask", # Mask a couple outer edge places
        # Dead pixel masks
        "./geometry/jungfrau_r223_deadLow.mask", 
        "./geometry/jungfrau_r223_highStd.mask",
    ]
    jungfrau4m = dict(
        pad_id='jungfrau4M',
        geometry=PADGeometryList(filepath=jungfrau_geometry_file),
        data_type='calib',
        mask=jungfrau_masks,
    )

    #best not to do both detectors at the same time
    if detector == "jungfrau":
        config['pad_detectors'] = [jungfrau4m]  # list allows for multiple detectors
    elif detector == "epix":
        config['pad_detectors'] = [epix10ka_0]
    else:
        print(f"ERROR: {detector} detector unknown. Either jungfrau or epix.")

    # radial profiler configurations
    config['profiles'] = dict(n_bins=500,
                              q_range=[0, 3e10])
    # runstats configurations **REVISIT**
    histogram_config = dict(bin_min=-5, bin_max=50, n_bins=100, zero_photon_peak=0, one_photon_peak=8)
    runstats_config = dict(log_file=None,
                           checkpoint_file=None,
                           checkpoint_interval=250,
                           message_prefix='',
                           debug=False,
                           histogram_params=histogram_config)
    config['runstats'] = runstats_config

    # ** CHECK - update these params? **
    pvs = {"photonBeam_rate": "EVNT:SYS0:1:LCLSBEAMRATE",
           "photonBeam_wavelength": "SIOC:SYS0:ML00:AO192",
           "photonBeam_energy": "SIOC:SYS0:ML00:AO627",
           "photonBeam_pulse_energy": "SIOC:SYS0:ML00:AO541",
           "eBeam_pulse_length": "SIOC:SYS0:ML00:AO820",
           "Acqiris": "CxiEndstation.0:Acqiris.0"}
    config["pvs"] = pvs
    config["beam"] = source.load_beam("geometry/beam.json")
    return config


# Run-specific modifications go here, e.g. if you want to manually set the
# geometry for a set of runs.
def get_config(run_number, detector="jungfrau"):
    # This is the place to modify the config according to run number (e.g. detector geometry, etc.)
    config = default_config(detector)
    config['run_number'] = run_number

    # RUNSTATS CONFIG INFO
    run = f"r{run_number:04d}"
    results = (config['results_directory'] + '/runstats/' # **
            + detector + '_' + run + '/')  # e.g. ./results/runstats/jungfrau_r0045/
    config['runstats']['checkpoint_file'] = results + "checkpoints/" + run
    config['runstats']['log_file'] = results + "logs/" + run


    # ** RUN-SPECIFIC DETECTOR CONFIGS
    if detector == "epix":
        # Add kapton mask for later runs
        if run_number >= 33:
            config['pad_detectors'][0]['mask'].append(
                "./geometry/epix_r33_kapton.mask"
            )

        # Runs 61-64: DNA standard, 100 uM 25mer
        if run_number in range(61,65):
            config['pad_detectors'][0]['mask'].extend([
                "./geometry/epix_r061_flare.mask",
                "./geometry/epix_r061_shadows.mask",
            ])

        # Runs 222-> 274: masking
        if run_number in range(222,274):
            config['pad_detectors'][0]['mask'].extend([
                "./geometry/epix_r222_injector.mask",
                "./geometry/epix_r222_leftShadow.mask",
                # "./geometry/epix_r222_llShadow.mask",
                "./geometry/epix_r222_urShadow.mask",
                "./geometry/epix_r223_flare.mask", # Use on 222 for background
            ])

    elif detector == "jungfrau":
        if run_number in [174,177]:
            config['pad_detectors'][0]['mask'].append(
                "./geometry/jungfrau_r177_shadow.mask"
            )
        elif run_number in range(61,64):
            config['pad_detectors'][0]['mask'].append(
                "./geometry/jungfrau_r063_flare.mask",
            )
        elif run_number in range(211,221):
            config['pad_detectors'][0]['mask'].append(
                "./geometry/jungfrau_r211_flare.mask",
            )
        elif run_number in range(222,281):
            config['pad_detectors'][0]['mask'].append(
                "./geometry/jungfrau_r223_flare.mask",
            )

    config['runstats']['message_prefix'] = f"Run {run_number}: "
    return config

# geometry from config via detector.load_pad_geometry_list
def get_geometry(run_number=None):
    # our convention is for the primary (saxs in this experiment)
    # detector to be first in the list
    c = get_config(run_number=run_number)
    pads = c['pad_detectors'][0]['geometry']
    if isinstance(pads, str):
        return detector.load_pad_geometry_list(pads)
    elif isinstance(pads, detector.PADGeometryList):
        return pads
    else:
        print('The geometry is not understood, please review the config file.')

# Main: Just print base_config?
if __name__ == '__main__':
    print(f'Base Configurations:\n\t{base_config()}')

