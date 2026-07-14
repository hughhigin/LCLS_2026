# Draft config file for 2024 data - Hugh

# Marked parameter locations with asterisks

from reborn import detector, source
from reborn.detector import epix100_pad_geometry_list, PADGeometryList
from reborn.external.crystfel import geometry_file_to_pad_geometry_list
import numpy as np

# default method from 2024, called by get_config to be modified
# ** CHANGE FILEPATHS **
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
                  hdf5_directory="/sdf/data/lcls/ds/cxi/cxi101672626/results/hhig/results/reb_hdf5/", #*
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

    # Hugh geom file
    # 2024 files
    jungfrau_geometry_file = './geometry/jungfrau_run46_AgBeh_powderFit.json' #*
    jungfrau_masks = [
        "./geometry/jungfrau_edges.mask",
    ]
    jungfrau4m = dict(pad_id='jungfrau4M',
                      geometry=PADGeometryList(filepath=jungfrau_geometry_file),
                      data_type='calib',
                      mask=jungfrau_masks,
                      )

    # Hugh files
    # epix_geometry_file = './geometry/epix_singlepanel.json'
    epix_masks = [
        "./geometry/epix_edges.mask",
        "./geometry/epix_llPanel.mask",
    ]
    # epix_geometry_file = './geometry/epix_recentered_postmove_r163.json'
    epix_geometry_file = './geometry/epix_recenter_run12.json'
    # epix_geometry_file = './epix.json'
    epix10ka_0 = dict(pad_id='epix10ka_0',
                      geometry=PADGeometryList(filepath=epix_geometry_file),
                      data_type='calib',
                      mask=epix_masks,
                      )
    # epix100 = dict(pad_id='epix100',
    #                geometry=epix100_pad_geometry_list(detector_distance=1),
    #                data_type='raw')

    #best not to do both detectors at the same time
    if detector == "jungfrau":
        config['pad_detectors'] = [jungfrau4m]  # list allows for multiple detectors
    elif detector == "epix":
        config['pad_detectors'] = [epix10ka_0]
    else:
        print(f"ERROR: {detector} detector unknown. Either jungfrau or epix.")

    # radial profiler configurations **WHAT MEANS**
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
def get_config(run_number, detector="jungfrau", maxes=False):
    print(maxes)
    # This is the place to modify the config according to run number (e.g. detector geometry, etc.)
    config = default_config(detector)
    run = f"r{run_number:04d}"
    if maxes:
        results = (config['results_directory'] + '/runstats_maxes/' # **
                + detector + '_' + run + '/')  # e.g. ./results/runstats/jungfrau_r0045/
    else:
        results = (config['results_directory'] + '/runstats/' # **
                + detector + '_' + run + '/')  # e.g. ./results/runstats/jungfrau_r0045/
    print(results)
    # config['runstats']['results_directory'] = results
    config['run_number'] = run_number
    config['runstats']['checkpoint_file'] = results + "checkpoints/" + run
    config['runstats']['log_file'] = results + "logs/" + run
    # if detector == "jungfrau":
        # if run_number in range(17,19):
        #     config['pad_detectors'][0]['mask'].append('geometry/samantha_run21-25.mask')
        # if run_number in range(263,266):
            # config['pad_detectors'][0]['mask'].append('geometry/maryellen_run263-265.mask')
            # config['pad_detectors'][0]['mask'].append('geometry/jungfrau_diode_mask_run263-265.mask')
    if detector == "epix":
        if (run_number in range(0,15) or run_number == 18):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(filepath='./geometry/epix_recenter_run12.json')
        elif run_number in range(15,19):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(filepath='./geometry/epix_recenter_run15.json')
        elif run_number in range(19,21):
            config['pad_detectors'][0]['mask'].append("./geometry/epix_collagen_band.mask")
            config['pad_detectors'][0]['geometry'] = PADGeometryList(filepath='./geometry/epix_recenter_run15.json')
        elif run_number in range(21, 25):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(filepath='./geometry/epix_recenter_run15.json')
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r22_kapton.mask")
        elif run_number in range(25,30):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(filepath='./geometry/epix_recenter_run25.json')
        elif run_number in range(30,31):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(filepath='./geometry/epix_recenter_run25.json')
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r30_kapton.mask")
        elif run_number in range(31,52):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(filepath='./geometry/epix_recenter_run32.json')
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r33_kapton.mask")
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r36_flare.mask")
        elif (run_number in range(52,56) or run_number == 59):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(filepath='./geometry/epix_recenter_run32.json')
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r52_flare.mask")
        # Skip 56 so dark has no masks
        elif run_number in range(59,60):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(filepath='./geometry/epix_recenter_run32.json')
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r59_flare.mask")
        elif run_number in range(60,61):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(filepath='./geometry/epix_recenter_run32.json')
            # config['pad_detectors'][0]['mask'].append("./geometry/epix_r60_flare.mask")
        elif run_number in range(61,63):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(filepath='./geometry/epix_recenter_run32.json')
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r61_flare.mask")
        elif run_number in range(63,66):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(filepath='./geometry/epix_recenter_run32.json')
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r63_flare.mask")
        elif run_number in range(66,67):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(filepath='./geometry/epix_recenter_run32.json')
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r66_shadow.mask") # ** SHADOW NAME
        elif run_number in range(67,68):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(filepath='./geometry/epix_recenter_run32.json')
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r67_flare.mask") # ** SHADOW NAME
        elif run_number in range(68,69):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run32.json'
            )
            # config['pad_detectors'][0]['mask'].append("./geometry/epix_r68_flare.mask")
        elif run_number in range(69,72):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run32.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r67_flare.mask")
        elif run_number in range(72,73):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r72_flare.mask")
        elif run_number in range(73,74):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r73_flare.mask")
        elif run_number in range(74,75):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r74_flare.mask")
        elif run_number in range(78,80):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r78_flare.mask")
        elif run_number in range(80,81):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r80_flare.mask")
        elif run_number in range(81,84):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r82_flare.mask")
        elif run_number in range(84,87):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r84_shadow.mask")
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r84_crossFlares.mask")
        elif run_number in range(95,96):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r95_flare.mask")
        elif run_number in range(98,99):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r95_flare.mask")
        elif run_number in range(119,120):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r138_flare.mask")
        elif run_number in range(124,125):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r138_flare.mask")
        elif run_number in range(126,127):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r126_flare.mask")
        elif run_number in range(130,131):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r130_flare.mask")
        elif run_number in range(132,133):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r130_flare.mask")
        elif run_number in range(135,136):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r130_flare.mask")
        elif run_number in range(138,139):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r138_flare.mask")
        elif run_number in range(141,142):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r138_flare.mask")
        elif run_number in range(146,147):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r146_flare.mask")
        elif run_number in range(150,151):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r146_flare.mask")
        elif run_number in range(152,153):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r152_flare.mask")
        elif run_number in range(156,157):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r156_flare.mask")
        elif run_number in range(158,159):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r156_flare.mask")
        elif run_number in range(162,163):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r156_flare.mask")
        elif run_number in range(168,169):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r168_flare.mask")
        elif run_number in range(174,175):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r174_flare.mask")
        elif run_number in range(177,178):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            # config['pad_detectors'][0]['mask'].append("./geometry/epix_r177_flare.mask")
        elif run_number in range(180,181):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r180_flare.mask")
        elif run_number in range(184,185):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r188_flare.mask")
        elif run_number in range(188,189):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r188_flare.mask")
        elif run_number in range(194,197):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r188_flare.mask")
        elif run_number in range(198,199):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r188_flare.mask")
        elif run_number in range(202,199):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r202_flare.mask")
        elif run_number in range(205,206):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r205_flare.mask")
        elif run_number in range(209,210):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r205_flare.mask")
        elif run_number in range(211,212):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r211_flare.mask")
        elif run_number in range(213,214):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r211_flare.mask")
        elif run_number in range(215,216):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r211_flare.mask")
        elif run_number in range(216,217):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(218,219):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(222,223):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r202_flare.mask")
            # NOTE: New background, but same 202 mask looks good
        elif run_number in range(223,224):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(225,226):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(229,230):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(230,231):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(232,233):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(234,235):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(236,237):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(238,239):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(240,241):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(241,242):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(244,245):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(246,247):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(250,252):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(252,273):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(275,276):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r202_flare.mask")
        elif run_number in range(276,279):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r216_flare.mask")
        elif run_number in range(279,280):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r279_flare.mask")
        elif run_number in range(281,283):
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            config['pad_detectors'][0]['mask'].append("./geometry/epix_r281_flare.mask")
        else:
            config['pad_detectors'][0]['geometry'] = PADGeometryList(
                filepath='./geometry/epix_recenter_run74_xyManual_up200um.json'
            )
            # config['pad_detectors'][0]['mask'].append("./geometry/epix_r78_flare.mask")

    elif detector == "jungfrau":
        if run_number in range(174,175):
            config['pad_detectors'][0]['mask'].append("./geometry/jungfrau_r177_shadow.mask")
        elif run_number in range(177,178):
            config['pad_detectors'][0]['mask'].append("./geometry/jungfrau_r177_shadow.mask")
        elif run_number in range(177,178):
            config['pad_detectors'][0]['mask'].append("./geometry/jungfrau_r177_shadow.mask")

        # epix10ka_0 = dict(pad_id='epix10ka_0',
        #                 geometry=PADGeometryList(filepath=epix_geometry_file),
        #                 data_type='calib',
        #                 mask=epix_masks,
        #                 )
        # config['pad_detectors'] = [epix10ka_0]

    config['runstats']['message_prefix'] = f"Run {run_number}: "
    return config


# **Where/why to use?**
# geometry from config via detector.load_pad_geometry_list
def get_geometry(run_number=None):
    # our convention is for the primary (saxs in this experiment) detector to be first in the list
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

