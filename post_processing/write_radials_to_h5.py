#!/usr/bin/env python

"""
Main function to write radials using reborn, called in batch jobs
NOTE: use notes and asterisks to mark key changes as much as possible
NOTE: Automatically calls runstats, which creates large files in the runstats dir
NOTE: Filepath and masking details set in "config.py"
TODO: Time difference between adding:
      np.max(data) vs. profiler.get_profile_statistic(data, statistic=np.max)
"""

import sys, shutil
import argparse
import numpy as np
import pyqtgraph as pg
import joblib
from reborn.dataframe import DataFrame
from reborn.external.pyqtgraph import imview
from reborn.external.lcls import LCLSFrameGetter
from reborn import analysis
from reborn.viewers.qtviews import PADView
from reborn.analysis.parallel import ParallelAnalyzer
from reborn.analysis.saxs import RadialProfiler
from reborn import fortran
# from reborn import detector
import config
# import config_noMasks as config # NO MASK VERSION OF CONFIG FILE**
import reborn
import os
import psana
from scipy import ndimage
import h5py

default_config = config.default_config()
memory = joblib.Memory(default_config["joblib_directory"])

# HH: Subclass RadialProfiler in order to properly add max

class MyRadialProfiler(RadialProfiler):

    def quickstats(self, data, weights=None, matrix=False):
        r"""
        Compute per-bin sums, weighted means, and weighted standard deviations.

        Let :math:`b` denote a radial bin and let :math:`i \in b` index the
        detector pixels that fall into that bin.  The input ``weights`` are
        interpreted as the per-pixel bin weights :math:`w_i`.  Internally, this
        method forms the corrected per-pixel quantity

        .. math::

            x_i =
            \begin{cases}
                d_i / w_i, & w_i > 0 \\
                0, & w_i = 0
            \end{cases}

        where :math:`d_i` is the input ``data`` value for pixel :math:`i`.

        The following quantities are then accumulated in each radial bin:

        .. math::

            W_b = \sum_{i \in b} w_i

        .. math::

            S_b = \sum_{i \in b} w_i x_i = \sum_{i \in b} d_i

        .. math::

            S^{(2)}_b = \sum_{i \in b} w_i x_i^2

        The returned mean and standard deviation are computed from these
        weighted moments:

        .. math::

            \mu_b = S_b / W_b

        .. math::

            \sigma_b = \sqrt{S^{(2)}_b / W_b - \mu_b^2}

        for bins with :math:`W_b > 0`.  Empty bins return zeros.

        This convention is useful when ``data`` already contains multiplicative
        per-pixel factors such as solid angle and polarization, and ``weights``
        contains those same factors times a mask.  In that case:

        - ``mean`` is the weighted average of the corrected quantity
          :math:`x_i = d_i / w_i`
        - ``sum`` is still the sum of the original input ``data`` values in the
          bin
        - ``weight_sum`` is :math:`W_b`
        - ``sum2`` is :math:`S^{(2)}_b`, the weighted second raw moment
          numerator of the corrected quantity

        In the scattering notation used elsewhere in the documentation, one
        often has

        .. math::

            I_i = w_i |F_i(q)|^2

        with :math:`w_i = J r_e^2 P_i \Delta \Omega_i` up to any additional
        per-pixel factors.  If you pass ``data = I`` and ``weights = w``, this
        method first forms

        .. math::

            x_i = I_i / w_i = |F_i(q)|^2

        and then accumulates weighted statistics of :math:`|F_i(q)|^2`.  This
        internal division is essential; without it, the accumulation would
        contain an unwanted extra factor of :math:`w_i`, for example
        :math:`\sum_i w_i^2 |F_i(q)|^2`.

        Therefore, do not apply the same physical weights twice.  If ``data``
        already has the form :math:`I_i = w_i |F_i(q)|^2`, pass those raw
        intensities together with ``weights = w`` and let :meth:`quickstats`
        perform the internal division.  If you have already corrected the data
        to :math:`|F_i(q)|^2`, then do not pass the same :math:`w_i` again as
        ``weights``.

        If ``weights`` is a binary mask, this reduces to the ordinary mean and
        standard deviation over unmasked pixels.  If ``weights`` is omitted, the
        mask configured with :meth:`set_mask` is used.

        The Fortran and sparse-matrix implementations compute the same
        quantities; ``matrix=True`` only changes the backend used to accumulate
        them.

        Arguments:
            data (|ndarray|): Diffraction intensities or other per-pixel values
                to be reduced radially.
            weights (|ndarray|): Optional per-pixel weights.  A binary mask is
                the special case ``weights in {0, 1}``.
            matrix (bool): Optional. Compute radial statistics via sparse matrix
                multiplication instead of the Fortran backend.

        Returns:
            dict: Dictionary containing ``mean``, ``sdev``, ``sum``, ``sum2``,
            and ``weight_sum`` for each radial bin.
        """
        sum_ = np.zeros(self.n_bins, dtype=np.float64)
        sum2 = np.zeros(self.n_bins, dtype=np.float64)
        w_sum = np.zeros(self.n_bins, dtype=np.float64)
        meen = np.empty(self.n_bins, dtype=np.float64)
        std = np.empty(self.n_bins, dtype=np.float64)
        data = self.concat_data(data).astype(np.float64).copy()
        if weights is None:
            weights = self._mask.astype(np.float64)
        else:
            weights = self.concat_data(weights).astype(np.float64)
        data[weights == 0] = 0
        data[weights != 0] /= weights[weights != 0]  # since the data already has weights referred above
        if self._matrix or matrix:
            meen *= 0
            std *= 0
            _s2_ = np.zeros(self.n_bins, dtype=np.float64)
            sum_[:] = self._ro @ (weights * data)
            sum2[:] = self._ro @ (weights * data**2)
            w_sum[:] = self._ro @ weights
            meen[w_sum > 0] = sum_[w_sum > 0] / w_sum[w_sum > 0]
            _s2_[w_sum > 0] = sum2[w_sum > 0] / w_sum[w_sum > 0]
            std = np.sqrt(_s2_ - meen**2)
        else:
            fortran.require_fortran("saxs radial profiles (set matrix=True to use the sparse matrix path)")
            fortran.scatter_f.profile_stats_indexed(data, self._fortran_indices, weights, sum_, sum2, w_sum)
            fortran.scatter_f.profile_stats_avg(sum_, sum2, w_sum, meen, std)

        # HH Addition
        fmax = np.max(data.flatten())
        out = dict(mean=meen, sdev=std, sum=sum_, sum2=sum2, weight_sum=w_sum,
                   frame_max = fmax)
        return out



class MyParallelRadialProfiler(ParallelAnalyzer):
    r"""
    Parallelized class for creating radial profiles from x-ray diffraction data.
    """

    include_median = False
    profiler = None
    radials = None

    def __init__(
        self,
        framegetter=None,
        n_q_bins=1000,
        q_range=None,
        pad_geometry=None,
        beam=None,
        mask=None,
        include_median=False,
        other_detectors_pv=None,
        **kwargs
    ):
        r"""
        Parallel analyzer that produces scattering ("radial") profiles from diffraction patterns.
        See the :meth:`~to_dict` method for details of what the output is.

        Standard profiles are computed using fortran code. Bin indices are cached for
        speed, provided that the |PADGeometry| and |Beam| do not change.

        Arguments:
            framegetter (|FrameGetter|): The FrameGetter that serves the data for analysis.
            n_bins (int): Number of radial bins (default is 1000).
            q_range (list-like): The minimum and maximum of the *centers* of the q bins.
            pad_geometry (|PADGeometryList|): Detector geometry, used to generate q magnitudes.
                                              If None, then automatically retrieved from raw data.
            beam (|Beam|): X-ray beam. Wavelength and beam direction required to calculate q magnitudes.
                           If None, then automatically retrieved from raw data.
            mask (List or |ndarray|): Optional (default is no masked pixels). Data will be multiplied by this mask,
                                      and the counts per radial bin will come from this (e.g. use values of
                                      0 and 1 if you want a normal average, otherwise you get a weighted average).
            include_median (bool): Set to True to include median profile (default is False).
            start (int): Which frame to start with.
            stop (int): Which frame to stop at.
            step (int): Step size between frames (default 1).
            n_processes (int): How many processes to run in parallel (if parallel=True).
            **kwargs: Any key-word arguments you would like to pass to the base class.
                      See: ..:py:class::`~reborn.analysis.parallel.ParallelAnalyzer`
        """
        super().__init__(framegetter=framegetter, **kwargs)
        self.logger.debug("set up ParallelAnalyser superclass")
        self.kwargs["n_q_bins"] = n_q_bins
        self.kwargs["q_range"] = q_range
        self.kwargs["pad_geometry"] = pad_geometry
        self.kwargs["beam"] = beam
        self.kwargs["mask"] = mask
        self.kwargs["include_median"] = include_median
        self.analyzer_name = "ParallelRadialProfiler"
        if pad_geometry is None:
            raise ValueError("pad_geometry cannot be None")
        if beam is None:
            raise ValueError("beam cannot be None")
        if mask is None:
            mask = pad_geometry.ones()
        mask = pad_geometry.concat_data(mask)
        self.mask = mask
        self.sap = None #Solid angle multiplied by polarization factor

        self.logger.debug("setting up profiler")
        self.profiler = MyRadialProfiler(
            mask=mask,
            n_bins=n_q_bins,
            q_range=q_range,
            pad_geometry=pad_geometry,
            beam=beam,
        )
        self.radials = { # ** REMOVE EXTRA INFO
            "sum": [],
            "sum2": [],
            "wsum": [],
            "mean": [],
            "sdev": [],
            "fmax": [], # **HH**
            "median": [],
            "frame_id": [],
            "n_frames": 0,
            "n_q_bins": n_q_bins,
            "q_bins": self.profiler.q_bin_centers,
            "experiment_id": self.framegetter.experiment_id,
            "run_id": self.framegetter.run_id,
            "pad_geometry": pad_geometry,
            "beam": beam,
            "mask": mask,
        }

        #add some entries to the dictionary for my pv values I want to save
        self.other_detectors_pv = other_detectors_pv
        if other_detectors_pv is not None:
            for d in other_detectors_pv:
                self.radials[d] = []

    # Subclassing version - slightly inefficient
    # def quickstats(self, data, weights=None):
    #     out = self.profiler.quickstats(data=data, weights=weights)
    #     if weights:
    #         data[weights == 0] == 0
    #         data[weights != 0] /= weights[weights != 0]
    #     out["frame_max"] = np.max(data)
    #     return out

    def add_frame(self, dat: DataFrame):
        if dat.validate():
            #data = dat.get_raw_data_flat()
            data = dat.get_processed_data_flat()
            # mask = dat.get_mask_flat()
            mask = self.mask
            if self.sap is None:
                g = dat.get_pad_geometry()
                b = dat.get_beam()
                self.sap = g.polarization_factors(beam=b)*g.solid_angles()
            weights= self.sap*mask
            out = self.profiler.quickstats(data=data, weights=weights)
            # out = self.quickstats(data=data, weights=weights)
            self.radials["sum"].append(out["sum"])
            self.radials["sum2"].append(out["sum2"])
            self.radials["wsum"].append(out["weight_sum"])
            self.radials["mean"].append(out["mean"])
            self.radials["sdev"].append(out["sdev"])
            self.radials["fmax"].append(out["frame_max"])
            self.radials["frame_id"].append(dat.get_frame_id())
            self.radials["n_frames"] += 1
            if self.kwargs["include_median"]:
                m = self.profiler.get_median_profile(data=data, mask=mask)
                self.radials["median"].append(m)
            #my additions:
            #edit LCLSFrameGetter code to add in other_detectors in init.
            #then in get_data loop through those other_detectors and grab the values
            #and add them to the df.parameters dictionary
            if self.other_detectors_pv is not None:
                for d in self.other_detectors_pv:
                    # print(dat.parameters[d])
                    self.radials[d].append(dat.parameters[d])


    def concatenate(self, stats: dict):
        self.radials["sum"].extend(stats["sum"])
        self.radials["sum2"].extend(stats["sum2"])
        self.radials["wsum"].extend(stats["wsum"])
        self.radials["mean"].extend(stats["mean"])
        self.radials["sdev"].extend(stats["sdev"])
        self.radials["fmax"].extend(stats["fmax"]) # ** HH **
        self.radials["median"].extend(stats["median"])
        self.radials["frame_id"].extend(stats["frame_id"])
        self.radials["n_frames"] += stats["n_frames"]


    def to_dict(self):
        """
        Create a dictionary of the results.

        Returns:
            (dict):
                - **mean** (|ndarray|) -- Mean of unmasked intensities
                - **sdev** (|ndarray|) -- Standard deviation of unmasked intensities
                - **median** (|ndarray|) -- Median of unmasked intensities (only if requested; this is slow)
                - **sum** (|ndarray|) -- Sum of unmasked intensities
                - **sum2** (|ndarray|) -- Sum of squared unmasked intensities
                - **wsum** (|ndarray|) -- Weighted sum (if no weights are provided,
                          then this is the number of unmasked pixels in the q bin)
                - **n_frames** (int) -- Number of frames analyzed.
                - **initial_frame** (int) -- First frame from which geometry and beam data are extracted.
                - **n_q_bins** (int) -- Number of q bins in radial profiles.
                - **q_bins** (|ndarray|) -- The centers of the q bins.
                - **experiment_id** (str) -- Identifier for experiment being analyzed.
                - **run_id** (str) -- Identifier for run being analyzed
                - **pad_geometry** (|PADGeometryList|) -- Detector geometry used to set up RadialProfiler.
                - **beam** (|Beam|) -- X-ray beam used to set up RadialProfiler.
                - **mask** (|ndarray|) -- Mask used to set up RadialProfiler.
        """
        return self.radials

    def from_dict(self, stats: dict):
        self.radials["sum"] = stats["sum"]
        self.radials["sum2"] = stats["sum2"]
        self.radials["wsum"] = stats["wsum"]
        self.radials["mean"] = stats["mean"]
        self.radials["sdev"] = stats["sdev"]
        self.radials["fmax"] = stats["fmax"] # ** HH **
        self.radials["median"] = stats["median"]
        self.radials["frame_id"] = stats["frame_id"]
        self.radials["n_frames"] = stats["n_frames"]
        self.radials["n_q_bins"] = stats["n_q_bins"]
        self.radials["q_bins"] = stats["q_bins"]
        self.radials["experiment_id"] = stats["experiment_id"]
        self.radials["run_id"] = stats["run_id"]
        self.radials["pad_geometry"] = stats["pad_geometry"]
        self.radials["beam"] = stats["beam"]
        self.radials["mask"] = stats["mask"]
        # self.radials["wave8"] = stats["mask"]


class MyLCLSFrameGetter(LCLSFrameGetter):
    def __init__(self, other_detectors_pv=None, mask=None, wave8_dict=None, **kwargs):
        """sub class of LCLSFrameGetter which allows additional detectors to be read from data.
        Give other_detectors as list of strings containing PV names desired."""
        super().__init__(**kwargs)
        if other_detectors_pv is not None:
            #save the pv names
            self.other_detectors_pv = other_detectors_pv
            #grab the actual psana detector instance
            self.other_detectors = [
                psana.Detector(d) for d in other_detectors_pv
            ]
        else:
            self.other_detectors_pv = None
            self.other_detectors = None

        print("Hello from MyLCLSFrameGetter")

    def get_data(self, frame_number=0):
        df = super().get_data(frame_number)
        if self.other_detectors_pv is not None:
            for i in range(len(self.other_detectors_pv)):
                data = self.other_detectors[i](self.event)
                df.parameters[self.other_detectors_pv[i]] = data
        df.set_mask(self.mask)
        return df

# @memory.cache(ignore=["n_processes"])
def get_radials(run_number=1, n_processes=1, start=0, stop=None, step=1, detector="jungfrau"):
    max_events = 1e7
    conf = config.get_config(run_number, detector)
    log_file = os.path.dirname(conf["runstats"]["log_file"])
    if not os.path.isdir(log_file):
        if not os.path.isdir(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.mkdir(log_file)
    RUNNING_file = log_file + "/RUNNING"
    with open(RUNNING_file, 'w') as f:
        f.close()
    runstats_conf = conf["runstats"]
    if stop is None:
        runstats_conf["stop"] = max_events
    else:
        runstats_conf["stop"] = stop
    runstats_conf["start"] = start
    runstats_conf["step"] = step
    runstats_conf["n_processes"] = n_processes
    detectors = conf["pad_detectors"]
    # for d in detectors:
    #     d["mask"] = None
    # framegetter = LCLSFrameGetter(
    #     run_number=run_number,
    #     max_events=max_events,
    #     experiment_id=conf["experiment_id"],
    #     pad_detectors=detectors,
    #     cachedir=conf["cachedir"],
    #     postprocessors=None,
    #     photon_wavelength_pv=conf["photon_wavelength_pv"]
    # )

    #remove histogram parameters from dictionary
    del runstats_conf["histogram_params"]

    # mask = reborn.detector.load_pad_masks("geometry/combined_masks.mask")

    #grab all masks to perform binary dilation
    mask = None
    if "mask" in conf["pad_detectors"][0]:
        masks = []
        # mask = reborn.detector.load_pad_masks("geometry/jungfrau_edges_belowstd-outer_abovestd-inner.mask")
        for mask_fn in conf["pad_detectors"][0]["mask"]:
            print(mask_fn)
            mask = reborn.detector.load_pad_masks(mask_fn)
            #now loop through each panel of each mask and perform binary erosion
            print(np.sum(mask))
            for i in range(len(mask)):
                #expand each pixel in the mask by the cross shape because many pixels were
                #causing bleeding to neighboring pixels above, below, to the sides of central masked pixel
                mask[i] = ndimage.binary_erosion(mask[i])
            print(np.sum(mask))
            masks.append(mask)

        #multiply all masks together to make one mask
        new_mask = masks[0].copy()
        for i in range(len(masks)):
            #loop through each panel
            for j in range(len(new_mask)):
                new_mask[j] *= masks[i][j]

        mask = new_mask
        print("Total mask: %d" % np.sum(mask))

    # reborn.detector.save_pad_masks("geometry/combined_masks.mask", mask)
    # print("Diode dict")
    # print(conf["diode_dict"])
    # print("Photon wavelength")
    # print(conf["photon_wavelength_pv"])

    print("setting up framegetter:")
    framegetter = MyLCLSFrameGetter(
        run_number=run_number,
        max_events=max_events,
        experiment_id=conf["experiment_id"],
        pad_detectors=detectors,
        cachedir=conf["cachedir"],
        postprocessors=None,
        photon_wavelength_pv=conf["photon_wavelength_pv"],
        #other_detectors_pv=["Acqiris"],
        mask=mask,
        # wave8_dict=conf["diode_dict"]
    )
    print("finished framegetter set up")

    #ParallelRadialProfiler does not appear to retrieve the geometry, beam, masks from
    #the framegetter (though it probably should). Retrieve and set them explicitly here:
    df = framegetter.get_frame(0)
    geom = df.get_pad_geometry()
    beam = df.get_beam()
    # # mask = df.get_mask_flat()
    # mask = df.get_mask_list()

    q = geom.q_mags(beam)

    # pv = PADView(frame_getter=framegetter, mask=mask)
    # pv.start()

    print("setting up radialprofiler...")
    # profiler = analysis.saxs.RadialProfiler(
    #     pad_geometry = geom,
    #     beam = beam,
    #     mask = mask,
    # )
    # profiler = analysis.saxs.ParallelRadialProfiler(
    #     framegetter=framegetter,
    #     pad_geometry = geom,
    #     beam = beam,
    #     mask = mask,
    #     other_detectors_pv=["Acqiris"],
    #     clear_checkpoints=True,
    #     reduce_from_checkpoints=False,
    #     **runstats_conf
    # )
    profiler = MyParallelRadialProfiler(
        framegetter=framegetter,
        pad_geometry = geom,
        beam = beam,
        mask = mask,
        #other_detectors_pv=["Acqiris"],
        clear_checkpoints=True,
        reduce_from_checkpoints=False,
        **runstats_conf
    )

    print("radialprofiler setup.")
    print("processing frames...")
    profiler.process_frames()
    print("frames processed")
    radials_dict = profiler.to_dict()
    if os.path.isfile(RUNNING_file):
        os.remove(RUNNING_file)
    return radials_dict

if __name__ == "__main__":
    # TODO: rename post_sample_intensity to dg2?
    # TODO: waveform -> "digitizer", scalar field from __
    # TODO: include upstream diodes
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=int, default=None, help="Run number")
    parser.add_argument("--start", type=int, default=0, help="start frame number")
    parser.add_argument("--stop",type=int, default=None, help="stop frame number")
    parser.add_argument("--step",type=int, default=1, help="Every Nth frame")
    parser.add_argument("-j", "--n_processes", type=int, default=12, help="Number of parallel processes")
    parser.add_argument("-d", "--detector", type=str, default='jungfrau', help="Detector to analyze (default=jungfrau, can also be epix.)")
    args = parser.parse_args()

    conf = config.get_config(args.run, args.detector)

    # output = conf["hdf5_directory"] + conf["experiment_id"] + "_r" + str(args.run) + "_" + args.detector + ".h5"
    output = f'{conf["hdf5_directory"]}{conf["experiment_id"]}_r{args.run:04d}_{args.detector}_step{args.step}.h5'

    print(output)

    print(f"Fetching radials...")
    radials_dict = get_radials(
        run_number=args.run, n_processes=args.n_processes,
        start=args.start, stop=args.stop,step=args.step,
        detector=args.detector
    )
    print(radials_dict.keys())
    print(np.sum(radials_dict['mask']))

    print(f"Fetching waveforms and post-sample-intensities")
    exp_id = "cxi101672626"
    run = str(args.run)

    re_fid = np.array(radials_dict["frame_id"])
    seconds = re_fid[:,0]
    nanoseconds = re_fid[:,1]
    fiducials = re_fid[:,2]

    nevents = len(seconds)

    # waveforms = np.zeros((nevents, 4, 1000)) #waveform array is (4,1000) shape
    # wftimes = np.zeros((nevents, 4, 1000))

    #create arrays for DG2
    dg2 = np.zeros((nevents))
    digi = np.zeros((nevents))
    # digi_times = np.zeros((nevents, 32000))
    fee = np.zeros((nevents,6))

    j = 0

    #jumping to events by timestamp seemed slow, maybe because of using idx.
    #Instead, lets just go through events in fastest order using smd,
    #then match up the timestamp using numpy
    #first, make a dictionary of reborn frame ids:
    #
    #
    # NOTE: HH REWORKING
    # NOTE: CURRENTLY DG2 ONLY FOR SPEED
    re_fid_dict = {tuple(value): idx for idx, value in enumerate(re_fid)}
    ds = psana.DataSource(f"exp={exp_id}:run={args.run}:smd")
    dg2_det = psana.Detector("CXI-DG2-BMMON")
    # digi_det = psana.Detector("qadc1") #
    # fee_det = psana.Detector("FEEGasDetEnergy") # event code

    for nevent,evt in enumerate(ds.events()):
        evtId = evt.get(psana.EventId)
        sec = evtId.time()[0]
        ns = evtId.time()[1]
        fid = evtId.fiducials()
        #grab the reborn index associated with that timestamp
        if tuple((sec,ns,fid)) in re_fid_dict:
            idx = re_fid_dict[tuple((sec,ns,fid))]
            sys.stdout.write('\r%d/%d   %d'%(j,nevents, idx))
            sys.stdout.flush()
            try:
                dg2[idx] = dg2_det.get(evt).TotalIntensity()
            except:
                dg2[idx] = -1
            # try:
            #     digi[idx] = digi_det.raw(evt)[0][4800:5500].mean()
            # except:
            #     digi[idx] = -1
            # try:
            #     fee[idx,0] = fee_det.get(evt).f_11_ENRC()
            #     fee[idx,1] = fee_det.get(evt).f_12_ENRC()
            #     fee[idx,2] = fee_det.get(evt).f_21_ENRC()
            #     fee[idx,3] = fee_det.get(evt).f_22_ENRC()
            #     fee[idx,4] = fee_det.get(evt).f_63_ENRC()
            #     fee[idx,5] = fee_det.get(evt).f_64_ENRC()
            # except:
            #     fee[idx,0] = -1
            #     fee[idx,1] = -1
            #     fee[idx,2] = -1
            #     fee[idx,3] = -1
            #     fee[idx,4] = -1
                # fee[idx,5] = -1
            j+=1
            # if j>100: break

    skip_list = ["sum","wsum","sum2"]
    print(f"Writing h5 file...")
    h5 = h5py.File(output, 'w')
    for key in radials_dict.keys():
        if key not in skip_list:
            print(key)
            try:
                h5.create_dataset(key, data = radials_dict[key])
            except:
                pass

    # waveform and post_sample intensity fields to add
    # Acqiris - failed attempt, deleted in latest version?
    h5.create_dataset('/dg2', data=dg2)
    # h5.create_dataset('/digitizer', data=digi)
    # h5.create_dataset('/digi_wftimes', data=digi_times)
    # h5.create_dataset('/FEE', data=fee)
    h5.close()

    # print(radials_dict["sum"])
    #
    # pw = pg.plot(title="Radials")
    # pw.addLegend()
    # c = ['r', 'g', 'b', 'c', 'm', 'y', 'w']
    # for i in range(radials_dict['n_frames']):
    #     pw.plot(radials_dict['q_bins'], radials_dict['mean'][i], name=i, pen=c[i%len(c)])
    # #putting this input() line here keeps the plot open until you press enter on the terminal,
    # #otherwise the pyqtgraph plot will immediately close as the script exits.
    # #can also just put python -i script.py at the command line to enter interpreter state instead.
    # input()
