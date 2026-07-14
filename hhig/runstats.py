#!/usr/bin/env python
#
import logging
import os
import threading
import numpy as np
import pyqtgraph as pg
from scipy.signal import find_peaks, peak_prominences

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None
from reborn import detector
from reborn.dataframe import DataFrame
from reborn.external.pyqtgraph import imview
from reborn.fileio import misc
from reborn.fileio.getters import ListFrameGetter
import time
import matplotlib.pyplot as plt
from pathlib import Path

#
import argparse
import joblib
from reborn.external.pyqtgraph import imview
from reborn.external.lcls import LCLSFrameGetter
from reborn import analysis
from reborn.analysis.parallel import ParallelAnalyzer
import config
# import config_noMasks as config
import reborn
import os
from scipy import ndimage

from reborn.analysis import runstats
from reborn.analysis.runstats import PixelHistogram

logger = logging.getLogger(__name__)

default_config = config.default_config()
memory = joblib.Memory(default_config["joblib_directory"])

class MyParallelPADStats(ParallelAnalyzer):
    r"""
    Parallelized class for giving different runstats
    """
    def __init__(self, framegetter=None, histogram_params=None, **kwargs):
        r"""Gather PAD statistics for a dataset.

        Given a |FrameGetter| subclass instance, fetch the mean intensity, mean squared intensity, minimum,
        and maximum intensities, and optionally a pixel-by-pixel intensity histogram.  The function can run on multiple
        processors via the joblib library.  Logfiles and checkpoint files are created.

        The return of this function is a dictionary with the following keys:

        * 'sum': Sum of PAD intensities
        * 'dataset_id': A unique identifier for the data set (e.g. 'run0154')
        * 'pad_geometry': PADGeometryList
        * 'mask': Bad pixel mask
        * 'n_frames': Number of valid frames contributing to the statistics
        * 'sum': Sum of PAD intensities
        * 'sum2': Sum of squared PAD intensities
        * 'min': Pixel-by-pixel minimum of PAD intensities
        * 'max': Pixel-by-pixel maximum of PAD intensities
        * 'counts': Sum of the PAD mask
        * 'beam': Beam info
        * 'start': Frame at which processing started (global framegetter index)
        * 'stop': Frame at which processing stopped (global framegetter index)
        * 'step': Step size between frames (helpful to speed up processing by sub-sampling)
        * 'wavelengths': An array of wavelengths
        * 'histogram_params': Dictionary with histogram parameters
        * 'histogram': Pixel-by-pixel intensity histogram (MxN array, with M the number of pixels)

        There is a corresponding view_padstats function to view the results in this dictionary.

        padstats needs a configuration dictionary with the following contents:

        * 'log_file': Base path/name for status logging.  Set to None to skip logging.
        * 'checkpoint_file': Base path/name for saving check points.  Set to None to skip checkpoints.
        * 'checkpoint_interval': How many frames between checkpoints.
        * 'message_prefix': Prefix added to all logging messages.  For example: "Run 35" might be helpful
        * 'debug': Set to True for more logging messages.
        * 'reduce_from_checkpoints': True by default, this indicates that data produced by multiple processors should be
          compiled by loading the checkpoints from disk.  Without this, you might have memory problems.  (The need for this
          is due to the joblib package; normally the reduce functions from MPI would be used to avoid hitting the disk.)
        * 'histogram_params': If not None, triggers production of a pixel-by-pixel histogram.  This is a dictionary with
          the following entries: dict(bin_min=-30, bin_max=100, n_bins=100, zero_photon_peak=0, one_photon_peak=30)

        Arguments:
            framegetter (|FrameGetter|): A FrameGetter subclass.  If running in parallel, you should instead pass a
                                         dictionary with keys 'framegetter' (with reference to FrameGetter subclass,
                                         not an actual class instance) and 'kwargs' containing a dictionary of keyword
                                         arguments needed to create a class instance.
            histogram_params (dict):
            start (int): Which frame to start with.
            stop (int): Which frame to stop at.
            step (int): Step size between frames (default 1).
            n_processes (int): How many processes to run in parallel (if parallel=True).
            **kwargs: Any additional key-word arguments you would like to pass to the base class.
                      See: ..:py:class::`~reborn.analysis.parallel.ParallelAnalyzer`

        Returns: dict"""
        super().__init__(framegetter=framegetter, **kwargs)
        self.kwargs["histogram_params"] = histogram_params
        self.logger.debug(f"histogram_params: {histogram_params}")
        self.analyzer_name = "ParallelPADStats"
        self.initialized = False
        # Data for pixel-by-pixel histograms
        self.histogrammer = None  # Instance of |PixelHistogram|
        self.histogram_params = histogram_params  # Configs for |PixelHistogram|
        # Simple pass-through data for convenience
        self.dataset_id = None  # Often something like "r0045" or similar
        self.pad_geometry = None  # First PadGeometryList found in the FrameGetter
        self.beam = None  # Will have the median wavelength
        self.mask = None  # First mask found in the FrameGetter
        # Cumulative pixel-by-pixel data
        # self.min_pad = None  # Array of minimum intensities
        # self.max_pad = None  # Array of maximum intensities
        self.sum_pad = None  # Sum of all intensities
        # self.sum_pad2 = None  # Sum of all squared intensities
        self.counts = None  # Counts
        # Shot-by-shot data
        self.wavelengths = None  # Array of all wavelengths
        self.percentiles = None  # Shot-by-shot percentiles
        # self.sums = None  # Shot-by-shot sums over the whole PAD
        self.maxes = None  # Shot-by-shot sums over the whole PAD
        self.frame_ids = None  # Shot-by-shot frame IDs (not including skipped frames)
        # # Results derived at the end of data gathering
        # self.gain = None  # A crude estimate of detector gain (probably wrong)
        # self.offset = None  # A crude estimate of detector offset (probably wrong)

    def _running_file_path(self):
        for base in (self.kwargs.get("checkpoint_file"), self.kwargs.get("log_file")):
            if not base:
                continue
            p = Path(base)
            if p.is_dir():
                run_dir = p
            else:
                run_dir = p.parent
                if run_dir.name in ("logs", "checkpoints"):
                    run_dir = run_dir.parent
            return run_dir / "RUNNING"
        return None

    def process_frames(self):
        if self.process_id != 0:
            return super().process_frames()
        running_file = self._running_file_path()
        if running_file is None:
            return super().process_frames()
        running_file.parent.mkdir(parents=True, exist_ok=True)
        if running_file.exists():
            self.logger.warning(f"RUNNING file exists: {running_file}. Skipping.")
            return None
        created = False
        start_time = time.time()
        stop_event = threading.Event()

        def _heartbeat():
            while not stop_event.wait(10):
                try:
                    running_file.write_text(
                        f"pid={os.getpid()} start={start_time} updated={time.time()}\n"
                    )
                except Exception:
                    pass

        try:
            running_file.write_text(
                f"pid={os.getpid()} start={start_time} updated={start_time}\n"
            )
            created = True
            thread = threading.Thread(target=_heartbeat, daemon=True)
            thread.start()
            return super().process_frames()
        finally:
            stop_event.set()
            if created:
                try:
                    running_file.unlink()
                except FileNotFoundError:
                    pass

    def add_frame(self, dat):
        tic = time.time()
        if dat is None:
            self.logger.warning(
                f"DataFrame {self.processing_index} is None.  Skipping frame"
            )
            return
        rdat = dat.get_raw_data_flat()
        logger.debug(f"add_frame: get_raw_data_flat: {time.time() - tic} seconds.")
        tic = time.time()
        mask = dat.get_mask_flat().copy()
        logger.debug(f"add_frame: get_mask_flat: {time.time() - tic} seconds.")
        if np.sum(mask) == 0:
            self.logger.warning(
                f"DataFrame {self.processing_index} is fully masked.  Skipping frame"
            )
            return
        tic = time.time()
        idx = np.where(mask > 0)
        rdat = rdat * mask
        logger.debug(f"add_frame: where mask: {time.time() - tic} seconds.")
        if rdat is None:
            self.logger.warning(f"Raw data is None.  Skipping frame.")
            return
        if not self.initialized:
            self.initialize_data(rdat)
        if dat.validate():
            tic = time.time()
            beam = dat.get_beam()
            self.wavelengths[self.processing_index] = beam.wavelength
            if self.beam is None:
                self.beam = beam
            logger.debug(f"add_frame: validate: {time.time() - tic} seconds.")
        else:
            self.logger.warning(
                "DataFrame is invalid.  If it is a dark run this could be due to missing Beam info."
            )
        tic = time.time()
        self.sum_pad += rdat
        # self.sum_pad2 += rdat**2
        # self.min_pad[idx] = np.minimum(self.min_pad[idx], rdat[idx])
        # self.max_pad[idx] = np.maximum(self.max_pad[idx], rdat[idx])
        self.counts += mask
        logger.debug(f"add_frame: operands counts: {time.time() - tic} seconds.")
        tic = time.time()
        # self.sums[self.processing_index] = np.sum(rdat[idx])
        self.maxes[self.processing_index] = np.max(rdat[idx])
        self.percentiles[self.processing_index, :] = np.percentile(
            rdat[idx], [50, 90, 95, 99]
        )
        logger.debug(f"add_frame: operands percentiles: {time.time() - tic} seconds.")
        tic = time.time()
        # self.sums[self.processing_index] = np.sum(rdat[idx])
        self.maxes[self.processing_index] = np.max(rdat[idx])
        self.frame_ids.append(dat.get_frame_id())
        if self.dataset_id is None:
            self.dataset_id = dat.get_dataset_id()
        if self.pad_geometry is None:
            self.pad_geometry = dat.get_pad_geometry()
        if self.mask is None:
            self.mask = dat.get_mask_flat()
        logger.debug(f"add_frame: operands mask: {time.time() - tic} seconds.")
        tic = time.time()
        # self.sums[self.processing_index] = np.sum(rdat[idx])
        self.maxes[self.processing_index] = np.max(rdat[idx])
        # if self.histogrammer is not None:
        #     self.histogrammer.add_frame(rdat, mask=mask)
        # logger.debug(f"add_frame: operands histogram: {time.time() - tic} seconds.")

    def finalize(self):
        r"""Compute gain and offset from histogram, trim arrays."""
        self.logger.info("Finalizing analysis")
        tic = time.time()
        # if self.histogrammer is not None:
        #     self.logger.info("Attempting to get gain and offset from histogram")
        #     self.gain, self.offset = self.histogrammer.gain_and_offset()
        self.wavelengths = self.wavelengths[0 : self.n_processed]
        # self.sums = self.sums[0 : self.n_processed]
        self.maxes = self.maxes[0 : self.n_processed]
        self.percentiles = self.percentiles[0 : self.n_processed, :]
        logger.debug(f"finalize: {time.time()-tic} seconds.")

    # def clear_data(self):
    #     self.wavelengths = None
    #     self.sum_pad = None
    #     self.sum_pad2 = None
    #     self.max_pad = None
    #     self.min_pad = None
    #     self.n_processed = 0
    #     self.histogrammer = None
    #     self.initialized = False

    def initialize_data(self, rdat):
        r"""Allocate memory, initialize counters."""
        self.logger.debug("Initializing arrays")
        tic = time.time()
        s = rdat.size
        self.wavelengths = np.zeros(self.n_chunk)
        self.sum_pad = np.zeros(s)
        # self.sum_pad2 = np.zeros(s)
        self.counts = np.zeros(s)
        # self.max_pad = rdat.copy()
        # self.min_pad = rdat.copy()
        # self.sums = np.zeros(self.n_chunk)
        self.maxes = np.zeros(self.n_chunk)
        self.percentiles = np.zeros([self.n_chunk, 4])
        self.frame_ids = []
        self.n_processed = 0
        # if self.histogram_params is not None:
        #     self.logger.info("Setting up histogram")
        #     if self.histogram_params.get("n_pixels", None) is None:
        #         self.histogram_params["n_pixels"] = self.sum_pad.shape[0]
        #     self.histogrammer = PixelHistogram(**self.histogram_params)
        self.initialized = True
        logger.debug(f"initialize_data: {time.time()-tic} seconds.")

    def to_dict(self):
        tic = time.time()
        stats = dict()
        stats["dataset_id"] = self.dataset_id
        stats["pad_geometry"] = self.pad_geometry
        stats["mask"] = self.mask
        stats["n_frames"] = self.n_processed
        stats["sum"] = self.sum_pad
        # stats["sum2"] = self.sum_pad2
        # stats["min"] = self.min_pad
        # stats["max"] = self.max_pad
        stats["counts"] = self.counts
        stats["beam"] = self.beam
        stats["start"] = self.start
        stats["stop"] = self.stop
        stats["step"] = self.step
        stats["wavelengths"] = self.wavelengths
        # stats["sums"] = self.sums
        stats["maxes"] = self.maxes
        stats["percentiles"] = self.percentiles
        stats["frame_ids"] = self.frame_ids
        # if self.histogrammer is not None:
        #     stats["histogram"] = self.histogrammer.histogram
        #     stats["histogram_params"] = self.histogram_params
        # if self.gain is not None:
        #     stats["gain"] = self.gain
        #     stats["offset"] = self.offset
        logger.debug(f"to_dict: {time.time() - tic} seconds.")
        return stats

    def from_dict(self, stats):
        # self.clear_data()
        tic = time.time()
        # if stats.get("sum") is None:
        #     self.logger.warning("Stats dictionary is empty!")
        #     return
        self.dataset_id = stats["dataset_id"]
        self.pad_geometry = stats["pad_geometry"]
        self.mask = stats["mask"]
        self.n_processed = stats["n_frames"]
        self.sum_pad = stats["sum"]
        # self.sum_pad2 = stats["sum2"]
        # self.min_pad = stats["min"]
        # self.max_pad = stats["max"]
        self.counts = stats.get(
            "counts", np.zeros(stats["sum"].size) + stats["n_frames"]
        )
        self.beam = stats["beam"]
        self.start = stats["start"]
        self.stop = stats["stop"]
        self.step = stats["step"]
        self.wavelengths = stats["wavelengths"]
        self.frame_ids = stats["frame_ids"]
        self.percentiles = stats["percentiles"]
        # self.sums = stats["sums"]
        self.maxes = stats["maxes"]
        # if stats.get("histogram_params"):
        #     self.histogram_params = stats["histogram_params"]
        #     self.histogrammer = PixelHistogram(**stats["histogram_params"])
        #     self.histogrammer.histogram = stats.get("histogram")
        self.initialized = True
        logger.debug(f"from_dict: {time.time()-tic} seconds.")

    def concatenate(self, stats):
        r"""Take the existing data in this class instance and 'add' the date in the incoming
        stats dictionary.  E.g. extend arrays as needed, or sum them.  Also known as a 'reduce' step.
        One can think of 'merging' the data."""
        tic = time.time()
        if not self.initialized:
            self.from_dict(stats)
            self.initialized = True
            return
        if stats["n_frames"] == 0:
            self.logger.debug("No frames to concatentate!")
            return
        self.start = min(self.start, stats["start"])
        self.stop = max(self.stop, stats["stop"])
        self.n_processed += stats["n_frames"]
        self.wavelengths = np.concatenate([self.wavelengths, stats["wavelengths"]])
        if stats["sum"] is None:
            return
        self.sum_pad += stats["sum"]
        # self.sum_pad2 += stats["sum2"]
        # self.min_pad = np.minimum(self.min_pad, stats["min"])
        # self.max_pad = np.minimum(self.max_pad, stats["max"])
        c = stats.get("counts", np.zeros(stats["sum"].size) + stats["n_frames"])
        self.counts += c
        self.frame_ids = self.frame_ids + stats["frame_ids"]
        self.percentiles = np.concatenate([self.percentiles, stats["percentiles"]])
        # self.sums = np.concatenate([self.sums, stats["sums"]])
        self.maxes = np.concatenate([self.maxes, stats["maxes"]])
        # if stats.get("histogram") is not None:
        #     self.histogrammer.histogram += stats["histogram"]
        logger.debug(f"concatenate: {time.time() - tic} seconds.")


@memory.cache(ignore=["n_processes"])
def get_runstats(run_number=1, det="jungfrau", n_processes=1, max_frames=1e7,
                 max_sum=None, min_sum=None, start=0, stop=None, step=1,
                 histogram=False, pixel_threshold=None, maxes=False): # NOTE: HH MADE HIST FALSE
    r"""Fetches some PAD statistics for a run.  See reborn docs."""
    conf = config.get_config(run_number, detector=det, maxes=maxes)
    print(conf['runstats']['log_file'])
    if pixel_threshold is not None:
        print(pixel_threshold)
        pp_suffix = f"_pt_{pixel_threshold}"
        conf['runstats']['checkpoint_file'] =conf['runstats']['checkpoint_file'] + pp_suffix
        conf['runstats']['log_file'] = conf['runstats']['log_file'] + pp_suffix
        print(conf["runstats"]["log_file"])
    log_file = os.path.dirname(conf["runstats"]["log_file"])
    if not os.path.isdir(log_file):
        if not os.path.isdir(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.mkdir(log_file)
    RUNNING_file = log_file + "/RUNNING"
#    if os.path.isfile(RUNNING_file):
#        raise ValueError(f"somebody else is running {run_number}!\n remove file {RUNNING_file} if the job has for sure terminated")
    with open(RUNNING_file, 'w') as f:
        f.close()
    runstats_conf = conf["runstats"]
    if stop is None:
        runstats_conf["stop"] = int(max_frames)
    else:
        runstats_conf["stop"] = stop
    runstats_conf["start"] = start
    runstats_conf["step"] = step
    runstats_conf["n_processes"] = n_processes
    if histogram == False:
        runstats_conf["histogram_params"] = None
    detectors = conf["pad_detectors"]
    for d in detectors:
        d["mask"] = None
    pp = None
    if pixel_threshold is not None:
        def _thresh(self, dat):
            x = dat.get_raw_data_flat()
            x[x < pixel_threshold] = 0
            dat.set_raw_data(x)
            return dat
        pp = [_thresh]
    framegetter = LCLSFrameGetter(
        run_number=run_number,
        max_events=max_frames,
        experiment_id=conf["experiment_id"],
        pad_detectors=detectors,
        cachedir=conf["cachedir"],
        postprocessors=pp,
        photon_wavelength_pv=conf["photon_wavelength_pv"]
    )
    # padstats = runstats.ParallelPADStats(
    #     framegetter=framegetter,  # max_sum=max_sum, min_sum=min_sum,
    #     **runstats_conf
    # )
    if maxes:
        padstats = MyParallelPADStats(
            framegetter=framegetter,  # max_sum=max_sum, min_sum=min_sum,
            **runstats_conf
        )
    else:
        padstats = ParallelPADStats(
            framegetter=framegetter,  # max_sum=max_sum, min_sum=min_sum,
            **runstats_conf
        )
    padstats.process_frames()
    stats = padstats.to_dict()
    if os.path.isfile(RUNNING_file):
        os.remove(RUNNING_file)
    return stats


def combine_runstats(run_numbers, max_frames=1e7):
    r""" Combine runstats from several runs.  Note that it makes no sense to combine some things such
    as the beam and PAD geometry.  As of now, we only combine the pixel histograms for the purpose of
    calibrating the detector across multiple runs."""
    stats = get_runstats(run_number=run_numbers[0])
    stats["histogram"] = 0
    stats["sum"] = 0
    # stats["sum2"] = 0
    stats["counts"] = 0
    stats["n_frames"] = 0
    stats["wavelengths"] = np.zeros(0)
    print('run_numbers', run_numbers)
    for r in run_numbers:
        s = get_runstats(run_number=r, max_frames=max_frames)
        for k in ["histogram", "sum", "sum2", "n_frames", "counts"]:
            stats[k] += s[k]
        for k in ["wavelengths", "percentiles"]:
            stats[k] = np.concatenate([stats[k], s[k]])
        stats["min"] = np.minimum(stats["min"], s["min"])
        stats["max"] = np.maximum(stats["max"], s["max"])
    return stats



def view_runstats(stats=None, geom=None, mask=None, hstgrm=True, **kwargs):
    """Convenience viewer for get_runstats. Accepts same arguments as get_runstats, along with a couple more:

    Arguments:
        geom (PADGeometryList): PAD geometry.
        mask (ndarray): PAD mask.
    """
    for key in stats:
        print(key)

    if stats is None:
        stats = get_runstats(**kwargs)
    if mask is not None:
        stats["mask"] = mask
    if geom is not None:
        stats["pad_geometry"] = geom

    # print(stats)
    geom = stats["pad_geometry"]
    beam = stats["beam"]
    c = config.default_config()
    n_q = c["profiles"]["n_bins"]
    q_range = np.array(c["profiles"]["q_range"])
    if hstgrm:
        hist = runstats.PixelHistogram(**stats["histogram_params"])
        hist.histogram = stats["histogram"].astype(float)
        php = stats["histogram_params"]
        adu_range = (php["bin_min"], php["bin_max"])
        if beam is not None:
            qb = hist.convert_to_q_histogram(
                pad_geometry=geom, n_q_bins=n_q, q_range=q_range, beam=beam, normalized=True
            )
            imv = imview(
                qb,
                ss_lims=q_range / 1e10,
                fs_lims=adu_range,
                ss_label="Q",
                fs_label="ADU",
                hold=False,
            )
            q_bins, median_profile = hist.get_median_profile(
                pad_geometry=geom, n_q_bins=n_q, q_range=q_range, beam=beam
            )
            imv.add_plot(q_bins / 1e10, median_profile)
    # plot = pg.plot(stats["sums"], pen=None, symbol="o", symbolBrush="w")
    # plot.setLabel("bottom", "Frame Number")
    # plot.setLabel("left", "Integrated PAD Intensiy")
    runstats.view_padstats(stats, start=True, histogram=hstgrm)
    '''
    pv = runstats.view_padstats(stats, start=False, histogram=true)
    name = f"results/runstats/r{kwargs.get('run'):04d}/"
    for i in range(pv.frame_getter.n_frames):
        name_ = f"{name}{pv.dataframe.get_frame_id()}.jpg"
        #print('saving screenshot ',name_,)
        pv.save_screenshot(name_)
        pv.show_next_frame()
    #pv.start()
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", type=int, default=154, help="Run number")
    parser.add_argument("-d", "--detector", type=str, default="jungfrau", help="Detector name")
    parser.add_argument("--view", action="store_true", help="View stats")
    parser.add_argument("--start", type=int, default=0, help="start frame number")
    parser.add_argument("--stop",type=int, default=None, help="stop frame number")
    parser.add_argument("--step",type=int, default=1, help="number to skip between frames")
    parser.add_argument("--no_histogram", action="store_true", help="turn off histograms")
    parser.add_argument("-t", "--pixel_threshold", type=float, default=None,
                        help="Threshold runstats.")
    parser.add_argument("--mask",type=str, default=None, nargs="*",
                        help="pick .mask path")
    parser.add_argument(
        "--max_sum",
        type=float,
        default=None,
        help="Maximum sum to include image in mean calculation",
    )
    parser.add_argument("--geom", action="store_true", default=False,
                        help="Use conf geom")
    parser.add_argument(
        "--min_sum",
        type=float,
        default=None,
        help="Minimum sum to include image in mean calculation",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=1e7,
        help="Maximum number of events to process",
    )
    parser.add_argument(
        "-j", "--n_processes", type=int, default=12, help="Number of parallel processes"
    )

    # ** PRINT
    args = parser.parse_args()

    if args.view:
        stats = get_runstats(
            run_number=args.run, det=args.detector, n_processes=args.n_processes,
            max_frames=args.max_events,
            max_sum=args.max_sum, min_sum=args.min_sum,
            start=args.start, stop=args.stop, step=args.step,
            histogram=args.no_histogram,
            pixel_threshold=args.pixel_threshold,
            maxes=False
        )
        for key in stats:
            print(key)
        # Hugh - loading in masks, copied from write_radials_to_h5.py

        #grab all masks to perform binary dilation
        mask_arr = None
        if args.mask:
            for mask_fn in args.mask:
                print(mask_fn)
                mask = reborn.detector.load_pad_masks(mask_fn)
                #now loop through each panel of each mask and perform binary erosion
                print(np.sum(mask))
                fo

        if args.geom:
            conf = config.get_config(args.run, args.detector)
            geometry = conf['pad_detectors'][0]['geometry']
        else:
            geometry = None

        print("Viewing runstats...")
        view_runstats(stats, hstgrm=args.no_histogram, run=args.run, mask=mask_arr,
                      geom=geometry)
    else:
         stats = get_runstats(
            run_number=args.run, det=args.detector, n_processes=args.n_processes,
            max_frames=args.max_events,
            max_sum=args.max_sum, min_sum=args.min_sum,
            start=args.start, stop=args.stop, step=args.step,
            histogram=args.no_histogram,
            pixel_threshold=args.pixel_threshold,
            maxes=True
        )


#    else:
#        print("Saving screenshots...")
#        pv = analysis.runstats.view_padstats(stats, start=False, histogram=args.no_histogram)
#        name = f"results/runstats/r{args.run:04d}/"
#        for i in range(pv.frame_getter.n_frames):
#            name_ = f"{name}{pv.dataframe.get_frame_id()}.jpg"
#            pv.save_screenshot(name_)
#            pv.show_next_frame()
