import sys
import numpy as np
from scipy import ndimage
import reborn
from reborn import detector
import config

run_number = int(sys.argv[1])

default_config = config.default_config()
# conf = config.get_config(run_number, "jungfrau")
conf = config.get_config(run_number, "epix")

# geom = detector.load_pad_geometry_list(file_name)
geom = conf["pad_detectors"][0]["geometry"]

#grab all masks to perform the binary dilation we want
# masks = []
mask = reborn.detector.load_pad_masks("geometry/epix_run7_all.mask")
# for mask_fn in conf["pad_detectors"][0]["mask"]:
    # print(mask_fn)
    # mask = detector.load_pad_masks(mask_fn)
    # #now loop through each panel of each mask and perform binary erosion
    # print(np.sum(mask))
    # for i in range(len(mask)):
        # #expand each pixel in the mask by the cross shape because many pixels were
        # #causing bleeding to neighboring pixels above, below, to the sides of central masked pixel
        # mask[i] = ndimage.binary_erosion(mask[i])
    # print(np.sum(mask))
    # masks.append(mask)
#
# #multiply all masks together to make one mask
# new_mask = masks[0].copy()
# for i in range(len(masks)):
    # #loop through each panel
    # for j in range(len(new_mask)):
        # new_mask[j] *= masks[i][j]
#
# # mask = new_mask
print("Total mask: %d" % np.sum(mask))

om_filename = f'masks4OM/r{run_number:04d}_combined_mask.h5'
detector.save_mask_as_om_h5(mask, geom, om_filename)
