import argparse
from reborn.external import crystfel
from reborn import detector

cspad_crystfel_geom_file = "./masks4OM/epix.geom"
geom = crystfel.geometry_file_to_pad_geometry_list(cspad_crystfel_geom_file)
filename = "epix.json"
detector.save_pad_geometry_list(geom_list=geom, file_name=filename)
