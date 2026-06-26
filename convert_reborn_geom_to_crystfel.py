import argparse
import config
from reborn import detector
from reborn.external import crystfel

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run", type=int, default=154,
                    help="Run number")
parser.add_argument("--template", type=str, default="",
                    help="Path to crystfel geometry file")
args = parser.parse_args()
run_number = args.run

conf = config.get_config(run_number, "epix")
geom = conf['pad_detectors'][0]['geometry']
om_filename = f'masks4OM/r{run_number:04d}_epix_padGeom.geom'

if not args.template:
    crystfel.write_geom_file_from_pad_geometry_list(
        pad_geometry=geom, file_path=om_filename
    )
else:
    crystfel.write_geom_file_from_template(
        pad_geometry=geom, template_file=args.template, out_file=om_filename
    )
