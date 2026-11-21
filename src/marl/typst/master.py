from parser import generate_lilaq_string
from rich.console import Console
import pathlib
import argparse


DIRECTORY = "./data/"

parser = argparse.ArgumentParser(description="Convert JSON logs to Typst Lilaq plot.")

parser.add_argument("output", type=str, help="Name of output file")
parser.add_argument("--experiments", nargs="+", required=True, help="Name of experiments to include")
parser.add_argument("--metric", type=str, required=True, help="Name of metric to plot on y-axis")
parser.add_argument("--resolution", type=int, default=1, help="Downsample factor")
parser.add_argument("--smooth", type=int, default=1, help="Smoothing factor")
parser.add_argument("--every", type=int, default=None, help="Place a marker every n steps on the x-axis")
parser.add_argument("--xmin", type=float, default=None, help="X minimum")
parser.add_argument("--xmax", type=float, default=None, help="X maximum")
parser.add_argument("--ymin", type=float, default=None, help="Y minimum")
parser.add_argument("--ymax", type=float, default=None, help="Y maximum")

args = parser.parse_args()

lilaq_string = generate_lilaq_string(   
    directory=DIRECTORY,
    experiments=args.experiments,
    metric=args.metric,
    resolution=args.resolution,
    smooth_window=args.smooth,
    every=args.every,
    x_min=args.xmin,
    x_max=args.xmax,
    y_min=args.ymin,
    y_max=args.ymax,
)

output_path = pathlib.Path.joinpath(pathlib.Path(DIRECTORY), args.output + ".txt")
with open(output_path, "w") as f:
    f.write(lilaq_string)

Console().print(f"Created typst plot at: '{output_path}'")