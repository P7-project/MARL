# Typst plotting tool

Turns TensorBoard scalar exports (JSON) into [Lilaq](https://typst.app/universe/package/lilaq)
plot code that can be pasted into the Typst report. It is a standalone helper
for the report, not part of the `marl` package.

- `master.py`: command line entry point.
- `parser.py`: reads the JSON files and builds the Lilaq code.

## Input: TensorBoard JSON exports

Each input file is one scalar of one run, exported from TensorBoard as JSON.
The file is a list of `[wall_time, step, value]` entries:

```json
[[1700000000.0, 0, 1.5], [1700000010.0, 1, 2.0], [1700000020.0, 2, 2.5]]
```

The step is used as the x value and the value as the y value (the wall time is
ignored in the plot).

To export a scalar, start TensorBoard on the log folder
(`uv run tensorboard --logdir runs`, or wherever your run wrote its logs),
open the Scalars tab, tick "Show data download links" in the settings on the
left, and use the download link below the chart to save it as JSON.
This flow is how the TensorBoard Scalars dashboard is known to work; it was not
checked click by click against the installed version (2.21), so the exact
labels may differ slightly.

## Data folder and file names

Put the JSON files in `tools/typst/data/` (gitignored, create it if it does not
exist), or in another folder given with `--data-dir`. Files must be named

```
EXPERIMENT.METRIC.json
```

for example `exp1.reward.json`. `--experiments` and `--metric` select the
files: every experiment name is combined with the metric name. Experiments
without a matching file are skipped silently. The experiment name, capitalised,
becomes the legend label; the metric name, capitalised, becomes the y-axis label
(the x-axis label is always "Episodes").

## Usage

Run from the repository root (it also works from any other directory, because
the data folder and `parser.py` are found relative to the script):

```console
uv run tools/typst/master.py OUTPUT --experiments EXPERIMENT [EXPERIMENT ...] --metric METRIC
```

| Argument | Default | Meaning |
|---|---|---|
| `OUTPUT` | required | Output file name without extension; `.txt` is added. |
| `--experiments` | required | One or more experiment names to plot, one line each. |
| `--metric` | required | Metric name (second part of the file name), used as y-axis label. |
| `--resolution` | `1` | Downsample factor: keep every n-th point (1 = keep all). |
| `--smooth` | `1` | Moving-average window in points, centred (1 = no smoothing). |
| `--every` | none | Place a marker every n points (Lilaq `every`). |
| `--xmin`, `--xmax` | auto | X-axis limits. |
| `--ymin`, `--ymax` | auto | Y-axis limits. |
| `--data-dir` | `tools/typst/data` | Folder with the JSON files; the output is written here too. |

Downsampling is applied before smoothing. Use `--help` to list the arguments.

## Output

The Lilaq code is written to `DATA_DIR/OUTPUT.txt` (by default
`tools/typst/data/OUTPUT.txt`). It starts with the Lilaq import and a
`#figure(...)` with a placeholder caption (`GIVE ME A CAPTION`) to replace.

## Example

With `tools/typst/data/exp1.reward.json` and `tools/typst/data/exp2.reward.json`:

```console
uv run tools/typst/master.py reward_plot --experiments exp1 exp2 --metric reward --smooth 5 --ymin 0
```

This writes `tools/typst/data/reward_plot.txt`, beginning with

```typst
#import "@preview/lilaq:0.5.0" as lq
#figure(
	caption: [GIVE ME A CAPTION],
	lq.diagram(
		legend: (position: bottom + center),
		xlim: (auto, auto),
		xlabel: [Episodes],

		ylim: (0.0, auto),
		ylabel: [Reward],
```
