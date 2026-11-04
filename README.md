# MAPPO

## Install

```bash
uv sync
```

## Train

```bash
uv run scripts/train.py --config configs/highway.yaml --save checkpoints/highway.pt
uv run scripts/train.py --config configs/intersection.yaml --save checkpoints/intersection.pt
```

Add `--render` to show one live window of the shared training environment, with all controlled cars.
Bare `--save` filenames are placed in `checkpoints/`; parent directories are created automatically.

## Evaluate

```bash
uv run scripts/evaluate.py --eval checkpoints/highway.pt --render
uv run scripts/evaluate.py --eval checkpoints/intersection.pt --config configs/intersection.yaml --render
```

Pass the training config when evaluating older checkpoints.
With `--render`, evaluation opens up to 12 windows. Each shows a separate copy of the full multi-agent environment.
