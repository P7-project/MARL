"""Checkpoint output path helpers."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def checkpoint_output_path(path: str | Path) -> Path:
    """Create parent directories and put bare filenames under ``checkpoints/``."""
    output_path = Path(path)
    if not output_path.is_absolute() and output_path.parent == Path("."):
        output_path = PROJECT_ROOT / "checkpoints" / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
