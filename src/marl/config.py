"""Helpers for loading project YAML configuration files."""

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML mapping; relative paths are resolved from the repo root."""
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return config
