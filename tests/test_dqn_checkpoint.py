import subprocess
import sys
from pathlib import Path

import torch

ROOT: Path = Path(__file__).resolve().parents[1]


def test_checkpoint_trained_with_config_loads_with_default_torch_load(tmp_path) -> None:
    save: Path = tmp_path / "model.pt"
    command: list[str] = [
        sys.executable, str(ROOT / "scripts" / "train.py"),
        "--config", str(ROOT / "configs" / "wocar_dqn_highway.json"),
        "--total-steps", "40", "--warmup", "20", "--batch-size", "8",
        "--logdir", str(tmp_path / "runs"), "--run-name", "run", "--save", str(save),
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    checkpoint: dict = torch.load(save)  # weights_only by default
    assert checkpoint["args"]["config"] == str(ROOT / "configs" / "wocar_dqn_highway.json")
    assert checkpoint["args"]["total_steps"] == 40
