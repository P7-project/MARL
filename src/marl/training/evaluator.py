"""Evaluation of saved policies. Picks the player from the checkpoint's algorithm name."""

from argparse import Namespace
from functools import partial
from typing import Callable

import torch

from ..rendering import PolicyPlayer, evaluate_in_windows

DEFAULT_ALGORITHM: str = "mappo"  # for checkpoints saved before the field existed


def evaluate(args: Namespace) -> None:
    evaluate_in_windows(
        partial(_build_player, args.eval), args.eval_episodes, args.eval_windows
    )


def _build_player(checkpoint_path: str) -> PolicyPlayer:
    """Top-level so it pickles under spawn. Runs in each child process."""
    checkpoint: dict = torch.load(checkpoint_path, map_location="cpu")
    algorithm: str = checkpoint.get("algorithm", DEFAULT_ALGORITHM)
    return _builder_for(algorithm)(checkpoint)


def _builder_for(algorithm: str) -> Callable[[dict], PolicyPlayer]:
    # Imported lazily, so evaluating one algorithm doesn't import the others' dependencies.
    if algorithm == "mappo":
        from .trainer import build_mappo_player
        return build_mappo_player
    if algorithm == "wocar_dqn":
        from ..woca_r_rl.dqn import build_dqn_player
        return build_dqn_player
    if algorithm == "wocar_ppo":
        from ..woca_r_rl.ppo import build_ppo_player
        return build_ppo_player
    raise ValueError(f"Unknown algorithm in checkpoint: {algorithm}")