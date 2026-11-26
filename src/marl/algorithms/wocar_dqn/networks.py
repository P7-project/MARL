"""The robust Q-network Q_r and the observation-to-tensor helper, shared by training and the player."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from auto_LiRPA import BoundedModule

from ...common import mlp


def _build_robust_q(input_dim: int, n_actions: int, hidden: int, device: torch.device) -> BoundedModule:
    return BoundedModule(
        mlp(input_dim, n_actions, hidden, out_gain=1.0).to(device),
        torch.zeros(1, input_dim, device=device),
        device=device.type,
    )


def _to_tensor(array: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)
