"""Networks that need auto_LiRPA. Kept out of marl.models so MAPPO does not depend on it."""

import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule
from torch.distributions import Independent, Normal

from ..models import mlp


class GaussianActor(nn.Module):
    """Gaussian policy whose mean network is a BoundedModule, used by every WocaR component."""

    def __init__(
        self, in_dim: int, action_dim: int, device: torch.device, hidden: int = 64
    ) -> None:
        super().__init__()
        self.mean_network: BoundedModule = BoundedModule(
            mlp(in_dim, action_dim, hidden, out_gain=0.01),
            torch.zeros(1, in_dim, device=device),  # dummy input, only used to trace the graph
            device=device.type,
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def dist(self, x: torch.Tensor) -> Independent:
        return Independent(Normal(self.mean_network(x), self.log_std.exp()), 1)
