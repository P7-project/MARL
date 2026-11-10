"""Neural network modules used by the MAPPO baseline."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, Normal
from auto_LiRPA import BoundedModule


def layer_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def mlp(in_dim: int, out_dim: int, hidden: int, out_gain: float) -> nn.Sequential:
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden, hidden)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden, out_dim), gain=out_gain),
    )


class Actor(nn.Module):
    """Shared policy mapping each agent observation to categorical action logits."""

    def __init__(self, in_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = mlp(in_dim, n_actions, hidden, out_gain=0.01)

    def dist(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(x))


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


class Critic(nn.Module):
    """Centralized value function mapping the joint observation to V(s)."""

    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(state_dim, 1, hidden, out_gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
