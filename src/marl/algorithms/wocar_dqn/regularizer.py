"""The SA-DQN state regularizer L_reg on Q_r."""

from __future__ import annotations

import torch
from torch import Tensor


class SARegularizer:
    """
        SA-DQN regularizer on Q_r.

        Math: L_reg(s) = max{ -min_{a != a*} LB[Q(a*) - Q(a)], -c },  a* = argmax_a Q_r(s, a).
        Zero gradient once the clean action is certified to stay best by margin c.
    """

    def __init__(self, margin: float) -> None:
        self._margin = margin

    def per_state_loss(self, margin_lower_bounds: Tensor) -> Tensor:
        """max(-min_a LB[Q(a*) - Q(a)], -c), the official hinge."""
        worst_margin: Tensor = margin_lower_bounds.min(dim=1).values
        return torch.clamp(-worst_margin, min=-self._margin)
