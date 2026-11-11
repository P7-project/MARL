"""Ramp schedule shared by WocaR-PPO and WocaR-DQN."""

from dataclasses import dataclass
from math import exp
from typing import Literal


@dataclass(frozen=True)
class RampSchedule:
    """0 until start_step, then ramps to final_value over ramp_steps, then stays there."""

    start_step: int
    ramp_steps: int
    final_value: float
    shape: Literal["linear", "exponential"] = "linear"
    growth: float = 5.0  # steepness of the exponential shape

    def __post_init__(self) -> None:
        if self.ramp_steps <= 0:
            raise ValueError("ramp_steps must be positive")

    def value(self, step: int) -> float:
        if step <= self.start_step:
            return 0.0
        progress: float = min((step - self.start_step) / self.ramp_steps, 1.0)
        if self.shape == "linear":
            return self.final_value * progress
        normalizer: float = exp(self.growth) - 1.0
        return self.final_value * (exp(self.growth * progress) - 1.0) / normalizer
