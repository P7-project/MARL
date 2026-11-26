"""The interface between a trained policy and the processes that play it (live view, evaluation)."""

from abc import ABC, abstractmethod
from typing import Callable, NamedTuple

from torch import Tensor


Weights = dict[str, Tensor]


class StepResult(NamedTuple):
    finished: bool
    reward: float
    crashed: bool | None = None  # None when the env has no crash notion


class PolicyPlayer(ABC):
    """What a trainer implements. Built inside the render process, so everything it holds
    (env, networks) lives there. Keep its constructor arguments plain and picklable."""

    @abstractmethod
    def load_weights(self, weights: Weights) -> None:
        """Copy the newest weights into the player's policy."""

    @abstractmethod
    def start_episode(self) -> None:
        """Reset the env."""

    @abstractmethod
    def step(self) -> StepResult:
        """Act once, render one frame, and return the result."""

    @abstractmethod
    def close(self) -> None:
        """Release the env."""


PlayerFactory = Callable[[], PolicyPlayer]  # must be picklable: a top-level class or a functools.partial of one
