"""Reward functions that turn the controlled vehicles' driving into one reward per vehicle."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .env import TeamHighwayEnv


class RewardFunction(ABC):
    """Maps the current env state to one reward per controlled vehicle."""

    @abstractmethod
    def __call__(self, env: "TeamHighwayEnv") -> tuple[float, ...]:
        """Return one reward per controlled vehicle, in controlled-vehicle order."""


class MeanReward(RewardFunction):
    """Every vehicle gets the mean of the per-vehicle driving rewards."""

    def __call__(self, env: "TeamHighwayEnv") -> tuple[float, ...]:
        vehicles = env.controlled_vehicles
        team_reward = float(np.mean([env._agent_reward(vehicle) for vehicle in vehicles]))
        return tuple(team_reward for _ in vehicles)


REWARD_FUNCTIONS: dict[str, type[RewardFunction]] = {"mean": MeanReward}


def get_reward_function(name: str) -> RewardFunction:
    """Build the reward function registered under the given name."""
    if name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function: {name!r}. "
            f"Available: {', '.join(sorted(REWARD_FUNCTIONS))}"
        )
    return REWARD_FUNCTIONS[name]()
