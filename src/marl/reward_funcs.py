"""Reward functions that turn the controlled vehicles' driving into one reward per vehicle."""

from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np
from highway_env.vehicle.kinematics import Vehicle


class TeamEnv(Protocol):
    """What a reward function may read from a multi-agent driving environment."""

    config: dict[str, Any]
    controlled_vehicles: list[Vehicle]

    def raw_vehicle_rewards(self) -> tuple[float, ...]:
        """Return one stock reward per controlled vehicle, in controlled-vehicle order."""


class RewardFunction(ABC):
    """Maps the current env state to one reward per controlled vehicle."""

    @abstractmethod
    def __call__(self, env: TeamEnv) -> tuple[float, ...]:
        """Return one reward per controlled vehicle, in controlled-vehicle order."""


class MeanReward(RewardFunction):
    """Every vehicle gets the mean of the per-vehicle driving rewards."""

    def __call__(self, env: TeamEnv) -> tuple[float, ...]:
        raw_rewards = env.raw_vehicle_rewards()
        team_reward = float(np.mean(raw_rewards))
        return tuple(team_reward for _ in raw_rewards)


class TeamReward(RewardFunction):
    """Each vehicle's own reward mixed with the mean reward of its teammates.

    With one controlled vehicle there is no teammate to average over, so the raw
    reward is returned unchanged.
    """

    def __call__(self, env: TeamEnv) -> tuple[float, ...]:
        raw_rewards = env.raw_vehicle_rewards()
        if len(raw_rewards) == 1:
            return raw_rewards
        weight: float = float(env.config["team_reward_weight"])
        total: float = sum(raw_rewards)
        teammate_count: int = len(raw_rewards) - 1
        mixed_rewards: list[float] = []
        for own_reward in raw_rewards:
            teammate_mean: float = (total - own_reward) / teammate_count
            mixed_rewards.append((1 - weight) * own_reward + weight * teammate_mean)
        return tuple(mixed_rewards)


REWARD_FUNCTIONS: dict[str, type[RewardFunction]] = {
    "mean": MeanReward,
    "team": TeamReward,
}


def get_reward_function(name: str) -> RewardFunction:
    """Build the reward function registered under the given name."""
    if name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function: {name!r}. "
            f"Available: {', '.join(sorted(REWARD_FUNCTIONS))}"
        )
    return REWARD_FUNCTIONS[name]()
