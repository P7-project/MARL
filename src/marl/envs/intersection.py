"""Intersection environment with one reward per controlled vehicle and per-agent termination."""

from typing import Any

import gymnasium as gym
import numpy as np
from highway_env.envs.intersection_env import ConnectedLaneMultiAgentIntersectionEnv

from .rewards import get_reward_function


# An intersection episode ends per agent, so a crashed teammate's negative
# reward is mixed into the survivors again on every remaining step and those
# steps accumulate: a smaller per-step weight already makes a crash costly.
DEFAULT_INTERSECTION_TEAM_WEIGHT: float = 0.1


class TeamIntersectionEnv(ConnectedLaneMultiAgentIntersectionEnv):
    """Intersection driving with one reward per controlled vehicle and per-agent termination.

    ``step`` replaces ``MultiAgentWrapper`` (which ``intersection-multi-agent-v2``
    uses) on purpose: it returns the per-agent rewards computed by the configured
    reward function together with ``info["agents_terminated"]``, so the reward can
    come from ``marl.envs.rewards`` instead of the stock cooperative mean.
    """

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """V2's config plus the reward function name and its team weight."""
        config: dict[str, Any] = super().default_config()
        config["reward_function"] = "team"
        config["team_reward_weight"] = DEFAULT_INTERSECTION_TEAM_WEIGHT
        return config

    def configure(self, config: dict) -> None:
        """Configure the env, build the reward function and reset the per-step caches."""
        super().configure(config)
        self.reward_function = get_reward_function(self.config["reward_function"])
        self._last_vehicle_rewards: tuple[float, ...] = ()
        self._last_action: Any = None

    def step(self, action) -> tuple:
        """Stock step, with per-agent rewards and per-agent termination instead of the team mean."""
        obs, _reward, _terminated, truncated, info = super().step(action)
        return obs, self._last_vehicle_rewards, info["agents_terminated"], truncated, info

    def _reward(self, action) -> float:
        """Run the reward function once for this step, store the per-vehicle rewards and return their mean."""
        self._last_action = action
        self._last_vehicle_rewards = self.reward_function(self)
        return float(np.mean(self._last_vehicle_rewards))

    def raw_vehicle_rewards(self) -> tuple[float, ...]:
        """The stock IntersectionEnv per-vehicle rewards for the action being rewarded."""
        return tuple(
            float(self._agent_reward(self._last_action, vehicle))
            for vehicle in self.controlled_vehicles
        )


gym.register(id="team-intersection-v0", entry_point=TeamIntersectionEnv)


def make_intersection_env(
    n_agents: int = 3,
    render_mode: str | None = None,
    reward_function: str = "team",
    team_weight: float | None = None,
) -> gym.Env:
    config: dict[str, Any] = {
        "controlled_vehicles": n_agents,
        "reward_function": reward_function,
    }
    if team_weight is not None:
        config["team_reward_weight"] = team_weight
    return gym.make("team-intersection-v0", config=config, render_mode=render_mode)
