"""Highway environment with one reward per controlled vehicle and a shared episode outcome."""

from typing import Any

import gymnasium as gym
import numpy as np
from highway_env import utils
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.controller import ControlledVehicle

from .rewards import get_reward_function


# A highway episode ends as soon as any controlled vehicle crashes, so a
# teammate's crash already ends the episode for everyone and the mixed term
# mostly shapes the driving reward: a larger weight keeps it visible.
DEFAULT_HIGHWAY_TEAM_WEIGHT: float = 0.25


class TeamHighwayEnv(HighwayEnv):
    """Highway driving with one reward per controlled vehicle and a shared episode outcome."""

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """HighwayEnv's config plus the reward function name and its team weight."""
        config: dict[str, Any] = super().default_config()
        config["reward_function"] = "team"
        config["team_reward_weight"] = DEFAULT_HIGHWAY_TEAM_WEIGHT
        config["render_agents"] = 1  # camera views in the window, one per controlled car
        return config

    def configure(self, config: dict) -> None:
        """HighwayEnv.configure, then build the configured reward function (fails early on a bad name)."""
        super().configure(config)
        self.reward_function = get_reward_function(self.config["reward_function"])

    def step(self, action) -> tuple:
        """HighwayEnv.step, but the reward is one value per controlled vehicle."""
        obs, _reward, terminated, truncated, info = super().step(action)
        return obs, self._last_vehicle_rewards, terminated, truncated, info

    def render(self):
        """HighwayEnv.render; in human mode drawn by the multi-agent viewer that highlights attacked cars.

        Other modes (e.g. rgb_array) keep the stock viewer, which returns a real frame.
        """
        if self.render_mode == "human" and self.viewer is None:
            from .highway_viewer import MultiAgentEnvViewer  # pygame only when rendering

            self.viewer = MultiAgentEnvViewer(self)
        return super().render()

    def set_attacked_agents(self, agent_indices: list[int]) -> None:
        """Set the agents whose vehicles should be highlighted during rendering."""
        self.attacked_agents = set(agent_indices)

        # Mark each controlled vehicle according to whether its agent is attacked.
        for agent_idx, vehicle in enumerate(self.controlled_vehicles):
            vehicle.attacked = agent_idx in self.attacked_agents

    def _reward(self, action) -> float:
        """Run the reward function once for this step, store the per-vehicle rewards and return their mean."""
        self._last_vehicle_rewards = self.reward_function(self)
        return float(np.mean(self._last_vehicle_rewards))

    def _rewards(self, action) -> dict[str, float]:
        """Average each reward term over all controlled vehicles."""
        agents_rewards = [
            self._agent_rewards(vehicle) for vehicle in self.controlled_vehicles
        ]
        return {
            name: float(
                np.mean([agent_rewards[name] for agent_rewards in agents_rewards])
            )
            for name in agents_rewards[0]
        }

    def _agent_reward(self, vehicle) -> float:
        """HighwayEnv._reward, applied to the given vehicle."""
        rewards = self._agent_rewards(vehicle)
        reward = sum(
            self.config.get(name, 0) * value for name, value in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        return reward * rewards["on_road_reward"]

    def _agent_rewards(self, vehicle) -> dict[str, float]:
        """HighwayEnv._rewards, applied to the given vehicle."""
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = (
            vehicle.target_lane_index[2]
            if isinstance(vehicle, ControlledVehicle)
            else vehicle.lane_index[2]
        )
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(vehicle.on_road),
        }

    def raw_vehicle_rewards(self) -> tuple[float, ...]:
        """The stock per-vehicle driving rewards, before the reward function mixes them."""
        return tuple(
            float(self._agent_reward(vehicle)) for vehicle in self.controlled_vehicles
        )

    def _is_terminated(self) -> bool:
        """The episode is over if any controlled vehicle crashed or, if configured, left the road."""
        return any(
            vehicle.crashed or (self.config["offroad_terminal"] and not vehicle.on_road)
            for vehicle in self.controlled_vehicles
        )

    def _info(self, obs, action=None) -> dict:
        """Report team values under the standard keys and per-vehicle values under agents_*."""
        vehicles = self.controlled_vehicles
        info = super()._info(obs, action)
        info["speed"] = float(np.mean([vehicle.speed for vehicle in vehicles]))
        info["crashed"] = any(vehicle.crashed for vehicle in vehicles)
        info["agents_speed"] = tuple(float(vehicle.speed) for vehicle in vehicles)
        info["agents_crashed"] = tuple(vehicle.crashed for vehicle in vehicles)
        info["agents_rewards"] = tuple(
            float(self._agent_reward(vehicle)) for vehicle in vehicles
        )
        return info


gym.register(id="team-highway-v0", entry_point=TeamHighwayEnv)


def make_highway_env(
    n_agents: int = 3,
    render_mode: str | None = None,
    reward_function: str = "team",
    team_weight: float | None = None,
    render_agents: int | None = None,
) -> gym.Env:
    config: dict[str, Any] = {
        "controlled_vehicles": n_agents,
        "reward_function": reward_function,
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
            },
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {"type": "DiscreteMetaAction"},
        },
        "lanes_count": 4,
        "vehicles_count": 30,
        "duration": 40,
        "initial_lane_id": None,
        "policy_frequency": 1,
        "simulation_frequency": 15,
    }
    if team_weight is not None:
        config["team_reward_weight"] = team_weight
    if render_agents is not None:
        config["render_agents"] = render_agents
    return gym.make("team-highway-v0", config=config, render_mode=render_mode)
