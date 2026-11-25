"""Highway environment setup and observation helpers."""

from typing import Any

import gymnasium as gym
import numpy as np
from highway_env import utils
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.controller import ControlledVehicle

from .render_viewer import MultiAgentEnvViewer
from .reward_funcs import get_reward_function


class TeamHighwayEnv(HighwayEnv):
    """Highway driving with one reward per controlled vehicle and a shared episode outcome."""

    @classmethod
    def default_config(cls) -> dict:
        """HighwayEnv's config plus the name of the reward function (see marl.reward_funcs)."""
        config = super().default_config()
        config["reward_function"] = "mean"
        return config

    def configure(self, config: dict) -> None:
        """HighwayEnv.configure, then build the configured reward function (fails early on a bad name)."""
        super().configure(config)
        self.reward_function = get_reward_function(self.config["reward_function"])

    def render(self) -> np.ndarray | None:
        """HighwayEnv.render, with the viewer that outlines attacked vehicles."""
        if self.viewer is None and self.render_mode is not None:
            self.viewer = MultiAgentEnvViewer(self)
        return super().render()

    def set_attacked_agents(self, agent_indices: list[int]) -> None:
        """Set the agents whose vehicles should be highlighted during rendering."""
        self.attacked_agents = set(agent_indices)
        for agent_idx, vehicle in enumerate(self.controlled_vehicles):
            vehicle.attacked = agent_idx in self.attacked_agents

    def step(self, action) -> tuple:
        """HighwayEnv.step, but the reward is one value per controlled vehicle."""
        obs, _reward, terminated, truncated, info = super().step(action)
        return obs, self._last_vehicle_rewards, terminated, truncated, info

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

def make_env(
    n_agents: int = 3,
    render_mode=None,
    name: str = "highway",
    reward_function: str = "mean",
    render_agents: int = 1,
) -> gym.Env:
    if name == "intersection":
        return make_intersection_env(n_agents, render_mode)
    if name == "highway":
        return make_highway_env(n_agents, render_mode, reward_function, render_agents)
    raise ValueError(f"Unknown environment: {name}")


def make_intersection_env(n_agents: int = 3, render_mode=None) -> gym.Env:
    return gym.make(
        "intersection-multi-agent-v2",
        config={"controlled_vehicles": n_agents},
        render_mode=render_mode,
    )


def make_highway_env(
    n_agents: int = 3,
    render_mode=None,
    reward_function: str = "mean",
    render_agents: int = 1,
) -> gym.Env:
    config = {
        "controlled_vehicles": n_agents,
        "reward_function": reward_function,
        "render_agents": render_agents,
        "screen_width": 900,
        "screen_height": 450,
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
    return gym.make("team-highway-v0", config=config, render_mode=render_mode)


def flatten_obs(obs) -> np.ndarray:
    """Convert one observation per agent to a float32 (agents, features) array."""
    return np.stack(
        [np.asarray(agent_obs, dtype=np.float32).reshape(-1) for agent_obs in obs]
    )


def any_vehicle_crashed(env: gym.Env) -> bool:
    unwrapped: Any = env.unwrapped  # highway-env specific attributes are unknown to the type checker
    return any(vehicle.crashed for vehicle in unwrapped.controlled_vehicles)


def build_actor_obs(obs_flat: np.ndarray, n_agents: int) -> np.ndarray:
    """Append one-hot agent IDs to the per-agent observations."""
    agent_ids = np.eye(n_agents, dtype=np.float32)
    return np.concatenate([obs_flat, agent_ids], axis=-1)
