"""Highway environment setup and observation helpers."""

from typing import Any

import gymnasium as gym
import numpy as np
from highway_env import utils
from highway_env.envs.highway_env import HighwayEnv
from highway_env.envs.intersection_env import ConnectedLaneMultiAgentIntersectionEnv
from highway_env.vehicle.controller import ControlledVehicle

from .reward_funcs import get_reward_function


# A highway episode ends as soon as any controlled vehicle crashes, so a
# teammate's crash already ends the episode for everyone and the mixed term
# mostly shapes the driving reward: a larger weight keeps it visible.
DEFAULT_HIGHWAY_TEAM_WEIGHT: float = 0.25
# An intersection episode ends per agent, so a crashed teammate's negative
# reward is mixed into the survivors again on every remaining step and those
# steps accumulate: a smaller per-step weight already makes a crash costly.
DEFAULT_INTERSECTION_TEAM_WEIGHT: float = 0.1


class TeamHighwayEnv(HighwayEnv):
    """Highway driving with one reward per controlled vehicle.

    By default the episode outcome is shared: any crash ends the episode for every
    vehicle. With ``per_agent_termination`` each vehicle gets its own terminated
    flag and the episode ends once every vehicle is terminated (or at the time limit).
    """

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """HighwayEnv's config plus the reward function name, its team weight and the termination mode."""
        config: dict[str, Any] = super().default_config()
        config["reward_function"] = "team"
        config["team_reward_weight"] = DEFAULT_HIGHWAY_TEAM_WEIGHT
        config["per_agent_termination"] = False
        return config

    def configure(self, config: dict) -> None:
        """HighwayEnv.configure, then build the configured reward function (fails early on a bad name)."""
        super().configure(config)
        self.reward_function = get_reward_function(self.config["reward_function"])

    def step(self, action) -> tuple:
        """HighwayEnv.step, but the reward is one value per controlled vehicle.

        With ``per_agent_termination`` the terminated flag is one bool per controlled
        vehicle, also stored in ``info["agents_terminated"]``.
        """
        obs, _reward, terminated, truncated, info = super().step(action)
        if self.config["per_agent_termination"]:
            terminated = self._agents_terminated()
            info["agents_terminated"] = terminated
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

    def raw_vehicle_rewards(self) -> tuple[float, ...]:
        """The stock per-vehicle driving rewards, before the reward function mixes them."""
        return tuple(
            float(self._agent_reward(vehicle)) for vehicle in self.controlled_vehicles
        )

    def _agents_terminated(self) -> tuple[bool, ...]:
        """One flag per controlled vehicle: it crashed or, if configured, left the road."""
        return tuple(
            bool(vehicle.crashed or (self.config["offroad_terminal"] and not vehicle.on_road))
            for vehicle in self.controlled_vehicles
        )

    def _is_terminated(self) -> bool:
        """The episode is over if any controlled vehicle is terminated, or all of them with per_agent_termination."""
        if self.config["per_agent_termination"]:
            return all(self._agents_terminated())
        return any(self._agents_terminated())

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


class TeamIntersectionEnv(ConnectedLaneMultiAgentIntersectionEnv):
    """Intersection driving with one reward per controlled vehicle and per-agent termination.

    ``step`` replaces ``MultiAgentWrapper`` (which ``intersection-multi-agent-v2``
    uses) on purpose: it returns the per-agent rewards computed by the configured
    reward function together with ``info["agents_terminated"]``, so the reward can
    come from ``marl.reward_funcs`` instead of the stock cooperative mean.
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


gym.register(id="team-highway-v0", entry_point=TeamHighwayEnv)
gym.register(id="team-intersection-v0", entry_point=TeamIntersectionEnv)


def make_env(
    n_agents: int = 3,
    render_mode: str | None = None,
    name: str = "highway",
    reward_function: str = "team",
    team_weight: float | None = None,
    per_agent_termination: bool = False,
) -> gym.Env:
    """Build a driving env; ``per_agent_termination`` only applies to highway (intersection always ends per agent)."""
    if team_weight is not None and (team_weight < 0.0 or team_weight > 1.0):
        raise ValueError(f"team_weight must be in [0, 1], got {team_weight}")
    if name == "intersection":
        return make_intersection_env(n_agents, render_mode, reward_function, team_weight)
    if name == "highway":
        return make_highway_env(
            n_agents, render_mode, reward_function, team_weight, per_agent_termination
        )
    raise ValueError(f"Unknown environment: {name}")


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


def make_highway_env(
    n_agents: int = 3,
    render_mode: str | None = None,
    reward_function: str = "team",
    team_weight: float | None = None,
    per_agent_termination: bool = False,
) -> gym.Env:
    config: dict[str, Any] = {
        "controlled_vehicles": n_agents,
        "reward_function": reward_function,
        "per_agent_termination": per_agent_termination,
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
    return gym.make("team-highway-v0", config=config, render_mode=render_mode)


def flatten_obs(obs) -> np.ndarray:
    """Convert one observation per agent to a float32 (agents, features) array."""
    return np.stack(
        [np.asarray(agent_obs, dtype=np.float32).reshape(-1) for agent_obs in obs]
    )


def any_vehicle_crashed(env: gym.Env) -> bool:
    unwrapped: Any = env.unwrapped  # highway-env specific attributes are unknown to the type checker
    return any(vehicle.crashed for vehicle in unwrapped.controlled_vehicles)


def crashed_vehicle_fraction(env: gym.Env) -> float:
    """Fraction of the controlled vehicles that have crashed."""
    unwrapped: Any = env.unwrapped  # highway-env specific attributes are unknown to the type checker
    vehicles = unwrapped.controlled_vehicles
    return sum(vehicle.crashed for vehicle in vehicles) / len(vehicles)


def build_actor_obs(obs_flat: np.ndarray, n_agents: int) -> np.ndarray:
    """Append one-hot agent IDs to the per-agent observations."""
    agent_ids = np.eye(n_agents, dtype=np.float32)
    return np.concatenate([obs_flat, agent_ids], axis=-1)
