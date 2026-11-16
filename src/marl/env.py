"""Highway environment setup and observation helpers."""

import gymnasium as gym
import numpy as np
from highway_env import utils
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.controller import ControlledVehicle


class TeamHighwayEnv(HighwayEnv):
    """Highway driving with one reward and episode outcome shared by all controlled vehicles."""

    def _reward(self, action) -> float:
        """Average the per-vehicle driving reward over all controlled vehicles."""
        return float(
            np.mean(
                [self._agent_reward(vehicle) for vehicle in self.controlled_vehicles]
            )
        )

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

def make_env(n_agents: int = 3, render_mode=None, name: str = "highway",) -> gym.Env:
    if name == "intersection":
        return make_intersection_env(n_agents, render_mode)
    if name == "highway":
        return make_highway_env(n_agents, render_mode)
    raise ValueError(f"Unknown environment: {name}")


def make_intersection_env(n_agents: int = 3, render_mode=None) -> gym.Env:
    return gym.make(
        "intersection-multi-agent-v2",
        config={"controlled_vehicles": n_agents},
        render_mode=render_mode,
    )


def make_highway_env(n_agents: int = 3, render_mode=None) -> gym.Env:
    config = {
        "controlled_vehicles": n_agents,
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


def build_actor_obs(obs_flat: np.ndarray, n_agents: int) -> np.ndarray:
    """Append one-hot agent IDs to the per-agent observations."""
    agent_ids = np.eye(n_agents, dtype=np.float32)
    return np.concatenate([obs_flat, agent_ids], axis=-1)
