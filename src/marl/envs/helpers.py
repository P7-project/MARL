"""Observation and crash helpers shared by the trainers."""

from typing import Any

import gymnasium as gym
import numpy as np


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
