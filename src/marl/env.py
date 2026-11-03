"""Highway environment setup and observation helpers."""

import gymnasium as gym
import highway_env  # noqa: F401  (registers the highway-* environments)
import numpy as np


if hasattr(gym, "register_envs"):
    gym.register_envs(highway_env)


def make_env(n_agents: int = 3, render_mode=None) -> gym.Env:
    """Create the project's multi-agent highway environment."""
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
    return gym.make("highway-v0", config=config, render_mode=render_mode)


def flatten_obs(obs) -> np.ndarray:
    """Convert one observation per agent to a float32 (agents, features) array."""
    return np.stack([np.asarray(agent_obs, dtype=np.float32).reshape(-1) for agent_obs in obs])


def build_actor_obs(obs_flat: np.ndarray, n_agents: int) -> np.ndarray:
    """Append one-hot agent IDs to the per-agent observations."""
    agent_ids = np.eye(n_agents, dtype=np.float32)
    return np.concatenate([obs_flat, agent_ids], axis=-1)
