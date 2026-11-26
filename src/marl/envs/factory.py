"""Build an environment by name."""

import gymnasium as gym

from .highway import make_highway_env
from .intersection import make_intersection_env


def make_env(
    n_agents: int = 3,
    render_mode: str | None = None,
    name: str = "highway",
    reward_function: str = "team",
    team_weight: float | None = None,
    render_agents: int | None = None,
) -> gym.Env:
    """render_agents (camera views in the window) only applies to highway."""
    if team_weight is not None and (team_weight < 0.0 or team_weight > 1.0):
        raise ValueError(f"team_weight must be in [0, 1], got {team_weight}")
    if name == "intersection":
        return make_intersection_env(n_agents, render_mode, reward_function, team_weight)
    if name == "highway":
        return make_highway_env(
            n_agents, render_mode, reward_function, team_weight, render_agents
        )
    raise ValueError(f"Unknown environment: {name}")
