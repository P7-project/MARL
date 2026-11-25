"""Multi-agent driving environments (highway, intersection), their reward functions and helpers.

Importing this package registers ``team-highway-v0`` and ``team-intersection-v0``.
"""

from .factory import make_env
from .helpers import any_vehicle_crashed, build_actor_obs, flatten_obs
from .highway import TeamHighwayEnv, make_highway_env
from .intersection import TeamIntersectionEnv, make_intersection_env
from .rewards import REWARD_FUNCTIONS, get_reward_function

__all__ = [
    "REWARD_FUNCTIONS",
    "TeamHighwayEnv",
    "TeamIntersectionEnv",
    "any_vehicle_crashed",
    "build_actor_obs",
    "flatten_obs",
    "get_reward_function",
    "make_env",
    "make_highway_env",
    "make_intersection_env",
]
