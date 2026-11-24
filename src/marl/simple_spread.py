"""Gymnasium adapter for the cooperative MPE2 Simple Spread task."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from mpe2 import simple_spread_v3

# 0 makes every agent receive only the shared landmark-coverage reward. The local
# collision penalty (-1 per collision) is weighted by local_ratio, so it is omitted.
SIMPLE_SPREAD_LOCAL_RATIO: float = 0.0
SIMPLE_SPREAD_MAX_CYCLES: int = 25  # MPE2 default episode length

TeamObs = tuple[np.ndarray, ...]  # one (obs_dim,) array per agent, in agent order
TeamAction = tuple[int, ...]  # one discrete action per agent, in agent order


class SimpleSpreadEnv(gym.Env[TeamObs, TeamAction]):
    """MPE2 simple_spread_v3 as a team Gymnasium env.

    Observations and actions are tuples in the fixed order agent_0 ... agent_{N-1}.
    The reward is the shared global reward, returned once per step rather than once per
    agent. Termination and truncation stay separate, so time limits can be bootstrapped.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_agents: int,
        render_mode: str | None = None,
        max_cycles: int = SIMPLE_SPREAD_MAX_CYCLES,
    ) -> None:
        self.render_mode = render_mode
        self._env = simple_spread_v3.parallel_env(
            N=n_agents,
            local_ratio=SIMPLE_SPREAD_LOCAL_RATIO,
            max_cycles=max_cycles,
            continuous_actions=False,
            render_mode=render_mode,
        )
        self._agent_names: list[str] = list(self._env.possible_agents)
        self.observation_space = spaces.Tuple(
            [self._env.observation_space(name) for name in self._agent_names]
        )
        self.action_space = spaces.Tuple(
            [self._env.action_space(name) for name in self._agent_names]
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[TeamObs, dict[str, object]]:
        """Start an episode. A seed reseeds MPE2; later unseeded resets continue its RNG."""
        super().reset(seed=seed)
        observations, _infos = self._env.reset(seed=seed, options=options)
        return self._ordered_obs(observations), {}

    def step(
        self, action: TeamAction
    ) -> tuple[TeamObs, float, bool, bool, dict[str, object]]:
        """Apply one action per agent and return the team transition."""
        actions = {
            name: int(agent_action)
            for name, agent_action in zip(self._agent_names, action, strict=True)
        }
        observations, rewards, terminations, truncations, _infos = self._env.step(actions)
        # local_ratio=0 gives every agent the same global reward, so count it once.
        reward = float(rewards[self._agent_names[0]])
        terminated = any(terminations[name] for name in self._agent_names)
        truncated = any(truncations[name] for name in self._agent_names)
        return self._ordered_obs(observations), reward, terminated, truncated, {}

    # gymnasium declares render() with an unbound RenderFrame TypeVar, so no concrete
    # return type can match it.
    def render(self) -> np.ndarray | None:  # type: ignore[override]
        """Return an RGB frame in rgb_array mode. Human mode already draws inside step."""
        if self.render_mode != "rgb_array":
            return None
        return self._env.render()

    def close(self) -> None:
        """Close the MPE2 env and its window."""
        self._env.close()

    def _ordered_obs(self, observations: dict[str, np.ndarray]) -> TeamObs:
        # MPE2's parallel step observes before dropping truncated agents, so the final
        # observation is present; a missing agent raises KeyError instead of being skipped.
        return tuple(observations[name] for name in self._agent_names)
