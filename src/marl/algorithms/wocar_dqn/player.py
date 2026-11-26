"""Plays a trained WocaR-DQN policy (greedy Q_r), optionally under an observation attack."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from auto_LiRPA import BoundedModule

from ...attacks.registry import ObservationAttack
from ...common import PolicyPlayer, StepResult, Weights
from ...envs import any_vehicle_crashed, build_actor_obs, flatten_obs, make_env
from .networks import _build_robust_q, _to_tensor


class DqnPlayer(PolicyPlayer):
    """Plays the greedy Q_r policy in a multi-agent driving environment."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        hidden: int,
        env_name: str = "highway",
        reward_function: str = "mean",
        team_weight: float | None = None,
        render: bool = True,
        render_agents: int | None = None,
    ) -> None:
        self._n_agents: int = n_agents
        self._obs_dim: int = obs_dim
        self._render: bool = render
        self._device: torch.device = torch.device("cpu")
        self._env = make_env(
            n_agents,
            render_mode="human" if render else None,
            name=env_name,
            reward_function=reward_function,
            team_weight=team_weight,
            render_agents=render_agents,
        )
        self._policy: BoundedModule = _build_robust_q(
            obs_dim + n_agents, n_actions, hidden, self._device
        )
        self._policy.eval()
        self._obs: np.ndarray | None = None
        self._attack: ObservationAttack | None = None  # None means no attack
        self._attack_epsilon: float = 0.0
        self._attacked_mask: Tensor | None = None  # [n_agents], bool
        self._attacked_decisions: int = 0
        self._changed_decisions: int = 0

    def load_weights(self, weights: Weights) -> None:
        self._policy.load_state_dict(weights)

    def set_attack(
        self, attack: ObservationAttack, epsilon: float, attacked_mask: Tensor
    ) -> None:
        """Attack the observations of the masked cars from now on."""
        self._attack = attack
        self._attack_epsilon = epsilon
        self._attacked_mask = attacked_mask

    @property
    def attack_counts(self) -> tuple[int, int]:
        """Attacked and changed decisions in the current episode."""
        return self._attacked_decisions, self._changed_decisions

    def start_episode(self, seed: int | None = None) -> None:
        self._obs = flatten_obs(self._env.reset(seed=seed)[0])
        self._attacked_decisions = 0
        self._changed_decisions = 0
        if self._render and self._attacked_mask is not None:
            self._highlight_attacked_cars(self._attacked_mask)

    def step(self) -> StepResult:
        if self._obs is None:
            raise RuntimeError("start_episode must be called before step")
        actions: list[int] = self._greedy_actions(self._obs)
        next_obs, _reward, terminated, truncated, _info = self._env.step(tuple(actions))
        if self._render:
            self._env.render()
        self._obs = flatten_obs(next_obs)
        agent_terminated = np.asarray(terminated, dtype=np.bool_)
        episode_truncated = np.asarray(truncated, dtype=np.bool_)
        finished: bool = bool(agent_terminated.all()) or bool(episode_truncated.any())
        reward: float = float(np.asarray(_reward, dtype=np.float32).mean())
        crashed: bool = any_vehicle_crashed(self._env)
        return StepResult(finished=finished, reward=reward, crashed=crashed)

    def close(self) -> None:
        self._env.close()

    def _highlight_attacked_cars(self, attacked_mask: Tensor) -> None:
        """Mark the attacked cars in the window; reset creates new cars, so call after it."""
        unwrapped: Any = self._env.unwrapped
        if hasattr(unwrapped, "set_attacked_agents"):  # only the highway viewer highlights cars
            unwrapped.set_attacked_agents(attacked_mask.nonzero().flatten().tolist())

    @torch.no_grad()
    def _greedy_actions(self, obs: np.ndarray) -> list[int]:
        actor_obs: Tensor = _to_tensor(build_actor_obs(obs, self._n_agents), self._device)
        actions: Tensor = self._policy(actor_obs).argmax(dim=1)  # [n_agents]
        if self._attack is not None and self._attacked_mask is not None:
            actions = self._attacked_actions(
                self._attack, actor_obs, actions, self._attacked_mask
            )
        return [int(action) for action in actions]

    def _attacked_actions(
        self,
        attack: ObservationAttack,
        actor_obs: Tensor,
        clean_actions: Tensor,
        attacked_mask: Tensor,
    ) -> Tensor:
        """Greedy actions under the attack; also counts the changed decisions."""
        perturbed: Tensor = attack(
            self._policy, actor_obs, self._attack_epsilon, attacked_mask, self._obs_dim
        )
        actions: Tensor = self._policy(perturbed).argmax(dim=1)
        self._attacked_decisions += int(attacked_mask.sum())
        self._changed_decisions += int((actions != clean_actions)[attacked_mask].sum())
        return actions


def build_dqn_player(
    checkpoint: dict, render: bool = True, render_agents: int | None = None
) -> DqnPlayer:
    player = DqnPlayer(
        checkpoint["n_agents"],
        checkpoint["obs_dim"],
        checkpoint["n_actions"],
        checkpoint["hidden"],
        env_name=checkpoint.get("dqn_env", "highway"),
        reward_function=checkpoint.get("reward_function", "mean"),
        team_weight=checkpoint.get("team_weight"),
        render=render,
        render_agents=render_agents,
    )
    player.load_weights(checkpoint["robust_q"])
    return player
