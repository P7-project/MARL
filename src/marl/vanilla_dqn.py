"""
    Vanilla MARL DQN, the baseline for WocaR-DQN: the same setup without WocaR.

    One Q-network Q, which also acts (epsilon-greedy), plus a soft-updated target Q'.
    It is the Q_v path of WocaRDQNTrainer, with Q_v acting instead of Q_r.

    Per update(batch, step):
    + y = r + gamma * (1 - terminated) * max_a' Q'(s', a')
      (double DQN: r + gamma * (1 - terminated) * Q'(s', argmax_a' Q(s', a'))).
    + MSE between Q(s, a) and y, one Adam step on Q.
    + Soft update: theta' <- tau * theta + (1 - tau) * theta'.

    No Q_r, no Q_phi, no IBP bounds and no regularizer.
    The env loop, shared replay buffer, one-hot agent ID, exploration, seeds and reward
    are those of train_dqn in woca_r_rl/dqn.py.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.optim import Adam

from auto_LiRPA import BoundedModule

from .woca_r_rl.dqn import Batch, _build_robust_q


@dataclass(frozen=True)
class VanillaDQNConfig:
    num_actions: int
    discount_factor: float
    target_update_coefficient: float  # tau
    learning_rate: float
    use_double_dqn: bool = False


class VanillaMetrics(NamedTuple):
    td_loss: float


class VanillaDQNTrainer:
    def __init__(
        self,
        q_network: BoundedModule,
        target_network: BoundedModule,
        config: VanillaDQNConfig,
    ) -> None:
        """target_network is a second network of the same shape (a BoundedModule cannot be deep-copied)."""
        self._q = q_network
        self._target_q = target_network
        self._target_q.load_state_dict(q_network.state_dict())
        self._target_q.eval()  # auto_LiRPA resets requires_grad to self.training on every forward
        for parameter in self._target_q.parameters():
            parameter.requires_grad_(False)
        self._config = config
        self._optimizer = Adam(q_network.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def select_action(self, state: Tensor, exploration_probability: float) -> int:
        """Epsilon-greedy w.r.t. Q (the acting policy)."""
        if float(torch.rand(())) < exploration_probability:
            return int(torch.randint(self._config.num_actions, ()))
        q_values = self._q(state.unsqueeze(0))
        return int(q_values.argmax(dim=1))

    def update(self, batch: Batch, step: int) -> VanillaMetrics:
        """One TD step on Q, then a soft target update. `step` is unused (same interface as WocaRDQNTrainer)."""
        targets = self._targets(batch)
        predictions = self._gather(self._q(batch.states), batch.actions)
        loss = nn.functional.mse_loss(predictions, targets)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._soft_update_target()
        return VanillaMetrics(td_loss=float(loss.detach()))

    def q_state_dict(self) -> dict[str, Tensor]:
        return self._q.state_dict()

    @torch.no_grad()
    def _targets(self, batch: Batch) -> Tensor:
        """r + gamma * Q'(s', argmax_a' Q(s', a')) (double DQN) or r + gamma * max_a' Q'(s', a')."""
        target_q_values = self._target_q(batch.next_states)
        if self._config.use_double_dqn:
            best_actions = self._q(batch.next_states).argmax(dim=1, keepdim=True)
            next_values = target_q_values.gather(1, best_actions).squeeze(1)
        else:
            next_values = target_q_values.max(dim=1).values
        not_terminated = 1.0 - batch.terminated
        return batch.rewards + self._config.discount_factor * not_terminated * next_values

    @torch.no_grad()
    def _soft_update_target(self) -> None:
        """theta' <- tau * theta + (1 - tau) * theta'."""
        tau = self._config.target_update_coefficient
        for target_parameter, online_parameter in zip(
            self._target_q.parameters(), self._q.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(online_parameter, alpha=tau)

    def _gather(self, q_values: Tensor, actions: Tensor) -> Tensor:
        return q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)


def build_vanilla_trainer(
    obs_dim: int, n_agents: int, n_actions: int, args: Namespace, device: torch.device
) -> VanillaDQNTrainer:
    """Q and Q' with the same shape as Q_r, so the checkpoint loads into DqnPlayer."""
    input_dim: int = obs_dim + n_agents
    config: VanillaDQNConfig = VanillaDQNConfig(
        num_actions=n_actions,
        discount_factor=args.gamma,
        target_update_coefficient=args.tau,
        learning_rate=args.lr,
        use_double_dqn=args.double_dqn,
    )
    q_network: BoundedModule = _build_robust_q(input_dim, n_actions, args.hidden, device)
    target_network: BoundedModule = _build_robust_q(input_dim, n_actions, args.hidden, device)
    return VanillaDQNTrainer(q_network, target_network, config)
