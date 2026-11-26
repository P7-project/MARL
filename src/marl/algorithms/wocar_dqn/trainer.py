"""
    WocaR-DQN (Algorithm 2 of Liang et al., NeurIPS 2022) for discrete actions.

    Networks:
    Q_v plus a soft-updated target Q_v', trained as a vanilla (double) DQN.
    Q_r, the robust Q-network. It acts, and its bounds define A_adv_hat.
    Q_phi, the worst-attack critic (one value per action).

    Per update(batch, step):
    + Update Q_v with the DQN target, then soft-update Q_v'.
    + IBP output bounds of Q_r at s' with the scheduled eps_t: Q_l(a), Q_u(a).
    + A_adv_hat(s') = {a : Q_u(a) >= max_{a' != a} Q_l(a')}.
    + a_hat = argmin over A_adv_hat of Q_phi(s', a); y = r + gamma * Q_phi(s', a_hat). MSE update of Q_phi.
    + w(s) = normalized(max_a Q_v(s, a) - min_a Q_v(s, a)).
    + Train Q_r on y_r = r + gamma * max_a' [(1 - kappa_wst) * Q_v'(s', a') + kappa_wst * Q_phi(s', a')]

    Math                                         Code
    (T_worst^pi Q)(s, a)                         _worst_attack_targets
    E_{s' ~ P(s, a)}                             replay-buffer sample next_states
    R(s, a)                                      batch.rewards
    gamma                                        config.discount_factor
    min_{a' in A_adv(s', pi)} Q(s', a')          _worst_case_values
    A_adv(s, pi)                                 ReachableActionEstimator.mask
    B_eps(s)                                     RobustQBounder.bounds (clipped L-inf ball)
    Q (worst-attack critic)                      worst_attack_q
    pi = argmax_a Q_r(s, a)                      robust_q
    eps_t, kappa_wst(t)                          RampSchedule
    w(s)                                         _state_importance
    L_reg                                        SARegularizer
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn
from torch.optim import Adam

from auto_LiRPA import BoundedModule

from ...common import RampSchedule
from .bounds import ReachableActionEstimator, RobustQBounder
from .regularizer import SARegularizer


class Batch(NamedTuple):
    states: Tensor  # [batch, *observation_shape]
    actions: Tensor  # [batch], integer
    rewards: Tensor  # [batch]
    next_states: Tensor  # [batch, *observation_shape]
    terminated: Tensor  # [batch], float 0/1; must exclude time-limit truncation


@dataclass(frozen=True)
class WocaRDQNConfig:
    num_actions: int
    discount_factor: float
    target_update_coefficient: float  # tau
    learning_rate: float
    epsilon_schedule: RampSchedule
    kappa_wst_schedule: RampSchedule
    kappa_reg: float
    observation_low: float
    observation_high: float
    regularizer_margin: float = 1.0  # hinge margin c of the SA-DQN regularizer
    use_double_dqn: bool = False
    bound_method: str = "IBP"
    perturbation_scale: Optional[Tensor] = None


class TrainingMetrics(NamedTuple):
    vanilla_loss: float
    worst_attack_loss: float
    robust_td_loss: float
    regularizer_loss: float
    epsilon: float
    kappa_wst: float
    mean_reachable_actions: float


class WocaRDQNTrainer:
    def __init__(
        self,
        vanilla_q: nn.Module,
        robust_q: BoundedModule,
        worst_attack_q: nn.Module,
        config: WocaRDQNConfig,
    ) -> None:
        self._vanilla_q = vanilla_q
        self._target_vanilla_q = self._frozen_copy(vanilla_q)
        self._robust_q = robust_q
        self._worst_attack_q = worst_attack_q
        self._config = config
        self._bounder = RobustQBounder(
            robust_q,
            config.observation_low,
            config.observation_high,
            config.perturbation_scale,
            config.bound_method,
        )
        self._estimator = ReachableActionEstimator()
        self._regularizer = SARegularizer(config.regularizer_margin)
        self._vanilla_optimizer = Adam(vanilla_q.parameters(), lr=config.learning_rate)
        self._robust_optimizer = Adam(robust_q.parameters(), lr=config.learning_rate)
        self._worst_attack_optimizer = Adam(worst_attack_q.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def select_action(self, state: Tensor, exploration_probability: float) -> int:
        """Epsilon-greedy w.r.t. Q_r (the acting policy)."""
        if float(torch.rand(())) < exploration_probability:
            return int(torch.randint(self._config.num_actions, ()))
        q_values = self._robust_q(state.unsqueeze(0))
        return int(q_values.argmax(dim=1))

    def update(self, batch: Batch, step: int) -> TrainingMetrics:
        epsilon = self._config.epsilon_schedule.value(step)
        kappa_wst = self._config.kappa_wst_schedule.value(step)

        vanilla_loss = self._update_vanilla_q(batch)
        self._soft_update_target()
        worst_attack_loss, mean_reachable = self._update_worst_attack_critic(batch, epsilon)
        robust_td_loss, regularizer_loss = self._update_robust_q(batch, epsilon, kappa_wst)

        return TrainingMetrics(
            vanilla_loss=vanilla_loss,
            worst_attack_loss=worst_attack_loss,
            robust_td_loss=robust_td_loss,
            regularizer_loss=regularizer_loss,
            epsilon=epsilon,
            kappa_wst=kappa_wst,
            mean_reachable_actions=mean_reachable,
        )

    def robust_q_state_dict(self) -> dict[str, Tensor]:
        return self._robust_q.state_dict()

    # ----------------------------------------------------------------- Q_v ---

    def _update_vanilla_q(self, batch: Batch) -> float:
        targets = self._vanilla_targets(batch)
        predictions = self._gather(self._vanilla_q(batch.states), batch.actions)
        loss = nn.functional.mse_loss(predictions, targets)
        self._step(self._vanilla_optimizer, loss)
        return float(loss.detach())

    @torch.no_grad()
    def _vanilla_targets(self, batch: Batch) -> Tensor:
        """r + gamma * Q_v'(s', argmax_a' Q_v(s', a')) (double DQN) or r + gamma * max_a' Q_v'(s', a')."""
        target_q_values = self._target_vanilla_q(batch.next_states)
        if self._config.use_double_dqn:
            best_actions = self._vanilla_q(batch.next_states).argmax(dim=1, keepdim=True)
            next_values = target_q_values.gather(1, best_actions).squeeze(1)
        else:
            next_values = target_q_values.max(dim=1).values
        return self._bootstrap(batch, next_values)

    @torch.no_grad()
    def _soft_update_target(self) -> None:
        """theta' <- tau * theta + (1 - tau) * theta'."""
        tau = self._config.target_update_coefficient
        for target_parameter, online_parameter in zip(
            self._target_vanilla_q.parameters(), self._vanilla_q.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(online_parameter, alpha=tau)

    # --------------------------------------------------------------- Q_phi ---

    def _update_worst_attack_critic(self, batch: Batch, epsilon: float) -> tuple[float, float]:
        targets, reachable = self._worst_attack_targets(batch, epsilon)
        predictions = self._gather(self._worst_attack_q(batch.states), batch.actions)
        loss = nn.functional.mse_loss(predictions, targets)
        self._step(self._worst_attack_optimizer, loss)
        mean_reachable = float(reachable.float().sum(dim=1).mean())
        return float(loss.detach()), mean_reachable

    @torch.no_grad()
    def _worst_attack_targets(self, batch: Batch, epsilon: float) -> tuple[Tensor, Tensor]:
        """
        r + gamma * min_{a' in A_adv_hat(s')} Q_phi(s', a'), with A_adv_hat from the
        IBP bounds of Q_r at the scheduled epsilon. Uses Q_phi itself, no target network.
        `terminated` must exclude time-limit truncation.
        """
        bounds = self._bounder.bounds(batch.next_states, epsilon)
        reachable = self._estimator.mask(bounds)
        q_values = self._worst_attack_q(batch.next_states)
        next_values = self._worst_case_values(q_values, reachable)
        return self._bootstrap(batch, next_values), reachable

    def _worst_case_values(self, q_values: Tensor, reachable: Tensor) -> Tensor:
        unreachable = torch.logical_not(reachable)
        return q_values.masked_fill(unreachable, float("inf")).min(dim=1).values

    # ----------------------------------------------------------------- Q_r ---

    def _update_robust_q(
        self, batch: Batch, epsilon: float, kappa_wst: float
    ) -> tuple[float, float]:
        targets = self._robust_targets(batch, kappa_wst)
        clean_q_values = self._robust_q(batch.states)
        td_loss = nn.functional.mse_loss(self._gather(clean_q_values, batch.actions), targets)
        regularizer_loss = self._weighted_regularizer(batch.states, clean_q_values, epsilon)
        loss = td_loss + self._config.kappa_reg * regularizer_loss
        self._step(self._robust_optimizer, loss)
        return float(td_loss.detach()), float(regularizer_loss.detach())

    @torch.no_grad()
    def _robust_targets(self, batch: Batch, kappa_wst: float) -> Tensor:
        """
        y_r = r + gamma * max_a' [(1 - kappa_wst) * Q_v'(s', a') + kappa_wst * Q_phi(s', a')].
        The two weights are swapped relative to the published algorithm (see the paper, Appendix B).
        """
        vanilla_values = self._target_vanilla_q(batch.next_states)
        worst_attack_values = self._worst_attack_q(batch.next_states)
        mixed_values = (1.0 - kappa_wst) * vanilla_values + kappa_wst * worst_attack_values
        return self._bootstrap(batch, mixed_values.max(dim=1).values)

    def _weighted_regularizer(
        self, states: Tensor, clean_q_values: Tensor, epsilon: float
    ) -> Tensor:
        """mean_s [ w(s) * L_reg(s) ]. Gradients flow into Q_r through the bounds, not through w."""
        best_actions: Tensor = clean_q_values.detach().argmax(dim=1)
        margin_lower: Tensor = self._bounder.margin_lower_bounds(
            states, best_actions, self._config.num_actions, epsilon
        )
        per_state_loss: Tensor = self._regularizer.per_state_loss(margin_lower)
        return (self._state_importance(states) * per_state_loss).mean()

    @torch.no_grad()
    def _state_importance(self, states: Tensor) -> Tensor:
        """w(s) = max_a Q_v(s, a) - min_a Q_v(s, a), divided by its batch maximum."""
        q_values = self._vanilla_q(states)
        spread = q_values.max(dim=1).values - q_values.min(dim=1).values
        return spread / spread.max().clamp(min=1e-8)

    # ------------------------------------------------------------- helpers ---

    def _bootstrap(self, batch: Batch, next_values: Tensor) -> Tensor:
        not_terminated = 1.0 - batch.terminated
        return batch.rewards + self._config.discount_factor * not_terminated * next_values

    def _gather(self, q_values: Tensor, actions: Tensor) -> Tensor:
        return q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)

    def _step(self, optimizer: Adam, loss: Tensor) -> None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _frozen_copy(self, network: nn.Module) -> nn.Module:
        copy = deepcopy(network)
        for parameter in copy.parameters():
            parameter.requires_grad_(False)
        return copy
