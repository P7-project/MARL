"""
Networks: Gaussian policy π_θ (mean net μ), value V, worst-attack critic Q_φ(s, a).

Per iteration:

+ Roll out π_θ on clean observations. Compute returns and advantages, then fit V.
+ Compute IBP bounds of μ over B_ε(s') for the next states. This gives the box [μ_lo, μ_hi].
+ Find â = argmin_{a in box} Q_φ(s', a) with 50 steps of gradient descent, clamped to the box.
+ Set y = r + γ·Q_φ(s', â), with y = r at terminal states. Update φ with MSE.
+ Compute w(s) = V(s) − min_a Q_φ(s, a).
+ Compute L_reg with SGLD, using the SA-PPO regularizer, weighted by w(s).
+ Update π_θ with the clipped PPO objective, using advantage Â + κ_wst·Q_φ(s, a) at the sampled action, plus κ_reg·w(s)·L_reg.

Hyperparameters from the paper:

- κ_wst ramps from 0, linearly to 0.8 on Hopper, Walker2d and HalfCheetah, and exponentially to 0.5 on Ant.
- ε ramps up slowly over training.
- κ_reg is grid-searched over {0.01, 0.03, 0.1, 0.3, 1.0}.
- Training length is 2M steps, or 10M on Ant.
"""





from typing import NamedTuple

import torch
from torch import Tensor, nn

from auto_LiRPA.auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.auto_LiRPA.perturbations import PerturbationLpNorm


class ActionBox(NamedTuple):
    lower: Tensor  # [batch, action_dim]
    upper: Tensor  # [batch, action_dim]


class WorstAttackPPOLoss:
    """Combines the standard PPO loss with the worst-attack value term.

    Math: maximize  E_s[ min_{a in A_adv(s, pi)} Q_worst(s, a) ],
    where the min is lower-bounded over the box containing A_adv(s, pi).
    Gradients flow into pi through the box bounds.
    """

    def __init__(
        self,
        worst_value_estimator: "WorstAttackValueEstimator",
        policy_bounder: "PolicyActionBounder",
        worst_value_coefficient: float,
        regularization_coefficient: float,
    ) -> None:
        self._worst_value_estimator = worst_value_estimator
        self._policy_bounder = policy_bounder
        self._worst_value_coefficient = worst_value_coefficient
        self._regularization_coefficient = regularization_coefficient

    def compute(self, ppo_loss: Tensor, states: Tensor) -> Tensor:
        """total = L_PPO - kappa_wst * E[V_worst-bound] + kappa_reg * E[box width]."""
        box = self._policy_bounder.action_box(states)
        worst_value = self._worst_value_estimator.lower_bound_over_box(states, box)
        box_width = (box.upper - box.lower).mean()
        return (
            ppo_loss
            - self._worst_value_coefficient * worst_value.mean()
            + self._regularization_coefficient * box_width
        )


class WorstAttackQTrainer:
    """Regresses Q_worst onto targets of the worst-attack Bellman operator (Definition 4.1)."""

    def __init__(
        self,
        q_network: nn.Module,
        target_estimator: "WorstAttackValueEstimator",
        policy_bounder: "PolicyActionBounder",
        discount_factor: float,
    ) -> None:
        self._q_network = q_network
        self._target_estimator = target_estimator
        self._policy_bounder = policy_bounder
        self._discount_factor = discount_factor

    def loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        terminated: Tensor,
    ) -> Tensor:
        targets = self._targets(rewards, next_states, terminated)
        predictions = self._q_network(torch.cat([states, actions], dim=1)).squeeze(-1)
        return nn.functional.mse_loss(predictions, targets)

    @torch.no_grad()
    def _targets(self, rewards: Tensor, next_states: Tensor, terminated: Tensor) -> Tensor:
        """Math: R(s,a) + gamma * min_{a' in A_adv(s', pi)} Q(s', a'), one sampled s'.

        `terminated` is float 0/1 and excludes time-limit truncation.
        """
        box = self._policy_bounder.action_box(next_states)
        worst_values = self._target_estimator.lower_bound_over_box(next_states, box)
        not_terminated = 1.0 - terminated
        return rewards + self._discount_factor * not_terminated * worst_values


class WorstAttackValueEstimator:
    """Lower-bounds min_{a' in box} Q(s, a') by propagating bounds through Q's action input.

    Sound: the result is <= the true min over the box, and the box contains
    A_adv, so the result is <= min_{a' in A_adv} Q(s, a'). Conservative.
    """

    def __init__(self, bounded_q_network: BoundedModule) -> None:
        # Wraps a Q network that takes cat([state, action]).
        self._bounded_q_network = bounded_q_network

    def lower_bound_over_box(self, states: Tensor, box: ActionBox) -> Tensor:
        lower_input = torch.cat([states, box.lower], dim=1)
        upper_input = torch.cat([states, box.upper], dim=1)
        center = (lower_input + upper_input) / 2
        # State dims have zero width (x_L == x_U); only the action dims vary.
        perturbation = PerturbationLpNorm(x_L=lower_input, x_U=upper_input)
        bounded_input = BoundedTensor(center, perturbation)
        lower, _ = self._bounded_q_network.compute_bounds(x=(bounded_input,), method="CROWN")
        return lower.squeeze(-1)


class PolicyActionBounder:
    """Bounds the policy's mean action over B_eps(s).

    Math: [a_lo(s), a_hi(s)] with
        a_lo <= pi(s~) <= a_hi  for all s~ in B_eps(s) (L-inf ball, radius epsilon),
    hence A_adv(s, pi) is contained in the box. For a Gaussian policy, `pi` here
    is the mean network (the deterministic action the adversary can shift).
    """

    def __init__(
        self,
        bounded_policy_mean_network: BoundedModule,
        epsilon: float,
        method: str = "IBP",
    ) -> None:
        self._bounded_policy_mean_network = bounded_policy_mean_network
        self._epsilon = epsilon
        self._method = method

    def action_box(self, states: Tensor) -> ActionBox:
        perturbation = PerturbationLpNorm(norm=float("inf"), eps=self._epsilon) # type: ignore[arg-type]  # runtime accepts float
        bounded_states = BoundedTensor(states, perturbation)
        lower, upper = self._bounded_policy_mean_network.compute_bounds(
            x=(bounded_states,), method=self._method
        )
        return ActionBox(lower, upper)