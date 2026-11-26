"""IBP output bounds of Q_r and the estimated adversarial action set A_adv_hat."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


class QValueBounds(NamedTuple):
    lower: Tensor  # [batch, num_actions]
    upper: Tensor  # [batch, num_actions]


class RobustQBounder:
    """
        Output bounds of Q_r over B_eps(s), clipped to the valid observation range.

        Math: Q_l(s, a) <= Q_r(s~, a) <= Q_u(s, a) for all s~ in B_eps(s).
        B_eps is the L-inf ball: {s~ : ||s~ - s||_inf <= eps}, scaled per entry by perturbation_scale
        and clipped to the valid range.
        Assumes observations already lie inside [observation_low, observation_high].
    """

    def __init__(
        self,
        bounded_q_network: BoundedModule,
        observation_low: float,
        observation_high: float,
        perturbation_scale: Tensor | None,
        method: str,
    ) -> None:
        self._bounded_q_network = bounded_q_network
        self._observation_low = observation_low
        self._observation_high = observation_high
        self._perturbation_scale = perturbation_scale
        self._method = method

    def bounds(self, states: Tensor, epsilon: float) -> QValueBounds:
        lower, upper = self._bounded_q_network.compute_bounds(
            x=(self._bounded_input(states, epsilon),), method=self._method
        )
        return QValueBounds(lower, upper)

    def margin_lower_bounds(
        self, states: Tensor, best_actions: Tensor, num_actions: int, epsilon: float
    ) -> Tensor:
        specification: Tensor = self._margin_specification(best_actions, num_actions)
        lower, _ = self._bounded_q_network.compute_bounds(
            x=(self._bounded_input(states, epsilon),), C=specification, method=self._method
        )
        return lower

    def _bounded_input(self, states: Tensor, epsilon: float) -> BoundedTensor:
        radius = self._radius(states, epsilon)
        lower_corner = (states - radius).clamp(self._observation_low, self._observation_high)
        upper_corner = (states + radius).clamp(self._observation_low, self._observation_high)
        return BoundedTensor(states, PerturbationLpNorm(x_L=lower_corner, x_U=upper_corner))

    def _margin_specification(self, best_actions: Tensor, num_actions: int) -> Tensor:
        """[batch, A-1, A]: one row per a != a*, computing Q(a*) - Q(a)."""
        identity: Tensor = torch.eye(num_actions, device=best_actions.device)
        rows: Tensor = identity[best_actions].unsqueeze(1) - identity.unsqueeze(0)
        is_best: Tensor = nn.functional.one_hot(best_actions, num_actions).bool()
        others: Tensor = torch.logical_not(is_best)
        return rows[others].view(best_actions.shape[0], num_actions - 1, num_actions)

    def _radius(self, states: Tensor, epsilon: float) -> Tensor:
        if self._perturbation_scale is None:
            return torch.full_like(states, epsilon)
        return self._perturbation_scale.to(states.device) * epsilon


class ReachableActionEstimator:
    """
        Over-approximates A_adv(s, pi) for pi = argmax_a Q_r(s, a) from per-action output bounds.

        Math: A_adv(s, pi) := {a in A : exists s~ in B_eps(s) s.t. pi(s~) = a}.
        a in A_adv_hat(s) iff Q_u(s, a) >= max_{a' != a} Q_l(s, a').
        If Q_u(a) < Q_l(a') for some a', then a' beats a everywhere in B_eps(s),
        so argmax_a Q_r(s~, a) != a for every s~ in B_eps(s), hence a not in A_adv(s, pi).

        Soundness: the returned set contains A_adv (it only removes actions that are
        provably unreachable), so min over it is <= min over the true A_adv.
        The target therefore lower-bounds the true worst-attack Bellman target:
        conservative, never optimistic.

        The clean greedy action always satisfies the condition, so the set is never empty.
    """

    def mask(self, bounds: QValueBounds) -> Tensor:
        """
            Bool tensor [batch, num_actions]: True where a is in A_adv_hat.
        """
        competitor_lower = self._highest_competitor_lower_bound(bounds.lower)
        reachable = bounds.upper >= competitor_lower
        if not bool(reachable.any(dim=1).all()):
            raise RuntimeError("Empty reachable action set; check bounds for NaN or inf.")
        return reachable

    def _highest_competitor_lower_bound(self, lower: Tensor) -> Tensor:
        """
            [batch, A]: entry (b, i) = max_{j != i} lower[b, j] (-inf if there is one action).
        """
        num_actions = lower.shape[1]
        diagonal = torch.eye(num_actions, dtype=torch.bool, device=lower.device)
        expanded = lower.unsqueeze(1).expand(-1, num_actions, -1)
        others = expanded.masked_fill(diagonal.unsqueeze(0), float("-inf"))
        return others.max(dim=2).values
