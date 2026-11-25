"""Untargeted PGD observation attack.
Madry et al. 2018: https://arxiv.org/abs/1706.06083, as used in WocaR-RL Sec. 5.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

PGD_STEPS: int = 10


class PgdAttack:
    """PGD whose random start comes from its own seeded generator."""

    def __init__(self, seed: int, steps: int = PGD_STEPS) -> None:
        self._generator: torch.Generator = torch.Generator().manual_seed(seed)
        self._steps: int = steps

    def __call__(
        self,
        q_network: nn.Module,
        actor_obs: Tensor,
        epsilon: float,
        attacked_mask: Tensor,
        obs_dim: int,
    ) -> Tensor:
        return pgd(
            q_network, actor_obs, epsilon, attacked_mask, obs_dim, self._generator, self._steps
        )


def pgd_gradient(q_network: nn.Module, actor_obs: Tensor, clean_actions: Tensor) -> Tensor:
    """Gradient of the cross-entropy against the clean greedy actions, per car."""
    with torch.enable_grad():
        obs_input: Tensor = actor_obs.detach().clone().requires_grad_(True)
        q_values: Tensor = q_network(obs_input)
        loss: Tensor = F.cross_entropy(q_values, clean_actions, reduction="sum")
        (gradient,) = torch.autograd.grad(loss, obs_input)
    return gradient


def pgd_step(
    adversarial: Tensor, clean: Tensor, gradient: Tensor, step_size: float, epsilon: float
) -> Tensor:
    """One signed ascent step, projected to the L-inf eps-ball around clean and clamped to [-1, 1]."""
    stepped: Tensor = adversarial + step_size * gradient.sign()
    projected: Tensor = clean + (stepped - clean).clamp(-epsilon, epsilon)
    return projected.clamp(-1.0, 1.0)


def pgd(
    q_network: nn.Module,
    actor_obs: Tensor,
    epsilon: float,
    attacked_mask: Tensor,
    obs_dim: int,
    generator: torch.Generator | None = None,
    steps: int = PGD_STEPS,
) -> Tensor:
    """Untargeted PGD on the first obs_dim features of the attacked cars; agent IDs stay unchanged."""
    clean: Tensor = actor_obs.detach()
    perturbed: Tensor = clean.clone()
    if epsilon == 0 or not bool(attacked_mask.any()):
        return perturbed

    rows: Tensor = attacked_mask.to(device=clean.device, dtype=torch.bool)
    clean_rows: Tensor = clean[rows]
    with torch.no_grad():
        clean_actions: Tensor = q_network(clean_rows).argmax(dim=-1)

    clean_obs: Tensor = clean_rows[:, :obs_dim]
    noise: Tensor = torch.rand(clean_obs.shape, generator=generator, dtype=clean_obs.dtype)
    adversarial_rows: Tensor = clean_rows.clone()
    adversarial_rows[:, :obs_dim] = (
        clean_obs + epsilon * (2.0 * noise.to(clean_obs.device) - 1.0)
    ).clamp(-1.0, 1.0)

    step_size: float = epsilon / steps
    for _ in range(steps):
        gradient: Tensor = pgd_gradient(q_network, adversarial_rows, clean_actions)
        adversarial_rows[:, :obs_dim] = pgd_step(
            adversarial_rows[:, :obs_dim], clean_obs, gradient[:, :obs_dim], step_size, epsilon
        )
    perturbed[rows] = adversarial_rows
    return perturbed
