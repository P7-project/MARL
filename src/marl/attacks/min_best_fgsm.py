"""MinBest observation attack (FGSM).
Huang et al. 2017: https://arxiv.org/abs/1702.02284
"""

import torch
from torch import Tensor, nn


def min_best_loss(q_values: Tensor, best_actions: Tensor) -> Tensor:
    """Huang et al. 2017, Sec. 4.1."""
    log_probs: Tensor = torch.log_softmax(q_values, dim=-1)
    return -log_probs.gather(1, best_actions.unsqueeze(1)).squeeze(1)


def min_best_gradient(q_network: nn.Module, actor_obs: Tensor) -> Tensor:
    """Huang et al. 2017, Sec. 4.1."""
    with torch.enable_grad():
        obs_input: Tensor = actor_obs.detach().clone().requires_grad_(True)
        q_values: Tensor = q_network(obs_input)
        best_actions: Tensor = q_values.detach().argmax(dim=-1)
        loss: Tensor = min_best_loss(q_values, best_actions).sum()
        (gradient,) = torch.autograd.grad(loss, obs_input)
    return gradient


def fgsm_step(obs: Tensor, gradient: Tensor, epsilon: float) -> Tensor:
    """Huang et al. 2017, Eq. 2 (Sec. 3.1) and Eq. 3 L-inf case (Sec. 4.2)."""
    return (obs + epsilon * gradient.sign()).clamp(-1.0, 1.0)


def min_best_fgsm(
    q_network: nn.Module,
    actor_obs: Tensor,
    epsilon: float,
    attacked_mask: Tensor,
    obs_dim: int,
) -> Tensor:
    """WocaR-RL, Appendix D.1.2."""
    clean: Tensor = actor_obs.detach()
    perturbed: Tensor = clean.clone()
    if epsilon == 0 or not bool(attacked_mask.any()):
        return perturbed

    rows: Tensor = attacked_mask.to(device=clean.device, dtype=torch.bool)
    gradient: Tensor = min_best_gradient(q_network, clean)
    perturbed[rows, :obs_dim] = fgsm_step(
        clean[rows, :obs_dim], gradient[rows, :obs_dim], epsilon
    )
    return perturbed
