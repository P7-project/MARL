"""Random observation noise: the baseline that uses no knowledge of the policy."""

import torch
from torch import Tensor, nn


class RandomNoiseAttack:
    """Uniform noise from its own seeded generator, so a run is reproducible from its seed."""

    def __init__(self, seed: int) -> None:
        self._generator: torch.Generator = torch.Generator().manual_seed(seed)

    def __call__(
        self,
        q_network: nn.Module,
        actor_obs: Tensor,
        epsilon: float,
        attacked_mask: Tensor,
        obs_dim: int,
    ) -> Tensor:
        return random_noise(actor_obs, epsilon, attacked_mask, obs_dim, self._generator)


def random_noise(
    actor_obs: Tensor,
    epsilon: float,
    attacked_mask: Tensor,
    obs_dim: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Uniform noise in [-eps, eps] on the first obs_dim features of the attacked cars, clamped to [-1, 1]."""
    clean: Tensor = actor_obs.detach()
    perturbed: Tensor = clean.clone()
    if epsilon == 0 or not bool(attacked_mask.any()):
        return perturbed

    rows: Tensor = attacked_mask.to(device=clean.device, dtype=torch.bool)
    clean_obs: Tensor = clean[rows, :obs_dim]
    noise: Tensor = torch.rand(clean_obs.shape, generator=generator, dtype=clean_obs.dtype)
    perturbed[rows, :obs_dim] = (
        clean_obs + epsilon * (2.0 * noise.to(clean_obs.device) - 1.0)
    ).clamp(-1.0, 1.0)
    return perturbed
