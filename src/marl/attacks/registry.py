"""Observation attacks that can be selected by name."""

from typing import Callable, Protocol

from torch import Tensor, nn

from .min_best_fgsm import min_best_fgsm


class ObservationAttack(Protocol):
    """Perturbs the observations of the masked cars."""

    def __call__(
        self,
        q_network: nn.Module,
        actor_obs: Tensor,
        epsilon: float,
        attacked_mask: Tensor,
        obs_dim: int,
    ) -> Tensor:
        """Return the perturbed observations of all cars."""
        ...


def _build_min_best() -> ObservationAttack:
    """MinBest needs no setup, so the function itself is the attack."""
    return min_best_fgsm


ATTACK_BUILDERS: dict[str, Callable[[], ObservationAttack]] = {"minbest": _build_min_best}


def build_attack(name: str) -> ObservationAttack:
    """Build the attack registered under the given name."""
    if name not in ATTACK_BUILDERS:
        raise ValueError(
            f"Unknown attack: {name!r}. Available: {', '.join(sorted(ATTACK_BUILDERS))}"
        )
    return ATTACK_BUILDERS[name]()
