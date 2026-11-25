"""Observation attacks that can be selected by name."""

from typing import Callable, Protocol

from torch import Tensor, nn

from .min_best_fgsm import min_best_fgsm
from .pgd import PgdAttack
from .random_noise import RandomNoiseAttack


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


def _build_min_best(seed: int) -> ObservationAttack:
    """MinBest is deterministic and needs no setup, so the function itself is the attack."""
    return min_best_fgsm


def _build_pgd(seed: int) -> ObservationAttack:
    """PGD draws its random start from a generator seeded with the evaluation seed."""
    return PgdAttack(seed)


def _build_random(seed: int) -> ObservationAttack:
    """The noise comes from a generator seeded with the evaluation seed."""
    return RandomNoiseAttack(seed)


ATTACK_BUILDERS: dict[str, Callable[[int], ObservationAttack]] = {
    "minbest": _build_min_best,
    "pgd": _build_pgd,
    "random": _build_random,
}


def build_attack(name: str, seed: int) -> ObservationAttack:
    """Build the attack registered under the given name, seeded with the evaluation seed."""
    if name not in ATTACK_BUILDERS:
        raise ValueError(
            f"Unknown attack: {name!r}. Available: {', '.join(sorted(ATTACK_BUILDERS))}"
        )
    return ATTACK_BUILDERS[name](seed)
