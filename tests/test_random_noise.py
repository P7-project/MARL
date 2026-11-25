import torch
from torch import Tensor, nn

from src.marl.attacks.random_noise import RandomNoiseAttack, random_noise

N_AGENTS: int = 3
OBS_DIM: int = 25
EPSILON: float = 0.3
ATOL: float = 1e-6


def _actor_obs() -> Tensor:
    torch.manual_seed(0)
    obs: Tensor = torch.empty(N_AGENTS, OBS_DIM).uniform_(-0.9, 0.9)
    obs[0, :2] = torch.tensor([1.0, -1.0])  # on the bounds, so the clamp is exercised
    return torch.cat([obs, torch.eye(N_AGENTS)], dim=-1)


def _generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_noise_bounded_and_clipped() -> None:
    actor_obs: Tensor = _actor_obs()
    mask: Tensor = torch.ones(N_AGENTS, dtype=torch.bool)
    perturbed: Tensor = random_noise(actor_obs, EPSILON, mask, OBS_DIM, _generator())
    assert (perturbed - actor_obs).abs().max() <= EPSILON + ATOL
    assert perturbed[:, :OBS_DIM].min() >= -1.0
    assert perturbed[:, :OBS_DIM].max() <= 1.0
    assert (perturbed[:, :OBS_DIM] != actor_obs[:, :OBS_DIM]).float().mean() > 0.9


def test_agent_ids_and_unattacked_rows_unchanged() -> None:
    actor_obs: Tensor = _actor_obs()
    mask: Tensor = torch.tensor([False, True, False])
    perturbed: Tensor = random_noise(actor_obs, EPSILON, mask, OBS_DIM, _generator())
    assert torch.equal(perturbed[:, OBS_DIM:], actor_obs[:, OBS_DIM:])
    assert torch.equal(perturbed[0], actor_obs[0])
    assert torch.equal(perturbed[2], actor_obs[2])
    assert not torch.equal(perturbed[1, :OBS_DIM], actor_obs[1, :OBS_DIM])


def test_zero_epsilon_and_empty_mask_return_input() -> None:
    actor_obs: Tensor = _actor_obs()
    everyone: Tensor = torch.ones(N_AGENTS, dtype=torch.bool)
    nobody: Tensor = torch.zeros(N_AGENTS, dtype=torch.bool)
    assert torch.equal(random_noise(actor_obs, 0.0, everyone, OBS_DIM, _generator()), actor_obs)
    assert torch.equal(random_noise(actor_obs, EPSILON, nobody, OBS_DIM, _generator()), actor_obs)


def test_same_seed_gives_the_same_noise() -> None:
    actor_obs: Tensor = _actor_obs()
    mask: Tensor = torch.ones(N_AGENTS, dtype=torch.bool)
    network: nn.Module = nn.Identity()  # the noise ignores the network
    first: Tensor = RandomNoiseAttack(seed=3)(network, actor_obs, EPSILON, mask, OBS_DIM)
    assert torch.equal(first, RandomNoiseAttack(seed=3)(network, actor_obs, EPSILON, mask, OBS_DIM))
    assert not torch.equal(first, RandomNoiseAttack(seed=4)(network, actor_obs, EPSILON, mask, OBS_DIM))


def test_global_rng_is_not_used() -> None:
    actor_obs: Tensor = _actor_obs()
    mask: Tensor = torch.ones(N_AGENTS, dtype=torch.bool)
    state: Tensor = torch.get_rng_state()
    random_noise(actor_obs, EPSILON, mask, OBS_DIM, _generator())
    assert torch.equal(torch.get_rng_state(), state)
