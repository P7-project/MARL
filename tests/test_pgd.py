import pytest
import torch
from auto_LiRPA import BoundedModule
from torch import Tensor, nn

from src.marl.attacks.min_best_fgsm import min_best_fgsm
from src.marl.attacks.pgd import PGD_STEPS, PgdAttack, pgd, pgd_step
from src.marl.common import mlp

N_AGENTS: int = 3
HIDDEN: int = 32
EPSILON: float = 0.05
ATOL: float = 1e-6


def _build_robust_q(input_dim: int, n_actions: int) -> BoundedModule:
    """Same construction as _build_robust_q in src/marl/algorithms/wocar_dqn/networks.py."""
    return BoundedModule(
        mlp(input_dim, n_actions, HIDDEN, out_gain=1.0),
        torch.zeros(1, input_dim),
        device="cpu",
    )


@pytest.fixture(params=[(25, 5), (105, 3)], ids=["highway", "intersection"])
def setup(request) -> tuple[BoundedModule, Tensor, int]:
    obs_dim, n_actions = request.param
    torch.manual_seed(0)
    q_network: BoundedModule = _build_robust_q(obs_dim + N_AGENTS, n_actions)
    obs: Tensor = torch.empty(N_AGENTS, obs_dim).uniform_(-0.9, 0.9)
    obs[0, :3] = torch.tensor([1.0, -1.0, 0.99])  # on and near the bounds, so the clamp is exercised
    actor_obs: Tensor = torch.cat([obs, torch.eye(N_AGENTS)], dim=-1)
    return q_network, actor_obs, obs_dim


def _generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_perturbation_bounded_and_clipped(setup) -> None:
    q_network, actor_obs, obs_dim = setup
    mask: Tensor = torch.ones(N_AGENTS, dtype=torch.bool)
    epsilon: float = 0.3
    perturbed: Tensor = pgd(q_network, actor_obs, epsilon, mask, obs_dim, _generator())
    assert (perturbed - actor_obs).abs().max() <= epsilon + ATOL
    assert perturbed[:, :obs_dim].min() >= -1.0
    assert perturbed[:, :obs_dim].max() <= 1.0
    assert not torch.equal(perturbed[:, :obs_dim], actor_obs[:, :obs_dim])


def test_agent_ids_and_unattacked_rows_unchanged(setup) -> None:
    q_network, actor_obs, obs_dim = setup
    mask: Tensor = torch.tensor([True, False, True])
    perturbed: Tensor = pgd(q_network, actor_obs, EPSILON, mask, obs_dim, _generator())
    assert torch.equal(perturbed[:, obs_dim:], actor_obs[:, obs_dim:])
    assert torch.equal(perturbed[1], actor_obs[1])
    assert not torch.equal(perturbed[0, :obs_dim], actor_obs[0, :obs_dim])
    assert not torch.equal(perturbed[2, :obs_dim], actor_obs[2, :obs_dim])


def test_zero_epsilon_and_empty_mask_return_input(setup) -> None:
    q_network, actor_obs, obs_dim = setup
    everyone: Tensor = torch.ones(N_AGENTS, dtype=torch.bool)
    nobody: Tensor = torch.zeros(N_AGENTS, dtype=torch.bool)
    assert torch.equal(pgd(q_network, actor_obs, 0.0, everyone, obs_dim, _generator()), actor_obs)
    assert torch.equal(pgd(q_network, actor_obs, EPSILON, nobody, obs_dim, _generator()), actor_obs)


def test_same_seed_gives_the_same_perturbation(setup) -> None:
    q_network, actor_obs, obs_dim = setup
    mask: Tensor = torch.ones(N_AGENTS, dtype=torch.bool)
    first: Tensor = PgdAttack(seed=3)(q_network, actor_obs, EPSILON, mask, obs_dim)
    assert torch.equal(first, PgdAttack(seed=3)(q_network, actor_obs, EPSILON, mask, obs_dim))
    assert not torch.equal(first, PgdAttack(seed=4)(q_network, actor_obs, EPSILON, mask, obs_dim))


def test_works_under_no_grad_without_param_grads(setup) -> None:
    q_network, actor_obs, obs_dim = setup
    mask: Tensor = torch.ones(N_AGENTS, dtype=torch.bool)
    with torch.no_grad():
        perturbed: Tensor = pgd(q_network, actor_obs, EPSILON, mask, obs_dim, _generator())
    assert not perturbed.requires_grad
    assert all(parameter.grad is None for parameter in q_network.parameters())


def test_pgd_step_on_hand_written_numbers() -> None:
    clean: Tensor = torch.tensor([[0.0, 0.0, 0.95, 0.0]])
    adversarial: Tensor = torch.tensor([[0.08, -0.08, 0.95, 0.0]])
    gradient: Tensor = torch.tensor([[1.0, -2.0, 3.0, 0.0]])
    stepped: Tensor = pgd_step(adversarial, clean, gradient, step_size=0.05, epsilon=0.1)
    # projected back to the eps-ball, clamped to 1, and unmoved where the gradient is 0
    assert torch.allclose(stepped, torch.tensor([[0.1, -0.1, 1.0, 0.0]]))


def _two_peak_network(epsilon: float, n_agents: int) -> nn.Sequential:
    """Q = [0, f(x)] on the first feature x, f piecewise linear with slopes +-1.

    f(x) = |x + eps/10| - 2 relu(x - eps/2) - 2 relu(-x - eps/2) - eps/5: peaks at
    x = +-eps/2 (f >= 0.1 eps within eps/10 of them, action 1 wins), f(0) = f(eps) = -0.1 eps.
    At x = 0 f rises, so one signed step goes to x = eps, where action 0 is still best.
    From any start in [-eps, eps], ascent reaches a peak within eps/2.
    """
    input_dim: int = 1 + n_agents
    hidden = nn.Linear(input_dim, 4)
    output = nn.Linear(4, 2)
    with torch.no_grad():
        hidden.weight.zero_()
        hidden.weight[:, 0] = torch.tensor([1.0, -1.0, 1.0, -1.0])
        hidden.bias.copy_(torch.tensor([epsilon / 10, -epsilon / 10, -epsilon / 2, -epsilon / 2]))
        output.weight.copy_(torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, -2.0, -2.0]]))
        output.bias.copy_(torch.tensor([0.0, -epsilon / 5]))
    return nn.Sequential(hidden, nn.ReLU(), output)


@pytest.mark.parametrize("seed", range(20))
def test_pgd_changes_an_action_one_signed_step_cannot(seed) -> None:
    epsilon: float = 0.5
    network: nn.Sequential = _two_peak_network(epsilon, N_AGENTS)
    actor_obs: Tensor = torch.cat([torch.zeros(N_AGENTS, 1), torch.eye(N_AGENTS)], dim=-1)
    mask: Tensor = torch.tensor([True, False, True])
    clean_actions: Tensor = network(actor_obs).argmax(dim=1)
    assert clean_actions.tolist() == [0, 0, 0]

    one_step: Tensor = min_best_fgsm(network, actor_obs, epsilon, mask, 1)
    assert torch.allclose(one_step[mask, 0], torch.full((2,), epsilon))
    assert torch.equal(network(one_step).argmax(dim=1), clean_actions)

    perturbed: Tensor = pgd(network, actor_obs, epsilon, mask, 1, _generator(seed))
    assert network(perturbed).argmax(dim=1).tolist() == [1, 0, 1]
    distance_to_peak: Tensor = (perturbed[mask, 0].abs() - epsilon / 2).abs()
    assert distance_to_peak.max() <= epsilon / PGD_STEPS + ATOL
