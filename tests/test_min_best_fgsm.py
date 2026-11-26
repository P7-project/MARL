import pytest
import torch
import torch.nn.functional as F
from auto_LiRPA import BoundedModule
from torch import Tensor

from src.marl.attacks.min_best_fgsm import (
    fgsm_step,
    min_best_fgsm,
    min_best_gradient,
    min_best_loss,
)
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
    actor_obs: Tensor = torch.cat([obs, torch.eye(N_AGENTS)], dim=-1)
    return q_network, actor_obs, obs_dim


def test_perturbation_bounded_and_clipped(setup) -> None:
    q_network, actor_obs, obs_dim = setup
    edge_obs: Tensor = actor_obs.clone()
    edge_obs[:, :2] = torch.tensor([1.0, -1.0])  # values on the clip boundary
    mask: Tensor = torch.ones(N_AGENTS, dtype=torch.bool)
    for clean in (actor_obs, edge_obs):
        attacked: Tensor = min_best_fgsm(q_network, clean, EPSILON, mask, obs_dim)
        delta: Tensor = (attacked[:, :obs_dim] - clean[:, :obs_dim]).abs()
        assert delta.max() <= EPSILON + ATOL
        assert attacked[:, :obs_dim].min() >= -1.0
        assert attacked[:, :obs_dim].max() <= 1.0
        assert delta.max() > 0


def test_agent_ids_and_unattacked_rows_unchanged(setup) -> None:
    q_network, actor_obs, obs_dim = setup
    original: Tensor = actor_obs.clone()
    mask: Tensor = torch.tensor([True, False, True])
    attacked: Tensor = min_best_fgsm(q_network, actor_obs, EPSILON, mask, obs_dim)
    assert torch.equal(attacked[:, obs_dim:], actor_obs[:, obs_dim:])
    assert torch.equal(attacked[1], actor_obs[1])
    assert not torch.equal(attacked[0], actor_obs[0])
    assert torch.equal(actor_obs, original)  # input not modified in place


def test_zero_epsilon_and_empty_mask_return_input(setup) -> None:
    q_network, actor_obs, obs_dim = setup
    all_true: Tensor = torch.ones(N_AGENTS, dtype=torch.bool)
    all_false: Tensor = torch.zeros(N_AGENTS, dtype=torch.bool)
    zero_eps: Tensor = min_best_fgsm(q_network, actor_obs, 0.0, all_true, obs_dim)
    no_agents: Tensor = min_best_fgsm(q_network, actor_obs, EPSILON, all_false, obs_dim)
    assert torch.equal(zero_eps, actor_obs)
    assert torch.equal(no_agents, actor_obs)
    assert zero_eps is not actor_obs and no_agents is not actor_obs


def test_small_epsilon_lowers_best_action_probability(setup) -> None:
    q_network, actor_obs, obs_dim = setup
    mask: Tensor = torch.ones(N_AGENTS, dtype=torch.bool)
    attacked: Tensor = min_best_fgsm(q_network, actor_obs, 1e-3, mask, obs_dim)
    with torch.no_grad():
        clean_q: Tensor = q_network(actor_obs).double()
        attacked_q: Tensor = q_network(attacked).double()
    best: Tensor = clean_q.argmax(dim=-1, keepdim=True)
    clean_p: Tensor = torch.softmax(clean_q, dim=-1).gather(1, best).squeeze(1)
    attacked_p: Tensor = torch.softmax(attacked_q, dim=-1).gather(1, best).squeeze(1)
    assert (attacked_p < clean_p).all()


def test_works_under_no_grad_without_param_grads(setup) -> None:
    q_network, actor_obs, obs_dim = setup
    mask: Tensor = torch.ones(N_AGENTS, dtype=torch.bool)
    with torch.no_grad():
        attacked: Tensor = min_best_fgsm(q_network, actor_obs, EPSILON, mask, obs_dim)
    assert not torch.equal(attacked, actor_obs)
    assert not attacked.requires_grad
    assert all(param.grad is None for param in q_network.parameters())


def test_loss_matches_cross_entropy(setup) -> None:
    q_network, actor_obs, _ = setup
    with torch.no_grad():
        q_values: Tensor = q_network(actor_obs)
    best_actions: Tensor = q_values.argmax(dim=-1)
    loss: Tensor = min_best_loss(q_values, best_actions)
    expected: Tensor = F.cross_entropy(q_values, best_actions, reduction="none")
    assert loss.shape == (N_AGENTS,)
    assert torch.allclose(loss, expected, atol=ATOL)


def test_matches_independent_per_agent_cross_entropy(setup) -> None:
    q_network, actor_obs, obs_dim = setup
    mask: Tensor = torch.ones(N_AGENTS, dtype=torch.bool)
    expected: Tensor = actor_obs.clone()
    for i in range(N_AGENTS):
        row: Tensor = actor_obs[i : i + 1].clone().requires_grad_(True)
        q_values: Tensor = q_network(row)
        loss: Tensor = F.cross_entropy(q_values, q_values.detach().argmax(dim=-1))
        (grad,) = torch.autograd.grad(loss, row)
        step: Tensor = EPSILON * grad[0, :obs_dim].sign()
        expected[i, :obs_dim] = (actor_obs[i, :obs_dim] + step).clamp(-1.0, 1.0)
    attacked: Tensor = min_best_fgsm(q_network, actor_obs, EPSILON, mask, obs_dim)
    assert torch.allclose(attacked, expected, atol=ATOL)


def test_fgsm_step_on_hand_written_numbers() -> None:
    obs: Tensor = torch.tensor([[0.0, 0.5, -0.3, 0.95, -0.95]])
    gradient: Tensor = torch.tensor([[2.0, -0.001, 0.0, 5.0, -5.0]])
    expected: Tensor = torch.tensor([[0.1, 0.4, -0.3, 1.0, -1.0]])
    assert torch.allclose(fgsm_step(obs, gradient, 0.1), expected, atol=ATOL)


def test_gradient_shape_under_no_grad_without_param_grads(setup) -> None:
    q_network, actor_obs, _ = setup
    with torch.no_grad():
        gradient: Tensor = min_best_gradient(q_network, actor_obs)
    assert gradient.shape == actor_obs.shape
    assert gradient.abs().max() > 0
    assert not gradient.requires_grad
    assert all(param.grad is None for param in q_network.parameters())
