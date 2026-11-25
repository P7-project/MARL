import math
from argparse import Namespace

import pytest
import torch
from auto_LiRPA import BoundedModule
from torch import Tensor, nn

# Adjust to the real module path of ppo.py.
from src.marl.woca_r_rl import ppo
from src.marl.woca_r_rl.ppo import (
    ActionBox,
    PolicyActionBounder,
    SgldRegularizer,
    WocaRLosses,
    WorstActionSearcher,
)

STATE_DIM: int = 4
ACTION_DIM: int = 2
BATCH_SIZE: int = 8
NUM_SAMPLES: int = 256
TOLERANCE: float = 1e-5


# --------------------------------------------------------------------------- #
# PolicyActionBounder
# --------------------------------------------------------------------------- #


def test_action_box_contains_policy_outputs_on_perturbed_states() -> None:
    epsilon = 0.1
    states = _random_states()
    policy = _make_bounded_policy(states)

    box = PolicyActionBounder(policy).action_box(states, epsilon)

    lower = box.lower.detach()
    upper = box.upper.detach()
    with torch.no_grad():
        for perturbed_states in _sample_ball(states, epsilon, NUM_SAMPLES):
            outputs = policy(perturbed_states)
            assert bool((outputs >= lower - TOLERANCE).all())
            assert bool((outputs <= upper + TOLERANCE).all())


def test_action_box_lower_never_exceeds_upper() -> None:
    states = _random_states()

    box = PolicyActionBounder(_make_bounded_policy(states)).action_box(states, 0.1)

    assert bool((box.lower <= box.upper + TOLERANCE).all())


def test_action_box_has_positive_width_for_fractional_epsilon() -> None:
    """Regression test: casting epsilon to int turns 0.05 into 0 and collapses the box."""
    states = _random_states()

    box = PolicyActionBounder(_make_bounded_policy(states)).action_box(states, 0.05)

    assert bool(((box.upper - box.lower) > 0.0).all())


def test_action_box_collapses_to_policy_output_for_zero_epsilon() -> None:
    states = _random_states()
    policy = _make_bounded_policy(states)

    box = PolicyActionBounder(policy).action_box(states, 0.0)

    with torch.no_grad():
        expected = policy(states)
    torch.testing.assert_close(box.lower.detach(), expected, atol=TOLERANCE, rtol=0.0)
    torch.testing.assert_close(box.upper.detach(), expected, atol=TOLERANCE, rtol=0.0)


def test_action_box_width_grows_with_epsilon() -> None:
    states = _random_states()
    bounder = PolicyActionBounder(_make_bounded_policy(states))

    small_box = bounder.action_box(states, 0.02)
    large_box = bounder.action_box(states, 0.2)

    small_width = (small_box.upper - small_box.lower).detach()
    large_width = (large_box.upper - large_box.lower).detach()
    assert bool((large_width >= small_width - TOLERANCE).all())
    assert float(large_width.mean()) > float(small_width.mean())


def test_action_box_gradients_reach_the_policy() -> None:
    states = _random_states()
    policy = _make_bounded_policy(states)

    box = PolicyActionBounder(policy).action_box(states, 0.1)
    (box.upper - box.lower).sum().backward()

    assert _has_nonzero_gradient(policy)


# --------------------------------------------------------------------------- #
# WorstActionSearcher
# --------------------------------------------------------------------------- #


def test_search_result_stays_inside_the_box() -> None:
    states = _random_states()
    box = _fixed_box()
    searcher = WorstActionSearcher(_make_mlp(STATE_DIM + ACTION_DIM, 1))

    actions = searcher.search(states, box)

    assert actions.shape == (BATCH_SIZE, ACTION_DIM)
    assert bool((actions >= box.lower - TOLERANCE).all())
    assert bool((actions <= box.upper + TOLERANCE).all())


def test_search_result_is_detached() -> None:
    searcher = WorstActionSearcher(_make_mlp(STATE_DIM + ACTION_DIM, 1))

    actions = searcher.search(_random_states(), _fixed_box())

    assert not actions.requires_grad


def test_search_does_not_increase_q_compared_to_the_box_center() -> None:
    states = _random_states()
    box = _fixed_box()
    q_network = _make_mlp(STATE_DIM + ACTION_DIM, 1)

    actions = WorstActionSearcher(q_network).search(states, box)

    center = (box.lower + box.upper) / 2
    with torch.no_grad():
        found = q_network(torch.cat([states, actions], dim=1)).squeeze(-1)
        baseline = q_network(torch.cat([states, center], dim=1)).squeeze(-1)
    assert bool((found <= baseline + TOLERANCE).all())


def test_search_reaches_the_lower_corner_when_q_increases_with_the_action() -> None:
    states = _random_states()
    box = _fixed_box()
    q_network = _linear_q(state_weight=0.0, action_weight=1.0)

    actions = WorstActionSearcher(q_network).search(states, box)

    torch.testing.assert_close(actions, box.lower, atol=TOLERANCE, rtol=0.0)


def test_search_reaches_the_upper_corner_when_q_decreases_with_the_action() -> None:
    states = _random_states()
    box = _fixed_box()
    q_network = _linear_q(state_weight=0.0, action_weight=-1.0)

    actions = WorstActionSearcher(q_network).search(states, box)

    torch.testing.assert_close(actions, box.upper, atol=TOLERANCE, rtol=0.0)


def test_search_returns_the_point_for_a_degenerate_box() -> None:
    point = torch.full((BATCH_SIZE, ACTION_DIM), 0.3)
    searcher = WorstActionSearcher(_make_mlp(STATE_DIM + ACTION_DIM, 1))

    actions = searcher.search(_random_states(), ActionBox(lower=point, upper=point))

    torch.testing.assert_close(actions, point, atol=TOLERANCE, rtol=0.0)


# --------------------------------------------------------------------------- #
# SgldRegularizer
# --------------------------------------------------------------------------- #


def test_regularizer_has_one_nonnegative_entry_per_state() -> None:
    states = _random_states()
    regularizer = SgldRegularizer(_make_mlp(STATE_DIM, ACTION_DIM))

    loss = regularizer.per_state_loss(states, 0.1)

    assert loss.shape == (BATCH_SIZE,)
    assert bool((loss >= 0.0).all())


def test_regularizer_is_zero_for_zero_epsilon() -> None:
    states = _random_states()
    regularizer = SgldRegularizer(_make_mlp(STATE_DIM, ACTION_DIM))

    loss = regularizer.per_state_loss(states, 0.0)

    torch.testing.assert_close(loss.detach(), torch.zeros(BATCH_SIZE), atol=TOLERANCE, rtol=0.0)


def test_regularizer_is_zero_for_a_constant_policy() -> None:
    states = _random_states()
    regularizer = SgldRegularizer(_constant_network(STATE_DIM, ACTION_DIM, 0.3))

    loss = regularizer.per_state_loss(states, 0.1)

    torch.testing.assert_close(loss.detach(), torch.zeros(BATCH_SIZE), atol=TOLERANCE, rtol=0.0)


def test_regularizer_is_positive_for_a_state_dependent_policy() -> None:
    states = _random_states()
    regularizer = SgldRegularizer(_make_mlp(STATE_DIM, ACTION_DIM))

    loss = regularizer.per_state_loss(states, 0.1)

    assert float(loss.sum()) > 0.0


def test_regularizer_gradients_reach_the_policy() -> None:
    states = _random_states()
    policy = _make_mlp(STATE_DIM, ACTION_DIM)

    SgldRegularizer(policy).per_state_loss(states, 0.1).sum().backward()

    assert _has_nonzero_gradient(policy)


# --------------------------------------------------------------------------- #
# WocaRLosses.shift_advantages
# --------------------------------------------------------------------------- #


def test_shift_with_zero_kappa_returns_the_advantages() -> None:
    states = _random_states()
    losses = _make_losses(_make_bounded_policy(states), _constant_q(2.0), _ConstantValue(0.0))
    advantages = torch.randn(BATCH_SIZE)

    shifted = losses.shift_advantages(advantages, states, _zero_actions(), 0.0)

    torch.testing.assert_close(shifted, advantages, atol=TOLERANCE, rtol=0.0)


def test_shift_adds_kappa_times_q_at_the_sampled_action() -> None:
    states = _random_states()
    losses = _make_losses(_make_bounded_policy(states), _constant_q(2.0), _ConstantValue(0.0))
    advantages = torch.randn(BATCH_SIZE)

    shifted = losses.shift_advantages(advantages, states, _zero_actions(), 0.5)

    torch.testing.assert_close(shifted, advantages + 1.0, atol=TOLERANCE, rtol=0.0)


def test_shift_does_not_track_gradients() -> None:
    states = _random_states()
    losses = _make_losses(_make_bounded_policy(states), _make_mlp(STATE_DIM + ACTION_DIM, 1), _ConstantValue(0.0))

    shifted = losses.shift_advantages(torch.zeros(BATCH_SIZE), states, _zero_actions(), 0.5)

    assert not shifted.requires_grad


# --------------------------------------------------------------------------- #
# WocaRLosses.critic_loss
# --------------------------------------------------------------------------- #


def test_critic_loss_matches_worst_attack_bellman_backup_with_terminal_masking() -> None:
    discount = 0.9
    value = 1.0
    states = _random_states()
    rewards = torch.linspace(-1.0, 1.0, BATCH_SIZE)
    terminated = (torch.arange(BATCH_SIZE) % 2).float()
    losses = _make_losses(_make_bounded_policy(states), _constant_q(value), _ConstantValue(0.0), discount)

    loss = losses.critic_loss(states, _zero_actions(), rewards, states, states, terminated, 0.05)

    targets = rewards + discount * (1.0 - terminated) * value
    expected = ((value - targets) ** 2).mean()
    torch.testing.assert_close(loss.detach(), expected, atol=TOLERANCE, rtol=0.0)


def test_critic_loss_is_zero_at_the_bellman_fixed_point() -> None:
    discount = 0.9
    value = 2.0
    states = _random_states()
    rewards = torch.full((BATCH_SIZE,), value * (1.0 - discount))
    losses = _make_losses(_make_bounded_policy(states), _constant_q(value), _ConstantValue(0.0), discount)

    loss = losses.critic_loss(
        states, _zero_actions(), rewards, states, states, torch.zeros(BATCH_SIZE), 0.05
    )

    assert float(loss.detach()) == pytest.approx(0.0, abs=TOLERANCE)


def test_critic_loss_ignores_the_next_state_for_terminal_transitions() -> None:
    states = _random_states()
    actions = _zero_actions()
    rewards = torch.linspace(-1.0, 1.0, BATCH_SIZE)
    q_network = _linear_q(state_weight=1.0, action_weight=0.0)
    losses = _make_losses(_make_bounded_policy(states), q_network, _ConstantValue(0.0))

    loss = losses.critic_loss(
        states, actions, rewards, states + 100.0, states + 100.0, torch.ones(BATCH_SIZE), 0.05
    )

    with torch.no_grad():
        predictions = q_network(torch.cat([states, actions], dim=1)).squeeze(-1)
    expected = ((predictions - rewards) ** 2).mean()
    torch.testing.assert_close(loss.detach(), expected, atol=TOLERANCE, rtol=0.0)


def test_critic_loss_gradients_reach_only_the_q_network() -> None:
    states = _random_states()
    policy = _make_bounded_policy(states)
    q_network = _make_mlp(STATE_DIM + ACTION_DIM, 1)
    losses = _make_losses(policy, q_network, _ConstantValue(0.0))

    loss = losses.critic_loss(
        states, _zero_actions(), torch.ones(BATCH_SIZE), states, states, torch.zeros(BATCH_SIZE), 0.05
    )
    loss.backward()

    assert _has_nonzero_gradient(q_network)
    assert not _has_any_gradient(policy)


# --------------------------------------------------------------------------- #
# WocaRLosses.regularizer_loss
# --------------------------------------------------------------------------- #


def test_regularizer_loss_is_zero_when_value_is_below_the_worst_case_q() -> None:
    """w(s) is clamped at 0, so states where an attack costs nothing are not penalized."""
    states = _random_states()
    losses = _make_losses(_make_bounded_policy(states), _constant_q(1.0), _ConstantValue(0.0))

    loss = losses.regularizer_loss(states, states, 0.1)

    assert float(loss.detach()) == pytest.approx(0.0, abs=TOLERANCE)


def test_regularizer_loss_is_zero_for_zero_epsilon() -> None:
    states = _random_states()
    losses = _make_losses(_make_bounded_policy(states), _constant_q(1.0), _ConstantValue(3.0))

    loss = losses.regularizer_loss(states, states, 0.0)

    assert float(loss.detach()) == pytest.approx(0.0, abs=TOLERANCE)


def test_regularizer_loss_is_positive_when_value_exceeds_the_worst_case_q() -> None:
    states = _random_states()
    losses = _make_losses(_make_bounded_policy(states), _constant_q(1.0), _ConstantValue(3.0))

    loss = losses.regularizer_loss(states, states, 0.1)

    assert float(loss.detach()) > 0.0


def test_regularizer_loss_gradients_reach_the_policy_and_not_the_q_network() -> None:
    states = _random_states()
    policy = _make_bounded_policy(states)
    q_network = _make_mlp(STATE_DIM + ACTION_DIM, 1)
    losses = _make_losses(policy, q_network, _ConstantValue(100.0))

    losses.regularizer_loss(states, states, 0.1).backward()

    assert _has_nonzero_gradient(policy)
    assert not _has_any_gradient(q_network)


# --------------------------------------------------------------------------- #
# Algorithm 1 behaviour: w(s), Dist, gradients, robust advantage, update order
# --------------------------------------------------------------------------- #


def test_state_importance_minimizes_q_over_the_action_space_not_the_ibp_box() -> None:
    """
    pi(s) = 0.3 everywhere, so its IBP box is the point (0.3, 0.3) for any epsilon.
    Q(s, a) = a_1 + a_2 and the action space is [-1, 1] x [-2, 2], so
    min_a Q = -3 and w = V - (-3) = 1 + 3 = 4. Over the box it would be 1 - 0.6 = 0.4.
    """
    states = _random_states()
    policy = BoundedModule(_constant_network(STATE_DIM, ACTION_DIM, 0.3), torch.zeros_like(states))
    q_network = _linear_q(state_weight=0.0, action_weight=1.0)
    losses = WocaRLosses(
        value_network=_ConstantValue(1.0),
        q_network=q_network,
        bounder=PolicyActionBounder(policy),
        searcher=WorstActionSearcher(q_network),
        regularizer=SgldRegularizer(policy),
        discount_factor=0.9,
        action_lower=torch.tensor([-1.0, -2.0]),
        action_upper=torch.tensor([1.0, 2.0]),
    )

    weights = losses.state_importance(states)

    torch.testing.assert_close(weights, torch.full((BATCH_SIZE,), 4.0), atol=TOLERANCE, rtol=0.0)


def test_regularizer_distance_divides_by_sigma_squared() -> None:
    """
    mu(s) = (2 s, s) with a 1-d state, sigma = (0.5, 2), epsilon = 0.1.
    The distance grows with |s~ - s|, so SGLD ends on the ball edge |s~ - s| = 0.1:
    Dist = (2 * 0.1)^2 / 0.5^2 + (0.1)^2 / 2^2 = 0.16 + 0.0025 = 0.1625.
    """
    mean_network = nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        mean_network.weight.copy_(torch.tensor([[2.0], [1.0]]))
    log_std = nn.Parameter(torch.log(torch.tensor([0.5, 2.0])))
    states = torch.tensor([[1.0], [-0.5], [0.0]])

    loss = SgldRegularizer(mean_network, log_std=log_std).per_state_loss(states, 0.1)

    torch.testing.assert_close(loss.detach(), torch.full((3,), 0.1625), atol=TOLERANCE, rtol=0.0)


def test_regularizer_gradient_flows_only_through_the_perturbed_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    mu(s) = w s with w = 2, sigma = 0.5, s = 1 and s~ fixed to 1.1.
    With mu(s) and sigma detached, L = (c - w s~)^2 / sigma^2 with c = w s = 2, so
    dL/dw = 2 (c - w s~)(-s~) / sigma^2 = 2 * (-0.2) * (-1.1) / 0.25 = 1.76,
    and log_std gets no gradient. (Without detaching mu(s), dL/dw would be 0.16.)
    """
    mean_network = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        mean_network.weight.fill_(2.0)
    log_std = nn.Parameter(torch.log(torch.tensor([0.5])))
    regularizer = SgldRegularizer(mean_network, log_std=log_std)
    monkeypatch.setattr(
        regularizer, "_find_perturbation", lambda states, epsilon: (states + 0.1).detach()
    )

    regularizer.per_state_loss(torch.tensor([[1.0]]), 0.1).sum().backward()

    assert log_std.grad is None or float(log_std.grad.abs().sum()) == 0.0
    assert mean_network.weight.grad is not None
    assert float(mean_network.weight.grad) == pytest.approx(1.76, abs=TOLERANCE)


def test_robust_advantage_is_the_normalized_sum() -> None:
    """
    A_hat = (1, -1, 1, -1), Q_phi = (2, 2, -2, -2), kappa_wst = 0.5:
    A_hat + kappa Q = (2, 0, 0, -2), mean 0, unbiased std sqrt(8 / 3),
    so the result is (sqrt(3/2), 0, 0, -sqrt(3/2)).
    """
    states = torch.zeros(4, STATE_DIM)
    states[:, 0] = torch.tensor([2.0, 2.0, -2.0, -2.0])  # Q(s, a) = sum(s)
    actions = torch.zeros(4, ACTION_DIM)
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])
    losses = _make_losses(
        _make_bounded_policy(states), _linear_q(state_weight=1.0, action_weight=0.0), _ConstantValue(0.0)
    )

    robust = ppo._robust_advantages(losses, advantages, states, actions, 0.5)

    root = math.sqrt(1.5)
    torch.testing.assert_close(robust, torch.tensor([root, 0.0, 0.0, -root]), atol=1e-4, rtol=0.0)


def test_update_fits_value_then_worst_attack_critic_then_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fit_value(*args: object, **kwargs: object) -> float:
        calls.append("value")
        return 0.0

    def fit_q(*args: object, **kwargs: object) -> float:
        calls.append("q")
        return 0.0

    def fit_policy(*args: object, **kwargs: object) -> dict[str, float]:
        calls.append("policy")
        return {"actor": 0.0, "reg": 0.0, "entropy": 0.0}

    monkeypatch.setattr(ppo, "_rollout_tensors", lambda *args, **kwargs: None)
    monkeypatch.setattr(ppo, "_fit_value_function", fit_value)
    monkeypatch.setattr(ppo, "_fit_worst_attack_critic", fit_q)
    monkeypatch.setattr(ppo, "_fit_policy", fit_policy)

    ppo.ppo_update_wocar(
        None, None, None, None, None, None,  # type: ignore[arg-type]
        0.1, 0.5, 0.1, {}, 0.0, Namespace(), torch.device("cpu"),
    )

    assert calls == ["value", "q", "policy"]


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _make_losses(
    policy: BoundedModule,
    q_network: nn.Module,
    value_network: nn.Module,
    discount: float = 0.9,
) -> WocaRLosses:
    return WocaRLosses(
        value_network=value_network,
        q_network=q_network,
        bounder=PolicyActionBounder(policy),
        searcher=WorstActionSearcher(q_network),
        regularizer=SgldRegularizer(policy),
        discount_factor=discount,
    )


def _make_bounded_policy(states: Tensor) -> BoundedModule:
    return BoundedModule(_make_mlp(STATE_DIM, ACTION_DIM), torch.zeros_like(states))


def _make_mlp(input_dim: int, output_dim: int) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.Tanh(),
        nn.Linear(16, output_dim),
    )


def _constant_q(value: float) -> nn.Module:
    return _constant_network(STATE_DIM + ACTION_DIM, 1, value)


def _constant_network(input_dim: int, output_dim: int, value: float) -> nn.Module:
    layer = nn.Linear(input_dim, output_dim)
    nn.init.zeros_(layer.weight)
    nn.init.constant_(layer.bias, value)
    return nn.Sequential(layer)


def _linear_q(state_weight: float, action_weight: float) -> nn.Module:
    """Q(s, a) = state_weight * sum(s) + action_weight * sum(a)."""
    layer = nn.Linear(STATE_DIM + ACTION_DIM, 1)
    with torch.no_grad():
        layer.weight[:, :STATE_DIM] = state_weight
        layer.weight[:, STATE_DIM:] = action_weight
        layer.bias.zero_()
    return nn.Sequential(layer)


class _ConstantValue(nn.Module):
    """Stands in for the critic V: returns a constant per state, shape [batch]."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self._value: float = value

    def forward(self, states: Tensor) -> Tensor:
        return torch.full((states.shape[0],), self._value)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #


def _random_states() -> Tensor:
    torch.manual_seed(1)
    return torch.rand(BATCH_SIZE, STATE_DIM) * 2.0 - 1.0


def _zero_actions() -> Tensor:
    return torch.zeros(BATCH_SIZE, ACTION_DIM)


def _fixed_box() -> ActionBox:
    lower = torch.full((BATCH_SIZE, ACTION_DIM), -0.5)
    upper = torch.full((BATCH_SIZE, ACTION_DIM), 0.5)
    return ActionBox(lower=lower, upper=upper)


def _sample_ball(states: Tensor, epsilon: float, num_samples: int) -> Tensor:
    """[num_samples, batch, state_dim]: uniform samples with ||s~ - s||_inf <= epsilon."""
    noise = (torch.rand(num_samples, *states.shape) * 2.0 - 1.0) * epsilon
    return states.unsqueeze(0) + noise


def _has_nonzero_gradient(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in module.parameters()
    )


def _has_any_gradient(module: nn.Module) -> bool:
    return any(parameter.grad is not None for parameter in module.parameters())
