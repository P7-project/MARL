import pytest
import torch
from torch import Tensor, nn

from auto_LiRPA import BoundedModule

# Adjust to the real module name of the file containing the classes.
from src.marl.woca_r_rl.ppo import (
    ActionBox,
    PolicyActionBounder,
    WorstAttackPPOLoss,
    WorstAttackQTrainer,
    WorstAttackValueEstimator,
)

STATE_DIM: int = 4
ACTION_DIM: int = 2
BATCH_SIZE: int = 8
NUM_SAMPLES: int = 256
TOLERANCE: float = 1e-5


# --------------------------------------------------------------------------- #
# PolicyActionBounder
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["IBP", "CROWN"])
def test_action_box_contains_policy_outputs_on_perturbed_states(method: str) -> None:
    epsilon = 0.1
    states = _random_states()
    policy = _make_mlp(STATE_DIM, ACTION_DIM)
    bounder = _make_bounder(policy, states, epsilon, method)

    box = bounder.action_box(states)

    lower = box.lower.detach()
    upper = box.upper.detach()
    with torch.no_grad():
        for perturbed_states in _sample_ball(states, epsilon, NUM_SAMPLES):
            outputs = policy(perturbed_states)
            assert bool((outputs >= lower - TOLERANCE).all())
            assert bool((outputs <= upper + TOLERANCE).all())


@pytest.mark.parametrize("method", ["IBP", "CROWN"])
def test_action_box_lower_never_exceeds_upper(method: str) -> None:
    states = _random_states()
    policy = _make_mlp(STATE_DIM, ACTION_DIM)
    bounder = _make_bounder(policy, states, 0.1, method)

    box = bounder.action_box(states)

    assert bool((box.lower <= box.upper + TOLERANCE).all())


def test_action_box_has_positive_width_for_fractional_epsilon() -> None:
    """Regression test: casting epsilon to int turns 0.05 into 0 and collapses the box."""
    states = _random_states()
    policy = _make_mlp(STATE_DIM, ACTION_DIM)
    bounder = _make_bounder(policy, states, 0.05, "IBP")

    box = bounder.action_box(states)

    assert bool(((box.upper - box.lower) > 0.0).all())


def test_action_box_collapses_to_policy_output_for_zero_epsilon() -> None:
    states = _random_states()
    policy = _make_mlp(STATE_DIM, ACTION_DIM)
    bounder = _make_bounder(policy, states, 0.0, "IBP")

    box = bounder.action_box(states)

    with torch.no_grad():
        expected = policy(states)
    torch.testing.assert_close(box.lower.detach(), expected, atol=TOLERANCE, rtol=0.0)
    torch.testing.assert_close(box.upper.detach(), expected, atol=TOLERANCE, rtol=0.0)


def test_action_box_width_grows_with_epsilon() -> None:
    states = _random_states()
    policy = _make_mlp(STATE_DIM, ACTION_DIM)
    small_box = _make_bounder(policy, states, 0.02, "IBP").action_box(states)
    large_box = _make_bounder(policy, states, 0.2, "IBP").action_box(states)

    small_width = (small_box.upper - small_box.lower).detach()
    large_width = (large_box.upper - large_box.lower).detach()

    assert bool((large_width >= small_width - TOLERANCE).all())
    assert float(large_width.mean()) > float(small_width.mean())


# --------------------------------------------------------------------------- #
# WorstAttackValueEstimator
# --------------------------------------------------------------------------- #


def test_value_lower_bound_has_one_entry_per_state() -> None:
    states = _random_states()
    estimator = _make_estimator(_make_mlp(STATE_DIM + ACTION_DIM, 1), states)

    lower_bound = estimator.lower_bound_over_box(states, _fixed_box())

    assert lower_bound.shape == (BATCH_SIZE,)


def test_value_lower_bound_is_below_q_values_inside_box() -> None:
    states = _random_states()
    q_network = _make_mlp(STATE_DIM + ACTION_DIM, 1)
    estimator = _make_estimator(q_network, states)
    box = _fixed_box()

    lower_bound = estimator.lower_bound_over_box(states, box).detach()

    with torch.no_grad():
        for actions in _sample_box(box, NUM_SAMPLES):
            values = q_network(torch.cat([states, actions], dim=1)).squeeze(-1)
            assert bool((values >= lower_bound - TOLERANCE).all())


def test_value_lower_bound_equals_q_value_for_degenerate_box() -> None:
    states = _random_states()
    q_network = _make_mlp(STATE_DIM + ACTION_DIM, 1)
    estimator = _make_estimator(q_network, states)
    actions = torch.full((BATCH_SIZE, ACTION_DIM), 0.3)
    box = ActionBox(lower=actions, upper=actions)

    lower_bound = estimator.lower_bound_over_box(states, box).detach()

    with torch.no_grad():
        expected = q_network(torch.cat([states, actions], dim=1)).squeeze(-1)
    torch.testing.assert_close(lower_bound, expected, atol=TOLERANCE, rtol=0.0)


def test_value_lower_bound_of_constant_q_is_that_constant() -> None:
    states = _random_states()
    estimator = _make_estimator(_constant_q(1.5), states)

    lower_bound = estimator.lower_bound_over_box(states, _fixed_box()).detach()

    torch.testing.assert_close(
        lower_bound, torch.full((BATCH_SIZE,), 1.5), atol=TOLERANCE, rtol=0.0
    )


# --------------------------------------------------------------------------- #
# WorstAttackQTrainer
# --------------------------------------------------------------------------- #


def test_loss_matches_worst_attack_bellman_backup_with_terminal_masking() -> None:
    discount = 0.9
    value = 1.0
    states = _random_states()
    actions = torch.zeros(BATCH_SIZE, ACTION_DIM)
    rewards = torch.linspace(-1.0, 1.0, BATCH_SIZE)
    terminated = (torch.arange(BATCH_SIZE) % 2).float()
    trainer = _make_trainer(_constant_q(value), _constant_q(value), states, discount)

    loss = trainer.loss(states, actions, rewards, states, terminated)

    targets = rewards + discount * (1.0 - terminated) * value
    expected = ((value - targets) ** 2).mean()
    torch.testing.assert_close(loss.detach(), expected, atol=TOLERANCE, rtol=0.0)


def test_loss_is_zero_at_the_bellman_fixed_point() -> None:
    discount = 0.9
    value = 2.0
    states = _random_states()
    actions = torch.zeros(BATCH_SIZE, ACTION_DIM)
    rewards = torch.full((BATCH_SIZE,), value * (1.0 - discount))
    terminated = torch.zeros(BATCH_SIZE)
    trainer = _make_trainer(_constant_q(value), _constant_q(value), states, discount)

    loss = trainer.loss(states, actions, rewards, states, terminated)

    assert float(loss.detach()) == pytest.approx(0.0, abs=TOLERANCE)


def test_terminal_transitions_ignore_the_next_state_value() -> None:
    discount = 0.9
    states = _random_states()
    actions = torch.zeros(BATCH_SIZE, ACTION_DIM)
    rewards = torch.zeros(BATCH_SIZE)
    terminated = torch.ones(BATCH_SIZE)
    low_target = _make_trainer(_constant_q(0.0), _constant_q(-100.0), states, discount)
    high_target = _make_trainer(_constant_q(0.0), _constant_q(100.0), states, discount)

    low_loss = low_target.loss(states, actions, rewards, states, terminated)
    high_loss = high_target.loss(states, actions, rewards, states, terminated)

    torch.testing.assert_close(low_loss.detach(), high_loss.detach())


def test_loss_gradients_reach_only_the_online_q_network() -> None:
    states = _random_states()
    actions = torch.zeros(BATCH_SIZE, ACTION_DIM)
    rewards = torch.zeros(BATCH_SIZE)
    terminated = torch.zeros(BATCH_SIZE)
    online_q = _constant_q(1.0)
    target_q = _constant_q(1.0)
    bounded_target = _bound(target_q, torch.cat([states, actions], dim=1))
    trainer = WorstAttackQTrainer(
        online_q,
        WorstAttackValueEstimator(bounded_target),
        _make_bounder(_make_mlp(STATE_DIM, ACTION_DIM), states, 0.05, "IBP"),
        0.9,
    )

    trainer.loss(states, actions, rewards, states, terminated).backward()

    assert _has_nonzero_gradient(online_q)
    assert not _has_any_gradient(bounded_target)


# --------------------------------------------------------------------------- #
# WorstAttackPPOLoss
# --------------------------------------------------------------------------- #


def test_total_loss_equals_ppo_loss_when_both_coefficients_are_zero() -> None:
    states = _random_states()
    bounder = _make_bounder(_make_mlp(STATE_DIM, ACTION_DIM), states, 0.05, "IBP")
    estimator = _make_estimator(_make_mlp(STATE_DIM + ACTION_DIM, 1), states)
    loss_function = WorstAttackPPOLoss(estimator, bounder, 0.0, 0.0)
    ppo_loss = torch.tensor(0.7)

    total = loss_function.compute(ppo_loss, states)

    torch.testing.assert_close(total.detach(), ppo_loss, atol=TOLERANCE, rtol=0.0)


def test_worst_value_term_subtracts_coefficient_times_mean_value() -> None:
    coefficient = 0.5
    value = 3.0
    states = _random_states()
    bounder = _make_bounder(_make_mlp(STATE_DIM, ACTION_DIM), states, 0.05, "IBP")
    estimator = _make_estimator(_constant_q(value), states)
    loss_function = WorstAttackPPOLoss(estimator, bounder, coefficient, 0.0)
    ppo_loss = torch.tensor(1.0)

    total = loss_function.compute(ppo_loss, states)

    expected = torch.tensor(1.0 - coefficient * value)
    torch.testing.assert_close(total.detach(), expected, atol=TOLERANCE, rtol=0.0)


def test_regularizer_adds_coefficient_times_mean_box_width() -> None:
    coefficient = 2.0
    states = _random_states()
    bounder = _make_bounder(_make_mlp(STATE_DIM, ACTION_DIM), states, 0.1, "IBP")
    estimator = _make_estimator(_constant_q(0.0), states)
    loss_function = WorstAttackPPOLoss(estimator, bounder, 0.0, coefficient)
    ppo_loss = torch.tensor(0.0)

    total = loss_function.compute(ppo_loss, states)

    box = bounder.action_box(states)
    expected_width = (box.upper - box.lower).mean().detach()
    torch.testing.assert_close(
        total.detach(), coefficient * expected_width, atol=TOLERANCE, rtol=0.0
    )


def test_policy_gradients_flow_through_worst_value_term() -> None:
    states = _random_states()
    bounded_policy = _bound(_make_mlp(STATE_DIM, ACTION_DIM), states)
    bounder = PolicyActionBounder(bounded_policy, 0.1, "IBP")
    estimator = _make_estimator(_make_mlp(STATE_DIM + ACTION_DIM, 1), states)
    loss_function = WorstAttackPPOLoss(estimator, bounder, 1.0, 0.0)

    loss_function.compute(torch.tensor(0.0), states).backward()

    assert _has_nonzero_gradient(bounded_policy)


def test_policy_gradients_flow_through_box_width_regularizer() -> None:
    states = _random_states()
    bounded_policy = _bound(_make_mlp(STATE_DIM, ACTION_DIM), states)
    bounder = PolicyActionBounder(bounded_policy, 0.1, "IBP")
    estimator = _make_estimator(_constant_q(0.0), states)
    loss_function = WorstAttackPPOLoss(estimator, bounder, 0.0, 1.0)

    loss_function.compute(torch.tensor(0.0), states).backward()

    assert _has_nonzero_gradient(bounded_policy)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _make_trainer(
    online_q: nn.Module,
    target_q: nn.Module,
    states: Tensor,
    discount: float,
) -> WorstAttackQTrainer:
    estimator = _make_estimator(target_q, states)
    bounder = _make_bounder(_make_mlp(STATE_DIM, ACTION_DIM), states, 0.05, "IBP")
    return WorstAttackQTrainer(online_q, estimator, bounder, discount)


def _make_estimator(q_network: nn.Module, states: Tensor) -> WorstAttackValueEstimator:
    example_input = torch.zeros(states.shape[0], STATE_DIM + ACTION_DIM)
    return WorstAttackValueEstimator(_bound(q_network, example_input))


def _make_bounder(
    policy: nn.Module,
    states: Tensor,
    epsilon: float,
    method: str,
) -> PolicyActionBounder:
    return PolicyActionBounder(_bound(policy, states), epsilon, method)


def _bound(model: nn.Module, example_input: Tensor) -> BoundedModule:
    return BoundedModule(model, torch.zeros_like(example_input))


def _make_mlp(input_dim: int, output_dim: int) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.Tanh(),
        nn.Linear(16, output_dim),
    )


def _constant_q(value: float) -> nn.Module:
    layer = nn.Linear(STATE_DIM + ACTION_DIM, 1)
    nn.init.zeros_(layer.weight)
    nn.init.constant_(layer.bias, value)
    return nn.Sequential(layer)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #


def _random_states() -> Tensor:
    torch.manual_seed(1)
    return torch.rand(BATCH_SIZE, STATE_DIM) * 2.0 - 1.0


def _fixed_box() -> ActionBox:
    lower = torch.full((BATCH_SIZE, ACTION_DIM), -0.5)
    upper = torch.full((BATCH_SIZE, ACTION_DIM), 0.5)
    return ActionBox(lower=lower, upper=upper)


def _sample_ball(states: Tensor, epsilon: float, num_samples: int) -> Tensor:
    """[num_samples, batch, state_dim]: uniform samples with ||s~ - s||_inf <= epsilon."""
    noise = (torch.rand(num_samples, *states.shape) * 2.0 - 1.0) * epsilon
    return states.unsqueeze(0) + noise


def _sample_box(box: ActionBox, num_samples: int) -> Tensor:
    """[num_samples, batch, action_dim]: uniform samples inside the box."""
    lower = box.lower.detach()
    upper = box.upper.detach()
    fractions = torch.rand(num_samples, *lower.shape)
    return lower + fractions * (upper - lower)


def _has_nonzero_gradient(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in module.parameters()
    )


def _has_any_gradient(module: nn.Module) -> bool:
    return any(parameter.grad is not None for parameter in module.parameters())