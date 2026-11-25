from dataclasses import replace
from typing import Literal

import pytest
import torch
from torch import Tensor, nn

from auto_LiRPA import BoundedModule

from src.marl.woca_r_rl.dqn import (
    Batch,
    QValueBounds,
    RampSchedule,
    ReachableActionEstimator,
    RobustQBounder,
    SARegularizer,
    WocaRDQNConfig,
    WocaRDQNTrainer,
)

STATE_DIM: int = 4
NUM_ACTIONS: int = 3
BATCH_SIZE: int = 4
NUM_SAMPLES: int = 256
DISCOUNT: float = 0.9
TOLERANCE: float = 1e-5


# --------------------------------------------------------------------------- #
# RampSchedule
# --------------------------------------------------------------------------- #


def test_ramp_schedule_is_zero_up_to_the_start_step() -> None:
    schedule = RampSchedule(start_step=10, ramp_steps=20, final_value=0.5)

    assert schedule.value(0) == 0.0
    assert schedule.value(10) == 0.0


@pytest.mark.parametrize("shape", ["linear", "exponential"])
def test_ramp_schedule_reaches_the_final_value_and_stays(
    shape: Literal["linear", "exponential"],
) -> None:
    schedule = RampSchedule(start_step=10, ramp_steps=20, final_value=0.5, shape=shape)

    assert schedule.value(30) == pytest.approx(0.5)
    assert schedule.value(1000) == pytest.approx(0.5)


def test_linear_ramp_is_proportional_to_progress() -> None:
    schedule = RampSchedule(start_step=10, ramp_steps=20, final_value=0.5, shape="linear")

    assert schedule.value(20) == pytest.approx(0.25)


def test_exponential_ramp_is_increasing_and_below_the_linear_ramp() -> None:
    exponential = RampSchedule(start_step=10, ramp_steps=20, final_value=0.5, shape="exponential")
    linear = RampSchedule(start_step=10, ramp_steps=20, final_value=0.5, shape="linear")

    values = [exponential.value(step) for step in range(10, 31)]

    assert values == sorted(values)
    assert 0.0 < exponential.value(20) < linear.value(20)


def test_ramp_schedule_rejects_non_positive_ramp_steps() -> None:
    with pytest.raises(ValueError):
        RampSchedule(start_step=0, ramp_steps=0, final_value=1.0)


# --------------------------------------------------------------------------- #
# ReachableActionEstimator
# --------------------------------------------------------------------------- #


def test_reachable_set_follows_the_interval_rule() -> None:
    bounds = QValueBounds(
        lower=torch.tensor([[1.0, 0.0, -1.0]]),
        upper=torch.tensor([[2.0, 0.5, 0.0]]),
    )

    mask = ReachableActionEstimator().mask(bounds)

    assert mask.tolist() == [[True, False, False]]


def test_reachable_set_keeps_actions_that_tie_with_the_best_lower_bound() -> None:
    bounds = QValueBounds(
        lower=torch.tensor([[1.0, 0.0]]),
        upper=torch.tensor([[2.0, 1.0]]),
    )

    mask = ReachableActionEstimator().mask(bounds)

    assert mask.tolist() == [[True, True]]


def test_reachable_set_contains_every_action_when_intervals_overlap() -> None:
    bounds = QValueBounds(
        lower=torch.zeros(1, NUM_ACTIONS),
        upper=torch.ones(1, NUM_ACTIONS),
    )

    mask = ReachableActionEstimator().mask(bounds)

    assert mask.tolist() == [[True, True, True]]


def test_reachable_set_with_a_single_action_contains_it() -> None:
    bounds = QValueBounds(lower=torch.tensor([[0.0]]), upper=torch.tensor([[1.0]]))

    mask = ReachableActionEstimator().mask(bounds)

    assert mask.tolist() == [[True]]


def test_reachable_set_of_exact_bounds_is_the_greedy_action_only() -> None:
    q_values = torch.tensor([[0.5, 0.1, 0.2], [0.0, 2.0, 1.0]])
    bounds = QValueBounds(lower=q_values, upper=q_values)

    mask = ReachableActionEstimator().mask(bounds)

    assert mask.tolist() == [[True, False, False], [False, True, False]]


def test_reachable_set_always_contains_the_clean_greedy_action() -> None:
    torch.manual_seed(0)
    q_values = torch.randn(64, NUM_ACTIONS)
    radius = torch.rand(64, NUM_ACTIONS)
    bounds = QValueBounds(lower=q_values - radius, upper=q_values + radius)

    mask = ReachableActionEstimator().mask(bounds)

    greedy_actions = q_values.argmax(dim=1, keepdim=True)
    assert bool(mask.gather(1, greedy_actions).all())


def test_reachable_set_raises_when_bounds_are_not_finite() -> None:
    bounds = QValueBounds(
        lower=torch.zeros(1, NUM_ACTIONS),
        upper=torch.full((1, NUM_ACTIONS), float("nan")),
    )

    with pytest.raises(RuntimeError):
        ReachableActionEstimator().mask(bounds)


# --------------------------------------------------------------------------- #
# SARegularizer
# --------------------------------------------------------------------------- #


def test_regularizer_is_the_largest_competing_upper_bound_minus_the_best_lower_bound() -> None:
    bounds = QValueBounds(
        lower=torch.tensor([[1.0, 0.0]]),
        upper=torch.tensor([[2.0, 1.5]]),
    )
    clean_q_values = torch.tensor([[1.5, 0.2]])
    margin_lower_bounds = _margin_lower_bounds(bounds, clean_q_values)

    loss = SARegularizer(margin=0.0).per_state_loss(margin_lower_bounds)

    torch.testing.assert_close(loss, torch.tensor([0.5]))


def test_regularizer_uses_the_clean_greedy_action_not_the_bound_maximum() -> None:
    bounds = QValueBounds(
        lower=torch.tensor([[1.0, 0.0]]),
        upper=torch.tensor([[2.0, 1.5]]),
    )
    clean_q_values = torch.tensor([[0.2, 1.5]])
    margin_lower_bounds = _margin_lower_bounds(bounds, clean_q_values)

    loss = SARegularizer(margin=0.0).per_state_loss(margin_lower_bounds)

    torch.testing.assert_close(loss, torch.tensor([2.0]))


@pytest.mark.parametrize("margin, expected", [(0.0, 0.0), (0.5, -0.5)])
def test_regularizer_is_floored_at_minus_the_margin_once_certified(
    margin: float,
    expected: float,
) -> None:
    bounds = QValueBounds(
        lower=torch.tensor([[3.0, 0.0]]),
        upper=torch.tensor([[4.0, 1.0]]),
    )
    clean_q_values = torch.tensor([[3.5, 0.5]])
    margin_lower_bounds = _margin_lower_bounds(bounds, clean_q_values)

    loss = SARegularizer(margin=margin).per_state_loss(margin_lower_bounds)

    torch.testing.assert_close(loss, torch.tensor([expected]))


def test_regularizer_gradient_touches_only_the_two_deciding_bounds() -> None:
    lower = torch.tensor([[1.0, 0.0]], requires_grad=True)
    upper = torch.tensor([[2.0, 1.5]], requires_grad=True)
    clean_q_values = torch.tensor([[1.5, 0.2]], requires_grad=True)
    margin_lower_bounds = _margin_lower_bounds(QValueBounds(lower, upper), clean_q_values)

    loss = SARegularizer(margin=0.0).per_state_loss(margin_lower_bounds)
    loss.sum().backward()

    assert lower.grad is not None and upper.grad is not None
    assert lower.grad.tolist() == [[-1.0, 0.0]]
    assert upper.grad.tolist() == [[0.0, 1.0]]
    assert clean_q_values.grad is None


def _margin_lower_bounds(bounds: QValueBounds, clean_q_values: Tensor) -> Tensor:
    """[batch, A-1]: LB[Q(a*) - Q(a)] for every competing action a, from exact bounds."""
    best_actions = clean_q_values.detach().argmax(dim=1)
    num_actions = bounds.upper.shape[1]
    best_mask = nn.functional.one_hot(best_actions, num_actions).bool()
    lower_of_best = bounds.lower.gather(1, best_actions.unsqueeze(1))
    upper_of_others = bounds.upper.masked_fill(best_mask, float("-inf"))
    margins = lower_of_best - upper_of_others
    return margins.masked_select(torch.logical_not(best_mask)).view(
        clean_q_values.shape[0], num_actions - 1
    )


# --------------------------------------------------------------------------- #
# RobustQBounder
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["IBP", "CROWN"])
def test_bounds_contain_q_values_on_the_clipped_perturbed_states(method: str) -> None:
    epsilon = 0.2
    states = _random_states()
    q_network = _make_mlp(STATE_DIM, NUM_ACTIONS)
    bounder = RobustQBounder(_bound(q_network, states), -1.0, 1.0, None, method)

    bounds = bounder.bounds(states, epsilon)

    lower = bounds.lower.detach()
    upper = bounds.upper.detach()
    with torch.no_grad():
        for perturbed_states in _sample_clipped_ball(states, epsilon, -1.0, 1.0, NUM_SAMPLES):
            q_values = q_network(perturbed_states)
            assert bool((q_values >= lower - TOLERANCE).all())
            assert bool((q_values <= upper + TOLERANCE).all())


def test_bounds_collapse_to_the_q_values_for_zero_epsilon() -> None:
    states = _random_states()
    q_network = _make_mlp(STATE_DIM, NUM_ACTIONS)
    bounder = RobustQBounder(_bound(q_network, states), -1.0, 1.0, None, "IBP")

    bounds = bounder.bounds(states, 0.0)

    with torch.no_grad():
        expected = q_network(states)
    torch.testing.assert_close(bounds.lower.detach(), expected, atol=TOLERANCE, rtol=0.0)
    torch.testing.assert_close(bounds.upper.detach(), expected, atol=TOLERANCE, rtol=0.0)


def test_bounds_with_range_clipping_are_not_wider_than_without() -> None:
    states = torch.ones(BATCH_SIZE, STATE_DIM)
    q_network = _make_mlp(STATE_DIM, NUM_ACTIONS)
    bounded = _bound(q_network, states)
    clipped = RobustQBounder(bounded, -1.0, 1.0, None, "IBP").bounds(states, 0.5)
    unclipped = RobustQBounder(bounded, -10.0, 10.0, None, "IBP").bounds(states, 0.5)

    clipped_width = (clipped.upper - clipped.lower).detach()
    unclipped_width = (unclipped.upper - unclipped.lower).detach()

    assert bool((clipped_width <= unclipped_width + TOLERANCE).all())
    assert float(clipped_width.mean()) < float(unclipped_width.mean())


def test_zero_perturbation_scale_collapses_the_bounds() -> None:
    states = _random_states()
    q_network = _make_mlp(STATE_DIM, NUM_ACTIONS)
    scale = torch.zeros(STATE_DIM)
    bounder = RobustQBounder(_bound(q_network, states), -1.0, 1.0, scale, "IBP")

    bounds = bounder.bounds(states, 0.5)

    with torch.no_grad():
        expected = q_network(states)
    torch.testing.assert_close(bounds.lower.detach(), expected, atol=TOLERANCE, rtol=0.0)
    torch.testing.assert_close(bounds.upper.detach(), expected, atol=TOLERANCE, rtol=0.0)


def test_partial_perturbation_scale_gives_narrower_bounds_than_full_scale() -> None:
    states = _random_states()
    bounded = _bound(_make_mlp(STATE_DIM, NUM_ACTIONS), states)
    partial_scale = torch.tensor([1.0, 1.0, 0.0, 0.0])
    full = RobustQBounder(bounded, -1.0, 1.0, None, "IBP").bounds(states, 0.2)
    partial = RobustQBounder(bounded, -1.0, 1.0, partial_scale, "IBP").bounds(states, 0.2)

    full_width = (full.upper - full.lower).detach()
    partial_width = (partial.upper - partial.lower).detach()

    assert bool((partial_width <= full_width + TOLERANCE).all())
    assert float(partial_width.mean()) < float(full_width.mean())


def test_bounds_propagate_gradients_to_the_bounded_network() -> None:
    states = _random_states()
    bounded = _bound(_make_mlp(STATE_DIM, NUM_ACTIONS), states)
    bounder = RobustQBounder(bounded, -1.0, 1.0, None, "IBP")

    bounder.bounds(states, 0.1).upper.sum().backward()

    assert _has_nonzero_gradient(bounded)

# --------------------------------------------------------------------------- #
# WocaRDQNTrainer: hand-computed values with constant networks
# --------------------------------------------------------------------------- #


def test_update_losses_match_hand_computed_values() -> None:
    """
        Constant networks and learning rate 0 make every reported loss exact.
        Q_v = [1, 0, 0.4], Q_r = [0.5, 0.1, 0.2], Q_phi = [0, -1, 5], eps = 0,
        kappa_wst = 0.25. The soft update leaves Q_v' equal to Q_v. Taken action 1,
        rewards [1, 0, -1, 2], terminated [0, 0, 1, 0], gamma 0.9.
    """
    trainer = _make_constant_trainer(
        vanilla_values=[1.0, 0.0, 0.4],
        robust_values=[0.5, 0.1, 0.2],
        worst_attack_values=[0.0, -1.0, 5.0],
        regularizer_margin=0.5,
        final_kappa_wst=0.25,
    )
    batch = _make_batch(
        rewards=[1.0, 0.0, -1.0, 2.0],
        terminated=[0.0, 0.0, 1.0, 0.0],
    )

    metrics = trainer.update(batch, step=100)

    # Q_v: y = r + 0.9 * max Q_v' = r + 0.9 * 1 on live states = [1.9, 0.9, -1, 2.9].
    # Prediction Q_v(s, 1) = 0: (1.9^2 + 0.9^2 + 1^2 + 2.9^2) / 4 = 13.83 / 4.
    expected_vanilla = 3.4575
    # Q_phi: eps 0 gives A_adv_hat = {argmax Q_r} = {0}, so y = r + 0.9 * Q_phi(s', 0) = r.
    # Prediction Q_phi(s, 1) = -1: (2^2 + 1^2 + 0^2 + 3^2) / 4 = 14 / 4.
    expected_worst_attack = 3.5
    # Q_r: mix = 0.75 * [1, 0, 0.4] + 0.25 * [0, -1, 5] = [0.75, -0.25, 1.55], max 1.55.
    # y = r + 0.9 * 1.55 on live states = [2.395, 1.395, -1, 3.395]; prediction Q_r(s, 1) = 0.1.
    # (2.295^2 + 1.295^2 + 1.1^2 + 3.295^2) / 4 = 19.011075 / 4.
    expected_td = 4.75276875
    # L_reg: a* = 0; exact margins to actions 1 and 2 are 0.4 and 0.3, so -min = -0.3 > -0.5.
    # w(s) = 1 because every state has the same Q_v spread 1.
    expected_regularizer = -0.3

    assert metrics.vanilla_loss == pytest.approx(expected_vanilla, abs=TOLERANCE)
    assert metrics.worst_attack_loss == pytest.approx(expected_worst_attack, abs=TOLERANCE)
    assert metrics.robust_td_loss == pytest.approx(expected_td, abs=TOLERANCE)
    assert metrics.regularizer_loss == pytest.approx(expected_regularizer, abs=TOLERANCE)
    assert metrics.mean_reachable_actions == pytest.approx(1.0)
    assert metrics.epsilon == pytest.approx(0.0)
    assert metrics.kappa_wst == pytest.approx(0.25)


def test_terminal_transitions_ignore_every_next_state_value() -> None:
    trainer = _make_constant_trainer(
        vanilla_values=[1.0, 0.0, 0.4],
        robust_values=[0.5, 0.1, 0.2],
        worst_attack_values=[0.0, -1.0, 5.0],
    )
    batch = _make_batch(
        rewards=[1.0, 0.0, -1.0, 2.0],
        terminated=[1.0, 1.0, 1.0, 1.0],
    )

    metrics = trainer.update(batch, step=100)

    expected_vanilla = nn.functional.mse_loss(torch.full((BATCH_SIZE,), 0.0), batch.rewards)
    expected_td = nn.functional.mse_loss(torch.full((BATCH_SIZE,), 0.1), batch.rewards)
    expected_worst_attack = nn.functional.mse_loss(
        torch.full((BATCH_SIZE,), -1.0), batch.rewards
    )

    assert metrics.vanilla_loss == pytest.approx(float(expected_vanilla), abs=TOLERANCE)
    assert metrics.robust_td_loss == pytest.approx(float(expected_td), abs=TOLERANCE)
    assert metrics.worst_attack_loss == pytest.approx(float(expected_worst_attack), abs=TOLERANCE)


def test_state_importance_uses_the_vanilla_network() -> None:
    """
        Linear networks without bias on one-hot states A = e_0 and B = e_1.
        Q_v(A) = [3, 1, 0] (spread 3), Q_v(B) = [1, 1, 0] (spread 1): w = [1, 1/3].
        Q_r(A) = [1, 0, 0] (spread 1), Q_r(B) = [5, 0, 0] (spread 5) would give w = [0.2, 1].
    """
    states = torch.eye(2, STATE_DIM)
    vanilla_q = _linear_network([[3.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    robust_q = _bound(_linear_network([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), states)
    worst_attack_q = _constant_network([0.0, 0.0, 0.0])
    trainer = WocaRDQNTrainer(vanilla_q, robust_q, worst_attack_q, _make_config())

    importance = trainer._state_importance(states)

    torch.testing.assert_close(importance, torch.tensor([1.0, 1.0 / 3.0]))


def test_soft_update_moves_the_target_by_tau() -> None:
    """
        Q_v = theta: zero weights, bias [1, 2, 3]. Q_v' = theta': weights 2, bias [-1, 0, 5].
        With tau = 0.1: weights 0.1 * 0 + 0.9 * 2 = 1.8, bias [0.1 - 0.9, 0.2, 0.3 + 4.5].
    """
    tau = 0.1
    vanilla_q = _constant_network([1.0, 2.0, 3.0])
    trainer = WocaRDQNTrainer(
        vanilla_q,
        _bound(_constant_network([0.0, 0.0, 0.0]), _random_states()),
        _constant_network([0.0, 0.0, 0.0]),
        replace(_make_config(), target_update_coefficient=tau),
    )
    target_q = trainer._target_vanilla_q
    assert isinstance(target_q, nn.Sequential)
    with torch.no_grad():
        target_q[0].weight.fill_(2.0)
    _set_bias(target_q, [-1.0, 0.0, 5.0])
    vanilla_before = _snapshot(vanilla_q)

    trainer._soft_update_target()

    torch.testing.assert_close(target_q[0].weight, torch.full((NUM_ACTIONS, STATE_DIM), 1.8))
    torch.testing.assert_close(target_q[0].bias, torch.tensor([-0.8, 0.2, 4.8]))
    assert not _parameters_changed(vanilla_q, vanilla_before)

    # update() with lr > 0 moves Q_v', but only part of the way to the new Q_v.
    states = _random_states()
    learning_vanilla_q = _make_mlp(STATE_DIM, NUM_ACTIONS, seed=4)
    learning_trainer = WocaRDQNTrainer(
        learning_vanilla_q,
        _bound(_make_mlp(STATE_DIM, NUM_ACTIONS, seed=2), states),
        _make_mlp(STATE_DIM, NUM_ACTIONS, seed=3),
        replace(_make_config(learning_rate=1e-2), target_update_coefficient=tau),
    )
    learning_target_q = learning_trainer._target_vanilla_q
    target_before = _snapshot(learning_target_q)
    batch = _make_batch(rewards=[1.0, 0.0, -1.0, 2.0], terminated=[0.0, 0.0, 1.0, 0.0])

    learning_trainer.update(batch, step=100)

    assert _parameters_changed(learning_target_q, target_before)
    assert _parameters_changed(learning_target_q, _snapshot(learning_vanilla_q))


# --------------------------------------------------------------------------- #
# WocaRDQNTrainer: behaviour with real networks
# --------------------------------------------------------------------------- #


def test_update_changes_the_robust_and_worst_attack_networks() -> None:
    states = _random_states()
    vanilla_q = _make_mlp(STATE_DIM, NUM_ACTIONS, seed=4)
    robust_q = _bound(_make_mlp(STATE_DIM, NUM_ACTIONS, seed=2), states)
    worst_attack_q = _make_mlp(STATE_DIM, NUM_ACTIONS, seed=3)
    trainer = WocaRDQNTrainer(
        vanilla_q,
        robust_q,
        worst_attack_q,
        _make_config(learning_rate=1e-2),
    )
    snapshots = [_snapshot(network) for network in (robust_q, worst_attack_q)]
    batch = _make_batch(rewards=[1.0, 0.0, -1.0, 2.0], terminated=[0.0, 0.0, 1.0, 0.0])

    trainer.update(batch, step=100)

    for network, snapshot in zip((robust_q, worst_attack_q), snapshots):
        assert _parameters_changed(network, snapshot)


def test_zero_perturbation_scale_makes_the_reachable_set_a_single_action() -> None:
    states = _random_states()
    trainer = WocaRDQNTrainer(
        _make_mlp(STATE_DIM, NUM_ACTIONS, seed=1),
        _bound(_make_mlp(STATE_DIM, NUM_ACTIONS, seed=2), states),
        _make_mlp(STATE_DIM, NUM_ACTIONS, seed=3),
        _make_config(
            epsilon_schedule=RampSchedule(start_step=0, ramp_steps=1, final_value=0.5),
            perturbation_scale=torch.zeros(STATE_DIM),
        ),
    )
    batch = _make_batch(rewards=[1.0, 0.0, -1.0, 2.0], terminated=[0.0, 0.0, 0.0, 0.0])

    metrics = trainer.update(batch, step=100)

    assert metrics.epsilon == pytest.approx(0.5)
    assert metrics.mean_reachable_actions == pytest.approx(1.0)


def test_select_action_without_exploration_is_the_greedy_action_of_the_robust_network() -> None:
    trainer = _make_constant_trainer(
        vanilla_values=[0.9, 0.5, 0.1],
        robust_values=[0.5, 0.1, 0.9],
        worst_attack_values=[0.0, 0.0, 0.0],
    )

    action = trainer.select_action(_random_states()[0], exploration_probability=0.0)

    assert action == 2


def test_select_action_with_full_exploration_covers_the_valid_actions() -> None:
    trainer = _make_constant_trainer(
        vanilla_values=[0.9, 0.5, 0.1],
        robust_values=[0.5, 0.1, 0.9],
        worst_attack_values=[0.0, 0.0, 0.0],
    )
    state = _random_states()[0]
    torch.manual_seed(0)

    actions = {trainer.select_action(state, exploration_probability=1.0) for _ in range(200)}

    assert actions == {0, 1, 2}


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _make_constant_trainer(
    vanilla_values: list[float],
    robust_values: list[float],
    worst_attack_values: list[float],
    regularizer_margin: float = 0.0,
    final_kappa_wst: float = 0.5,
) -> WocaRDQNTrainer:
    """
        Trainer with constant networks (zero weights, given biases) and learning rate 0,
        so no parameter moves during update() and every loss can be computed by hand.
    """
    vanilla_q = _constant_network(vanilla_values)
    robust_q = _bound(_constant_network(robust_values), _random_states())
    worst_attack_q = _constant_network(worst_attack_values)
    config = _make_config(
        learning_rate=0.0,
        regularizer_margin=regularizer_margin,
        kappa_wst_schedule=RampSchedule(start_step=0, ramp_steps=1, final_value=final_kappa_wst),
    )
    return WocaRDQNTrainer(vanilla_q, robust_q, worst_attack_q, config)


def _make_config(
    learning_rate: float = 0.0,
    regularizer_margin: float = 0.0,
    epsilon_schedule: RampSchedule | None = None,
    kappa_wst_schedule: RampSchedule | None = None,
    perturbation_scale: Tensor | None = None,
) -> WocaRDQNConfig:
    if epsilon_schedule is None:
        epsilon_schedule = RampSchedule(start_step=0, ramp_steps=1, final_value=0.0)
    if kappa_wst_schedule is None:
        kappa_wst_schedule = RampSchedule(start_step=0, ramp_steps=1, final_value=0.0)
    return WocaRDQNConfig(
        num_actions=NUM_ACTIONS,
        discount_factor=DISCOUNT,
        target_update_coefficient=0.005,
        learning_rate=learning_rate,
        epsilon_schedule=epsilon_schedule,
        kappa_wst_schedule=kappa_wst_schedule,
        kappa_reg=1.0,
        observation_low=-1.0,
        observation_high=1.0,
        regularizer_margin=regularizer_margin,
        perturbation_scale=perturbation_scale,
    )


def _make_batch(
    rewards: list[float],
    terminated: list[float],
) -> Batch:
    return Batch(
        states=_random_states(seed=1),
        actions=torch.ones(BATCH_SIZE, dtype=torch.long),
        rewards=torch.tensor(rewards),
        next_states=_random_states(seed=2),
        terminated=torch.tensor(terminated),
    )


def _bound(model: nn.Module, example_input: Tensor) -> BoundedModule:
    return BoundedModule(model, torch.zeros_like(example_input))


def _make_mlp(input_dim: int, output_dim: int, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.Tanh(),
        nn.Linear(16, output_dim),
    )


def _constant_network(values: list[float]) -> nn.Sequential:
    layer = nn.Linear(STATE_DIM, len(values))
    nn.init.zeros_(layer.weight)
    network = nn.Sequential(layer)
    _set_bias(network, values)
    return network


def _linear_network(columns: list[list[float]]) -> nn.Sequential:
    """
        Bias-free linear network whose weight column i is columns[i], so the
        one-hot state e_i maps to the Q-values columns[i].
    """
    layer = nn.Linear(STATE_DIM, len(columns[0]), bias=False)
    weight = torch.zeros(len(columns[0]), STATE_DIM)
    weight[:, : len(columns)] = torch.tensor(columns).T
    with torch.no_grad():
        layer.weight.copy_(weight)
    return nn.Sequential(layer)


def _set_bias(network: nn.Sequential, values: list[float]) -> None:
    with torch.no_grad():
        bias = network[0].bias
        assert isinstance(bias, Tensor)
        bias.copy_(torch.as_tensor(values, dtype=bias.dtype, device=bias.device))


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #


def _random_states(seed: int = 1) -> Tensor:
    torch.manual_seed(seed)
    return torch.rand(BATCH_SIZE, STATE_DIM) * 2.0 - 1.0


def _sample_clipped_ball(
    states: Tensor,
    epsilon: float,
    low: float,
    high: float,
    num_samples: int,
) -> Tensor:
    """
        [num_samples, batch, state_dim]: uniform samples of the L-inf ball clipped to [low, high].
    """
    noise = (torch.rand(num_samples, *states.shape) * 2.0 - 1.0) * epsilon
    return (states.unsqueeze(0) + noise).clamp(low, high)


def _snapshot(module: nn.Module) -> list[Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _parameters_changed(module: nn.Module, snapshot: list[Tensor]) -> bool:
    return any(
        not torch.equal(parameter.detach(), before)
        for parameter, before in zip(module.parameters(), snapshot)
    )


def _has_nonzero_gradient(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in module.parameters()
    )
