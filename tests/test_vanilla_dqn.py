from argparse import Namespace
from pathlib import Path

import pytest
import torch
from rich.console import Console
from torch import Tensor, nn

from src.marl.vanilla_dqn import VanillaDQNConfig, VanillaDQNTrainer
from src.marl.woca_r_rl.dqn import Batch, _build_robust_q, build_dqn_player, train_dqn

STATE_DIM: int = 4
NUM_ACTIONS: int = 3
HIDDEN: int = 16
BATCH_SIZE: int = 8
TAU: float = 0.1
DEVICE: torch.device = torch.device("cpu")


# --------------------------------------------------------------------------- #
# VanillaDQNTrainer
# --------------------------------------------------------------------------- #


def test_target_starts_equal_to_the_online_network() -> None:
    trainer = _make_trainer()

    assert _distance(trainer._target_q, trainer._q) == 0.0


def test_update_on_a_random_batch_gives_a_finite_loss() -> None:
    trainer = _make_trainer()

    metrics = trainer.update(_random_batch(), step=0)

    assert torch.isfinite(torch.tensor(metrics.td_loss))
    for parameter in trainer._q.parameters():
        assert torch.isfinite(parameter).all()


def test_soft_update_moves_the_target_by_tau_toward_the_online_network() -> None:
    trainer = _make_trainer()
    old_target: list[Tensor] = _snapshot(trainer._target_q)

    trainer.update(_random_batch(), step=0)

    assert _distance(trainer._target_q, trainer._q) > 0.0
    for old, target, online in zip(old_target, trainer._target_q.parameters(), trainer._q.parameters()):
        assert torch.allclose(target, old + TAU * (online - old), atol=1e-6)
        assert torch.linalg.norm(target - online) <= torch.linalg.norm(old - online)


def test_target_network_is_not_trained_by_the_optimizer() -> None:
    trainer = _make_trainer()

    trainer.update(_random_batch(), step=0)

    for parameter in trainer._target_q.parameters():
        assert not parameter.requires_grad
        assert parameter.grad is None


def test_terminal_transitions_use_only_the_reward() -> None:
    trainer = _make_trainer()
    batch = _random_batch()._replace(terminated=torch.ones(BATCH_SIZE))

    assert torch.allclose(trainer._targets(batch), batch.rewards)


@pytest.mark.parametrize("use_double_dqn", [False, True])
def test_targets_match_the_dqn_formula(use_double_dqn: bool) -> None:
    trainer = _make_trainer(use_double_dqn=use_double_dqn)
    trainer.update(_random_batch(seed=5), step=0)  # makes Q and Q' differ
    batch = _random_batch()

    with torch.no_grad():
        target_values = trainer._target_q(batch.next_states)
        if use_double_dqn:
            best = trainer._q(batch.next_states).argmax(dim=1, keepdim=True)
            next_values = target_values.gather(1, best).squeeze(1)
        else:
            next_values = target_values.max(dim=1).values
    expected = batch.rewards + 0.9 * (1.0 - batch.terminated) * next_values

    assert torch.allclose(trainer._targets(batch), expected)


def test_select_action_without_exploration_is_the_greedy_action_of_q() -> None:
    trainer = _make_trainer()
    state = _random_batch().states[0]

    with torch.no_grad():
        greedy = int(trainer._q(state.unsqueeze(0)).argmax(dim=1))

    assert trainer.select_action(state, exploration_probability=0.0) == greedy


# --------------------------------------------------------------------------- #
# train_dqn with update_function "dqn"
# --------------------------------------------------------------------------- #


def test_short_dqn_training_saves_a_checkpoint_that_loads(tmp_path: Path) -> None:
    save: Path = tmp_path / "model.pt"

    train_dqn(_train_args(tmp_path, save), Console(quiet=True))

    checkpoint: dict = torch.load(save)  # weights_only by default
    assert checkpoint["algorithm"] == "dqn"
    assert "q" in checkpoint
    assert "robust_q" not in checkpoint
    player = build_dqn_player(checkpoint, render=False)
    player.start_episode(seed=0)
    player.step()
    player.close()


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _make_trainer(use_double_dqn: bool = False) -> VanillaDQNTrainer:
    torch.manual_seed(0)
    q_network = _build_robust_q(STATE_DIM, NUM_ACTIONS, HIDDEN, DEVICE)
    target_network = _build_robust_q(STATE_DIM, NUM_ACTIONS, HIDDEN, DEVICE)
    config = VanillaDQNConfig(
        num_actions=NUM_ACTIONS,
        discount_factor=0.9,
        target_update_coefficient=TAU,
        learning_rate=1e-2,
        use_double_dqn=use_double_dqn,
    )
    return VanillaDQNTrainer(q_network, target_network, config)


def _random_batch(seed: int = 1) -> Batch:
    generator = torch.Generator().manual_seed(seed)
    return Batch(
        states=torch.rand(BATCH_SIZE, STATE_DIM, generator=generator) * 2 - 1,
        actions=torch.randint(NUM_ACTIONS, (BATCH_SIZE,), generator=generator),
        rewards=torch.rand(BATCH_SIZE, generator=generator),
        next_states=torch.rand(BATCH_SIZE, STATE_DIM, generator=generator) * 2 - 1,
        terminated=torch.tensor([0.0, 1.0] * (BATCH_SIZE // 2)),
    )


def _train_args(tmp_path: Path, save: Path) -> Namespace:
    return Namespace(
        update_function="dqn",
        dqn_env="highway",
        agents=2,
        reward="team",
        team_weight=0.0,
        seed=0,
        total_steps=40,
        warmup=20,
        buffer_size=1000,
        batch_size=8,
        hidden=HIDDEN,
        lr=1e-4,
        gamma=0.99,
        tau=0.005,
        double_dqn=False,
        explore_steps=20,
        explore_final=0.05,
        logdir=str(tmp_path / "runs"),
        run_name="run",
        save=str(save),
        render=False,
    )


def _snapshot(module: nn.Module) -> list[Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _distance(first: nn.Module, second: nn.Module) -> float:
    return sum(
        float(torch.linalg.norm(a - b)) for a, b in zip(first.parameters(), second.parameters())
    )
