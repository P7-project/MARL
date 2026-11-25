from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from mpe2 import simple_spread_v3
from pettingzoo import ParallelEnv
from rich.console import Console

from src.marl.env import any_vehicle_crashed, build_actor_obs, flatten_obs, make_env
from src.marl.models import Actor
from src.marl.rendering import PolicyPlayer, StepResult
from src.marl.simple_spread import SIMPLE_SPREAD_MAX_CYCLES, SimpleSpreadEnv
from src.marl.training import evaluator, trainer
from src.marl.woca_r_rl import dqn

# scripts.train imports the package as `marl`, a second copy of `src.marl` that
# re-registers the team envs. Restore the registrations so every test module keeps
# building the `src.marl` classes.
_registered_envs = dict(gym.registry)
from scripts.train import build_parser  # noqa: E402

gym.registry.update(_registered_envs)

N_AGENTS: int = 4
N_ACTIONS: int = 5  # no_action, left, right, down, up
SEED: int = 3


def _reference_env(n_agents: int) -> ParallelEnv[str, np.ndarray, int]:
    return simple_spread_v3.parallel_env(
        N=n_agents,
        local_ratio=0.0,
        max_cycles=SIMPLE_SPREAD_MAX_CYCLES,
        continuous_actions=False,
    )


def _actions(step: int, n_agents: int) -> tuple[int, ...]:
    return tuple((step + agent) % N_ACTIONS for agent in range(n_agents))


def test_make_env_builds_simple_spread_with_one_space_per_agent() -> None:
    env = make_env(N_AGENTS, name="simple_spread")
    try:
        assert isinstance(env, SimpleSpreadEnv)
        assert isinstance(env.observation_space, spaces.Tuple)
        assert isinstance(env.action_space, spaces.Tuple)
        assert len(env.observation_space.spaces) == N_AGENTS
        assert all(space == spaces.Discrete(N_ACTIONS) for space in env.action_space.spaces)

        obs, _info = env.reset(seed=SEED)
        assert len(obs) == N_AGENTS
        for agent_obs, space in zip(obs, env.observation_space.spaces):
            assert agent_obs.shape == space.shape
        assert any_vehicle_crashed(env) is None
    finally:
        env.close()


def test_step_matches_mpe2_in_agent_order_and_counts_the_shared_reward_once() -> None:
    env = SimpleSpreadEnv(N_AGENTS)
    reference = _reference_env(N_AGENTS)
    try:
        obs, _ = env.reset(seed=SEED)
        reference_obs, _ = reference.reset(seed=SEED)
        names = [f"agent_{index}" for index in range(N_AGENTS)]
        for agent_obs, name in zip(obs, names):
            np.testing.assert_array_equal(agent_obs, reference_obs[name])

        for step in range(3):
            actions = _actions(step, N_AGENTS)
            obs, reward, terminated, truncated, _ = env.step(actions)
            reference_obs, rewards, _, _, _ = reference.step(dict(zip(names, actions)))

            for agent_obs, name in zip(obs, names):
                np.testing.assert_array_equal(agent_obs, reference_obs[name])
            # local_ratio=0: every agent gets the same global reward, reported once.
            assert len({rewards[name] for name in names}) == 1
            assert reward == pytest.approx(rewards["agent_0"])
            assert reward < 0.0
            assert not terminated
            assert not truncated
    finally:
        env.close()
        reference.close()


def test_time_limit_truncates_without_terminating_and_keeps_the_final_obs() -> None:
    env = SimpleSpreadEnv(N_AGENTS)
    reference = _reference_env(N_AGENTS)
    try:
        env.reset(seed=SEED)
        reference.reset(seed=SEED)
        names = [f"agent_{index}" for index in range(N_AGENTS)]
        for step in range(SIMPLE_SPREAD_MAX_CYCLES):
            actions = _actions(step, N_AGENTS)
            obs, _, terminated, truncated, _ = env.step(actions)
            reference_obs, _, _, _, _ = reference.step(dict(zip(names, actions)))
            assert not terminated
            assert truncated == (step == SIMPLE_SPREAD_MAX_CYCLES - 1)

        assert len(obs) == N_AGENTS
        for agent_obs, name in zip(obs, names):
            np.testing.assert_array_equal(agent_obs, reference_obs[name])
    finally:
        env.close()
        reference.close()


def test_train_cli_defaults_to_simple_spread_and_keeps_the_old_env_flag() -> None:
    parser = build_parser()
    assert parser.parse_args([]).env == "simple_spread"
    assert parser.parse_args(["--env", "highway"]).env == "highway"
    assert parser.parse_args(["--dqn-env", "intersection"]).env == "intersection"


def test_mappo_rejects_the_dqn_only_intersection_env() -> None:
    args = build_parser().parse_args(["--env", "intersection"])
    with pytest.raises(ValueError, match="MAPPO supports"):
        trainer.train(args)


def test_dqn_still_accepts_the_legacy_namespace_field(monkeypatch: pytest.MonkeyPatch) -> None:
    args = build_parser().parse_args([])
    del args.env
    args.dqn_env = "intersection"
    make_legacy_env = Mock(side_effect=RuntimeError("environment selected"))
    monkeypatch.setattr(dqn, "make_env", make_legacy_env)

    with pytest.raises(RuntimeError, match="environment selected"):
        dqn.train_dqn(args, Console(quiet=True))

    make_legacy_env.assert_called_once_with(
        args.agents,
        name="intersection",
        reward_function=args.reward,
        team_weight=args.team_weight,
    )


def _record_env_names(monkeypatch: pytest.MonkeyPatch, module: object) -> list[str]:
    """Replace module.make_env with a headless one that records the requested env."""
    names: list[str] = []

    def fake_make_env(
        n_agents: int,
        render_mode: str | None = None,
        name: str = "highway",
        reward_function: str = "team",
        team_weight: float | None = None,
    ) -> gym.Env:
        names.append(name)
        return make_env(n_agents, name=name)

    monkeypatch.setattr(module, "make_env", fake_make_env)
    return names


def test_checkpoints_without_env_metadata_still_play_highway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mappo_names = _record_env_names(monkeypatch, trainer)
    dqn_names = _record_env_names(monkeypatch, dqn)
    obs_dim, n_actions, hidden = 25, 5, 8
    mappo_checkpoint = {
        "actor": Actor(obs_dim + 1, n_actions).state_dict(),
        "n_agents": 1,
        "obs_dim": obs_dim,
        "n_actions": n_actions,
    }
    dqn_checkpoint = {
        "robust_q": dqn._build_robust_q(
            obs_dim + 1, n_actions, hidden, torch.device("cpu")
        ).state_dict(),
        "n_agents": 1,
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "hidden": hidden,
    }

    trainer.build_mappo_player(mappo_checkpoint).close()
    dqn.build_dqn_player(dqn_checkpoint).close()

    assert mappo_names == ["highway"]
    assert dqn_names == ["highway"]


def _play_one_episode(player: PolicyPlayer) -> tuple[int, StepResult]:
    player.start_episode()
    steps = 0
    result = StepResult(finished=False, reward=0.0)
    while not result.finished:
        result = player.step()
        steps += 1
    player.close()
    return steps, result


@pytest.mark.parametrize(
    ("update_function", "extra_args"),
    [
        ("mappo", ["--rollout", "25", "--epochs", "1", "--minibatch", "25"]),
        ("wocar_dqn", ["--warmup", "20", "--batch-size", "8", "--explore-steps", "25"]),
    ],
)
def test_simple_spread_checkpoint_trains_and_plays_back(
    update_function: str,
    extra_args: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SDL_VIDEO_DRIVER", "dummy")  # human rendering without a window
    checkpoint_path = tmp_path / f"{update_function}.pt"
    args: Namespace = build_parser().parse_args(
        [
            "--update-function", update_function,
            "--agents", str(N_AGENTS),
            "--total-steps", "50",
            "--save", str(checkpoint_path),
            "--logdir", str(tmp_path / "runs"),
            "--run-name", update_function,
            *extra_args,
        ]
    )

    if update_function == "mappo":
        trainer.train(args)
    else:
        build_trainer = dqn._build_trainer
        trained: list[dqn.WocaRDQNTrainer] = []

        def record_trainer(
            obs_dim: int,
            n_agents: int,
            n_actions: int,
            args: Namespace,
            device: torch.device,
            observation_bounds: tuple[float, float],
        ) -> dqn.WocaRDQNTrainer:
            instance = build_trainer(obs_dim, n_agents, n_actions, args, device, observation_bounds)
            trained.append(instance)
            return instance

        monkeypatch.setattr(dqn, "_build_trainer", record_trainer)
        dqn.train_dqn(args, Console(quiet=True))
        _assert_zero_radius_bounds(trained[0])

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint.get("env", checkpoint.get("dqn_env")) == "simple_spread"
    steps, result = _play_one_episode(evaluator._build_player(str(checkpoint_path)))
    assert steps == SIMPLE_SPREAD_MAX_CYCLES
    assert result.crashed is None


def _assert_zero_radius_bounds(training: dqn.WocaRDQNTrainer) -> None:
    env = SimpleSpreadEnv(N_AGENTS)
    try:
        obs, _ = env.reset(seed=SEED)
        device = next(training._robust_q.parameters()).device
        states = torch.as_tensor(build_actor_obs(flatten_obs(obs), N_AGENTS), device=device)
        assert states.abs().max().item() > 1.0  # this state would be clipped by the driving bounds
        with torch.no_grad():
            clean_q = training._robust_q(states)
            bounds = training._bounder.bounds(states, epsilon=0.0)
        torch.testing.assert_close(bounds.lower, clean_q)
        torch.testing.assert_close(bounds.upper, clean_q)
    finally:
        env.close()
