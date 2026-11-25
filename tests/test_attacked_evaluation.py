import json
import subprocess
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import highway_env.envs.common.abstract as highway_abstract
import numpy as np
import pytest
import torch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts.evaluate import build_parser
from src.marl.attacks.registry import ObservationAttack, build_attack
from src.marl.training import attack_logging
from src.marl.training.evaluator import attacked_cars_per_episode, evaluate_attacked
from src.marl.woca_r_rl.dqn import DqnPlayer, _build_robust_q, build_dqn_player

N_AGENTS: int = 5
HIDDEN: int = 16
EPISODES: int = 2
ENV_SIZES: dict[str, tuple[int, int]] = {"highway": (25, 5), "intersection": (105, 3)}
SCRIPT: Path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"


@pytest.fixture(autouse=True)
def no_window(monkeypatch) -> None:
    """Creating a viewer makes any of these tests fail."""

    def fail(*args, **kwargs) -> None:
        raise AssertionError("a window was created")

    monkeypatch.setattr(highway_abstract, "EnvViewer", fail)


@pytest.fixture(params=["highway", "intersection"])
def checkpoint(request) -> dict:
    return _checkpoint(request.param)


def _checkpoint(env_name: str) -> dict:
    """An untrained policy for the given environment."""
    obs_dim, n_actions = ENV_SIZES[env_name]
    torch.manual_seed(0)
    network = _build_robust_q(obs_dim + N_AGENTS, n_actions, HIDDEN, torch.device("cpu"))
    return {
        "algorithm": "wocar_dqn",
        "robust_q": network.state_dict(),
        "n_agents": N_AGENTS,
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "hidden": HIDDEN,
        "dqn_env": env_name,
    }


def _fake_attack(
    q_network: torch.nn.Module,
    actor_obs: torch.Tensor,
    epsilon: float,
    attacked_mask: torch.Tensor,
    obs_dim: int,
) -> torch.Tensor:
    """Replace the observations of the masked cars by epsilon."""
    perturbed: torch.Tensor = actor_obs.clone()
    perturbed[attacked_mask, :obs_dim] = epsilon
    return perturbed


class _FakeAttack:
    """An attack with its own state, such as a loaded network."""

    def __init__(self, scale: float) -> None:
        self._scale: float = scale

    def __call__(
        self,
        q_network: torch.nn.Module,
        actor_obs: torch.Tensor,
        epsilon: float,
        attacked_mask: torch.Tensor,
        obs_dim: int,
    ) -> torch.Tensor:
        return _fake_attack(q_network, actor_obs, self._scale * epsilon, attacked_mask, obs_dim)


def _args(checkpoint: dict, tmp_path: Path, **overrides) -> Namespace:
    path: Path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    values: dict = {
        "eval": str(path),
        "attack": "minbest",
        "logdir": str(tmp_path / "runs"),
        "run_name": None,
        "attack_eps": [0.0, 0.3],
        "attacked_agents": 1,
        "eval_episodes": EPISODES,
        "seed": 0,
        "out": None,
    }
    return Namespace(**{**values, **overrides})


def _natural_episodes(checkpoint: dict) -> tuple[list[float], list[bool]]:
    """Episodes of the greedy policy without any attack set."""
    player: DqnPlayer = build_dqn_player(checkpoint, render=False)
    returns: list[float] = []
    crashes: list[bool] = []
    for episode in range(EPISODES):
        player.start_episode(seed=episode)
        results = [player.step()]
        while not results[-1].finished:
            results.append(player.step())
        returns.append(sum(result.reward for result in results))
        crashes.append(any(bool(result.crashed) for result in results))
    player.close()
    return returns, crashes


def _actions_under_attack(
    player: DqnPlayer, attack: ObservationAttack, epsilon: float, cars: list[int]
) -> list[int]:
    mask: torch.Tensor = torch.zeros(N_AGENTS, dtype=torch.bool)
    mask[cars] = True
    player.set_attack(attack, epsilon, mask)
    assert player._obs is not None
    return player._greedy_actions(player._obs)


@pytest.mark.parametrize(
    "old, new, values",
    [
        ("--eval", "--checkpoint", ["other.pt"]),
        ("--eval-episodes", "--episodes", ["7"]),
        ("--attack-eps", "--epsilon", ["0.05", "0.15"]),
    ],
)
def test_alias_equals_its_old_name(old, new, values) -> None:
    required: list[str] = ["--eval", "unused.pt"]
    parsed: Namespace = build_parser().parse_args(required + [new] + values)
    assert parsed == build_parser().parse_args(required + [old] + values)
    assert parsed != build_parser().parse_args(required)


def test_checkpoint_alone_satisfies_the_required_option() -> None:
    parsed: Namespace = build_parser().parse_args(["--checkpoint", "some.pt"])
    assert parsed.eval == "some.pt"
    assert (parsed.eval_episodes, parsed.attack_eps) == (None, None)


def test_attacked_cars_are_reproducible_and_nested() -> None:
    by_count: dict[int, list[list[int]]] = {
        count: attacked_cars_per_episode(N_AGENTS, count, 50, seed=3) for count in (0, 1, 3, 5)
    }
    for count, per_episode in by_count.items():
        assert per_episode == attacked_cars_per_episode(N_AGENTS, count, 50, seed=3)
        assert all(len(set(cars)) == count for cars in per_episode)
    for one, three in zip(by_count[1], by_count[3]):
        assert set(one) <= set(three)
    assert {cars[0] for cars in by_count[1]} == set(range(N_AGENTS))
    assert by_count[1] != attacked_cars_per_episode(N_AGENTS, 1, 50, seed=4)


def test_short_run_is_reproducible_and_saved(checkpoint, tmp_path) -> None:
    out: Path = tmp_path / "results.json"
    report: dict = evaluate_attacked(_args(checkpoint, tmp_path, out=str(out), run_name="run"))
    assert json.loads(out.read_text()) == report
    assert json.loads((tmp_path / "runs" / "attack_evaluation" / "run" / "results.json").read_text()) == report
    assert evaluate_attacked(_args(checkpoint, tmp_path)) == report
    assert report["settings"] == {
        "checkpoint": str(tmp_path / "checkpoint.pt"),
        "env": checkpoint["dqn_env"],
        "n_agents": N_AGENTS,
        "attack": "minbest",
        "attacked_agents": 1,
        "epsilons": [0.0, 0.3],
        "episodes": EPISODES,
        "seed": 0,
    }
    zero, attacked = report["results"]
    assert zero["episode_attacked_cars"] == attacked["episode_attacked_cars"]
    assert zero["changed_action_rate"] == 0.0
    for result in report["results"]:
        assert len(result["episode_returns"]) == EPISODES
        assert np.isfinite(result["episode_returns"]).all()
        assert result["return_mean"] == pytest.approx(np.mean(result["episode_returns"]))
        assert result["return_std"] == pytest.approx(np.std(result["episode_returns"]))
        assert result["crash_rate"] == np.mean(result["episode_crashes"])
        assert 0.0 <= result["changed_action_rate"] <= 1.0


def _logged_scalars(run: Path) -> dict[str, list[float]]:
    events: EventAccumulator = EventAccumulator(str(run))
    events.Reload()
    return {tag: [event.value for event in events.Scalars(tag)] for tag in events.Tags()["scalars"]}


def test_results_are_logged_to_tensorboard(tmp_path) -> None:
    args: Namespace = _args(_checkpoint("highway"), tmp_path, run_name="run")
    report: dict = evaluate_attacked(args)
    logged = _logged_scalars(tmp_path / "runs" / "attack_evaluation" / "run")
    for result in report["results"]:
        tag: str = f"minbest/attacked_1/epsilon_{result['epsilon']:g}"
        assert logged[f"{tag}/episode_reward"] == pytest.approx(result["episode_returns"])
        assert logged[f"{tag}/episode_crashed"] == result["episode_crashes"]
        assert logged[f"{tag}/reward_mean"] == pytest.approx([result["return_mean"]])
        assert logged[f"{tag}/reward_std"] == pytest.approx([result["return_std"]])
        assert logged[f"{tag}/crash_rate"] == pytest.approx([result["crash_rate"]])
        assert logged[f"{tag}/changed_action_rate"] == pytest.approx([result["changed_action_rate"]])


def test_each_run_gets_its_own_folder_with_results(tmp_path, monkeypatch) -> None:
    class _FixedTime(datetime):
        @classmethod
        def now(cls, tz=None) -> datetime:
            return datetime(2026, 11, 25, 22, 10, 5)

    monkeypatch.setattr(attack_logging, "datetime", _FixedTime)
    checkpoint: dict = _checkpoint("highway")
    reports: list[dict] = [evaluate_attacked(_args(checkpoint, tmp_path)) for _ in range(2)]
    root: Path = tmp_path / "runs" / "attack_evaluation"
    folders: list[str] = sorted(path.name for path in root.iterdir())
    assert folders == ["minbest_1of5_25_22-10-05", "minbest_1of5_25_22-10-05_2"]
    for name, report in zip(folders, reports):
        assert json.loads((root / name / "results.json").read_text()) == report
        assert _logged_scalars(root / name)


def test_explicit_run_name_is_not_reused(tmp_path) -> None:
    checkpoint: dict = _checkpoint("highway")
    for _ in range(2):
        evaluate_attacked(_args(checkpoint, tmp_path, run_name="run"))
    root: Path = tmp_path / "runs" / "attack_evaluation"
    assert sorted(path.name for path in root.iterdir()) == ["run", "run_2"]
    assert (root / "run_2" / "results.json").is_file()


def test_no_attacked_cars_are_logged_as_clean(tmp_path) -> None:
    evaluate_attacked(_args(_checkpoint("highway"), tmp_path, attacked_agents=0, run_name="run"))
    logged = _logged_scalars(tmp_path / "runs" / "attack_evaluation" / "run")
    assert set(logged) == {
        f"clean/{name}"
        for name in ("episode_reward", "episode_crashed", "reward_mean", "reward_std", "crash_rate")
    }


def test_env_seeds_are_the_same_for_every_epsilon_and_count(
    checkpoint, tmp_path, monkeypatch
) -> None:
    starts: list[tuple[int | None, np.ndarray]] = []
    start_episode = DqnPlayer.start_episode

    def record(self: DqnPlayer, seed: int | None = None) -> None:
        start_episode(self, seed)
        starts.append((seed, self._obs))

    monkeypatch.setattr(DqnPlayer, "start_episode", record)
    for count in (1, "all"):
        evaluate_attacked(_args(checkpoint, tmp_path, attacked_agents=count, seed=7))
    assert [seed for seed, _ in starts] == [7, 8] * 4
    for index, (_, obs) in enumerate(starts):
        np.testing.assert_array_equal(obs, starts[index % EPISODES][1])
    assert not np.array_equal(starts[0][1], starts[1][1])


def test_zero_cars_and_zero_epsilon_match_natural_policy(checkpoint, tmp_path) -> None:
    returns, crashes = _natural_episodes(checkpoint)
    no_cars: dict = evaluate_attacked(
        _args(checkpoint, tmp_path, attacked_agents=0, attack_eps=[0.1, 0.3])
    )
    no_eps: dict = evaluate_attacked(
        _args(checkpoint, tmp_path, attacked_agents=0, attack_eps=None)
    )
    zero_eps: dict = evaluate_attacked(
        _args(checkpoint, tmp_path, attacked_agents="all", attack_eps=[0.0])
    )
    assert no_cars["results"] == no_eps["results"]
    for report in (no_cars, zero_eps):
        assert len(report["results"]) == 1
        assert report["results"][0]["episode_returns"] == returns
        assert report["results"][0]["episode_crashes"] == crashes
    assert no_cars["results"][0]["changed_action_rate"] is None
    assert no_cars["results"][0]["episode_attacked_cars"] == [[]] * EPISODES


def test_epsilons_ignore_the_checkpoint_eps_final(checkpoint, tmp_path) -> None:
    checkpoint["eps_final"] = 0.05
    report: dict = evaluate_attacked(_args(checkpoint, tmp_path, attack_eps=[0.02]))
    assert report["settings"]["epsilons"] == [0.02]
    assert [result["epsilon"] for result in report["results"]] == [0.02]


def test_missing_epsilon_is_rejected_before_creating_the_player(
    checkpoint, tmp_path, monkeypatch
) -> None:
    def fail(*args, **kwargs) -> None:
        raise AssertionError("a player was created")

    checkpoint["eps_final"] = 0.05
    monkeypatch.setattr(DqnPlayer, "__init__", fail)
    with pytest.raises(ValueError, match="--attack-eps is required"):
        evaluate_attacked(_args(checkpoint, tmp_path, attack_eps=None))
    assert not (tmp_path / "runs" / "attack_evaluation").exists()


def test_zero_cars_need_no_epsilon(checkpoint, tmp_path) -> None:
    report: dict = evaluate_attacked(
        _args(checkpoint, tmp_path, attacked_agents=0, attack_eps=None)
    )
    assert report["settings"]["epsilons"] == [0.0]
    assert [result["epsilon"] for result in report["results"]] == [0.0]


def test_attack_changes_only_the_attacked_cars(checkpoint) -> None:
    player: DqnPlayer = build_dqn_player(checkpoint, render=False)
    player.start_episode(seed=0)
    assert player._obs is not None
    clean: list[int] = player._greedy_actions(player._obs)
    assert player.attack_counts == (0, 0)
    attack: ObservationAttack = build_attack("minbest")
    everyone: list[int] = _actions_under_attack(player, attack, 1.0, list(range(N_AGENTS)))
    changed: list[int] = [car for car in range(N_AGENTS) if everyone[car] != clean[car]]
    assert changed
    assert player.attack_counts == (N_AGENTS, len(changed))

    expected: list[int] = list(clean)
    expected[changed[0]] = everyone[changed[0]]
    assert _actions_under_attack(player, attack, 1.0, [changed[0]]) == expected
    assert player.attack_counts == (N_AGENTS + 1, len(changed) + 1)
    player.close()


@pytest.mark.parametrize(
    "attack, constant", [(_fake_attack, 0.5), (_FakeAttack(scale=-2.0), -1.0)]
)
def test_player_uses_the_attack_it_is_given(checkpoint, monkeypatch, attack, constant) -> None:
    player: DqnPlayer = build_dqn_player(checkpoint, render=False)
    player.start_episode(seed=0)
    assert player._obs is not None
    seen: list[torch.Tensor] = []
    forward = player._policy.forward

    def record(actor_obs: torch.Tensor) -> torch.Tensor:
        seen.append(actor_obs)
        return forward(actor_obs)

    monkeypatch.setattr(player._policy, "forward", record)
    cars: list[int] = [0, 2]
    actions: list[int] = _actions_under_attack(player, attack, 0.5, cars)

    clean, perturbed = seen
    expected: torch.Tensor = clean.clone()
    expected[cars, : checkpoint["obs_dim"]] = constant
    assert torch.equal(perturbed, expected)
    with torch.no_grad():
        assert actions == forward(expected).argmax(dim=1).tolist()
    player.close()


def test_unknown_attack_is_rejected_before_creating_the_player(
    checkpoint, tmp_path, monkeypatch
) -> None:
    def fail(*args, **kwargs) -> None:
        raise AssertionError("a player was created")

    monkeypatch.setattr(DqnPlayer, "__init__", fail)
    with pytest.raises(ValueError, match="Unknown attack: 'unknown'. Available: minbest"):
        evaluate_attacked(_args(checkpoint, tmp_path, attack="unknown"))


def test_headless_player_creates_no_window(checkpoint) -> None:
    player: DqnPlayer = build_dqn_player(checkpoint, render=False)
    player.start_episode(seed=0)
    player.step()
    assert player._env.unwrapped.render_mode is None
    assert player._env.unwrapped.viewer is None
    player.close()


@pytest.mark.parametrize("algorithm", ["mappo", "wocar_ppo", None])
def test_wrong_algorithm_is_rejected(tmp_path, algorithm) -> None:
    checkpoint: dict = {} if algorithm is None else {"algorithm": algorithm}
    with pytest.raises(ValueError, match="only supports wocar_dqn"):
        evaluate_attacked(_args(checkpoint, tmp_path))


@pytest.mark.parametrize("count", [-1, N_AGENTS + 1])
def test_invalid_car_count_is_rejected(tmp_path, count) -> None:
    checkpoint: dict = {"algorithm": "wocar_dqn", "n_agents": N_AGENTS}
    with pytest.raises(ValueError, match="--attacked-agents must be between 0 and 5"):
        evaluate_attacked(_args(checkpoint, tmp_path, attacked_agents=count))


@pytest.mark.parametrize(
    "flags, message",
    [
        (["--attack-eps=-0.1"], "must be a finite number >= 0"),
        (["--attack-eps", "nan"], "must be a finite number >= 0"),
        (["--attack-eps", "0.1", "--attacked-agents=-1"], "must be a count >= 0 or 'all'"),
        ([], "--attack-eps (or --epsilon) is required with --attack"),
        (["--attacked-agents", "0"], "--attack-eps (or --epsilon) is required with --attack"),
    ],
)
def test_script_rejects_invalid_flags(flags, message) -> None:
    command: list[str] = [sys.executable, str(SCRIPT), "--eval", "unused.pt", "--attack", "minbest"]
    result = subprocess.run(command + flags, capture_output=True, text=True)
    assert result.returncode == 2
    assert message in result.stderr
