"""Evaluation of saved policies. Picks the player from the checkpoint's algorithm name."""

from __future__ import annotations

import json
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NamedTuple

import numpy as np
import torch
from torch import Tensor

from ..attacks.registry import ObservationAttack, build_attack
from ..rendering import PolicyPlayer, StepResult, evaluate_in_windows
from .attack_logging import (
    RESULTS_FILE,
    create_attack_writer,
    default_run_name,
    log_attack_result,
    reserve_attack_folder,
)

if TYPE_CHECKING:
    from ..woca_r_rl.dqn import DqnPlayer

ATTACK_ALGORITHM: str = "wocar_dqn"
ATTACK_RNG_STREAM: int = 1  # keeps the car shuffling independent of the env seeds
DEFAULT_ALGORITHM: str = "mappo"  # for checkpoints saved before the field existed


def evaluate(args: Namespace) -> None:
    evaluate_in_windows(
        partial(_build_player, args.eval), args.eval_episodes, args.eval_windows
    )


def _build_player(checkpoint_path: str) -> PolicyPlayer:
    """Top-level so it pickles under spawn. Runs in each child process."""
    checkpoint: dict = torch.load(checkpoint_path, map_location="cpu")
    algorithm: str = checkpoint.get("algorithm", DEFAULT_ALGORITHM)
    return _builder_for(algorithm)(checkpoint)


def _builder_for(algorithm: str) -> Callable[[dict], PolicyPlayer]:
    # Imported lazily, so evaluating one algorithm doesn't import the others' dependencies.
    if algorithm == "mappo":
        from .trainer import build_mappo_player
        return build_mappo_player
    if algorithm == "wocar_dqn":
        from ..woca_r_rl.dqn import build_dqn_player
        return build_dqn_player
    if algorithm == "wocar_ppo":
        from ..woca_r_rl.ppo import build_ppo_player
        return build_ppo_player
    raise ValueError(f"Unknown algorithm in checkpoint: {algorithm}")


class EpisodeResult(NamedTuple):
    episode_return: float
    crashed: bool
    attacked_cars: list[int]
    attacked_decisions: int
    changed_decisions: int


def evaluate_attacked(args: Namespace) -> dict:
    """Evaluation under the attack (headless unless --render); prints, logs and saves the results in its own folder."""
    checkpoint: dict = torch.load(args.eval, map_location="cpu")
    n_agents: int = _checked_agent_count(checkpoint)
    attacked_agents: int = _attacked_count(args.attacked_agents, n_agents)
    if args.render_agents is not None and not 1 <= args.render_agents <= n_agents:
        raise ValueError(f"--render-agents must be between 1 and {n_agents}, got {args.render_agents}")
    if args.eval_episodes < 1:
        raise ValueError(f"--eval-episodes must be at least 1, got {args.eval_episodes}")
    if args.out and not Path(args.out).resolve().parent.is_dir():
        raise ValueError(f"--out directory does not exist: {Path(args.out).parent}")

    attacked_cars: list[list[int]] = attacked_cars_per_episode(
        n_agents, attacked_agents, args.eval_episodes, args.seed
    )
    epsilons: list[float] = _attack_epsilons(args.attack_eps, attacked_agents)
    build_attack(args.attack, args.seed)  # rejects an unknown name before the player is created
    report: dict = {
        "settings": {
            "checkpoint": args.eval,
            "env": checkpoint.get("dqn_env", "highway"),
            "n_agents": n_agents,
            "attack": args.attack,
            "attacked_agents": attacked_agents,
            "epsilons": epsilons,
            "episodes": args.eval_episodes,
            "seed": args.seed,
        },
        "results": [],
    }

    run_name: str = args.run_name or default_run_name(args.attack, attacked_agents, n_agents)
    folder: Path = reserve_attack_folder(args.logdir, run_name)
    print(f"Attack evaluation folder: {folder}", flush=True)
    player: DqnPlayer = _builder_for(ATTACK_ALGORITHM)(
        checkpoint, render=args.render, render_agents=args.render_agents
    )
    writer = create_attack_writer(folder)
    try:
        for epsilon in epsilons:
            # Rebuilt per radius, so a seeded attack draws the same numbers for every radius.
            attack: ObservationAttack = build_attack(args.attack, args.seed)
            episodes: list[EpisodeResult] = [
                _run_attacked_episode(
                    player, attack, epsilon, cars, n_agents, args.seed + episode
                )
                for episode, cars in enumerate(attacked_cars)
            ]
            result: dict = _summarize(epsilon, episodes)
            report["results"].append(result)
            _print_result(result, attacked_agents, n_agents)
            log_attack_result(writer, report["settings"], result)
    finally:
        writer.close()
        player.close()

    results: str = json.dumps(report, indent=2)
    (folder / RESULTS_FILE).write_text(results)
    if args.out:
        Path(args.out).write_text(results)
    return report


def attacked_cars_per_episode(
    n_agents: int, attacked_agents: int, episodes: int, seed: int
) -> list[list[int]]:
    """Attacked cars of every episode; a larger count keeps the cars of a smaller one."""
    rng: np.random.Generator = np.random.default_rng([seed, ATTACK_RNG_STREAM])
    return [rng.permutation(n_agents)[:attacked_agents].tolist() for _ in range(episodes)]


def _checked_agent_count(checkpoint: dict) -> int:
    """Number of cars in the checkpoint; only WocaR-DQN can be attacked."""
    algorithm: str = checkpoint.get("algorithm", DEFAULT_ALGORITHM)
    if algorithm != ATTACK_ALGORITHM:
        raise ValueError(f"--attack only supports {ATTACK_ALGORITHM} checkpoints, got {algorithm}")
    return checkpoint["n_agents"]


def _attacked_count(requested: int | str, n_agents: int) -> int:
    """Turn a count or "all" into a number of cars."""
    count: int = n_agents if requested == "all" else int(requested)
    if not 0 <= count <= n_agents:
        raise ValueError(f"--attacked-agents must be between 0 and {n_agents}, got {count}")
    return count


def _attack_epsilons(requested: list[float] | None, attacked_agents: int) -> list[float]:
    """The requested radii; 0 when no car is attacked."""
    if attacked_agents == 0:
        return [0.0]
    if requested is None:
        raise ValueError("--attack-eps is required with --attack")
    return requested


@torch.no_grad()
def _run_attacked_episode(
    player: DqnPlayer,
    attack: ObservationAttack,
    epsilon: float,
    attacked_cars: list[int],
    n_agents: int,
    env_seed: int,
) -> EpisodeResult:
    attacked_mask: Tensor = torch.zeros(n_agents, dtype=torch.bool)
    attacked_mask[attacked_cars] = True
    player.set_attack(attack, epsilon, attacked_mask)
    player.start_episode(seed=env_seed)

    episode_return: float = 0.0
    crashed: bool = False
    result: StepResult = StepResult(finished=False, reward=0.0)
    while not result.finished:
        result = player.step()
        episode_return += result.reward
        crashed = crashed or bool(result.crashed)
    return EpisodeResult(episode_return, crashed, attacked_cars, *player.attack_counts)


def _summarize(epsilon: float, episodes: list[EpisodeResult]) -> dict:
    """Averages over the episodes, plus the per-episode values."""
    returns: list[float] = [episode.episode_return for episode in episodes]
    crashes: list[bool] = [episode.crashed for episode in episodes]
    attacked: int = sum(episode.attacked_decisions for episode in episodes)
    changed: int = sum(episode.changed_decisions for episode in episodes)
    return {
        "epsilon": epsilon,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "crash_rate": float(np.mean(crashes)),
        "changed_action_rate": changed / attacked if attacked > 0 else None,
        "episode_returns": returns,
        "episode_crashes": crashes,
        "episode_attacked_cars": [episode.attacked_cars for episode in episodes],
    }


def _print_result(result: dict, attacked_agents: int, n_agents: int) -> None:
    changed: float | None = result["changed_action_rate"]
    changed_text: str = "n/a" if changed is None else f"{changed:.3f}"
    print(
        f"attacked={attacked_agents}/{n_agents}, eps={result['epsilon']:g}: "
        f"return={result['return_mean']:.2f} +/- {result['return_std']:.2f}, "
        f"crash_rate={result['crash_rate']:.3f}, changed_actions={changed_text}",
        flush=True,
    )
