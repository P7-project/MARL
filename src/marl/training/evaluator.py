"""Evaluation loop for saved MAPPO, WocaR-DQN and WocaR-PPO policies."""

import os
from argparse import Namespace
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from rich.console import Console

from ..env import build_actor_obs, flatten_obs, make_env
from ..models import Actor
from ..woca_r_rl.dqn import build_robust_q
from ..woca_r_rl.models import GaussianActor

console = Console()

Checkpoint = dict[str, Any]
TeamPolicy = Callable[[np.ndarray], tuple[int, ...]]

MAX_WINDOWS: int = 12
WINDOWS_PER_ROW: int = 3


def evaluate(args: Namespace) -> None:
    checkpoint: Checkpoint = torch.load(args.eval, map_location="cpu")
    # Checkpoints written before the "algorithm" key existed are MAPPO checkpoints.
    algorithm: str = checkpoint.get("algorithm", "mappo")

    if algorithm == "wocar_ppo":
        _evaluate_single_agent(args, checkpoint)
    elif algorithm in ("mappo", "wocar_dqn"):
        _evaluate_team(args, checkpoint, algorithm)
    else:
        raise NotImplementedError(f"Cannot evaluate a '{algorithm}' checkpoint.")


def _evaluate_team(args: Namespace, checkpoint: Checkpoint, algorithm: str) -> None:
    """One process per controlled car, each running the whole team with its window following that car."""
    n_agents: int = checkpoint["n_agents"]
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_team_worker, args=(args, view_index, checkpoint, algorithm))
        for view_index in range(min(n_agents, MAX_WINDOWS))
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()


@torch.no_grad()
def _evaluate_single_agent(args: Namespace, checkpoint: Checkpoint) -> None:
    actor: GaussianActor = GaussianActor(
        checkpoint["obs_dim"], checkpoint["action_dim"], torch.device("cpu")
    )
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env: gym.Env = gym.make(checkpoint["env_id"], render_mode=_render_mode(args))
    for episode in range(args.eval_episodes):
        obs, _ = env.reset()
        done: bool = False
        total_return: float = 0.0
        steps: int = 0

        while not done:
            obs_batch: torch.Tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mean_action: np.ndarray = actor.mean_network(obs_batch).squeeze(0).numpy()
            action: np.ndarray = np.clip(mean_action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_return += float(reward)
            steps += 1
            done = terminated or truncated

        console.print(f"Eval episode {episode + 1}: steps={steps}, return={total_return:.2f}")
    env.close()


@torch.no_grad()  # applies inside the spawned process, unlike a decorator on evaluate()
def _team_worker(args: Namespace, view_index: int, checkpoint: Checkpoint, algorithm: str) -> None:
    n_agents: int = checkpoint["n_agents"]
    _position_window(view_index)
    select_actions: TeamPolicy = _build_team_policy(checkpoint, algorithm)

    env: gym.Env = make_env(n_agents, render_mode=_render_mode(args))
    for episode in range(args.eval_episodes):
        obs, _ = env.reset()
        _follow_vehicle(env, view_index)
        obs = flatten_obs(obs)
        done: bool = False
        total_return: float = 0.0
        steps: int = 0
        info: dict[str, Any] = {}

        while not done:
            obs, reward, terminated, truncated, info = env.step(select_actions(obs))
            obs = flatten_obs(obs)
            total_return += float(reward)
            steps += 1
            done = terminated or truncated

        console.print(
            f"(Car {view_index + 1}): Eval episode {episode + 1}: steps={steps}, "
            f"return={total_return:.2f}, crashed={info.get('crashed')}"
        )
    env.close()


def _build_team_policy(checkpoint: Checkpoint, algorithm: str) -> TeamPolicy:
    """Greedy policy over the actor logits (MAPPO) or the Q_r values (WocaR-DQN)."""
    n_agents: int = checkpoint["n_agents"]
    in_dim: int = checkpoint["obs_dim"] + n_agents

    scores: torch.nn.Module
    if algorithm == "mappo":
        actor: Actor = Actor(in_dim, checkpoint["n_actions"])
        actor.load_state_dict(checkpoint["actor"])
        scores = actor.net
    else:
        scores = build_robust_q(in_dim, checkpoint["n_actions"], checkpoint["hidden"], torch.device("cpu"))
        scores.load_state_dict(checkpoint["robust_q"])
    scores.eval()

    def select_actions(obs: np.ndarray) -> tuple[int, ...]:
        actor_obs: torch.Tensor = torch.as_tensor(build_actor_obs(obs, n_agents))
        return tuple(int(action) for action in scores(actor_obs).argmax(dim=-1))

    return select_actions


def _render_mode(args: Namespace) -> str | None:
    return "human" if args.render else None


def _position_window(view_index: int) -> None:
    """Offset each window so they do not stack on top of each other."""
    column: int = view_index % WINDOWS_PER_ROW
    row: int = view_index // WINDOWS_PER_ROW
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{25 + column * 625},{25 + row * 175}"


def _follow_vehicle(env: gym.Env, view_index: int) -> None:
    """Point the render camera at one controlled car. The viewer exists once reset() has rendered."""
    viewer = getattr(env.unwrapped, "viewer", None)
    if viewer is not None:
        viewer.observer_vehicle = env.unwrapped.controlled_vehicles[view_index]
