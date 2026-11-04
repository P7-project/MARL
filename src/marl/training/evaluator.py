"""Evaluation loop for saved MAPPO policies."""

import os

import numpy as np
import torch
from rich.console import Console

from ..env import build_actor_obs, flatten_obs, make_env
from ..models import Actor


console = Console()


@torch.no_grad()
def _evaluate_in_env(checkpoint, env_cfg, n_agents, eval_episodes, render, label=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(checkpoint["obs_dim"] + n_agents, checkpoint["n_actions"]).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env = make_env(n_agents, render_mode="human" if render else None, env_cfg=env_cfg)
    obs_dim = int(np.prod(env.observation_space.spaces[0].shape))
    n_actions = int(env.action_space.spaces[0].n)
    if obs_dim != checkpoint["obs_dim"] or n_actions != checkpoint["n_actions"]:
        env.close()
        raise ValueError(
            f"Environment config does not match checkpoint: got obs_dim={obs_dim}, "
            f"n_actions={n_actions}; checkpoint expects obs_dim={checkpoint['obs_dim']}, "
            f"n_actions={checkpoint['n_actions']}. Pass the same --config used for training."
        )

    try:
        for episode in range(eval_episodes):
            obs, _ = env.reset()
            obs = flatten_obs(obs)
            done = False
            total_return = 0.0
            steps = 0

            while not done:
                actor_obs = torch.as_tensor(build_actor_obs(obs, n_agents), device=device)
                actions = tuple(
                    int(item)
                    for item in actor.net(actor_obs).argmax(-1).cpu().numpy()
                )
                obs, reward, terminated, truncated, _info = env.step(actions)
                obs = flatten_obs(obs)
                total_return += float(np.asarray(reward, dtype=np.float32).mean())
                steps += 1
                done = bool(np.any(terminated)) or bool(truncated)

            crashed = any(
                vehicle.crashed for vehicle in env.unwrapped.controlled_vehicles
            )
            console.print(
                f"{label}Eval episode {episode + 1}: steps={steps}, "
                f"return={total_return:.2f}, crashed={crashed}"
            )
    finally:
        env.close()


def _render_worker(eval_path, env_cfg, n_agents, eval_episodes, idx):
    # Place human-rendered windows in a grid so they remain visible together.
    screen_width = 600
    screen_height = 150
    if env_cfg:
        overrides = env_cfg.get("overrides", {})
        screen_width = overrides.get("screen_width", screen_width)
        screen_height = overrides.get("screen_height", screen_height)
    column, row = idx % 3, idx // 3
    os.environ["SDL_VIDEO_WINDOW_POS"] = (
        f"{25 + column * (screen_width + 25)},"
        f"{25 + row * (screen_height + 25)}"
    )

    checkpoint = torch.load(eval_path, map_location="cpu")
    _evaluate_in_env(
        checkpoint,
        env_cfg,
        n_agents,
        eval_episodes,
        render=True,
        label=f"View {idx + 1}: ",
    )


def evaluate(args):
    checkpoint = torch.load(args.eval, map_location="cpu")
    n_agents = checkpoint["n_agents"]
    env_cfg = getattr(args, "env_cfg", None) or checkpoint.get("env_cfg")

    if args.render:
        # One rendered evaluation process per controlled agent, matching the
        # historical parallel-view behavior. Cap windows to keep the desktop usable.
        import torch.multiprocessing as mp

        process_count = min(n_agents, 12)
        ctx = mp.get_context("spawn")
        workers = [
            ctx.Process(
                target=_render_worker,
                args=(args.eval, env_cfg, n_agents, args.eval_episodes, idx),
            )
            for idx in range(process_count)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        failed = [worker.exitcode for worker in workers if worker.exitcode != 0]
        if failed:
            raise RuntimeError(
                f"{len(failed)} parallel evaluation view(s) failed; exit codes: {failed}"
            )
    else:
        _evaluate_in_env(
            checkpoint,
            env_cfg,
            n_agents,
            args.eval_episodes,
            render=False,
        )
