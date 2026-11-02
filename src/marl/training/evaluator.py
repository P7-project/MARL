"""Evaluation loop for a saved MAPPO highway policy."""

import torch
from rich.console import Console

from ..env import build_actor_obs, flatten_obs, make_env
from ..models import Actor


console = Console()


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.eval, map_location=device)
    n_agents = checkpoint["n_agents"]

    actor = Actor(checkpoint["obs_dim"] + n_agents, checkpoint["n_actions"]).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env = make_env(n_agents, render_mode="human" if args.render else None)
    for episode in range(args.eval_episodes):
        obs, _ = env.reset()
        obs = flatten_obs(obs)
        done = False
        total_return = 0.0
        steps = 0
        info = {}

        while not done:
            actor_obs = torch.as_tensor(build_actor_obs(obs, n_agents), device=device)
            actions = tuple(int(item) for item in actor.net(actor_obs).argmax(-1).cpu().numpy())
            obs, reward, terminated, truncated, info = env.step(actions)
            obs = flatten_obs(obs)
            total_return += float(reward)
            steps += 1
            done = terminated or truncated

        console.print(
            f"Eval episode {episode + 1}: steps={steps}, return={total_return:.2f}, "
            f"crashed={info.get('crashed')}"
        )

    env.close()
