"""Evaluation loop for a saved MAPPO highway policy."""

import torch
import math
import os
import torch.multiprocessing as mp
from rich.console import Console
from ..env import build_actor_obs, flatten_obs, make_env
from ..models import Actor

console = Console()

def eval_worker(args, idx, checkpoint, n_agents, device):
    # Offset each window so they don't stack on top of each other
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{25 + idx % 3 * 625},{25 + 175*math.floor(idx/3)}"

    # Evaluate this car
    actor = Actor(checkpoint["obs_dim"] + n_agents, checkpoint["n_actions"]).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    env = make_env(n_agents, render_mode="human")
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

        # Show information about the car while evaluating
        console.print(
            f"(Car {idx + 1}): Eval episode {episode + 1}: steps={steps}, return={total_return:.2f}, crashed={info.get('crashed')}"
        )

    env.close()

@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.eval, map_location=device)
    n_agents = checkpoint["n_agents"]

    # Run eval for each car in parallel
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=eval_worker, args=(args, i, checkpoint, n_agents, device)) for i in range(min(n_agents, 12))]
    for p in procs: p.start()
    for p in procs: p.join()
