"""Training loop for the current MAPPO highway baseline."""

from datetime import datetime

import numpy as np
import torch
from rich.console import Console
from rich.live import Live
from torch.utils.tensorboard import SummaryWriter

from ..env import build_actor_obs, flatten_obs, make_env
from ..mappo import ppo_update
from ..models import Actor, Critic
from functools import partial

from ..rendering import AsyncPolicyRenderer, PolicyPlayer, StepResult, Weights, create_renderer


console = Console()


def train(args):
    """Train the shared-policy MAPPO baseline using the provided CLI args."""
    with console.status(f"Setting up training with {args.agents} agents..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        n_agents = args.agents
        env = make_env(n_agents)
        obs_dim = int(np.prod(env.observation_space.spaces[0].shape))
        n_actions = int(env.action_space.spaces[0].n)

        actor = Actor(obs_dim + n_agents, n_actions).to(device)
        critic = Critic(obs_dim * n_agents).to(device)
        opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor, eps=1e-5)
        opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic, eps=1e-5)

        renderer: AsyncPolicyRenderer | None = create_renderer(
            args, partial(MappoPlayer, n_agents, obs_dim, n_actions)
        )

        obs, _ = env.reset(seed=args.seed)
        obs = flatten_obs(obs)
        episode_return = 0.0
        finished_returns = []
        steps_done = 0
        iteration = 0

        run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(f"{args.logdir}/{run_name}")
        writer.add_text("hparams", "\n".join(f"{k}: {v}" for k, v in vars(args).items()))

    while steps_done < args.total_steps:
        with console.status(
            f"Training ({steps_done}/{args.total_steps} steps, "
            f"{len(finished_returns)} episodes, {iteration} iterations)..."
        ):
            buf = {key: [] for key in ("aobs", "state", "act", "logp", "rew", "done", "val")}

            with Live(console=console, refresh_per_second=0.1) as rollout_live:
                for rollout_step in range(args.rollout):
                    rollout_live.update(f"Rollout ({rollout_step}/{args.rollout})...")
                    actor_obs = build_actor_obs(obs, n_agents)
                    state = obs.reshape(-1)

                    with torch.no_grad():
                        dist = actor.dist(torch.as_tensor(actor_obs, device=device))
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                        value = critic(torch.as_tensor(state, device=device)).item()

                    actions = tuple(int(item) for item in action.cpu().numpy())
                    next_obs, reward, terminated, truncated, _info = env.step(actions)
                    next_obs = flatten_obs(next_obs)
                    reward = float(reward)
                    done = terminated or truncated

                    # Time-limit truncation is not terminal, so include its bootstrap.
                    train_reward = reward
                    if truncated and not terminated:
                        with torch.no_grad():
                            next_value = critic(
                                torch.as_tensor(next_obs.reshape(-1), device=device)
                            ).item()
                        train_reward += args.gamma * next_value

                    buf["aobs"].append(actor_obs)
                    buf["state"].append(state)
                    buf["act"].append(action.cpu().numpy())
                    buf["logp"].append(log_prob.cpu().numpy())
                    buf["rew"].append(train_reward)
                    buf["done"].append(float(done))
                    buf["val"].append(value)

                    episode_return += reward
                    steps_done += 1
                    

                    if done:
                        finished_returns.append(episode_return)
                        writer.add_scalar("episode/return", episode_return, steps_done)
                        crashed = any(
                            vehicle.crashed for vehicle in env.unwrapped.controlled_vehicles
                        )
                        writer.add_scalar("episode/crashed", float(crashed), steps_done)
                        episode_return = 0.0
                        next_obs, _ = env.reset()
                        next_obs = flatten_obs(next_obs)
                    obs = next_obs

            with torch.no_grad():
                last_value = critic(torch.as_tensor(obs.reshape(-1), device=device)).item()

            stats = ppo_update(
                actor, critic, opt_actor, opt_critic, buf, last_value, args, device
            )

            if renderer is not None:
                renderer.publish_if_due(steps_done, actor.state_dict)

            iteration += 1
            recent_returns = finished_returns[-10:]
            mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
            console.print(
                f"iter {iteration:4d} | steps {steps_done:7d} | "
                f"mean ep return (last {len(recent_returns)}) {mean_return:7.2f} | "
                f"actor {stats['actor']:.3f} critic {stats['critic']:.3f} "
                f"entropy {stats['entropy']:.3f}"
            )

            writer.add_scalar("loss/actor", stats["actor"], steps_done)
            writer.add_scalar("loss/critic", stats["critic"], steps_done)
            writer.add_scalar("policy/entropy", stats["entropy"], steps_done)
            if recent_returns:
                writer.add_scalar("episode/mean_return_last10", mean_return, steps_done)

    if args.save:
        torch.save(
            {
                "algorithm": "mappo",
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "n_agents": n_agents,
                "obs_dim": obs_dim,
                "n_actions": n_actions,
            },
            args.save,
        )
        console.print(f"Saved checkpoint to {args.save}")

    if renderer is not None:
        renderer.close()
    writer.close()
    env.close()


class MappoPlayer(PolicyPlayer):
    """Plays the shared categorical actor greedily in the multi-agent highway env."""

    def __init__(self, n_agents: int, obs_dim: int, n_actions: int) -> None:
        self._n_agents: int = n_agents
        self._env = make_env(n_agents, render_mode="human")  # needs the make_env change
        self._actor: Actor = Actor(obs_dim + n_agents, n_actions)
        self._actor.eval()
        self._obs: np.ndarray | None = None

    def load_weights(self, weights: Weights) -> None:
        self._actor.load_state_dict(weights)

    def start_episode(self) -> None:
        self._obs = flatten_obs(self._env.reset()[0])

    def step(self) -> StepResult:
        if self._obs is None:
            raise RuntimeError("start_episode must be called before step")
        actions: tuple[int, ...] = self._greedy_actions(self._obs)
        next_obs, _reward, terminated, truncated, _info = self._env.step(actions)
        self._env.render()  # remove if make_env's human mode already draws inside step
        self._obs = flatten_obs(next_obs)
        crashed: bool = any(
            vehicle.crashed for vehicle in self._env.unwrapped.controlled_vehicles
        )
        return StepResult(finished=terminated or truncated, reward=float(_reward), crashed=crashed)

    def close(self) -> None:
        self._env.close()

    @torch.no_grad()
    def _greedy_actions(self, obs: np.ndarray) -> tuple[int, ...]:
        actor_obs: torch.Tensor = torch.as_tensor(
            build_actor_obs(obs, self._n_agents), dtype=torch.float32
        )
        logits: torch.Tensor = self._actor.dist(actor_obs).logits  # [n_agents, n_actions]
        return tuple(int(action) for action in logits.argmax(dim=1))


def build_mappo_player(checkpoint: dict) -> PolicyPlayer:
    player = MappoPlayer(checkpoint["n_agents"], checkpoint["obs_dim"], checkpoint["n_actions"])
    player.load_weights(checkpoint["actor"])
    return player