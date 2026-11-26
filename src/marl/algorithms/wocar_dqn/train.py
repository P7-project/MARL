"""The WocaR-DQN training loop: environment steps, replay, updates and the checkpoint."""

from __future__ import annotations

from argparse import Namespace
from functools import partial
from typing import Any

from gymnasium import spaces
import numpy as np
from rich.console import Console
import torch
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from auto_LiRPA import BoundedModule

from ...common import AsyncPolicyRenderer, RampSchedule, create_renderer, mlp
from ...envs import build_actor_obs, flatten_obs, make_env
from .networks import _build_robust_q, _to_tensor
from .player import DqnPlayer
from .replay_buffer import ReplayBuffer
from .trainer import Batch, WocaRDQNConfig, WocaRDQNTrainer


def train_dqn(args: Namespace, console: Console) -> None:
    with console.status("[bold green]Setting up WocaR-DQN..."):
        if args.total_steps <= args.warmup:
            raise ValueError(f"--total_steps ({args.total_steps}) must be >= --warmup ({args.warmup})")

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        n_agents: int = args.agents
        env_name: str = getattr(args, "dqn_env", "highway")
        env = make_env(
            n_agents,
            name=env_name,
            reward_function=args.reward,
            team_weight=args.team_weight,
        )

        observation_space = env.observation_space
        action_space = env.action_space
        assert isinstance(observation_space, spaces.Tuple)
        assert isinstance(action_space, spaces.Tuple)
        first_observation_space = observation_space.spaces[0]
        first_action_space = action_space.spaces[0]
        assert isinstance(first_observation_space, spaces.Box)
        assert isinstance(first_action_space, spaces.Discrete)

        obs_dim: int = int(np.prod(first_observation_space.shape))
        n_actions: int = int(first_action_space.n)
        input_dim: int = obs_dim + n_agents

        trainer: WocaRDQNTrainer = _build_trainer(obs_dim, n_agents, n_actions, args, device)
        buffer: ReplayBuffer = ReplayBuffer(args.buffer_size, input_dim, device)
        writer: SummaryWriter = SummaryWriter(f"{args.logdir}/{args.run_name}")
        renderer: AsyncPolicyRenderer | None = create_renderer(
            args,
            partial(
                DqnPlayer,
                n_agents,
                obs_dim,
                n_actions,
                args.hidden,
                env_name,
                args.reward,
                args.team_weight,
            ),
        )

        obs: np.ndarray = flatten_obs(env.reset(seed=args.seed)[0])
        episode_return: float = 0.0

    with console.status(f"[bold green]Training WocaR-DQN (0/{args.total_steps})...") as training_status:
        for step in range(args.total_steps):
            training_status.update(f"[bold green]Training WocaR-DQN ({step}/{args.total_steps})...")
            actor_obs: Tensor = _to_tensor(build_actor_obs(obs, n_agents), device)  # [n_agents, input_dim]
            actions: list[int] = _select_actions(trainer, actor_obs, _exploration_probability(step, args))

            next_obs, reward, terminated, truncated, _info = env.step(tuple(actions))
            agent_rewards: np.ndarray = np.asarray(reward, dtype=np.float32).reshape(-1)
            if agent_rewards.size == 1:
                agent_rewards = np.full(n_agents, agent_rewards.item(), dtype=np.float32)
            elif agent_rewards.size != n_agents:
                raise ValueError(
                    f"Expected 1 or {n_agents} rewards, got {agent_rewards.size}"
                )

            agent_terminated: np.ndarray = np.asarray(terminated, dtype=np.bool_).reshape(-1)
            if agent_terminated.size == 1:
                agent_terminated = np.full(
                    n_agents, agent_terminated.item(), dtype=np.bool_
                )
            elif agent_terminated.size != n_agents:
                raise ValueError(
                    f"Expected 1 or {n_agents} termination flags, "
                    f"got {agent_terminated.size}"
                )

            episode_done: bool = bool(agent_terminated.all()) or bool(
                np.asarray(truncated, dtype=np.bool_).any()
            )
            next_obs = flatten_obs(next_obs)
            # Built before any reset, so it is the real next observation.
            next_actor_obs: Tensor = _to_tensor(build_actor_obs(next_obs, n_agents), device)

            for agent in range(n_agents):
                buffer.add(
                    actor_obs[agent],
                    actions[agent],
                    float(agent_rewards[agent]),
                    next_actor_obs[agent],
                    float(agent_terminated[agent]),
                )

            episode_return += float(agent_rewards.mean())
            if episode_done:
                writer.add_scalar("episode/return", episode_return, step)
                episode_return = 0.0
                next_obs = flatten_obs(env.reset()[0])
            obs = next_obs

            if len(buffer) >= args.warmup:
                batch: Batch = buffer.sample(args.batch_size)
                metrics = trainer.update(batch, step)  # global env step drives eps_t, kappa_wst
                if step % 100 == 0:
                    for name, value in metrics._asdict().items():
                      if np.isfinite(value):
                          writer.add_scalar(f"woca/{name}", value, step)

            if renderer is not None:
                renderer.publish_if_due(step, trainer.robust_q_state_dict)

    with console.status("[bold green]Saving WocaR-DQN..."):
        if args.save:
            torch.save(
                {
                    "algorithm": "wocar_dqn",
                    "robust_q": trainer.robust_q_state_dict(),
                    "n_agents": n_agents,
                    "obs_dim": obs_dim,
                    "n_actions": n_actions,
                    "hidden": args.hidden,
                    "dqn_env": env_name,
                    "args": _plain_args(args),
                    "reward_function": args.reward,
                    "team_weight": float(env.unwrapped.config["team_reward_weight"]),
                },
                args.save,
            )

    with console.status("[bold green]Cleaning up..."):
        if renderer is not None:
            renderer.close()
        writer.close()
        env.close()


def _plain_args(args: Namespace) -> dict:
    """The arguments as builtin values, so the checkpoint loads with torch.load's weights_only."""
    return {name: _plain_value(value) for name, value in vars(args).items()}


def _plain_value(value: Any) -> Any:
    """Keep None, bool, int, float, str and lists/tuples of them; anything else (e.g. Path) becomes str."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return str(value)


def _build_trainer(
    obs_dim: int, n_agents: int, n_actions: int, args: Namespace, device: torch.device
) -> WocaRDQNTrainer:
    input_dim: int = obs_dim + n_agents
    perturbation_scale: Tensor = torch.cat(
        [torch.ones(obs_dim), torch.zeros(n_agents)]
    ).to(device)
    vanilla_q: nn.Module = mlp(input_dim, n_actions, args.hidden, out_gain=1.0).to(device)
    robust_q: BoundedModule = _build_robust_q(input_dim, n_actions, args.hidden, device)
    worst_attack_q: nn.Module = mlp(input_dim, n_actions, args.hidden, out_gain=1.0).to(device)
    kappa_start: int = int(args.total_steps * args.kappa_wst_start_fraction)

    config: WocaRDQNConfig = WocaRDQNConfig(
        num_actions=n_actions,
        discount_factor=args.gamma,
        target_update_coefficient=args.tau,
        learning_rate=args.lr,
        epsilon_schedule=RampSchedule(
            start_step=args.warmup,
            ramp_steps=args.total_steps - args.warmup,
            final_value=args.eps_final,
        ),
        kappa_wst_schedule=RampSchedule(
            start_step=kappa_start,
            ramp_steps=args.total_steps - kappa_start,
            final_value=args.dqn_kappa_wst_final,
            shape="exponential",
        ),
        kappa_reg=args.kappa_reg,
        observation_low=-1.0,
        observation_high=1.0,
        regularizer_margin=args.regularizer_margin,
        use_double_dqn=args.double_dqn,
        perturbation_scale=perturbation_scale,
    )
    return WocaRDQNTrainer(vanilla_q, robust_q, worst_attack_q, config)


def _select_actions(trainer: WocaRDQNTrainer, actor_obs: Tensor, exploration_probability: float) -> list[int]:
    return [trainer.select_action(agent_obs, exploration_probability) for agent_obs in actor_obs]


def _exploration_probability(step: int, args: Namespace) -> float:
    """(mine) linear decay from 1.0 to explore_final over explore_steps."""
    progress: float = min(step / args.explore_steps, 1.0)
    return 1.0 + progress * (args.explore_final - 1.0)
