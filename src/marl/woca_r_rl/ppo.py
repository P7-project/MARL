"""
Networks: Gaussian policy π_θ (mean net μ), value V, worst-attack critic Q_φ(s, a).

Per iteration:

+ Roll out π_θ on clean observations. Compute returns and advantages, then fit V.
+ Compute IBP bounds of μ over B_ε(s') for the next states. This gives the box [μ_lo, μ_hi].
+ Find â = argmin_{a in box} Q_φ(s', a) with 50 steps of gradient descent, clamped to the box.
+ Set y = r + γ·Q_φ(s', â), with y = r at terminal states. Update φ with MSE.
+ Compute w(s) = V(s) − min_a Q_φ(s, a).
+ Compute L_reg with SGLD, using the SA-PPO regularizer, weighted by w(s).
+ Update π_θ with the clipped PPO objective, using advantage Â + κ_wst·Q_φ(s, a) at the sampled action, plus κ_reg·w(s)·L_reg.

Hyperparameters from the paper:

- κ_wst ramps from 0, linearly to 0.8 on Hopper, Walker2d and HalfCheetah, and exponentially to 0.5 on Ant.
- ε ramps up slowly over training.
- κ_reg is grid-searched over {0.01, 0.03, 0.1, 0.3, 1.0}.
- Training length is 2M steps, or 10M on Ant.
"""





from argparse import Namespace
from typing import NamedTuple

from gymnasium import spaces
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from datetime import datetime

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from ..mappo import compute_gae
from ..models import Critic, mlp
from .models import GaussianActor
from .schedule import RampSchedule


class ActionBox(NamedTuple):
    lower: Tensor  # [batch, action_dim]
    upper: Tensor  # [batch, action_dim]


class WocaRLosses:
    """Wires the pieces together. Everything below it is called by it."""

    def __init__(
        self,
        value_network: nn.Module,
        q_network: nn.Module,
        bounder: "PolicyActionBounder",
        searcher: "WorstActionSearcher",
        regularizer: "SgldRegularizer",
        discount_factor: float,
    ) -> None:
        self._value_network = value_network
        self._q_network = q_network
        self._bounder = bounder
        self._searcher = searcher
        self._regularizer = regularizer
        self._discount_factor = discount_factor

    @torch.no_grad()
    def shift_advantages(
        self, advantages: Tensor, states: Tensor, actions: Tensor, kappa_wst: float
    ) -> Tensor:
        """Advantage + kappa_wst * Q_phi(s, a) at the sampled action."""
        return advantages + kappa_wst * self._q_values(states, actions)

    def critic_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_aobs: Tensor,
        next_states: Tensor,
        terminated: Tensor,
        epsilon: float,
    ) -> Tensor:
        """MSE onto y = r + gamma * Q_phi(s', a_hat), y = r at terminal states."""
        worst_actions = self._worst_actions(next_aobs, next_states, epsilon)
        with torch.no_grad():
            next_values = self._q_values(next_states, worst_actions)
            targets = rewards + self._discount_factor * (1.0 - terminated) * next_values
        return nn.functional.mse_loss(self._q_values(states, actions), targets)

    def regularizer_loss(self, aobs: Tensor, states: Tensor, epsilon: float) -> Tensor:
        """mean_s[ w(s) * L_reg(s) ]. Gradients flow only through L_reg."""
        weights = self._weights(aobs, states, epsilon)
        return (weights * self._regularizer.per_state_loss(aobs, epsilon)).mean()

    def _weights(self, aobs: Tensor, states: Tensor, epsilon: float) -> Tensor:
        """w(s) = V(s) - min_a Q_phi(s, a)."""
        worst_actions = self._worst_actions(aobs, states, epsilon)
        with torch.no_grad():
            value = self._value_network(states)
            worst_q = self._q_values(states, worst_actions)
            # (mine) min over the IBP box at s, clamped at 0.
            return (value - worst_q).clamp(min=0.0)

    def _worst_actions(self, aobs: Tensor, states: Tensor, epsilon: float) -> Tensor:
        with torch.no_grad():
            box = self._bounder.action_box(aobs, epsilon)
        return self._searcher.search(states, box)

    def _q_values(self, states: Tensor, actions: Tensor) -> Tensor:
        return self._q_network(torch.cat([states, actions], dim=1)).squeeze(-1)


class WorstActionSearcher:
    """a_hat = argmin_{a in box} Q_phi(s, a): gradient descent, clamped to the box."""

    def __init__(self, q_network: nn.Module, steps: int = 50, learning_rate: float = 0.1) -> None:
        self._q_network = q_network
        self._steps = steps
        self._learning_rate = learning_rate  # (mine) not given in the paper

    def search(self, states: Tensor, box: ActionBox) -> Tensor:
        actions = ((box.lower + box.upper) / 2).clone().requires_grad_(True)
        for _ in range(self._steps):
            with torch.enable_grad():
                q_sum = self._q_network(torch.cat([states, actions], dim=1)).sum()
                gradient, = torch.autograd.grad(q_sum, actions)
            with torch.no_grad():
                actions -= self._learning_rate * gradient
                actions.copy_(torch.maximum(torch.minimum(actions, box.upper), box.lower))
        return actions.detach()


class SgldRegularizer:
    """SA-PPO regularizer: L_reg(s) = max_{s~ in B_eps(s)} ||mu(s) - mu(s~)||^2, found by SGLD."""

    def __init__(
        self,
        mean_network: nn.Module,
        steps: int = 10,
        step_size: float = 0.1,
        noise_scale: float = 0.01,
    ) -> None:
        self._mean_network = mean_network
        self._steps = steps
        self._step_size = step_size
        self._noise_scale = noise_scale

    def per_state_loss(self, states: Tensor, epsilon: float) -> Tensor:
        perturbed = self._find_perturbation(states, epsilon)
        deviation = self._mean_network(states) - self._mean_network(perturbed)
        return deviation.pow(2).sum(dim=1)

    def _find_perturbation(self, states: Tensor, epsilon: float) -> Tensor:
        clean_means = self._mean_network(states).detach()
        perturbed = (states + (torch.rand_like(states) * 2 - 1) * epsilon).detach()
        for _ in range(self._steps):
            perturbed.requires_grad_(True)
            deviation = (self._mean_network(perturbed) - clean_means).pow(2).sum()
            gradient, = torch.autograd.grad(deviation, perturbed)
            with torch.no_grad():
                perturbed = perturbed + self._step_size * gradient
                perturbed += self._noise_scale * torch.randn_like(perturbed)
                perturbed = torch.maximum(torch.minimum(perturbed, states + epsilon), states - epsilon)
        return perturbed.detach()


class PolicyActionBounder:
    """
        IBP box [mu_lo, mu_hi] of the mean network over B_eps(s).
        Bounds the policy's mean action over B_eps(s).
    
        Math: [a_lo(s), a_hi(s)] with
            a_lo <= pi(s~) <= a_hi  for all s~ in B_eps(s) (L-inf ball, radius epsilon),
        hence A_adv(s, pi) is contained in the box. For a Gaussian policy, `pi` here
        is the mean network (the deterministic action the adversary can shift).
    """

    def __init__(self, bounded_mean_network: BoundedModule) -> None:
        self._bounded_mean_network = bounded_mean_network

    def action_box(self, states: Tensor, epsilon: float) -> ActionBox:
        perturbation = PerturbationLpNorm(norm=float("inf"), eps=epsilon)  # type: ignore[arg-type]
        lower, upper = self._bounded_mean_network.compute_bounds(
            x=(BoundedTensor(states, perturbation),), method="IBP"
        )
        return ActionBox(lower, upper)


def ppo_update_wocar(
    actor: GaussianActor,
    critic: nn.Module,
    opt_actor: Optimizer,
    opt_critic: Optimizer,
    wocar: WocaRLosses, # @woca
    opt_q: Optimizer, # @woca
    epsilon: float, # @woca
    kappa_wst: float, # @woca
    kappa_reg: float, # @woca
    buf: dict[str, list],
    last_value: float,
    args: Namespace,
    device: torch.device,
) -> dict[str, float]:
    advantages, returns = compute_gae(
        buf["rew"], buf["val"], buf["done"], last_value, args.gamma, args.lam
    )

    actor_obs = torch.as_tensor(np.stack(buf["aobs"]), device=device)
    states = torch.as_tensor(np.stack(buf["state"]), device=device)
    actions = torch.as_tensor(np.stack(buf["act"]), device=device)
    old_log_probs = torch.as_tensor(np.stack(buf["logp"]), device=device)
    advantages = torch.as_tensor(advantages, device=device)
    returns = torch.as_tensor(returns, device=device)
    next_aobs = torch.as_tensor(np.stack(buf["next_aobs"]), device=device) # @woca
    next_states = torch.as_tensor(np.stack(buf["next_state"]), device=device) # @woca
    terminated = torch.as_tensor(np.stack(buf["terminated"]), device=device).float() # @woca
    rewards = torch.as_tensor(np.stack(buf["reward_raw"]), device=device) # @woca

    advantages = wocar.shift_advantages(advantages, states, actions, kappa_wst) # @woca
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    horizon = actor_obs.shape[0]
    stats = {"actor": 0.0, "critic": 0.0, "entropy": 0.0, "q": 0.0, "reg": 0.0}
    update_count = 0

    for _ in range(args.epochs):
        permutation = torch.randperm(horizon, device=device)

        for start in range(0, horizon, args.minibatch):
            indices = permutation[start : start + args.minibatch]

            # @woca: Q_phi step first, so the regularizer weights use the fresh Q_phi.
            q_loss: Tensor = wocar.critic_loss(
                states[indices], 
                actions[indices], 
                rewards[indices],
                next_aobs[indices],
                next_states[indices], 
                terminated[indices], 
                epsilon,
            )
            opt_q.zero_grad()
            q_loss.backward()
            opt_q.step()

            dist = actor.dist(actor_obs[indices])
            new_log_probs = dist.log_prob(actions[indices])
            ratio = torch.exp(new_log_probs - old_log_probs[indices])
            advantage_mb = advantages[indices]
            surrogate_1 = ratio * advantage_mb
            surrogate_2 = torch.clamp(
                ratio, 1 - args.clip, 1 + args.clip
            ) * advantage_mb
            entropy = dist.entropy().mean()

            reg_loss: Tensor = wocar.regularizer_loss(actor_obs[indices], states[indices], epsilon)  # @woca
            actor_loss = (
                -torch.min(surrogate_1, surrogate_2).mean()
                - args.ent * entropy
                + kappa_reg * reg_loss  # @woca
            )

            opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            opt_actor.step()

            value = critic(states[indices])
            critic_loss = 0.5 * (value - returns[indices]).pow(2).mean()

            opt_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            opt_critic.step()

            stats["actor"] += actor_loss.item()
            stats["critic"] += critic_loss.item()
            stats["entropy"] += entropy.item()
            stats["q"] += q_loss.item()  # @woca
            stats["reg"] += reg_loss.item()  # @woca
            update_count += 1

    return {key: value / max(update_count, 1) for key, value in stats.items()}


def build_wocar(
    actor: GaussianActor,
    critic: nn.Module,
    state_dim: int,
    action_dim: int,
    args: Namespace,
    device: torch.device,
) -> tuple[WocaRLosses, torch.optim.Optimizer]:
    q_network: nn.Module = mlp(state_dim + action_dim, 1, 64, out_gain=1.0).to(device)
    wocar = WocaRLosses(
        value_network=critic,
        q_network=q_network,
        bounder=PolicyActionBounder(actor.mean_network),  # same object as the actor uses
        searcher=WorstActionSearcher(q_network),
        regularizer=SgldRegularizer(actor.mean_network),  # same object again
        discount_factor=args.gamma,
    )
    opt_q = torch.optim.Adam(q_network.parameters(), lr=args.lr_critic, eps=1e-5)
    return wocar, opt_q


def train_ppo(args: Namespace) -> None:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env: gym.Env = gym.make(args.env_id)
    observation_space = env.observation_space
    action_space = env.action_space
    assert isinstance(observation_space, spaces.Box)
    assert isinstance(action_space, spaces.Box)

    obs_dim: int = int(np.prod(observation_space.shape))
    action_dim: int = int(np.prod(action_space.shape))

    actor: GaussianActor = GaussianActor(obs_dim, action_dim, device).to(device)
    critic: Critic = Critic(obs_dim).to(device)
    opt_actor: Optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr_actor, eps=1e-5)
    opt_critic: Optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr_critic, eps=1e-5)
    # Single-agent: the actor observation and the critic state are the same vector.
    wocar, opt_q = build_wocar(actor, critic, obs_dim, action_dim, args, device)

    # (mine) ramp windows: the paper only says epsilon ramps slowly.
    epsilon_schedule: RampSchedule = RampSchedule(
        start_step=0, ramp_steps=args.total_steps, final_value=args.eps_final
    )
    kappa_wst_schedule: RampSchedule = RampSchedule(
        start_step=0,
        ramp_steps=args.total_steps,
        final_value=args.kappa_wst_final,
        shape=args.kappa_wst_shape,
    )

    run_name: str = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    writer: SummaryWriter = SummaryWriter(f"{args.logdir}/{run_name}")

    obs: np.ndarray = env.reset(seed=args.seed)[0].astype(np.float32)
    episode_return: float = 0.0
    finished_returns: list[float] = []
    steps_done: int = 0
    iteration: int = 0

    while steps_done < args.total_steps:
        buf, obs, episode_return, steps_done = _collect_rollout(
            env, actor, critic, obs, action_space.low, action_space.high, episode_return, finished_returns,
            steps_done, args, device, writer,
        )

        with torch.no_grad():
            last_value: float = critic(_to_batch(obs, device)).item()

        epsilon: float = epsilon_schedule.value(steps_done)
        kappa_wst: float = kappa_wst_schedule.value(steps_done)
        stats: dict[str, float] = ppo_update_wocar(
            actor, critic, opt_actor, opt_critic, wocar, opt_q,
            epsilon, kappa_wst, args.kappa_reg, buf, last_value, args, device,
        )

        iteration += 1
        recent_returns: list[float] = finished_returns[-10:]
        mean_return: float = float(np.mean(recent_returns)) if recent_returns else float("nan")
        print(
            f"iter {iteration:4d} | steps {steps_done:8d} | return(last10) {mean_return:8.2f} | "
            f"eps {epsilon:.4f} kappa_wst {kappa_wst:.3f} | "
            f"actor {stats['actor']:.3f} critic {stats['critic']:.3f} "
            f"q {stats['q']:.3f} reg {stats['reg']:.4f}"
        )
        for name, value in stats.items():
            writer.add_scalar(f"loss/{name}", value, steps_done)
        writer.add_scalar("woca/epsilon", epsilon, steps_done)
        writer.add_scalar("woca/kappa_wst", kappa_wst, steps_done)
        if recent_returns:
            writer.add_scalar("episode/mean_return_last10", mean_return, steps_done)

    if args.save:
        torch.save(
            {
                "algorithm": "wocar_ppo",
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "env_id": args.env_id,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
            },
            args.save,
        )

    writer.close()
    env.close()


def _collect_rollout(
    env: gym.Env,
    actor: GaussianActor,
    critic: nn.Module,
    obs: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    episode_return: float,
    finished_returns: list[float],
    steps_done: int,
    args: Namespace,
    device: torch.device,
    writer: SummaryWriter,
) -> tuple[dict[str, list], np.ndarray, float, int]:
    buf: dict[str, list] = {
        key: []
        for key in (
            "aobs", "state", "act", "logp", "rew", "done", "val",
            "next_aobs", "next_state", "terminated", "reward_raw",
        )
    }

    for _ in range(args.rollout):
        with torch.no_grad():
            obs_batch: Tensor = _to_batch(obs, device)
            dist = actor.dist(obs_batch)
            action: Tensor = dist.sample()
            log_prob: Tensor = dist.log_prob(action)
            value: float = critic(obs_batch).item()

        action_np: np.ndarray = action.squeeze(0).cpu().numpy()
        # Store the unclipped action for the log-prob ratio, clip only for the env.
        env_action: np.ndarray = np.clip(action_np, action_low, action_high)
        next_obs, reward, terminated, truncated, _info = env.step(env_action)
        next_obs = next_obs.astype(np.float32)
        done: bool = terminated or truncated

        # Time-limit truncation is not terminal: bootstrap the GAE reward with V(s').
        train_reward: float = float(reward)
        if truncated and not terminated:
            with torch.no_grad():
                train_reward += args.gamma * critic(_to_batch(next_obs, device)).item()

        buf["aobs"].append(obs)
        buf["state"].append(obs)
        buf["act"].append(action_np.astype(np.float32))
        buf["logp"].append(np.float32(log_prob.item()))
        buf["rew"].append(np.float32(train_reward))
        buf["done"].append(np.float32(done))
        buf["val"].append(np.float32(value))
        # Stored before any reset, so this is the real next observation.
        buf["next_aobs"].append(next_obs)
        buf["next_state"].append(next_obs)
        buf["terminated"].append(np.float32(terminated))
        buf["reward_raw"].append(np.float32(reward))  # the Q_phi target adds its own bootstrap

        episode_return += float(reward)
        steps_done += 1

        if done:
            finished_returns.append(episode_return)
            writer.add_scalar("episode/return", episode_return, steps_done)
            episode_return = 0.0
            next_obs = env.reset()[0].astype(np.float32)
        obs = next_obs

    return buf, obs, episode_return, steps_done


def _to_batch(obs: np.ndarray, device: torch.device) -> Tensor:
    """[obs_dim] -> [1, obs_dim], the shape the BoundedModule was traced with."""
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
