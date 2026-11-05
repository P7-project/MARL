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





from multiprocessing.managers import Namespace
from typing import NamedTuple

from matplotlib.pyplot import box
import numpy as np
import torch
from torch import Tensor, nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from torch.optim import Optimizer

from marl.mappo import compute_gae
from marl.models import Actor


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
            value = self._value_network(states).squeeze(-1)
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
    actor: Actor,
    critic: nn.Module,
    opt_actor: Optimizer,
    opt_critic: Optimizer,
    wocar: WocaRLosses, # @woca
    opt_q: Optimizer, # @woca
    epsilon: float, # @woca
    kappa_wst: float, # @woca
    kappa_reg: float, # @woca
    buf: dict[str, list],
    last_value: Tensor,
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
    next_states = torch.as_tensor(np.stack(buf["next_state"]), device=device) # @woca
    terminated = torch.as_tensor(np.stack(buf["terminated"]), device=device).float() # @woca
    rewards = torch.as_tensor(np.stack(buf["reward_raw"]), device=device) # @woca

    advantages = wocar.shift_advantages(advantages, states, actions, kappa_wst) # @woca
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    horizon = actor_obs.shape[0]
    stats = {"actor": 0.0, "critic": 0.0, "entropy": 0.0}
    update_count = 0

    for _ in range(args.epochs):
        permutation = torch.randperm(horizon, device=device)

        for start in range(0, horizon, args.minibatch):
            indices = permutation[start : start + args.minibatch]

            # @woca: Q_phi step first, so the regularizer weights use the fresh Q_phi.
            q_loss: Tensor = wocar.critic_loss(
                states[indices], actions[indices], rewards[indices],
                next_states[indices], terminated[indices], epsilon,
            )
            opt_q.zero_grad()
            q_loss.backward()
            opt_q.step()

            dist = actor.dist(actor_obs[indices])
            new_log_probs = dist.log_prob(actions[indices])
            ratio = torch.exp(new_log_probs - old_log_probs[indices])
            advantage_mb = advantages[indices].unsqueeze(-1)
            surrogate_1 = ratio * advantage_mb
            surrogate_2 = torch.clamp(
                ratio, 1 - args.clip, 1 + args.clip
            ) * advantage_mb
            entropy = dist.entropy().mean()

            reg_loss: Tensor = wocar.regularizer_loss(states[indices], epsilon)  # @woca
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
            stats["q"] = q_loss.item()  # @woca
            stats["reg"] = reg_loss.item()  # @woca
            update_count += 1

    return {key: value / max(update_count, 1) for key, value in stats.items()}