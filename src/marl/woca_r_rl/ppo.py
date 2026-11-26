"""
WocaR-PPO (Algorithm 1 of Liang et al., NeurIPS 2022).

The update follows the line numbers of Algorithm 1:

    Algorithm 1 line 3      _collect_rollout
    Algorithm 1 lines 4-5   compute_gae (advantages and returns)
    Algorithm 1 line 6      _fit_value_function
    Algorithm 1 lines 7-10  _fit_worst_attack_critic
    Algorithm 1 line 11     WocaRLosses.state_importance
    Algorithm 1 lines 12-13 _fit_policy

The networks are:
    pi_theta       GaussianActor; its mean network is the bounded policy pi.
    V_theta_V      Critic.
    Q_phi          worst-attack critic; it takes (s, a) and returns one value.

For each iteration:
    + Roll out pi_theta and compute GAE advantages and returns.
    + Fit V with regression onto the returns.
    + IBP-bound pi over B_epsilon(s') and find
      a_hat = argmin_{a in [pi_lo, pi_hi]} Q_phi(s', a) by gradient descent.
    + Set y = r + gamma * Q_phi(s', a_hat); y = r at terminal transitions.
    + Fit Q_phi with MSE onto y.
    + Set w(s) = V(s) - min_a Q_phi(s, a), with the minimum found over the
      environment action space by gradient descent.
    + Fit pi with the robust clipped PPO objective using
      A_hat + kappa_wst * Q_phi(s, a) at the sampled action, plus
      kappa_reg * w(s) * L_reg, where L_reg is found by SGLD.

Hyperparameters from the paper:
    - kappa_wst ramps from 0, linearly to 0.8 on Hopper, Walker2d and
      HalfCheetah, and exponentially to 0.5 on Ant.
    - epsilon ramps up slowly over training.
    - kappa_reg is grid-searched over {0.01, 0.03, 0.1, 0.3, 1.0}.
    - Training length is 2M steps, or 10M on Ant.
"""

from argparse import Namespace
from datetime import datetime
from typing import NamedTuple

import gymnasium as gym
import numpy as np
import torch
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from gymnasium import spaces
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from ..common import (
    Critic,
    GaussianActor,
    PolicyPlayer,
    RampSchedule,
    StepResult,
    Weights,
    compute_gae,
    mlp,
)


class ActionBox(NamedTuple):
    lower: Tensor  # [batch, action_dim]
    upper: Tensor  # [batch, action_dim]


class Rollout(NamedTuple):
    """One collected rollout, already moved to the training device."""

    actor_obs: Tensor  # [horizon, state_dim]
    states: Tensor  # [horizon, state_dim]
    actions: Tensor  # [horizon, action_dim]
    old_log_probs: Tensor  # [horizon]
    rewards: Tensor  # [horizon], raw environment rewards
    next_actor_obs: Tensor  # [horizon, state_dim]
    next_states: Tensor  # [horizon, state_dim]
    terminated: Tensor  # [horizon], 1.0 only for true MDP terminals
    advantages: Tensor  # [horizon]
    returns: Tensor  # [horizon]


# --------------------------------------------------------------------------- #
# WocaR losses and the three building blocks from the paper
# --------------------------------------------------------------------------- #


class WocaRLosses:
    """The three WocaR mechanisms from Algorithm 1, lines 7-13."""

    def __init__(
        self,
        value_network: nn.Module,
        q_network: nn.Module,
        bounder: "PolicyActionBounder",
        searcher: "WorstActionSearcher",
        regularizer: "SgldRegularizer",
        discount_factor: float,
        action_lower: Tensor | None = None,
        action_upper: Tensor | None = None,
    ) -> None:
        self._value_network = value_network
        self._q_network = q_network
        self._bounder = bounder
        self._searcher = searcher
        self._regularizer = regularizer
        self._discount_factor = discount_factor
        self._action_lower = action_lower
        self._action_upper = action_upper

    # ---------------------------------------------------- Algorithm 1 line 11 ---

    def state_importance(self, states: Tensor) -> Tensor:
        """
        w(s) = V(s) - min_a Q_phi(s, a).

        The minimum is over the environment action space, not over the policy's
        perturbation box: state importance is a property of Q_phi at s, while
        the box is only used to build the next-state Bellman target.
        """
        worst_actions = self._worst_actions_in_action_space(states)
        with torch.no_grad():
            value = self._value_network(states)
            worst_q = self._q_values(states, worst_actions)
            # The clamp at 0 is our choice: the estimate V - min Q_phi can be
            # slightly negative from estimation error, which would turn the
            # regularizer into a reward for being unstable.
            return (value - worst_q).clamp(min=0.0)

    def regularizer_loss(
        self, aobs: Tensor, states: Tensor, epsilon: float, weights: Tensor | None = None
    ) -> Tensor:
        """
        mean_s[ w(s) * L_reg(s) ] (Algorithm 1, lines 12-13).

        `weights` are precomputed w(s) for these states; if None, they are
        computed here from `states`.
        """
        if weights is None:
            weights = self.state_importance(states)
        return (weights * self._regularizer.per_state_loss(aobs, epsilon)).mean()

    # --------------------------------------------------------- Algorithm 1 line 8 ---

    def worst_actions_for_bellman(self, next_aobs: Tensor, next_states: Tensor, epsilon: float) -> Tensor:
        """
        a_hat = argmin_{a in [pi_lo, pi_hi]} Q_phi(s', a).

        The box is the IBP bound of the policy mean over B_epsilon(s').
        """
        with torch.no_grad():
            box = self._bounder.action_box(next_aobs, epsilon)
        return self._searcher.search(next_states, box)

    # ----------------------------------------------------- Algorithm 1 lines 9-10 ---

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
        worst_actions = self.worst_actions_for_bellman(next_aobs, next_states, epsilon)
        with torch.no_grad():
            next_values = self._q_values(next_states, worst_actions)
            targets = rewards + self._discount_factor * (1.0 - terminated) * next_values
        return nn.functional.mse_loss(self._q_values(states, actions), targets)

    # ------------------------------------------------ Algorithm 1 lines 8-10 target ---

    @torch.no_grad()
    def shift_advantages(
        self, advantages: Tensor, states: Tensor, actions: Tensor, kappa_wst: float
    ) -> Tensor:
        """Advantage + kappa_wst * Q_phi(s, a) at the sampled action."""
        return advantages + kappa_wst * self._q_values(states, actions)

    # ----------------------------------------------------------------- helpers ---

    def _worst_actions_in_action_space(self, states: Tensor) -> Tensor:
        """argmin_a Q_phi(s, a) over the environment action space."""
        lower, upper = self._action_domain(states)
        return self._searcher.search(states, ActionBox(lower=lower, upper=upper))

    def _action_domain(self, states: Tensor) -> tuple[Tensor, Tensor]:
        if self._action_lower is None or self._action_upper is None:
            # Tests and other callers can construct WocaRLosses without an
            # environment; fall back to the usual normalized action range.
            reference = self._bounder.action_box(states, 0.0).lower
            return torch.full_like(reference, -1.0), torch.full_like(reference, 1.0)
        batch_size = states.shape[0]
        lower = self._action_lower.to(states.device).unsqueeze(0).expand(batch_size, -1)
        upper = self._action_upper.to(states.device).unsqueeze(0).expand(batch_size, -1)
        return lower, upper

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
    """
    L_reg(s) = max_{s~ in B_eps(s)} Dist(pi(s), pi(s~)) (Algorithm 1, line 12).

    Dist is the mean part of the Gaussian KL divergence:
        sum_i (mu_i(s) - mu_i(s~))^2 / sigma_i^2,
    which is what SA-PPO (and the official WocaR-PPO code) maximizes. The
    maximum is approximated by projected SGLD: a sign step of size
    epsilon / steps plus decaying Langevin noise, as in SA-PPO.

    As in SA-PPO's get_state_kl_bound_sgld, sigma and the clean means mu(s)
    are detached, so the gradient flows only through mu(s~).
    """

    # (1 / beta) from SA-PPO: their gradients are small, so beta = 1e-5.
    _INVERSE_BETA = 1e-5

    def __init__(
        self,
        mean_network: nn.Module,
        log_std: Tensor | None = None,
        steps: int = 10,
    ) -> None:
        self._mean_network = mean_network
        self._log_std = log_std
        self._steps = steps

    def per_state_loss(self, states: Tensor, epsilon: float) -> Tensor:
        perturbed = self._find_perturbation(states, epsilon)
        clean_means = self._mean_network(states).detach()
        return self._distance(clean_means, self._mean_network(perturbed))

    def _distance(self, means: Tensor, perturbed_means: Tensor) -> Tensor:
        deviation = means - perturbed_means
        if self._log_std is None:
            return deviation.pow(2).sum(dim=1)
        return (deviation / self._log_std.detach().exp()).pow(2).sum(dim=1)

    def _find_perturbation(self, states: Tensor, epsilon: float) -> Tensor:
        clean_means = self._mean_network(states).detach()
        step = epsilon / self._steps
        noise_factor = float(np.sqrt(2 * step * self._INVERSE_BETA))
        perturbed = (states + torch.randn_like(states).sign() * step).detach()
        lower = states - epsilon
        upper = states + epsilon

        for iteration in range(self._steps):
            perturbed.requires_grad_(True)
            # Maximize Dist: gradient ascent on the clean means, with the
            # policy parameters held fixed for the perturbation search.
            loss = self._distance(clean_means.detach(), self._mean_network(perturbed)).sum()
            gradient, = torch.autograd.grad(loss, perturbed)
            # Reduce the noise every step, as in SA-PPO.
            noise_factor = float(np.sqrt(2 * step * self._INVERSE_BETA)) / (iteration + 2)
            with torch.no_grad():
                update = (gradient + noise_factor * torch.randn_like(perturbed)).sign() * step
                perturbed = (perturbed + update).clamp(min=lower, max=upper)

        return perturbed.detach()


class PolicyActionBounder:
    """
    IBP box [mu_lo, mu_hi] of the mean network over B_eps(s).

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


# --------------------------------------------------------------------------- #
# Algorithm 1: one full WocaR-PPO iteration
# --------------------------------------------------------------------------- #


def ppo_update_wocar(
    actor: GaussianActor,
    critic: nn.Module,
    opt_actor: Optimizer,
    opt_critic: Optimizer,
    wocar: WocaRLosses,  # @woca
    opt_q: Optimizer,  # @woca
    epsilon: float,  # @woca
    kappa_wst: float,  # @woca
    kappa_reg: float,  # @woca
    buf: dict[str, list],
    last_value: float,
    args: Namespace,
    device: torch.device,
) -> dict[str, float]:
    rollout = _rollout_tensors(buf, last_value, args, device)

    # Algorithm 1, line 6. The paper updates V before Q_phi, because line 11
    # uses the updated V to weight the regularizer.
    critic_loss = _fit_value_function(critic, opt_critic, rollout, args)

    # Algorithm 1, lines 7-10.
    q_loss = _fit_worst_attack_critic(wocar, opt_q, rollout, epsilon, args)

    # Algorithm 1, lines 12-13. state_importance uses the fresh V and Q_phi.
    policy_stats = _fit_policy(
        actor, opt_actor, wocar, rollout, epsilon, kappa_wst, kappa_reg, args
    )

    return {
        "actor": policy_stats["actor"],
        "critic": critic_loss,
        "q": q_loss,
        "reg": policy_stats["reg"],
        "entropy": policy_stats["entropy"],
    }


def _rollout_tensors(
    buf: dict[str, list], last_value: float, args: Namespace, device: torch.device
) -> Rollout:
    """Algorithm 1, lines 4-5: GAE advantages and returns from the rollout."""
    advantages, returns = compute_gae(
        buf["rew"], buf["val"], buf["done"], last_value, args.gamma, args.lam
    )

    actor_obs = torch.as_tensor(np.stack(buf["aobs"]), device=device)
    states = torch.as_tensor(np.stack(buf["state"]), device=device)
    actions = torch.as_tensor(np.stack(buf["act"]), device=device)
    old_log_probs = torch.as_tensor(np.stack(buf["logp"]), device=device)
    next_actor_obs = torch.as_tensor(np.stack(buf["next_aobs"]), device=device)
    next_states = torch.as_tensor(np.stack(buf["next_state"]), device=device)
    terminated = torch.as_tensor(np.stack(buf["terminated"]), device=device).float()
    rewards = torch.as_tensor(np.stack(buf["reward_raw"]), device=device)

    return Rollout(
        actor_obs=actor_obs,
        states=states,
        actions=actions,
        old_log_probs=old_log_probs,
        rewards=rewards,
        next_actor_obs=next_actor_obs,
        next_states=next_states,
        terminated=terminated,
        advantages=torch.as_tensor(advantages, device=device),
        returns=torch.as_tensor(returns, device=device),
    )


def _fit_value_function(
    critic: nn.Module, opt_critic: Optimizer, rollout: Rollout, args: Namespace
) -> float:
    """Algorithm 1, line 6: regress V onto the returns."""
    horizon = rollout.states.shape[0]
    total_loss = 0.0
    update_count = 0

    for _ in range(args.epochs):
        permutation = torch.randperm(horizon, device=rollout.states.device)

        for start in range(0, horizon, args.minibatch):
            indices = permutation[start : start + args.minibatch]

            value = critic(rollout.states[indices])
            loss = (value - rollout.returns[indices]).pow(2).mean()

            opt_critic.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            opt_critic.step()

            total_loss += loss.item()
            update_count += 1

    return total_loss / max(update_count, 1)


def _fit_worst_attack_critic(
    wocar: WocaRLosses,
    opt_q: Optimizer,
    rollout: Rollout,
    epsilon: float,
    args: Namespace,
) -> float:
    """Algorithm 1, lines 7-10: IBP box, worst next action, TD regression of Q_phi."""
    horizon = rollout.states.shape[0]
    total_loss = 0.0
    update_count = 0

    for _ in range(args.epochs):
        permutation = torch.randperm(horizon, device=rollout.states.device)

        for start in range(0, horizon, args.minibatch):
            indices = permutation[start : start + args.minibatch]

            loss = wocar.critic_loss(
                rollout.states[indices],
                rollout.actions[indices],
                rollout.rewards[indices],
                rollout.next_actor_obs[indices],
                rollout.next_states[indices],
                rollout.terminated[indices],
                epsilon,
            )

            opt_q.zero_grad()
            loss.backward()
            opt_q.step()

            total_loss += loss.item()
            update_count += 1

    return total_loss / max(update_count, 1)


def _fit_policy(
    actor: GaussianActor,
    opt_actor: Optimizer,
    wocar: WocaRLosses,
    rollout: Rollout,
    epsilon: float,
    kappa_wst: float,
    kappa_reg: float,
    args: Namespace,
) -> dict[str, float]:
    """Algorithm 1, lines 12-13: robust clipped PPO plus weighted regularization."""
    robust_advantages = _robust_advantages(
        wocar, rollout.advantages, rollout.states, rollout.actions, kappa_wst
    )
    # Algorithm 1, line 11. V and Q_phi are fixed during the policy phase, so
    # w(s) is computed once for the whole rollout.
    state_weights = wocar.state_importance(rollout.states)

    horizon = rollout.states.shape[0]
    stats = {"actor": 0.0, "reg": 0.0, "entropy": 0.0}
    update_count = 0

    for _ in range(args.epochs):
        permutation = torch.randperm(horizon, device=rollout.states.device)

        for start in range(0, horizon, args.minibatch):
            indices = permutation[start : start + args.minibatch]

            dist = actor.dist(rollout.actor_obs[indices])
            new_log_probs = dist.log_prob(rollout.actions[indices])
            ratio = torch.exp(new_log_probs - rollout.old_log_probs[indices])

            advantage_mb = robust_advantages[indices]
            surrogate_1 = ratio * advantage_mb
            surrogate_2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * advantage_mb
            entropy = dist.entropy().mean()

            # Algorithm 1, line 12: w(s) * L_reg(s) by SGLD.
            reg_loss = wocar.regularizer_loss(
                rollout.actor_obs[indices],
                rollout.states[indices],
                epsilon,
                weights=state_weights[indices],
            )

            actor_loss = (
                -torch.min(surrogate_1, surrogate_2).mean()
                - args.ent * entropy
                + kappa_reg * reg_loss
            )

            opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            opt_actor.step()

            stats["actor"] += actor_loss.item()
            stats["reg"] += reg_loss.item()
            stats["entropy"] += entropy.item()
            update_count += 1

    return {key: value / max(update_count, 1) for key, value in stats.items()}


def _robust_advantages(
    wocar: WocaRLosses, advantages: Tensor, states: Tensor, actions: Tensor, kappa_wst: float
) -> Tensor:
    """
    normalize(A_hat + kappa_wst * Q_phi(s, a)) over the whole rollout.

    The raw GAE advantages and raw Q_phi are added first (Algorithm 1, line 13);
    the sum is normalized once, as PPO implementations normalize the advantage.
    """
    return _normalize(wocar.shift_advantages(advantages, states, actions, kappa_wst))


def _normalize(values: Tensor) -> Tensor:
    return (values - values.mean()) / (values.std() + 1e-8)


# --------------------------------------------------------------------------- #
# Construction and training loop
# --------------------------------------------------------------------------- #


def build_wocar(
    actor: GaussianActor,
    critic: nn.Module,
    state_dim: int,
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    args: Namespace,
    device: torch.device,
) -> tuple[WocaRLosses, torch.optim.Optimizer]:
    q_network: nn.Module = mlp(state_dim + action_dim, 1, 64, out_gain=1.0).to(device)
    wocar = WocaRLosses(
        value_network=critic,
        q_network=q_network,
        bounder=PolicyActionBounder(actor.mean_network),  # same object as the actor uses
        searcher=WorstActionSearcher(q_network),
        regularizer=SgldRegularizer(actor.mean_network, log_std=actor.log_std),  # same object again
        discount_factor=args.gamma,
        action_lower=torch.as_tensor(action_low, dtype=torch.float32, device=device),
        action_upper=torch.as_tensor(action_high, dtype=torch.float32, device=device),
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
    wocar, opt_q = build_wocar(
        actor, critic, obs_dim, action_dim, action_space.low, action_space.high, args, device
    )

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
            }, args.save)

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


class PpoPlayer(PolicyPlayer):
    def __init__(self, env_id: str, obs_dim: int, action_dim: int) -> None:
        self._device: torch.device = torch.device("cpu")
        self._env = gym.make(env_id, render_mode="human")
        self._actor: GaussianActor = GaussianActor(obs_dim, action_dim, self._device)
        self._obs: np.ndarray | None = None

    def load_weights(self, weights: Weights) -> None:
        self._actor.load_state_dict(weights)

    def start_episode(self) -> None:
        self._obs = self._env.reset()[0].astype(np.float32)

    def step(self) -> StepResult:
        if self._obs is None:
            raise RuntimeError("start_episode must be called before step")

        assert isinstance(self._env.action_space, spaces.Box)

        with torch.no_grad():
            mean: Tensor = self._actor.mean_network(_to_batch(self._obs, self._device))  # deterministic action
        action: np.ndarray = np.clip(
            mean.squeeze(0).numpy(), self._env.action_space.low, self._env.action_space.high
        )
        next_obs, _reward, terminated, truncated, _info = self._env.step(action)
        self._obs = next_obs.astype(np.float32)
        return StepResult(finished=terminated or truncated, reward=float(_reward))

    def close(self) -> None:
        self._env.close()


def build_ppo_player(checkpoint: dict) -> PolicyPlayer:
    player = PpoPlayer(checkpoint["env_id"], checkpoint["obs_dim"], checkpoint["action_dim"])
    player.load_weights(checkpoint["actor"])
    return player
