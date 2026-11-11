"""
    WocaR-DQN (Algorithm 2 of Liang et al., NeurIPS 2022) for discrete actions.

    Networks:

    Q_v plus a target Q_v', trained as a vanilla DQN.
    Q_r, the robust Q-network. It acts, and its bounds define A_adv_hat.
    Q_phi, the worst-attack critic (outputs one value per action).

    Per update(batch, step):

    + Update Q_v with the (double) DQN target, then soft-update Q_v'.
    + Compute IBP output bounds of Q_r at s' with the current eps_t: Q_l(a), Q_u(a).
    + A_adv_hat(s') = {a : Q_u(a) >= max_{a' != a} Q_l(a')}.
    + a_hat = argmin_{a in A_adv_hat} Q_phi(s', a); y = r + gamma * Q_phi(s', a_hat).
      Update phi with MSE.
    + w(s) = normalized(max_a Q_v(s, a) - min_a Q_v(s, a)).
    + Train Q_r on
          y_r = r + gamma * max_a' [kappa_wst * Q_v'(s', a') + (1 - kappa_wst) * Q_phi(s', a')]
          loss = (y_r - Q_r(s, a))^2 + kappa_reg * w(s) * L_reg(s)
      with L_reg the SA-DQN regularizer on Q_r.

    Math                                         Code
    (T_worst^pi Q)(s, a)                         _worst_attack_targets
    E_{s' ~ P(s, a)}                             replay-buffer sample next_states
    R(s, a)                                      batch.rewards
    gamma                                        config.discount_factor
    min_{a' in A_adv(s', pi)} Q(s', a')          _worst_case_values
    A_adv(s, pi)                                 ReachableActionEstimator.mask
    B_eps(s)                                     RobustQBounder.bounds (clipped L-inf ball)
    Q (worst-attack critic)                      worst_attack_q
    pi = argmax_a Q_r(s, a)                      robust_q
    eps_t, kappa_wst(t)                          RampSchedule
    w(s)                                         _state_importance
    L_reg                                        SARegularizer
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import exp
from typing import Any, Literal, NamedTuple

from gymnasium import spaces
import gymnasium
from rich.console import Console
import torch
from torch import Tensor, nn
from torch.optim import Adam

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from ..models import mlp

from argparse import Namespace

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ..env import build_actor_obs, flatten_obs, make_env

from functools import partial

from ..rendering import AsyncPolicyRenderer, PolicyPlayer, StepResult, Weights, create_renderer


class Batch(NamedTuple):
    states: Tensor  # [batch, *observation_shape]
    actions: Tensor  # [batch], integer
    rewards: Tensor  # [batch]
    next_states: Tensor  # [batch, *observation_shape]
    terminated: Tensor  # [batch], float 0/1; must exclude time-limit truncation


class TrainingMetrics(NamedTuple):
    vanilla_loss: float
    worst_attack_loss: float
    robust_td_loss: float
    regularizer_loss: float
    epsilon: float
    kappa_wst: float
    mean_reachable_actions: float


@dataclass(frozen=True)
class WocaRDQNConfig:
    """
        Schedule example (paper, Atari, 6M steps): kappa_wst_schedule =
        RampSchedule(start_step=2_000_000, ramp_steps=4_000_000, final_value=0.5, shape="exponential").
        Rescale the steps to your training length (for example start at one third).
    """

    num_actions: int
    discount_factor: float
    target_update_coefficient: float
    learning_rate: float
    epsilon_schedule: RampSchedule
    kappa_wst_schedule: RampSchedule
    kappa_reg: float
    observation_low: float
    observation_high: float
    regularizer_margin: float = 0.0
    use_double_dqn: bool = True
    bound_method: str = "IBP"
    # Per-entry multiplier on epsilon (same shape as one observation or broadcastable),
    # for example 0 on highway-env's `presence` column. None means 1 everywhere.
    perturbation_scale: Tensor | None = None


class WocaRDQNTrainer:
    """
        Trains Q_v, Q_phi and Q_r (Algorithm 2).

        Sample-based estimate of the worst-attack Bellman operator (Definition 4.1).

        (T_worst^pi Q)(s, a) := E_{s' ~ P(s, a)} [ R(s, a) + gamma * min_{a' in A_adv(s', pi)} Q(s', a') ]
        A_adv(s, pi) := {a in A : exists s~ in B_eps(s) s.t. pi(s~) = a}

        Fixed point of T_worst^pi: the worst-attack value Q_worst^pi, i.e. the
        expected return of pi when an adversary may perturb every observation
        within B_eps and picks the perturbation that minimizes the future value.
        Learning Q_worst means regressing Q_phi onto the targets of _worst_attack_targets.

        Here pi(s) = argmax_a Q_r(s, a), and A_adv is replaced by its superset A_adv_hat,
        so the fixed point of the modified operator is a lower bound on Q_worst^pi.
    """

    def __init__(
        self,
        vanilla_q: nn.Module,
        robust_q: BoundedModule,
        worst_attack_q: nn.Module,
        config: WocaRDQNConfig,
    ) -> None:
        self._vanilla_q = vanilla_q
        self._target_vanilla_q = self._frozen_copy(vanilla_q)
        # BoundedModule: the optimizer below must use *its* parameters.
        self._robust_q = robust_q
        # Q in the definition: the worst-attack critic Q_phi.
        self._worst_attack_q = worst_attack_q
        # config.discount_factor is gamma in the definition.
        self._config = config
        # Bounds of Q_r over B_eps(s'); these encode both pi and B_eps.
        self._bounder = RobustQBounder(
            robust_q,
            config.observation_low,
            config.observation_high,
            config.perturbation_scale,
            config.bound_method,
        )
        # Provides A_adv_hat(s', pi) from those bounds.
        self._estimator = ReachableActionEstimator()
        self._regularizer = SARegularizer(config.regularizer_margin)
        self._vanilla_optimizer = Adam(vanilla_q.parameters(), lr=config.learning_rate)
        self._robust_optimizer = Adam(robust_q.parameters(), lr=config.learning_rate)
        self._worst_attack_optimizer = Adam(worst_attack_q.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def select_action(self, state: Tensor, exploration_probability: float) -> int:
        """
            Epsilon-greedy w.r.t. Q_r (the acting policy).

            Math: pi(s) = argmax_a Q_r(s, a)
        """
        if float(torch.rand(())) < exploration_probability:
            return int(torch.randint(self._config.num_actions, ()))
        q_values = self._robust_q(state.unsqueeze(0))
        return int(q_values.argmax(dim=1))

    def update(self, batch: Batch, step: int) -> TrainingMetrics:
        epsilon = self._config.epsilon_schedule.value(step)
        kappa_wst = self._config.kappa_wst_schedule.value(step)

        vanilla_loss = self._update_vanilla_q(batch)
        self._soft_update_target()
        worst_attack_loss, mean_reachable = self._update_worst_attack_critic(batch, epsilon)
        robust_td_loss, regularizer_loss = self._update_robust_q(batch, epsilon, kappa_wst)

        return TrainingMetrics(
            vanilla_loss=vanilla_loss,
            worst_attack_loss=worst_attack_loss,
            robust_td_loss=robust_td_loss,
            regularizer_loss=regularizer_loss,
            epsilon=epsilon,
            kappa_wst=kappa_wst,
            mean_reachable_actions=mean_reachable,
        )

    def robust_q_state_dict(self) -> dict[str, Tensor]:
        return self._robust_q.state_dict()

    # ----------------------------------------------------------------- Q_v ---

    def _update_vanilla_q(self, batch: Batch) -> float:
        targets = self._vanilla_targets(batch)
        predictions = self._gather(self._vanilla_q(batch.states), batch.actions)
        loss = nn.functional.mse_loss(predictions, targets)
        self._step(self._vanilla_optimizer, loss)
        return float(loss.detach())

    @torch.no_grad()
    def _vanilla_targets(self, batch: Batch) -> Tensor:
        """
            Math: r + gamma * Q_v'(s', argmax_a' Q_v(s', a')) (double DQN) or max_a' Q_v'(s', a').
        """
        target_q_values = self._target_vanilla_q(batch.next_states)
        if self._config.use_double_dqn:
            best_actions = self._vanilla_q(batch.next_states).argmax(dim=1, keepdim=True)
            next_values = target_q_values.gather(1, best_actions).squeeze(1)
        else:
            next_values = target_q_values.max(dim=1).values
        return self._bootstrap(batch, next_values)

    @torch.no_grad()
    def _soft_update_target(self) -> None:
        """
            Math: theta' <- tau * theta + (1 - tau) * theta'.
        """
        tau = self._config.target_update_coefficient
        for target_parameter, online_parameter in zip(
            self._target_vanilla_q.parameters(), self._vanilla_q.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(online_parameter, alpha=tau)

    # --------------------------------------------------------------- Q_phi ---

    def _update_worst_attack_critic(self, batch: Batch, epsilon: float) -> tuple[float, float]:
        """
            Math: MSE regression of Q_phi onto (T_worst^pi Q_phi)(s, a) for a batch of (s, a, r, s').
        """
        targets, reachable = self._worst_attack_targets(batch, epsilon)
        predictions = self._gather(self._worst_attack_q(batch.states), batch.actions)
        loss = nn.functional.mse_loss(predictions, targets)
        self._step(self._worst_attack_optimizer, loss)
        mean_reachable = float(reachable.float().sum(dim=1).mean())
        return float(loss.detach()), mean_reachable

    @torch.no_grad()
    def _worst_attack_targets(self, batch: Batch, epsilon: float) -> tuple[Tensor, Tensor]:
        """
            Single-sample estimate of (T_worst^pi Q_phi)(s, a) for a batch of (s, a, r, s').

            Math: R(s, a) + gamma * min_{a' in A_adv_hat(s')} Q_phi(s', a')

            - The expectation E_{s' ~ P(s, a)} is replaced by the one sampled s'
            from the replay buffer. Because the min sits inside the expectation,
            this single-sample target is an unbiased estimate of the operator.
            - R(s, a) does not depend on s' or a', so it stays outside the min.
            - `terminated` is 1.0 on true terminal transitions (value 0 afterwards)
            and must exclude time-limit truncation, which does not end the MDP. MDP time limit truncation refers to the practice of ending a reinforcement learning episode after a fixed number of timesteps, which breaks the standard Markov Decision Process (MDP) framework if not handled correctly.
            - Uses Q_phi itself (no target network), as in Algorithm 2.
        """
        bounds = self._bounder.bounds(batch.next_states, epsilon)
        reachable = self._estimator.mask(bounds)
        q_values = self._worst_attack_q(batch.next_states)
        next_values = self._worst_case_values(q_values, reachable)
        return self._bootstrap(batch, next_values), reachable

    def _worst_case_values(self, q_values: Tensor, reachable: Tensor) -> Tensor:
        """
            Math: min over the reachable set; unreachable actions get +inf.
        """
        unreachable = torch.logical_not(reachable)
        return q_values.masked_fill(unreachable, float("inf")).min(dim=1).values

    # ----------------------------------------------------------------- Q_r ---

    def _update_robust_q(
        self,
        batch: Batch,
        epsilon: float,
        kappa_wst: float,
    ) -> tuple[float, float]:
        """
            y_r = r + gamma * max_a' [kappa * Q_v'(s', a') + (1 - kappa) * Q_phi(s', a')]
            loss = (Q_r(s, a) - y_r)^2 + kappa_reg * means_s [ w(s) * L_reg(s) ]. 
        """
        targets = self._robust_targets(batch, kappa_wst)
        clean_q_values = self._robust_q(batch.states)
        td_loss = nn.functional.mse_loss(self._gather(clean_q_values, batch.actions), targets)
        regularizer_loss = self._weighted_regularizer(batch.states, clean_q_values, epsilon)
        loss = td_loss + self._config.kappa_reg * regularizer_loss
        self._step(self._robust_optimizer, loss)
        return float(td_loss.detach()), float(regularizer_loss.detach())

    @torch.no_grad()
    def _robust_targets(self, batch: Batch, kappa_wst: float) -> Tensor:
        """
            Math: y_r = r + gamma * max_a' [kappa * Q_v'(s', a') + (1 - kappa) * Q_phi(s', a')].
        """
        vanilla_values = self._target_vanilla_q(batch.next_states)
        worst_attack_values = self._worst_attack_q(batch.next_states)
        mixed_values = kappa_wst * vanilla_values + (1.0 - kappa_wst) * worst_attack_values
        return self._bootstrap(batch, mixed_values.max(dim=1).values) 
    
    def _weighted_regularizer(
        self,
        states: Tensor,
        clean_q_values: Tensor,
        epsilon: float,
    ) -> Tensor:
        """
            Math: mean_s [ w(s) * L_reg(s) ]. Gradients flow into Q_r through the bounds.
        """
        bounds = self._bounder.bounds(states, epsilon)
        per_state_loss = self._regularizer.per_state_loss(bounds, clean_q_values)
        weights = self._state_importance(states)
        return (weights * per_state_loss).mean()

    @torch.no_grad()
    def _state_importance(self, states: Tensor) -> Tensor:
        """
            Math: w(s) = max_a Q_v(s, a) - min_a Q_v(s, a), divided by its batch maximum.
        """
        q_values = self._vanilla_q(states)
        spread = q_values.max(dim=1).values - q_values.min(dim=1).values
        return spread / spread.max().clamp(min=1e-8)

    # ------------------------------------------------------------- helpers ---

    def _bootstrap(self, batch: Batch, next_values: Tensor) -> Tensor:
        """
            Math: r + gamma * (1 - terminated) * next_values
        """
        not_terminated = 1.0 - batch.terminated
        return batch.rewards + self._config.discount_factor * not_terminated * next_values

    def _gather(self, q_values: Tensor, actions: Tensor) -> Tensor:
        """
            Math: Q(s, a) for a batch of (s, a).
        """
        return q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)

    def _step(self, optimizer: Adam, loss: Tensor) -> None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _frozen_copy(self, network: nn.Module) -> nn.Module:
        copy = deepcopy(network)
        for parameter in copy.parameters():
            parameter.requires_grad_(False)
        return copy


@dataclass(frozen=True)
class RampSchedule:
    """
        0 until start_step, then ramps to final_value over ramp_steps, then stays there.
    """

    start_step: int
    ramp_steps: int
    final_value: float
    shape: Literal["linear", "exponential"] = "linear"
    growth: float = 5.0  # steepness of the exponential shape

    def __post_init__(self) -> None:
        if self.ramp_steps <= 0:
            raise ValueError("ramp_steps must be positive")

    def value(self, step: int) -> float:
        if step <= self.start_step:
            return 0.0
        progress = min((step - self.start_step) / self.ramp_steps, 1.0)
        if self.shape == "linear":
            return self.final_value * progress
        normalizer = exp(self.growth) - 1.0
        return self.final_value * (exp(self.growth * progress) - 1.0) / normalizer


class QValueBounds(NamedTuple):
    lower: Tensor  # [batch, num_actions]
    upper: Tensor  # [batch, num_actions]


class RobustQBounder:
    """
        Output bounds of Q_r over B_eps(s), clipped to the valid observation range.

        Math: Q_l(s, a) <= Q_r(s~, a) <= Q_u(s, a) for all s~ in B_eps(s).
        B_eps is the L-inf ball: {s~ : ||s~ - s||_inf <= eps}, scaled per entry by perturbation_scale
        and clipped to the valid range.
        Assumes observations already lie inside [observation_low, observation_high].
    """

    def __init__(
        self,
        bounded_q_network: BoundedModule,
        observation_low: float,
        observation_high: float,
        perturbation_scale: Tensor | None,
        method: str,
    ) -> None:
        self._bounded_q_network = bounded_q_network
        self._observation_low = observation_low
        self._observation_high = observation_high
        self._perturbation_scale = perturbation_scale
        self._method = method

    def bounds(self, states: Tensor, epsilon: float) -> QValueBounds:
        radius = self._radius(states, epsilon)
        lower_corner = (states - radius).clamp(self._observation_low, self._observation_high)
        upper_corner = (states + radius).clamp(self._observation_low, self._observation_high)
        perturbation = PerturbationLpNorm(x_L=lower_corner, x_U=upper_corner)
        bounded_states = BoundedTensor(states, perturbation)
        lower, upper = self._bounded_q_network.compute_bounds(
            x=(bounded_states,), method=self._method
        )
        return QValueBounds(lower, upper)

    def _radius(self, states: Tensor, epsilon: float) -> Tensor:
        if self._perturbation_scale is None:
            return torch.full_like(states, epsilon)
        return self._perturbation_scale.to(states.device) * epsilon


class ReachableActionEstimator:
    """
        Over-approximates A_adv(s, pi) for pi = argmax_a Q_r(s, a) from per-action output bounds.

        Math: A_adv(s, pi) := {a in A : exists s~ in B_eps(s) s.t. pi(s~) = a}.
        a in A_adv_hat(s) iff Q_u(s, a) >= max_{a' != a} Q_l(s, a').
        If Q_u(a) < Q_l(a') for some a', then a' beats a everywhere in B_eps(s),
        so argmax_a Q_r(s~, a) != a for every s~ in B_eps(s), hence a not in A_adv(s, pi).

        Soundness: the returned set contains A_adv (it only removes actions that are
        provably unreachable), so min over it is <= min over the true A_adv.
        The target therefore lower-bounds the true worst-attack Bellman target:
        conservative, never optimistic.

        The clean greedy action always satisfies the condition, so the set is never empty.
    """

    def mask(self, bounds: QValueBounds) -> Tensor:
        """
            Bool tensor [batch, num_actions]: True where a is in A_adv_hat.
        """
        competitor_lower = self._highest_competitor_lower_bound(bounds.lower)
        reachable = bounds.upper >= competitor_lower
        if not bool(reachable.any(dim=1).all()):
            raise RuntimeError("Empty reachable action set; check bounds for NaN or inf.")
        return reachable

    def _highest_competitor_lower_bound(self, lower: Tensor) -> Tensor:
        """
            [batch, A]: entry (b, i) = max_{j != i} lower[b, j] (-inf if there is one action).
        """
        num_actions = lower.shape[1]
        diagonal = torch.eye(num_actions, dtype=torch.bool, device=lower.device)
        expanded = lower.unsqueeze(1).expand(-1, num_actions, -1)
        others = expanded.masked_fill(diagonal.unsqueeze(0), float("-inf"))
        return others.max(dim=2).values


class SARegularizer:
    """
        SA-DQN regularizer on Q_r.

        Math: L_reg(s) = max{ max_{a != a*} Q_u(s, a) - Q_l(s, a*), -c },  a* = argmax_a Q_r(s, a).
        Zero gradient once the clean action is certified to stay best by margin c.
    """

    def __init__(self, margin: float) -> None:
        self._margin = margin

    def per_state_loss(self, bounds: QValueBounds, clean_q_values: Tensor) -> Tensor:
        """
            [batch]: regularizer value per state.
        """
        best_actions = clean_q_values.detach().argmax(dim=1)
        lower_of_best = bounds.lower.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        best_mask = nn.functional.one_hot(best_actions, num_classes=bounds.upper.shape[1]).bool()
        upper_of_others = bounds.upper.masked_fill(best_mask, float("-inf")).max(dim=1).values
        return torch.clamp(upper_of_others - lower_of_best, min=-self._margin)


def train_dqn(args: Namespace, console: Console) -> None:
    with console.status("[bold green]Setting up WocaR-DQN..."):
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        n_agents: int = args.agents
        env = make_env(n_agents)
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
            args, partial(DqnPlayer, n_agents, obs_dim, n_actions, args.hidden)
        )

        obs: np.ndarray = flatten_obs(env.reset(seed=args.seed)[0])
        episode_return: float = 0.0

    with console.status(f"[bold green]Training WocaR-DQN (0/{args.total_steps})...") as training_status:
        for step in range(args.total_steps):
            training_status.update(f"[bold green]Training WocaR-DQN ({step}/{args.total_steps})...")
            actor_obs: Tensor = _to_tensor(build_actor_obs(obs, n_agents), device)  # [n_agents, input_dim]
            actions: list[int] = _select_actions(trainer, actor_obs, _exploration_probability(step, args))

            next_obs, reward, terminated, truncated, _info = env.step(tuple(actions))
            next_obs = flatten_obs(next_obs)
            # Built before any reset, so it is the real next observation.
            next_actor_obs: Tensor = _to_tensor(build_actor_obs(next_obs, n_agents), device)

            for agent in range(n_agents):
                buffer.add(actor_obs[agent], actions[agent], float(reward), next_actor_obs[agent], float(terminated))

            episode_return += float(reward)
            if terminated or truncated:
                writer.add_scalar("episode/return", episode_return, step)
                episode_return = 0.0
                next_obs = flatten_obs(env.reset()[0])
            obs = next_obs

            if len(buffer) >= args.warmup:
                metrics = trainer.update(buffer.sample(args.batch_size), step)  # global env step drives eps_t, kappa_wst
                if step % 100 == 0:
                    for name, value in metrics._asdict().items():
                        writer.add_scalar(f"woca/{name}", value, step)

            if renderer is not None:
                renderer.publish_if_due(step, trainer.robust_q_state_dict)

        if args.save:
            torch.save(
            {
                "algorithm": "wocar_dqn",
                "robust_q": trainer.robust_q_state_dict(),
                "n_agents": n_agents,
                "obs_dim": obs_dim,
                "n_actions": n_actions,
                "hidden": args.hidden,
            },
            args.save,
        )
    if renderer is not None:
        renderer.close()
    writer.close()
    env.close()


def _build_trainer(
    obs_dim: int, n_agents: int, n_actions: int, args: Namespace, device: torch.device
) -> WocaRDQNTrainer:
    input_dim: int = obs_dim + n_agents
    vanilla_q: nn.Module = mlp(input_dim, n_actions, args.hidden, out_gain=1.0).to(device)
    worst_attack_q: nn.Module = mlp(input_dim, n_actions, args.hidden, out_gain=1.0).to(device)
    robust_q: BoundedModule = _build_robust_q(input_dim, n_actions, args.hidden, device)
    # Perturb the observation part, never the one-hot agent id.
    perturbation_scale: Tensor = torch.cat(
        [torch.ones(obs_dim), torch.zeros(n_agents)]
    ).to(device)

    config: WocaRDQNConfig = WocaRDQNConfig(
        num_actions=n_actions,
        discount_factor=args.gamma,
        target_update_coefficient=args.tau,
        learning_rate=args.lr,
        # (our choice) ramp windows rescaled from the paper's 6M-step Atari schedule.
        epsilon_schedule=RampSchedule(
            start_step=args.warmup,
            ramp_steps=args.total_steps - args.warmup,
            final_value=args.eps_final,
        ),
        kappa_wst_schedule=RampSchedule(
            start_step=args.total_steps // 3,
            ramp_steps=args.total_steps - args.total_steps // 3,
            final_value=0.5,
            shape="exponential",
        ),
        kappa_reg=args.kappa_reg,
        observation_low=-1.0,  # verify against your env's observation config
        observation_high=1.0,
        perturbation_scale=perturbation_scale,
    )
    return WocaRDQNTrainer(vanilla_q, robust_q, worst_attack_q, config)

def _build_robust_q(input_dim: int, n_actions: int, hidden: int, device: torch.device) -> BoundedModule:
    return BoundedModule(
        mlp(input_dim, n_actions, hidden, out_gain=1.0).to(device),
        torch.zeros(1, input_dim, device=device),
        device=device.type,
    )


def _select_actions(trainer: WocaRDQNTrainer, actor_obs: Tensor, exploration_probability: float) -> list[int]:
    return [trainer.select_action(agent_obs, exploration_probability) for agent_obs in actor_obs]


def _exploration_probability(step: int, args: Namespace) -> float:
    """(mine) linear decay from 1.0 to explore_final over explore_steps."""
    progress: float = min(step / args.explore_steps, 1.0)
    return 1.0 + progress * (args.explore_final - 1.0)


def _to_tensor(array: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


class ReplayBuffer:
    def __init__(self, capacity: int, input_dim: int, device: torch.device) -> None:
        self._capacity: int = capacity
        self._device: torch.device = device
        self._size: int = 0
        self._cursor: int = 0
        self._states: Tensor = torch.zeros(capacity, input_dim, device=device)
        self._actions: Tensor = torch.zeros(capacity, dtype=torch.long, device=device)
        self._rewards: Tensor = torch.zeros(capacity, device=device)
        self._next_states: Tensor = torch.zeros(capacity, input_dim, device=device)
        self._terminated: Tensor = torch.zeros(capacity, device=device)

    def __len__(self) -> int:
        return self._size

    def add(self, state: Tensor, action: int, reward: float, next_state: Tensor, terminated: float) -> None:
        self._states[self._cursor] = state
        self._actions[self._cursor] = action
        self._rewards[self._cursor] = reward
        self._next_states[self._cursor] = next_state
        self._terminated[self._cursor] = terminated
        self._cursor = (self._cursor + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int) -> Batch:
        indices: Tensor = torch.randint(self._size, (batch_size,), device=self._device)
        return Batch(
            states=self._states[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_states=self._next_states[indices],
            terminated=self._terminated[indices],
        )


class DqnPlayer(PolicyPlayer):
    """Plays the greedy Q_r policy in the multi-agent highway env."""

    def __init__(self, n_agents: int, obs_dim: int, n_actions: int, hidden: int) -> None:
        self._n_agents: int = n_agents
        self._device: torch.device = torch.device("cpu")
        self._env = make_env(n_agents, render_mode="human")  # needs the make_env change
        self._policy: BoundedModule = _build_robust_q(
            obs_dim + n_agents, n_actions, hidden, self._device
        )
        self._policy.eval()
        self._obs: np.ndarray | None = None

    def load_weights(self, weights: Weights) -> None:
        self._policy.load_state_dict(weights)

    def start_episode(self) -> None:
        self._obs = flatten_obs(self._env.reset()[0])

    def step(self) -> StepResult:
        if self._obs is None:
            raise RuntimeError("start_episode must be called before step")
        actions: list[int] = self._greedy_actions(self._obs)
        next_obs, _reward, terminated, truncated, _info = self._env.step(tuple(actions))
        self._env.render()
        self._obs = flatten_obs(next_obs)
        crashed: bool = self.any_vehicle_crashed(self._env)

        return StepResult(finished=terminated or truncated, reward=float(_reward), crashed=crashed)

    def any_vehicle_crashed(self, env: gymnasium.Env) -> bool:
        unwrapped: Any = env.unwrapped  # highway-env specific attributes are unknown to the type checker
        return any(vehicle.crashed for vehicle in unwrapped.controlled_vehicles)

    def close(self) -> None:
        self._env.close()

    @torch.no_grad()
    def _greedy_actions(self, obs: np.ndarray) -> list[int]:
        actor_obs: Tensor = _to_tensor(build_actor_obs(obs, self._n_agents), self._device)
        q_values: Tensor = self._policy(actor_obs)  # [n_agents, n_actions]
        return [int(action) for action in q_values.argmax(dim=1)]


def build_dqn_player(checkpoint: dict) -> PolicyPlayer:
    player = DqnPlayer(
        checkpoint["n_agents"], checkpoint["obs_dim"], checkpoint["n_actions"], checkpoint["hidden"]
    )
    player.load_weights(checkpoint["robust_q"])
    return player