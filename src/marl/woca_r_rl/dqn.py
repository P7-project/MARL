"""
    WocaR-DQN (Algorithm 2 of Liang et al., NeurIPS 2022) for discrete actions.

    Networks:
    Q_r, the robust Q-network. It acts, and its bounds define A_adv_hat.
    Q_r', a hard-copied target of Q_r.
    Q_phi, the worst-attack critic (one value per action).

    Per update(batch, step):
    + Q_r target: a' = argmax_a [(1 - r) Q_r'(s', a) + r Q_phi(s', a)]; y_r = reward + gamma * Q_r'(s', a').
    + Q_r loss: worst_kappa * TD + (1 - worst_kappa) * worst_case_loss + kappa_reg * L_reg.
    + Every worst_rate-th update: A_adv_hat(s') from IBP bounds of Q_r; y = r + gamma * min over A_adv_hat of Q_phi(s', a'); Huber regression of Q_phi.
    + Every target_update_every updates: copy Q_r into Q_r'.

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

from dataclasses import dataclass
from math import exp
from typing import Literal, NamedTuple, Optional

from gymnasium import spaces
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

from ..env import any_vehicle_crashed, build_actor_obs, flatten_obs, make_env

from functools import partial

from ..rendering import AsyncPolicyRenderer, PolicyPlayer, StepResult, Weights, create_renderer


class Batch(NamedTuple):
    states: Tensor  # [batch, *observation_shape]
    actions: Tensor  # [batch], integer
    rewards: Tensor  # [batch]
    next_states: Tensor  # [batch, *observation_shape]
    terminated: Tensor  # [batch], float 0/1; must exclude time-limit truncation


@dataclass(frozen=True)
class WocaRDQNConfig:
    num_actions: int
    discount_factor: float
    learning_rate: float
    target_update_every: int  # updates between hard copies Q_r -> target
    worst_mix_ratio: float  # official worst_ratio: weight of Q_phi in the target-action mix
    worst_update_every: int  # official worst_rate: Q_phi steps every N updates
    reachable_epsilon: Optional[float]  # official 1e-7; None uses the scheduled epsilon
    epsilon_schedule: RampSchedule  # radius for the regularizer
    worst_epsilon_schedule: RampSchedule  # radius for the worst-case loss
    worst_kappa_schedule: RampSchedule  # final_value = 1 - worst_kappa_end; TD weight = 1 - value
    kappa_reg: float
    observation_low: float
    observation_high: float
    adam_epsilon: float = 1.5e-4
    use_huber_loss: bool = True
    regularizer_margin: float = 1.0  # official hinge_c
    bound_method: str = "IBP"
    perturbation_scale: Optional[Tensor] = None


class TrainingMetrics(NamedTuple):
    worst_attack_loss: float
    robust_td_loss: float
    worst_case_loss: float
    regularizer_loss: float
    epsilon: float
    worst_epsilon: float
    worst_kappa: float
    mean_reachable_actions: float


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
        robust_q: BoundedModule,
        target_q: BoundedModule,
        worst_attack_q: nn.Module,
        config: WocaRDQNConfig,
    ) -> None:
        self._robust_q = robust_q
        self._target_q = target_q
        self._worst_attack_q = worst_attack_q
        self._config = config
        self._update_count: int = 0
        self._last_worst_attack_loss: float = float("nan")
        self._last_mean_reachable: float = float("nan")
        self._bounder = RobustQBounder(
            robust_q,
            config.observation_low,
            config.observation_high,
            config.perturbation_scale,
            config.bound_method,
        )
        self._estimator = ReachableActionEstimator()
        self._regularizer = SARegularizer(config.regularizer_margin)
        self._robust_optimizer = Adam(
            robust_q.parameters(), lr=config.learning_rate, eps=config.adam_epsilon
        )
        self._worst_attack_optimizer = Adam(
            worst_attack_q.parameters(), lr=config.learning_rate, eps=config.adam_epsilon
        )
        for parameter in target_q.parameters():
            parameter.requires_grad_(False)
        self._sync_target()

    # select_action and robust_q_state_dict stay as they are.

    def update(self, batch: Batch, step: int) -> TrainingMetrics:
        self._update_count += 1
        epsilon = self._config.epsilon_schedule.value(step)
        worst_epsilon = self._config.worst_epsilon_schedule.value(step)
        worst_kappa = 1.0 - self._config.worst_kappa_schedule.value(step)

        # Q_r first, so its target uses Q_phi from before this step, as in the official code.
        td_loss, worst_case_loss, regularizer_loss = self._update_robust_q(
            batch, epsilon, worst_epsilon, worst_kappa
        )
        worst_attack_loss, mean_reachable = self._maybe_update_worst_attack_critic(batch, epsilon)
        self._maybe_sync_target()

        return TrainingMetrics(
            worst_attack_loss=worst_attack_loss,
            robust_td_loss=td_loss,
            worst_case_loss=worst_case_loss,
            regularizer_loss=regularizer_loss,
            epsilon=epsilon,
            worst_epsilon=worst_epsilon,
            worst_kappa=worst_kappa,
            mean_reachable_actions=mean_reachable,
        )


    @torch.no_grad()
    def mean_reachable_actions_at(self, batch: Batch, radius: float) -> float:
        bounds: QValueBounds = self._bounder.bounds(batch.next_states, radius)
        reachable: Tensor = self._estimator.mask(bounds)
        return float(reachable.float().sum(dim=1).mean())

    # ----------------------------------------------------------------- Q_r ---

    def _update_robust_q(
        self, batch: Batch, epsilon: float, worst_epsilon: float, worst_kappa: float
    ) -> tuple[float, float, float]:
        targets = self._robust_targets(batch)
        clean_q_values = self._robust_q(batch.states)
        td_loss = self._td_loss(self._gather(clean_q_values, batch.actions), targets)
        worst_case_loss = self._worst_case_loss(batch, clean_q_values, targets, worst_epsilon)
        regularizer_loss = self._regularizer_loss(batch.states, clean_q_values, epsilon)
        loss = (
            worst_kappa * td_loss
            + (1.0 - worst_kappa) * worst_case_loss
            + self._config.kappa_reg * regularizer_loss
        )
        self._step(self._robust_optimizer, loss)
        return float(td_loss.detach()), float(worst_case_loss.detach()), float(regularizer_loss.detach())

    @torch.no_grad()
    def _robust_targets(self, batch: Batch) -> Tensor:
        """Pick a' = argmax of the mix, then bootstrap from the natural target Q at a' (not from the mix)."""
        target_values = self._target_q(batch.next_states)
        worst_values = self._worst_attack_q(batch.next_states)
        ratio = self._config.worst_mix_ratio
        mixed_values = (1.0 - ratio) * target_values + ratio * worst_values
        best_actions = mixed_values.argmax(dim=1, keepdim=True)
        return self._bootstrap(batch, target_values.gather(1, best_actions).squeeze(1))

    def _worst_case_loss(
        self, batch: Batch, clean_q_values: Tensor, targets: Tensor, epsilon: float
    ) -> Tensor:
        """Taken action: bounds vs the TD target. Other actions: bounds vs their current Q."""
        bounds = self._bounder.bounds(batch.states, epsilon)
        taken = nn.functional.one_hot(batch.actions.long(), num_classes=self._config.num_actions).bool()
        reference = torch.where(taken, targets.unsqueeze(1), clean_q_values)
        deviation = torch.maximum(
            (bounds.upper - reference).abs(), (bounds.lower - reference).abs()
        )
        return torch.minimum(deviation.pow(2), deviation).sum(dim=1).mean()

    def _regularizer_loss(self, states: Tensor, clean_q_values: Tensor, epsilon: float) -> Tensor:
        best_actions: Tensor = clean_q_values.detach().argmax(dim=1)
        margin_lower: Tensor = self._bounder.margin_lower_bounds(
            states, best_actions, self._config.num_actions, epsilon
        )  # [batch, A-1]
        per_state_loss: Tensor = self._regularizer.per_state_loss(margin_lower)
        weights: Tensor = self._state_importance(clean_q_values)
        return (weights * per_state_loss).mean()

    @staticmethod
    @torch.no_grad()
    def _state_importance(clean_q_values: Tensor) -> Tensor:
        """
            Math: w(s) = (max_a Q(s, a) - min_a Q(s, a)) / max over the batch,
            the paper's state importance weight (Algorithm 2, line 16).
        """
        spread: Tensor = clean_q_values.max(dim=1).values - clean_q_values.min(dim=1).values
        return spread / spread.max().clamp(min=1e-8)

    # --------------------------------------------------------------- Q_phi ---

    def _maybe_update_worst_attack_critic(self, batch: Batch, epsilon: float) -> tuple[float, float]:
        if self._update_count % self._config.worst_update_every != 0:
            return self._last_worst_attack_loss, self._last_mean_reachable
        reachable_epsilon = self._config.reachable_epsilon
        radius = epsilon if reachable_epsilon is None else reachable_epsilon
        targets, reachable = self._worst_attack_targets(batch, radius)
        predictions = self._gather(self._worst_attack_q(batch.states), batch.actions)
        loss = self._td_loss(predictions, targets)
        self._step(self._worst_attack_optimizer, loss)
        self._last_worst_attack_loss = float(loss.detach())
        self._last_mean_reachable = float(reachable.float().sum(dim=1).mean())
        return self._last_worst_attack_loss, self._last_mean_reachable

    # ------------------------------------------------------------- target ---

    def _maybe_sync_target(self) -> None:
        if self._update_count % self._config.target_update_every == 0:
            self._sync_target()

    @torch.no_grad()
    def _sync_target(self) -> None:
        self._target_q.load_state_dict(self._robust_q.state_dict())


    def _td_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        if self._config.use_huber_loss:
            return nn.functional.smooth_l1_loss(predictions, targets)
        return nn.functional.mse_loss(predictions, targets)


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


    def robust_q_state_dict(self) -> dict[str, Tensor]:
        return self._robust_q.state_dict()


    @torch.no_grad()
    def _worst_attack_targets(self, batch: Batch, radius: float) -> tuple[Tensor, Tensor]:
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
        bounds = self._bounder.bounds(batch.next_states, radius)
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
        lower, upper = self._bounded_q_network.compute_bounds(
            x=(self._bounded_input(states, epsilon),), method=self._method
        )
        return QValueBounds(lower, upper)

    def margin_lower_bounds(
        self, states: Tensor, best_actions: Tensor, num_actions: int, epsilon: float
    ) -> Tensor:
        specification: Tensor = self._margin_specification(best_actions, num_actions)
        lower, _ = self._bounded_q_network.compute_bounds(
            x=(self._bounded_input(states, epsilon),), C=specification, method=self._method
        )
        return lower

    def _bounded_input(self, states: Tensor, epsilon: float) -> BoundedTensor:
        radius = self._radius(states, epsilon)
        lower_corner = (states - radius).clamp(self._observation_low, self._observation_high)
        upper_corner = (states + radius).clamp(self._observation_low, self._observation_high)
        return BoundedTensor(states, PerturbationLpNorm(x_L=lower_corner, x_U=upper_corner))

    def _margin_specification(self, best_actions: Tensor, num_actions: int) -> Tensor:
        """[batch, A-1, A]: one row per a != a*, computing Q(a*) - Q(a)."""
        identity: Tensor = torch.eye(num_actions, device=best_actions.device)
        rows: Tensor = identity[best_actions].unsqueeze(1) - identity.unsqueeze(0)
        is_best: Tensor = nn.functional.one_hot(best_actions, num_actions).bool()
        others: Tensor = torch.logical_not(is_best)
        return rows[others].view(best_actions.shape[0], num_actions - 1, num_actions)

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

        Math: L_reg(s) = max{ -min_{a != a*} LB[Q(a*) - Q(a)], -c },  a* = argmax_a Q_r(s, a).
        Zero gradient once the clean action is certified to stay best by margin c.
    """

    def __init__(self, margin: float) -> None:
        self._margin = margin

    def per_state_loss(self, margin_lower_bounds: Tensor) -> Tensor:
        """max(-min_a LB[Q(a*) - Q(a)], -c), the official hinge."""
        worst_margin: Tensor = margin_lower_bounds.min(dim=1).values
        return torch.clamp(-worst_margin, min=-self._margin)


def train_dqn(args: Namespace, console: Console) -> None:
    with console.status("[bold green]Setting up WocaR-DQN..."):
        if args.total_steps <= args.warmup:
            raise ValueError(f"--total_steps ({args.total_steps}) must be >= --warmup ({args.warmup})")

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        n_agents: int = args.agents
        env_name: str = getattr(args, "dqn_env", "highway")
        env = make_env(n_agents, name=env_name)
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
            partial(DqnPlayer, n_agents, obs_dim, n_actions, args.hidden, env_name),
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
                    writer.add_scalar(
                        "woca/reachable_at_eps_final",
                        trainer.mean_reachable_actions_at(batch, args.eps_final),
                        step,
                    )

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
                },
                args.save,
            )

    with console.status("[bold green]Cleaning up..."):
        if renderer is not None:
            renderer.close()
        writer.close()
        env.close()


def _build_trainer(
    obs_dim: int, n_agents: int, n_actions: int, args: Namespace, device: torch.device
) -> WocaRDQNTrainer:
    input_dim: int = obs_dim + n_agents
    perturbation_scale: Tensor = torch.cat(
        [torch.ones(obs_dim), torch.zeros(n_agents)]
    ).to(device)
    robust_q: BoundedModule = _build_robust_q(input_dim, n_actions, args.hidden, device)
    target_q: BoundedModule = _build_robust_q(input_dim, n_actions, args.hidden, device)
    worst_attack_q: nn.Module = mlp(input_dim, n_actions, args.hidden, out_gain=1.0).to(device)
    ramp_steps: int = args.worst_schedule if args.worst_schedule > 0 else args.total_steps # official: max(num_frames, worst_schedule)

    config: WocaRDQNConfig = WocaRDQNConfig(
        num_actions=n_actions,
        discount_factor=args.gamma,
        learning_rate=args.lr,
        target_update_every=args.target_update_every,
        worst_mix_ratio=args.worst_ratio,
        worst_update_every=args.worst_rate,
        reachable_epsilon=None if args.reachable_eps_scheduled else args.reachable_eps,
        epsilon_schedule=RampSchedule(
            start_step=args.warmup,
            ramp_steps=args.total_steps - args.warmup,
            final_value=args.eps_final,
        ),
        worst_epsilon_schedule=RampSchedule(0, ramp_steps, args.worst_eps_final),
        worst_kappa_schedule=RampSchedule(0, ramp_steps, 1.0 - args.worst_kappa_final),
        kappa_reg=args.kappa_reg,
        observation_low=-1.0,
        observation_high=1.0,
        perturbation_scale=perturbation_scale,
        adam_epsilon=args.adam_eps,
        use_huber_loss=not args.mse_loss,
        regularizer_margin=args.regularizer_margin,
    )
    return WocaRDQNTrainer(robust_q, target_q, worst_attack_q, config)

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
    """Plays the greedy Q_r policy in a multi-agent driving environment."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        hidden: int,
        env_name: str = "highway",
    ) -> None:
        self._n_agents: int = n_agents
        self._device: torch.device = torch.device("cpu")
        self._env = make_env(n_agents, render_mode="human", name=env_name)
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
        agent_terminated = np.asarray(terminated, dtype=np.bool_)
        episode_truncated = np.asarray(truncated, dtype=np.bool_)
        finished: bool = bool(agent_terminated.all()) or bool(episode_truncated.any())
        reward: float = float(np.asarray(_reward, dtype=np.float32).mean())
        crashed: bool = any_vehicle_crashed(self._env)
        return StepResult(finished=finished, reward=reward, crashed=crashed)

    def close(self) -> None:
        self._env.close()

    @torch.no_grad()
    def _greedy_actions(self, obs: np.ndarray) -> list[int]:
        actor_obs: Tensor = _to_tensor(build_actor_obs(obs, self._n_agents), self._device)
        q_values: Tensor = self._policy(actor_obs)  # [n_agents, n_actions]
        return [int(action) for action in q_values.argmax(dim=1)]
def build_dqn_player(checkpoint: dict) -> PolicyPlayer:
    player = DqnPlayer(
        checkpoint["n_agents"],
        checkpoint["obs_dim"],
        checkpoint["n_actions"],
        checkpoint["hidden"],
        env_name=checkpoint.get("dqn_env", "highway"),
    )
    player.load_weights(checkpoint["robust_q"])
    return player
