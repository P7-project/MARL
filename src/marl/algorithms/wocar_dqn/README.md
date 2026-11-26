# WocaR-DQN (`marl.algorithms.wocar_dqn`)

WocaR-DQN is Algorithm 2 of the WocaR-RL paper (Liang et al., NeurIPS 2022),
run on the multi-agent driving environments in [`marl.envs`](../../envs/README.md).
All cars share one set of networks. Training is started with
`uv run scripts/train.py --update-function wocar_dqn`; the flags are in
[`scripts/train.md`](../../../../scripts/train.md#wocar-dqn-only-wocar_dqn).

## Files

| File | What it holds |
|---|---|
| `__init__.py` | The public interface (see below). |
| `trainer.py` | `Batch`, `WocaRDQNConfig`, `TrainingMetrics` and `WocaRDQNTrainer`, which owns the four networks and does one update per call. Its module docstring maps the math to the code. |
| `bounds.py` | `QValueBounds`, `RobustQBounder` (IBP output bounds of `Q_r` over the perturbation ball) and `ReachableActionEstimator` (the estimated adversarial action set `A_adv_hat`). |
| `regularizer.py` | `SARegularizer`, the SA-DQN hinge regularizer `L_reg`. |
| `replay_buffer.py` | `ReplayBuffer`, uniform sampling with replacement, returns a `Batch`. |
| `networks.py` | `_build_robust_q` (builds `Q_r` as an auto_LiRPA `BoundedModule`) and `_to_tensor`. Used by both `train.py` and `player.py`, so neither imports the other's internals. |
| `train.py` | `train_dqn`, the training loop, and its helpers (`_build_trainer`, `_select_actions`, `_exploration_probability`, `_plain_args`, `_plain_value`). |
| `player.py` | `DqnPlayer` (plays the greedy `Q_r` policy, optionally under an observation attack) and `build_dqn_player` (builds a player from a checkpoint). |

## Public interface

```python
from marl.algorithms.wocar_dqn import train_dqn, build_dqn_player
```

| Name | What it does |
|---|---|
| `train_dqn(args, console)` | Trains WocaR-DQN with the parsed `scripts/train.py` arguments and saves the checkpoint to `args.save`. |
| `build_dqn_player(checkpoint, render, render_agents)` | Builds a `DqnPlayer` from a loaded checkpoint dict. Used by `training/evaluator.py`. |
| `DqnPlayer` | A `PolicyPlayer` for the evaluator and the live view. `set_attack(attack, epsilon, attacked_mask)` attacks the masked cars. |
| `WocaRDQNTrainer`, `WocaRDQNConfig`, `Batch` | The trainer, its settings, and one sampled minibatch. |
| `ReplayBuffer` | The shared replay buffer. |
| `RobustQBounder`, `ReachableActionEstimator`, `SARegularizer` | The building blocks of the trainer (also tested on their own). |

## The four networks

| Network | Code | Role |
|---|---|---|
| `Q_v` | `_vanilla_q` | Vanilla Q-network, trained with the plain DQN target. Gives `w(s)`. |
| `Q_v'` | `_target_vanilla_q` | Target copy of `Q_v`, soft-updated with `tau`, never trained directly. |
| `Q_r` | `_robust_q` | Robust Q-network. It acts (greedy, epsilon-greedy while training), its IBP bounds define `A_adv_hat`, it carries the regularizer. The only network saved. |
| `Q_phi` | `_worst_attack_q` | Worst-attack critic: estimates the value under the worst attack. It never perturbs observations itself. |

All four are `mlp(input_dim, n_actions, hidden)` from `marl.common`; `Q_r` is
wrapped in a `BoundedModule` so bounds can be computed on it. Each has its own
Adam optimizer with the same learning rate `--lr`.

`WocaRDQNTrainer.update(batch, step)` runs in this order, on the same minibatch:

1. Read `eps_t` and `kappa_wst` from their schedules at `step`.
2. Update `Q_v` on the DQN target, then soft-update `Q_v'`.
3. Bound `Q_r` at `s'`, build `A_adv_hat(s')`, update `Q_phi` on the worst-attack target.
4. Update `Q_r` on `y^r` plus `kappa_reg * w(s) * L_reg`.

So `y^r` uses the freshly soft-updated `Q_v'` and the freshly updated `Q_phi`,
and `w(s)` uses the freshly updated `Q_v`.

## Algorithm 2 line by line

Line numbers as in our appendix (the Input line is line 1, one higher than in
the WocaR PDF). All functions are in `trainer.py` unless the file is given.

| Line | Algorithm 2 | Code |
|---|---|---|
| 1 | Input: `T`, `tau`, schedule `eps_t`, schedule `kappa_wst(t)`, `kappa_reg` | `train.py: _build_trainer` fills `WocaRDQNConfig` (`T` = `--total-steps`) |
| 2 | Initialize `Q_v`, `Q_v'`, `Q_r`, `Q_phi` | `train.py: _build_trainer` (`Q_v`, `Q_phi`), `networks.py: _build_robust_q` (`Q_r`), `WocaRDQNTrainer.__init__` (`Q_v'` = `_frozen_copy` of `Q_v`) |
| 3 | Initialize replay buffer `B` | `train.py: train_dqn` (`ReplayBuffer(...)`) |
| 4 | for k = 0, ..., T | `train.py: train_dqn`, loop over `range(args.total_steps)` |
| 5 | With probability beta a random action, else `argmax_a Q_r(s_t, a)` | `train.py: _select_actions` -> `WocaRDQNTrainer.select_action`; beta from `train.py: _exploration_probability` |
| 6 | Execute `a_t`, observe `r_t`, `s_{t+1}` | `train.py: train_dqn` (`env.step`) |
| 7 | Store the transition in `B` | `train.py: train_dqn` (`buffer.add`, one per car) |
| 8 | Sample a minibatch of N transitions | `replay_buffer.py: ReplayBuffer.sample` (only once the buffer holds `--warmup` transitions) |
| 9 | `y_j = r_j + gamma max_a' Q_v'(s'_j, a')` (`r_j` if terminal) | `_vanilla_targets` (terminal handling in `_bootstrap`) |
| 10 | TD loss `(y_j - Q_v(s_j, a_j))^2`, optimize `Q_v` | `_update_vanilla_q` |
| 11 | Soft update `theta_v' <- tau theta_v + (1 - tau) theta_v'` | `_soft_update_target` |
| 12 | IBP bounds `Q_l(s'_j, a, eps_t)`, `Q_u(s'_j, a, eps_t)` of `Q_r` | `bounds.py: RobustQBounder.bounds`, called in `_worst_attack_targets` |
| 13 | Build `A_adv_hat` from the bounds | `bounds.py: ReachableActionEstimator.mask` |
| 14 | `a_hat = argmin_{a in A_adv_hat} Q_phi(s'_j, a)` | `_worst_case_values` (returns the minimum value directly, not the index) |
| 15 | `y^w_j = r_j + gamma Q_phi(s'_j, a_hat)` (`r_j` if terminal) | `_worst_attack_targets` |
| 16 | TD loss `(y^w_j - Q_phi(s_j, a_j))^2`, gradient step on `phi` | `_update_worst_attack_critic` |
| 17 | `w(s_j)` by normalizing `max_a Q_v(s_j, a) - min_a Q_v(s_j, a)` | `_state_importance` |
| 18 | `(y^r_j - Q_r(s_j, a_j))^2 + kappa_reg w(s_j) L_reg`, update `Q_r` | `_update_robust_q`; `y^r` in `_robust_targets`; `w * L_reg` in `_weighted_regularizer`, which uses `bounds.py: RobustQBounder.margin_lower_bounds` and `regularizer.py: SARegularizer.per_state_loss` |
| 19 | end for | |

`eps_t` and `kappa_wst(t)` are read at the start of `update`, from the global
env step. Only `terminated` stops
bootstrapping; a time-limit truncation does not.

Line 13 in code: `a` is in `A_adv_hat(s')` when `Q_u(s', a) >= max_{a' != a} Q_l(s', a')`,
i.e. no other action beats `a` everywhere in the ball. This keeps every action
that may be reachable, so the min in line 14 is never optimistic, and the clean
greedy action is always in the set. The PDF writes the test as a strict `>`
over all `a'`.

## Our corrections to the published algorithm

- **`kappa_wst` swap (line 18).** The PDF writes
  `y^r = r + gamma max_a' [kappa_wst Q_v' + (1 - kappa_wst) Q_phi]`. Its
  schedule starts `kappa_wst` at 0 and grows it, which only makes sense if
  `kappa_wst` weights the worst case. We use
  `y^r = r + gamma max_a' [(1 - kappa_wst) Q_v'(s', a') + kappa_wst Q_phi(s', a')]`.
- **Minibatch index (lines 12, 13, 17).** The PDF writes `s_t`, `s_{t+1}` where
  the minibatch sample is meant. The code works on the whole minibatch.
- **`w(s)` normalization (line 17).** The paper only says "by normalizing".
  We divide by the largest value in the minibatch (our choice), so `w` is in
  `[0, 1]` and `kappa_reg` keeps its meaning.

## Differences from the official WocaR code

The official code (`WocaR-DQN/SA-DQN/train.py`) differs from the published
algorithm. We follow the published algorithm, not the official code:

- it has 3 networks (no vanilla `Q_v`);
- `Q_phi` only enters through action selection, mixed with a fixed
  `worst_ratio`, and the target value comes from the target network;
- `kappa` weights the TD loss against an extra bound loss;
- no `w(s)` is computed;
- `Q_phi` is updated only every `worst_rate` steps.

The hinge regularizer with margin `c = 1` is the same as in the official code.

## Multi-agent part

- **Parameter sharing.** One set of the four networks for all cars.
- **Agent ID.** A one-hot agent ID is appended to each car's observation
  (`build_actor_obs`), so the input is `obs_dim + n_agents` values. The ID is
  never perturbed: `perturbation_scale` in `_build_trainer` is 1 on the
  observation and 0 on the ID, and the attacks only change the first `obs_dim`
  features.
- **Shared buffer.** Every env step adds one transition per car to the same
  buffer, so the buffer grows by `n_agents` per step. The minibatch samples
  transitions, not cars: a car can appear several times or not at all.
- **Per-car reward and termination.** Each transition stores that car's own
  reward (from `--reward`) and its own terminated flag. The episode resets when
  all cars have terminated or the episode is truncated.

## Schedules

- **Attack radius `eps_t`.** 0 until env step `--warmup`, then linear up to
  `--eps-final` at `--total-steps` (`RampSchedule`).
- **`kappa_wst`.** 0 until `--kappa-wst-start-fraction` times `--total-steps`,
  then exponential up to `--dqn-kappa-wst-final` at `--total-steps`.
- **Exploration beta (line 5).** Linear from 1.0 to `--explore-final` over
  `--explore-steps` env steps (our choice; the PDF does not give a schedule).

See [`scripts/train.md`](../../../../scripts/train.md#wocar-dqn-only-wocar_dqn)
for all WocaR-DQN flags, their defaults, and why `--warmup` has two units.

## `--double-dqn`

Off by default, which gives the plain DQN target of Algorithm 2, line 9:
`r + gamma max_a' Q_v'(s', a')`. On, `Q_v` picks the action and `Q_v'` values
it: `r + gamma Q_v'(s', argmax_a' Q_v(s', a'))`. It only changes the `Q_v`
target, not `y^w` or `y^r`.

## Checkpoint

`train_dqn` saves a dict with `algorithm` (`"wocar_dqn"`), `robust_q` (the
`Q_r` state dict), `n_agents`, `obs_dim`, `n_actions`, `hidden`, `dqn_env`,
`reward_function`, `team_weight` and `args` (all arguments as plain values, so
it loads with `torch.load(..., weights_only=True)`). `Q_v`, `Q_v'` and `Q_phi`
are not saved: only `Q_r` acts. `build_dqn_player` reads this dict.
