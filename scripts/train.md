# Training CLI

One entry point, three training modes selected by `--update-function`.

| Value | Algorithm | Trainer | Environment |
|---|---|---|---|
| `mappo` | MAPPO baseline | `train` | multi-agent highway, discrete actions |
| `wocar_ppo` | WocaR-PPO (Algorithm 1) | `train_ppo` | single-agent Mujoco, continuous actions |
| `wocar_dqn` | WocaR-DQN (Algorithm 2) | `train_dqn` | multi-agent highway or intersection, discrete actions |

Run commands from the repository root.

## Examples

```bash
# MAPPO baseline (defaults)
uv run scripts/train.py --update-function mappo

# WocaR-PPO on Hopper (mujoco is installed by `uv sync`)
uv run scripts/train.py --update-function wocar_ppo --env-id Hopper-v5 \
    --total-steps 2000000 --rollout 2048 --save woca_ppo.pt \
    --eps-final 0.075 --kappa-wst-final 0.8 --kappa-reg 0.1

# WocaR-DQN on highway
uv run scripts/train.py --update-function wocar_dqn --dqn-env highway --agents 3 \
    --total-steps 200000 --warmup 5000 --save woca_dqn.pt \
    --eps-final 0.01 --kappa-reg 0.01

# WocaR-DQN on the multi-agent intersection
uv run scripts/train.py --update-function wocar_dqn --dqn-env intersection --agents 2 \
    --total-steps 200000 --warmup 5000 --save woca_dqn_intersection.pt \
    --eps-final 0.01 --kappa-reg 0.01

# Rendered smoke run for WocaR-DQN on intersection (total-steps must exceed warmup)
uv run scripts/train.py --update-function wocar_dqn --dqn-env intersection \
    --agents 2 --total-steps 5000 --warmup 500 --render
```

## Which arguments each mode reads

| Argument group | `mappo` | `wocar_ppo` | `wocar_dqn` |
|---|---|---|---|
| Common | yes | yes | partly (see below) |
| PPO | yes | yes | no |
| WocaR shared (`--eps-final`, `--kappa-reg`) | no | yes | yes |
| WocaR-PPO only | no | yes | no |
| WocaR-DQN only | no | no | yes |

An argument a mode does not read is accepted and ignored.

## Common

| Argument | Default | Meaning |
|---|---|---|
| `--update-function` | `mappo` | Selects the mode: `mappo`, `wocar_ppo`, `wocar_dqn`. |
| `--agents` | 3 | Number of controlled vehicles. Ignored by `wocar_ppo`. |
| `--reward` | `team` | Reward function for the multi-agent environments: `mean` gives every vehicle the team mean, `team` mixes each vehicle's own reward with its teammates' mean. Comes from `REWARD_FUNCTIONS` in `src/marl/reward_funcs.py`. Ignored by `wocar_ppo`. |
| `--team-weight` | per environment | Weight `lambda_env` of the teammates' mean reward; must be in `[0, 1]`. Defaults: `0.25` for highway, `0.1` for intersection. Only used by `--reward team`. Ignored by `wocar_ppo`. |
| `--total-steps` | 200000 | Total environment steps. |
| `--seed` | 0 | Seeds numpy and torch. |
| `--gamma` | 0.99 | Discount factor. |
| `--save` | `default_save_name.N.pt` | Checkpoint path, relative to the current working directory. `N` is the next unused number, picked at launch. Pass a name per run to avoid two simultaneous runs choosing the same file. |
| `--logdir` | `runs` | TensorBoard root. Logs go to `<logdir>/<run-name>`. |
| `--run-name` | timestamp | Sub-directory name inside `--logdir`. |
| `--config` | none | JSON file with argument values; flags on the command line override it. See `configs/README.md`. |
| `--render` | off | Show the current policy in a separate window while training. Ignored by `wocar_ppo`. |
| `--render-fps` | 15.0 | Frames per second of the render process. |
| `--render-publish-every` | 100 | Training steps between weight publishes. |
| `--render-refresh-seconds` | 0 | Swap in the newest weights mid-episode every N seconds (0 = only at episode start). |

With `--reward team` (the default), each vehicle gets

`r_i = (1 - lambda_env) * r_i^env + lambda_env * mean_{j != i} r_j^env`

where `lambda_env` is `--team-weight`, the teammate mean includes crashed and arrived vehicles, and `lambda_env = 0` reproduces the stock rewards. The default differs per environment because a highway episode ends for every vehicle when any of them crashes (a larger weight keeps a teammate's crash visible), while an intersection episode continues per agent, so a crashed teammate's penalty is mixed in again on every remaining step and accumulates (a smaller weight already makes it costly). MAPPO trains on the mean of the returned per-vehicle rewards, which the mixing leaves unchanged.

## PPO (`mappo` and `wocar_ppo`)

| Argument | Default | Meaning |
|---|---|---|
| `--rollout` | 512 | Environment steps collected per update. For `wocar_ppo` this is single-agent steps; 2048 is a more typical Mujoco value. |
| `--epochs` | 10 | Passes over each rollout. |
| `--minibatch` | 128 | Minibatch size. |
| `--lr-actor` | 3e-4 | Actor learning rate. |
| `--lr-critic` | 1e-3 | Critic learning rate. Also used for `Q_phi` in `wocar_ppo`. |
| `--lam` | 0.95 | GAE lambda. |
| `--clip` | 0.2 | PPO clip range. |
| `--ent` | 0.01 | Entropy bonus coefficient. |
| `--max-grad-norm` | 0.5 | Gradient clipping norm. |

## WocaR shared (`wocar_ppo` and `wocar_dqn`)

These two defaults are PPO values. For `wocar_dqn`, pass them explicitly.

| Argument | Default | Meaning |
|---|---|---|
| `--eps-final` | 0.075 | Final attack radius ε. The schedule ramps ε from 0 to this value. The default suits Hopper-scale observations, not normalized highway observations in [-1, 1]; tune it for DQN. |
| `--kappa-reg` | 0.1 | Weight of the `w(s) * L_reg` regularizer. Paper grid for PPO: {0.01, 0.03, 0.1, 0.3, 1.0}. Paper range for DQN: {0.005, 0.01, 0.02}. |

## WocaR-PPO only (`wocar_ppo`)

| Argument | Default | Meaning |
|---|---|---|
| `--env-id` | `Hopper-v5` | Gymnasium environment id (Hopper, Walker2d, HalfCheetah, Ant). |
| `--kappa-wst-final` | 0.8 | Final weight of the `Q_phi` advantage shift. Paper: 0.8 on Hopper, Walker2d, HalfCheetah; 0.5 on Ant. |
| `--kappa-wst-shape` | `linear` | Ramp shape for `kappa_wst`: `linear` or `exponential`. Paper uses exponential only on Ant. |

Both ramps start at step 0 and end at `--total-steps`. That window is our choice, not from the paper.

## WocaR-DQN only (`wocar_dqn`)

Defaults are placeholders, not paper values, unless the row says so.

| Argument | Default | Meaning |
|---|---|---|
| `--dqn-env` | `highway` | Environment: `highway` or `intersection`. |
| `--hidden` | 64 | Width of `Q_v`, `Q_phi` and `Q_r`. |
| `--regularizer-margin` | 1 | Margin `c` of the regularizer hinge (official `hinge_c`). |
| `--tau` | 0.005 | Soft-update coefficient for the target `Q_v'`. |
| `--dqn-kappa-wst-final` | 0.5 | Final value of `kappa_wst`, the weight of `Q_phi` in the `Q_r` target. Paper value. |
| `--kappa-wst-start-fraction` | 1/3 | Fraction of `--total-steps` before the `kappa_wst` ramp starts. Paper: 0 for the first 2M of 6M steps. |
| `--double-dqn` | off | Double-DQN target for `Q_v`. Off gives the plain DQN target `r + gamma * max_a' Q_v'(s', a')` of Algorithm 2. |
| `--lr` | 1e-4 | Learning rate for all three Q-networks. |
| `--buffer-size` | 100000 | Replay capacity in transitions. Each env step adds one transition per agent. |
| `--warmup` | 5000 | See below. |
| `--batch-size` | 64 | Minibatch size per update. |
| `--explore-steps` | 50000 | Env steps over which the exploration probability decays from 1.0. |
| `--explore-final` | 0.05 | Final epsilon-greedy exploration probability. Unrelated to the attack radius ε. |

`kappa_wst` for DQN ramps exponentially from 0 to `--dqn-kappa-wst-final`. The ramp starts at `--kappa-wst-start-fraction` times `--total-steps` and ends at `--total-steps`. `--kappa-wst-final` and `--kappa-wst-shape` are WocaR-PPO only and are not read by `wocar_dqn`.

### `--warmup` has two units

- Updates start when the replay buffer holds `--warmup` transitions. With `n` agents the buffer grows by `n` per env step, so updates start after about `warmup / n` env steps.
- The attack radius ε starts ramping at env step `--warmup`.

With 3 agents, updates begin well before ε starts ramping.

`--total-steps` must be greater than `--warmup`. Otherwise the ε ramp length is zero or negative and construction fails with `ramp_steps must be positive`.

## Outputs

- Checkpoint at `--save`. `mappo` and `wocar_ppo` save actor and critic state dicts. `wocar_dqn` saves the `Q_r` state dict, which is the acting network.
- TensorBoard logs in `<logdir>/<run-name>`: `episode/return`, losses, and for WocaR runs `woca/epsilon` and `woca/kappa_wst`.

```bash
uv run tensorboard --logdir runs
```

## Sanity check

For `wocar_ppo`, `--eps-final 0` and `--kappa-wst-final 0` remove the WocaR terms and should match the plain PPO baseline. For `wocar_dqn`, `--eps-final 0`, `--dqn-kappa-wst-final 0` and `--kappa-reg 0` reduce the `Q_r` loss to a plain TD loss on `r + gamma * max_a' Q_v'(s', a')`. This is still not plain independent DQN: `Q_r` bootstraps from the separately trained `Q_v'`, not from a target copy of itself.
