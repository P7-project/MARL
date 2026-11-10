# Training CLI

One entry point, three training modes selected by `--update-function`.

| Value | Algorithm | Trainer | Environment |
|---|---|---|---|
| `mappo` | MAPPO baseline | `train` | multi-agent highway, discrete actions |
| `wocar_ppo` | WocaR-PPO (Algorithm 1) | `train_ppo` | single-agent Mujoco, continuous actions |
| `wocar_dqn` | WocaR-DQN (Algorithm 2) | `train_dqn` | multi-agent highway, discrete actions |

Replace `scripts/train.py` below with the real path of your entry script.

## Examples

```bash
# MAPPO baseline (defaults)
python scripts/train.py --update-function mappo

# WocaR-PPO on Hopper
python scripts/train.py --update-function wocar --env-id Hopper-v4 \
    --total-steps 2000000 --rollout 2048 --save woca_ppo.pt \
    --eps-final 0.075 --kappa-wst-final 0.8 --kappa-reg 0.1

# WocaR-DQN on highway
python scripts/train.py --update-function wocar_dqn --agents 3 \
    --total-steps 200000 --warmup 5000 --save woca_dqn.pt \
    --eps-final 0.01 --kappa-reg 0.01

# Short smoke test for WocaR-DQN (total-steps must exceed warmup)
python scripts/train.py --update-function wocar_dqn --total-steps 5000 --warmup 500
```

## Which arguments each mode reads

| Argument group | `mappo` | `wocar` | `wocar_dqn` |
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
| `--update-function` | `mappo` | Selects the mode: `mappo`, `wocar`, `wocar_dqn`. |
| `--agents` | 3 | Number of controlled vehicles. Ignored by `wocar`. |
| `--total-steps` | 200000 | Total environment steps. |
| `--seed` | 0 | Seeds numpy and torch. |
| `--gamma` | 0.99 | Discount factor. |
| `--save` | `default_save_name.N.pt` | Checkpoint path, relative to the current working directory. `N` is the next unused number, picked at launch. Pass a name per run to avoid two simultaneous runs choosing the same file. |
| `--logdir` | `runs` | TensorBoard root. Logs go to `<logdir>/<run-name>`. |
| `--run-name` | timestamp | Sub-directory name inside `--logdir`. |

## PPO (`mappo` and `wocar`)

| Argument | Default | Meaning |
|---|---|---|
| `--rollout` | 512 | Environment steps collected per update. For `wocar` this is single-agent steps; 2048 is a more typical Mujoco value. |
| `--epochs` | 10 | Passes over each rollout. |
| `--minibatch` | 128 | Minibatch size. |
| `--lr-actor` | 3e-4 | Actor learning rate. |
| `--lr-critic` | 1e-3 | Critic learning rate. Also used for `Q_phi` in `wocar`. |
| `--lam` | 0.95 | GAE lambda. |
| `--clip` | 0.2 | PPO clip range. |
| `--ent` | 0.01 | Entropy bonus coefficient. |
| `--max-grad-norm` | 0.5 | Gradient clipping norm. |

## WocaR shared (`wocar` and `wocar_dqn`)

These two defaults are PPO values. For `wocar_dqn`, pass them explicitly.

| Argument | Default | Meaning |
|---|---|---|
| `--eps-final` | 0.075 | Final attack radius ε. The schedule ramps ε from 0 to this value. The default suits Hopper-scale observations, not normalized highway observations in [-1, 1]; tune it for DQN. |
| `--kappa-reg` | 0.1 | Weight of the `w(s) * L_reg` regularizer. Paper grid for PPO: {0.01, 0.03, 0.1, 0.3, 1.0}. Paper range for DQN: {0.005, 0.01, 0.02}. |

## WocaR-PPO only (`wocar`)

| Argument | Default | Meaning |
|---|---|---|
| `--env-id` | `Hopper-v4` | Gymnasium environment id (Hopper, Walker2d, HalfCheetah, Ant). |
| `--kappa-wst-final` | 0.8 | Final weight of the `Q_phi` advantage shift. Paper: 0.8 on Hopper, Walker2d, HalfCheetah; 0.5 on Ant. |
| `--kappa-wst-shape` | `linear` | Ramp shape for `kappa_wst`: `linear` or `exponential`. Paper uses exponential only on Ant. |

Both ramps start at step 0 and end at `--total-steps`. That window is our choice, not from the paper.

## WocaR-DQN only (`wocar_dqn`)

All defaults are placeholders, not paper values.

| Argument | Default | Meaning |
|---|---|---|
| `--hidden` | 64 | Width of `Q_v`, `Q_phi` and `Q_r`. |
| `--tau` | 0.005 | Soft-update coefficient for the target `Q_v'`. |
| `--lr` | 1e-4 | Learning rate for all three Q-networks. |
| `--buffer-size` | 100000 | Replay capacity in transitions. Each env step adds one transition per agent. |
| `--warmup` | 5000 | See below. |
| `--batch-size` | 64 | Minibatch size per update. |
| `--explore-steps` | 50000 | Env steps over which the exploration probability decays from 1.0. |
| `--explore-final` | 0.05 | Final epsilon-greedy exploration probability. Unrelated to the attack radius ε. |

`kappa_wst` for DQN is fixed in code: exponential ramp to 0.5, starting at one third of `--total-steps`.

### `--warmup` has two units

- Updates start when the replay buffer holds `--warmup` transitions. With `n` agents the buffer grows by `n` per env step, so updates start after about `warmup / n` env steps.
- The attack radius ε starts ramping at env step `--warmup`.

With 3 agents, updates begin well before ε starts ramping.

`--total-steps` must be greater than `--warmup`. Otherwise the ε ramp length is zero or negative and construction fails with `ramp_steps must be positive`.

## Outputs

- Checkpoint at `--save`. `mappo` and `wocar` save actor and critic state dicts. `wocar_dqn` saves the `Q_r` state dict, which is the acting network.
- TensorBoard logs in `<logdir>/<run-name>`: `episode/return`, losses, and for WocaR runs `woca/epsilon` and `woca/kappa_wst`.

```bash
tensorboard --logdir runs
```

## Sanity check

With `--eps-final 0` and `--kappa-wst-final 0`, the WocaR path should behave like the plain baseline (plain PPO for `wocar`, plain independent DQN for `wocar_dqn`). If the learning curves differ, the bug is in the integration, not in the method.