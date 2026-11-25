# Config files

A config file is a JSON object that sets the arguments of `scripts/train.py`.
It replaces a long command line and keeps the settings of an experiment in one place.

## How to use

Run from the repository root:

```bash
uv run --locked python scripts/train.py --config configs/wocar_dqn_highway.json
```

- Values in the file replace the defaults of `build_parser` in `scripts/train.py`.
- Flags given on the command line override the file. For a short test run of the example:

  ```bash
  uv run --locked python scripts/train.py --config configs/wocar_dqn_highway.json --total-steps 6000 --warmup 500
  ```

- Keys are the argparse dest names: the flag without the leading `--`, with `-` replaced by `_`.
  `--kappa-reg` becomes `kappa_reg`, `--update-function` becomes `update_function`.
- Boolean switches (`render`, `reachable_eps_scheduled`, `mse_loss`) take `true` or `false`.
- JSON has no comments. Put notes in the file name or in this README, not in the file.
- An unknown key (e.g. a typo like `kapa_reg`), a value outside the allowed choices
  (e.g. `"update_function": "wocar-dqn"`), a missing file or invalid JSON stops the
  script with an error message before training starts.
- Only allowed choices are checked. Types are not: write numbers as JSON numbers
  (`0.01`, not `"0.01"`).

## How to make a new config

1. Copy the example: `cp configs/wocar_dqn_highway.json configs/<new_name>.json`.
2. Change the values you need, and add keys from the table below. Keys you leave out keep their default.
3. Use one file per experiment, named after what it runs (mode, environment, the setting that differs).

## Keys

"Used by" says which `--update-function` modes read the key. A mode that does not read a key accepts it and ignores it.
Defaults are the `build_parser` defaults (see also `scripts/train.md`).

### Shared

| Key | What it does | Type / allowed values | Default | Used by |
|---|---|---|---|---|
| `update_function` | Selects the training mode. | `mappo`, `wocar_ppo`, `wocar_dqn` | `mappo` | all |
| `total_steps` | Total environment steps. | int | 200000 | all |
| `gamma` | Discount factor. | float | 0.99 | all |
| `seed` | Seeds numpy and torch. | int | 0 | all |
| `save` | Checkpoint path, relative to the working directory. | str | `default_save_name.N.pt` (next unused `N`, picked at launch) | all |
| `logdir` | TensorBoard root; logs go to `<logdir>/<run_name>`. | str | `runs` | all |
| `run_name` | Sub-directory name inside `logdir`. | str | `null` (becomes `<update_function>_<timestamp>`) | all |
| `agents` | Number of controlled vehicles. | int | 3 | mappo, wocar_dqn |
| `reward` | Reward function for the multi-agent environments (`marl.reward_funcs`). | keys of `REWARD_FUNCTIONS` in `src/marl/reward_funcs.py` | `team` | mappo, wocar_dqn |
| `team_weight` | Weight of the teammates' mean reward in each agent's reward, in [0, 1]. Only used by `reward: team`. | float or `null` | `null` (0.25 for highway, 0.1 for intersection) | mappo, wocar_dqn |
| `render` | Show the current policy in a separate window while training. | bool | `false` | mappo, wocar_dqn |
| `render_fps` | Frames per second of the render process. | float | 15.0 | mappo, wocar_dqn |
| `render_publish_every` | Training steps between weight publishes. | int | 100 | mappo, wocar_dqn |
| `render_refresh_seconds` | Swap in the newest weights mid-episode every N seconds (0 = only at episode start). | float | 0 | mappo, wocar_dqn |
| `eps_final` | Final attack radius ε, the value the epsilon schedule ramps to. The default suits Hopper, not highway; pass it for DQN. | float | 0.075 | wocar_ppo, wocar_dqn |
| `kappa_reg` | Weight of the regularizer term in the loss. Paper grid for PPO: {0.01, 0.03, 0.1, 0.3, 1.0}; for DQN: {0.005, 0.01, 0.02}. | float | 0.1 | wocar_ppo, wocar_dqn |

`config` itself is not a valid key inside a config file.

### MAPPO (also read by WocaR-PPO)

| Key | What it does | Type / allowed values | Default | Used by |
|---|---|---|---|---|
| `rollout` | Joint env steps per update. | int | 512 | mappo, wocar_ppo |
| `epochs` | Passes over each rollout. | int | 10 | mappo, wocar_ppo |
| `minibatch` | Minibatch size. | int | 128 | mappo, wocar_ppo |
| `lr_actor` | Actor learning rate. | float | 3e-4 | mappo, wocar_ppo |
| `lr_critic` | Critic learning rate (also `Q_phi` in wocar_ppo). | float | 1e-3 | mappo, wocar_ppo |
| `lam` | GAE lambda. | float | 0.95 | mappo, wocar_ppo |
| `clip` | PPO clip range. | float | 0.2 | mappo, wocar_ppo |
| `ent` | Entropy bonus coefficient. | float | 0.01 | mappo, wocar_ppo |
| `max_grad_norm` | Gradient clipping norm. | float | 0.5 | mappo, wocar_ppo |

### WocaR-PPO

| Key | What it does | Type / allowed values | Default | Used by |
|---|---|---|---|---|
| `env_id` | Gymnasium environment id. | str | `Hopper-v5` | wocar_ppo |
| `kappa_wst_final` | Final weight of the `Q_phi` advantage shift. | float | 0.8 | wocar_ppo |
| `kappa_wst_shape` | Ramp shape for `kappa_wst`. Exponential only on Ant. | `linear`, `exponential` | `linear` | wocar_ppo |

### WocaR-DQN

| Key | What it does | Type / allowed values | Default | Used by |
|---|---|---|---|---|
| `dqn_env` | Environment for WocaR-DQN. | `highway`, `intersection` | `highway` | wocar_dqn |
| `lr` | Learning rate for `Q_v`, `Q_phi` and `Q_r`. | float | 1e-4 | wocar_dqn |
| `adam_eps` | Epsilon of the Adam optimizers. | float | 1.5e-4 | wocar_dqn |
| `regularizer_margin` | Margin of the regularizer hinge (official `hinge_c`). | float | 1 | wocar_dqn |
| `hidden` | Width of the three Q-networks. | int | 64 | wocar_dqn |
| `target_update_every` | Updates between hard copies of `Q_r` into the target network. Placeholder value. | int | 2000 | wocar_dqn |
| `worst_ratio` | Weight of `Q_phi` in the target-action mix (official `worst_ratio`). | float | 0.1 | wocar_dqn |
| `worst_rate` | `Q_phi` step every N updates. | int | 100 | wocar_dqn |
| `worst_eps_final` | Radius of the worst-case loss; pixel scale in the official config, tune for ours. | float | 0.005 | wocar_dqn |
| `worst_schedule` | Ramp length of the worst-case schedules: this value if > 0, otherwise `total_steps`. | int | 0 | wocar_dqn |
| `worst_kappa_final` | TD weight ramps 1 -> this. | float | 0.1 | wocar_dqn |
| `reachable_eps` | Radius for the reachable-action set (official value). | float | 1e-7 | wocar_dqn |
| `reachable_eps_scheduled` | Use the scheduled ε for the reachable-action set instead (paper description). | bool | `false` | wocar_dqn |
| `mse_loss` | MSE instead of Huber loss. | bool | `false` | wocar_dqn |
| `buffer_size` | Replay buffer capacity, in transitions (one per agent per env step). | int | 100000 | wocar_dqn |
| `warmup` | Transitions in the buffer before updates start; also the env step where ε starts ramping. Must be smaller than `total_steps`. | int | 5000 | wocar_dqn |
| `batch_size` | Minibatch size per update. | int | 64 | wocar_dqn |
| `explore_steps` | Env steps over which the exploration probability decays. | int | 50000 | wocar_dqn |
| `explore_final` | Final epsilon-greedy exploration probability (not the attack radius). | float | 0.05 | wocar_dqn |
