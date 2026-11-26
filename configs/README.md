# Config files

A config file is a JSON object that sets the arguments of `scripts/train.py`.
It replaces a long command line and keeps the settings of an experiment in one place.

## How to use

Run from the repository root:

```bash
uv run scripts/train.py --config configs/wocar_dqn_highway.json
```

- Values in the file replace the defaults of `build_parser` in `scripts/train.py`.
- Flags given on the command line override the file. For a short test run of the example:

  ```bash
  uv run scripts/train.py --config configs/wocar_dqn_highway.json --total-steps 6000 --warmup 500
  ```

- Keys are the argparse dest names: the flag without the leading `--`, with `-` replaced by `_`.
  `--kappa-reg` becomes `kappa_reg`, `--update-function` becomes `update_function`.
- Boolean switches (`render`, `double_dqn`) must be JSON `true` or `false`; a string such as `"false"` is rejected with an error.
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
| `update_function` | Selects the training mode. | `mappo`, `wocar_ppo`, `wocar_dqn`, `dqn` | `mappo` | all |
| `total_steps` | Total environment steps. | int | 200000 | all |
| `gamma` | Discount factor. | float | 0.99 | all |
| `seed` | Seeds numpy and torch. | int | 0 | all |
| `save` | Checkpoint path, relative to the working directory. Only set it when you want a fixed path; runs with the same path overwrite each other. Its folder is created at launch if it does not exist. | str or `null` | `null` (becomes `<logdir>/<run_name>/model.pt`, in the run's own folder) | all |
| `logdir` | TensorBoard root; logs go to `<logdir>/<run_name>`. | str | `runs` | all |
| `run_name` | Sub-directory name inside `logdir`. Every run gets its own new folder: if `<logdir>/<run_name>` already exists (same-second start or a reused name), `_2`, `_3`, ... is appended; the final name is printed at launch. | str | `null` (becomes `<update_function>_<day>_<HH-MM-SS>`) | all |
| `agents` | Number of controlled vehicles. | int | 3 | mappo, wocar_dqn, dqn |
| `reward` | Reward function for the multi-agent environments (`marl.reward_funcs`). | keys of `REWARD_FUNCTIONS` in `src/marl/reward_funcs.py` | `team` | mappo, wocar_dqn, dqn |
| `team_weight` | Weight of the teammates' mean reward in each agent's reward, in [0, 1]. Only used by `reward: team`. | float or `null` | `null` (0.25 for highway, 0.1 for intersection) | mappo, wocar_dqn, dqn |
| `render` | Show the current policy in a separate window while training. | bool | `false` | mappo, wocar_dqn, dqn |
| `render_fps` | Frames per second of the render process. | float | 15.0 | mappo, wocar_dqn, dqn |
| `render_publish_every` | Training steps between weight publishes. | int | 100 | mappo, wocar_dqn, dqn |
| `render_refresh_seconds` | Swap in the newest weights mid-episode every N seconds (0 = only at episode start). | float | 0 | mappo, wocar_dqn, dqn |
| `eps_final` | Final attack radius ε, the value the epsilon schedule ramps to. The default suits Hopper, not highway; pass it for DQN. | float | 0.075 | wocar_ppo, wocar_dqn |
| `kappa_reg` | Weight of the regularizer term in the loss. Paper grid for PPO: {0.01, 0.03, 0.1, 0.3, 1.0}; for DQN: {0.005, 0.01, 0.02}. | float | 0.1 | wocar_ppo, wocar_dqn |

`config` and `help` are not valid keys inside a config file and are rejected like unknown keys.

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
| `kappa_wst_final` | Final weight of the `Q_phi` advantage shift. WocaR-PPO only; WocaR-DQN uses `dqn_kappa_wst_final`. | float | 0.8 | wocar_ppo |
| `kappa_wst_shape` | Ramp shape for `kappa_wst`. Exponential only on Ant. | `linear`, `exponential` | `linear` | wocar_ppo |

### WocaR-DQN (partly also read by vanilla DQN)

| Key | What it does | Type / allowed values | Default | Used by |
|---|---|---|---|---|
| `dqn_env` | Environment for WocaR-DQN. | `highway`, `intersection` | `highway` | wocar_dqn, dqn |
| `lr` | Learning rate for `Q_v`, `Q_phi` and `Q_r`. | float | 1e-4 | wocar_dqn, dqn |
| `regularizer_margin` | Margin of the regularizer hinge (official `hinge_c`). | float | 1 | wocar_dqn |
| `hidden` | Width of the three Q-networks. | int | 64 | wocar_dqn, dqn |
| `tau` | Soft-update coefficient for the target network `Q_v'`. | float | 0.005 | wocar_dqn, dqn |
| `dqn_kappa_wst_final` | Final value of `kappa_wst`, the weight of `Q_phi` in the `Q_r` target (exponential ramp). WocaR-DQN only; WocaR-PPO uses `kappa_wst_final`. | float | 0.5 | wocar_dqn |
| `kappa_wst_start_fraction` | Fraction of `total_steps` before the `kappa_wst` ramp starts; it ends at `total_steps`. | float | 1/3 | wocar_dqn |
| `double_dqn` | Double-DQN target for `Q_v`. When off, the plain DQN target `r + gamma * max_a' Q_v'(s', a')` as in Algorithm 2. | bool | `false` | wocar_dqn, dqn |
| `buffer_size` | Replay buffer capacity, in transitions (one per agent per env step). | int | 100000 | wocar_dqn, dqn |
| `warmup` | Transitions in the buffer before updates start; also the env step where ε starts ramping. Must be smaller than `total_steps`. | int | 5000 | wocar_dqn, dqn |
| `batch_size` | Minibatch size per update. | int | 64 | wocar_dqn, dqn |
| `explore_steps` | Env steps over which the exploration probability decays. | int | 50000 | wocar_dqn, dqn |
| `explore_final` | Final epsilon-greedy exploration probability (not the attack radius). | float | 0.05 | wocar_dqn, dqn |
