# `marl.common`: shared building blocks

Code that more than one algorithm uses. Import it from the package:

```python
from marl.common import RampSchedule, compute_gae, mlp
```

Inside `src/marl` use the relative form, e.g. `from ..common import PolicyPlayer` (tests import via `src.marl.common`).

| File | Holds | Used by |
|---|---|---|
| `networks.py` | `layer_init`, `mlp`, `Actor`, `GaussianActor`, `Critic` | MAPPO (`training/trainer.py`), WocaR-DQN (`woca_r_rl/dqn.py`), WocaR-PPO (`woca_r_rl/ppo.py`), attack tests |
| `schedules.py` | `RampSchedule` | WocaR-DQN, WocaR-PPO |
| `advantages.py` | `compute_gae` | MAPPO (`mappo.py`), WocaR-PPO |
| `player.py` | `PolicyPlayer`, `StepResult`, `Weights`, `PlayerFactory` | every player, the live view, `training/evaluator.py` |
| `live_view.py` | `AsyncPolicyRenderer`, `add_render_arguments`, `create_renderer`, `evaluate_in_windows` | `scripts/train.py`, MAPPO and WocaR-DQN training, `training/evaluator.py` |

## Networks (`networks.py`)

`mlp(in_dim, out_dim, hidden, out_gain)` is a 3-layer network: two hidden layers of width `hidden` with tanh, then a linear output layer. Every linear layer goes through `layer_init`: orthogonal weights (gain sqrt(2), the output layer uses `out_gain`) and zero bias.

- `Actor`: categorical policy over discrete actions (MAPPO). `dist(x)` returns a `Categorical`.
- `GaussianActor`: Gaussian policy for continuous actions (WocaR-PPO). The mean network is an auto_LiRPA `BoundedModule`, so bounds can be computed on it; `log_std` is a free parameter.
- `Critic`: value function V(s), returns shape `[batch]`.

WocaR-DQN builds its Q-networks directly with `mlp(...)`.

## `RampSchedule` (`schedules.py`)

A value that is 0 at the start of training and grows to a final value. WocaR uses it for the attack radius eps_t and for kappa_wst(t).

Fields:

- `start_step`: the value is 0 for every `step <= start_step`.
- `ramp_steps`: how many steps the ramp takes. Must be positive (otherwise `ValueError`).
- `final_value`: the value reached at `start_step + ramp_steps` and kept after that.
- `shape`: `"linear"` (default) or `"exponential"`.
- `growth`: steepness of the exponential shape (default 5.0). Ignored for linear.

`value(step)` computes `progress = min((step - start_step) / ramp_steps, 1)` and returns

- linear: `final_value * progress`
- exponential: `final_value * (exp(growth * progress) - 1) / (exp(growth) - 1)`

The exponential shape starts at 0 and ends at `final_value` like the linear one, but stays small for most of the ramp and rises near the end. A larger `growth` delays the rise more.

## `compute_gae` (`advantages.py`)

`compute_gae(rewards, values, dones, last_value, gamma, lam)` computes generalized advantage estimates for one rollout.

- `rewards`, `values`, `dones`: one entry per step of the rollout (length `horizon`). `dones[t]` is 1 when the episode ended at step t, so the value after it is not used.
- `last_value`: V of the state after the last step, used to bootstrap the end of the rollout.
- `gamma`: discount factor. `lam`: the GAE lambda (0 = one-step TD error, 1 = Monte Carlo return minus V).

Per step, going backwards: `delta_t = r_t + gamma * V_{t+1} * (1 - done_t) - V_t` and `A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}`.

Returns `(advantages, returns)`, two float32 numpy arrays of length `horizon`, with `returns = advantages + values` (the value-function targets). MAPPO calls it with the team reward, so all agents get the same advantage.

## Players (`player.py`)

A player runs a trained policy in an environment, one step at a time. The live view and the evaluation both work only through this interface, so they do not need to know which algorithm made the policy.

A new algorithm (for example a vanilla DQN baseline) subclasses `PolicyPlayer` and implements four methods:

- `load_weights(weights: Weights) -> None`: copy the weights into the player's policy. `Weights` is `dict[str, Tensor]`, i.e. a `state_dict` of the network that acts.
- `start_episode() -> None`: reset the env. Must be called before the first `step`.
- `step() -> StepResult`: choose an action with the current policy (greedy, no exploration), step the env, render a frame when rendering is on, and return `StepResult(finished, reward, crashed)`. `finished` is true when the episode is over (terminated or truncated). `reward` is a float (the multi-agent players return the mean over agents). `crashed` is `None` if the env has no notion of a crash.
- `close() -> None`: close the env.

Rules that come from how the player is used:

- The player is built inside a child process (spawned, not forked). Everything it holds (env, networks) is created there. Its constructor arguments must be plain and picklable; the network is built on CPU in the constructor and filled later by `load_weights`.
- The trainer passes a `PlayerFactory`: a zero-argument callable that builds the player. It must pickle, so use a top-level class or `functools.partial` of one (see `partial(DqnPlayer, ...)` in `woca_r_rl/dqn.py`), not a lambda or a nested function.
- For `scripts/evaluate.py`, add a top-level `build_<name>_player(checkpoint: dict) -> PolicyPlayer` that builds the player from the saved checkpoint and loads its weights, save an `"algorithm"` field in the checkpoint, and add the name to `_builder_for` in `training/evaluator.py`.

Existing players to copy from: `MappoPlayer` (`training/trainer.py`), `DqnPlayer` (`woca_r_rl/dqn.py`, also supports attacks via `set_attack`), `PpoPlayer` (`woca_r_rl/ppo.py`).

## Live view and windowed evaluation (`live_view.py`)

**Live view during training (`--render`).** `add_render_arguments(parser)` adds `--render`, `--render-fps`, `--render-publish-every` and `--render-refresh-seconds` to `scripts/train.py`. `create_renderer(args, player_factory)` returns `None` without `--render`, otherwise an `AsyncPolicyRenderer`.

The renderer starts one child process that builds the player and plays episodes at `--render-fps`, so training is not slowed by the window. The training loop calls `renderer.publish_if_due(step, get_weights)` every iteration; at most every `--render-publish-every` steps it sends a CPU copy of the weights (`get_weights` is only called then). The child loads the newest weights at the start of each episode, and also mid-episode every `--render-refresh-seconds` seconds when that is above 0. Call `renderer.close()` at the end of training. MAPPO and WocaR-DQN support it; WocaR-PPO does not.

**Windowed evaluation.** `evaluate_in_windows(player_factory, episodes, windows)` starts `windows` independent processes, each with its own player and window (placed in a grid), plays `episodes` episodes in each and prints steps, return and crash per episode. `scripts/evaluate.py` uses it for evaluation without `--attack` (`--eval-windows`).
