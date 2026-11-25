# Evaluation CLI

A single entry point for evaluating a trained model, in windows or under an observation attack. What works depends on the algorithm of the checkpoint:

| Algorithm | Windows | `--attack` | Environment | Number of cars |
|---|---|---|---|---|
| WocaR-DQN (`wocar_dqn`) | yes | yes | From the checkpoint | From the checkpoint |
| MAPPO (`mappo`) | yes | no | Always highway | From the checkpoint |
| WocaR-PPO (`wocar_ppo`) | no | no | - | - |

WocaR-PPO checkpoints don't load: they are saved without the environment and network sizes needed to rebuild the policy. Checkpoints without an algorithm name are treated as MAPPO.

The observation attack is selected with `--attack`:

| Value | Description |
|---|---|
| not given | No attack. The trained policy is shown in windows. |
| `minbest` | MinBest (FGSM): one gradient step that makes the car's best action less likely. Runs without a window unless `--render` is given. |

MinBest moves each feature by at most ε and then clips it to [-1, 1].

Run commands from the repository root. `uv run --locked` uses dependencies from `uv.lock`.

## Examples

```bash
# Exp-1: no car is attacked (natural performance, without windows)
uv run --locked python scripts/evaluate.py --checkpoint woca_dqn.pt --attack minbest --attacked-agents 0 --episodes 100

# Exp-2: one random car is attacked per episode
uv run --locked python scripts/evaluate.py --checkpoint woca_dqn.pt --attack minbest --attacked-agents 1 --episodes 100

# Exp-3: all cars are attacked
uv run --locked python scripts/evaluate.py --checkpoint woca_dqn.pt --attack minbest --attacked-agents all --episodes 100

# Custom epsilons
uv run --locked python scripts/evaluate.py --checkpoint woca_dqn.pt --attack minbest --epsilon 0.01 0.03 0.05 --episodes 100

# Show one window with a view of each of the first 3 cars; attacked cars are outlined in red
uv run --locked python scripts/evaluate.py --checkpoint woca_dqn.pt --attack minbest --attacked-agents 1 --render --render-agents 3 --episodes 2

# Save the results as JSON
uv run --locked python scripts/evaluate.py --checkpoint woca_dqn.pt --attack minbest --attacked-agents 1 --episodes 100 --out results.json
```

## Arguments

### General evaluation

| Argument | Default | Meaning |
|---|---|---|
| `--checkpoint` (or `--eval`) | Required | Path to a WocaR-DQN or MAPPO checkpoint (see the table above). |
| `--episodes` (or `--eval-episodes`) | 1000 with `--attack`, otherwise 5 | Number of evaluation episodes: per epsilon with `--attack`, otherwise per window. |
| `--eval-windows` | 3 | Number of windows. Only without `--attack`. |
| `--render` | off | With `--attack`: show one window (highway only). Has no effect otherwise (see Rendering). |
| `--render-agents` | 1 | Only with `--attack` and `--render`. Number of views in the window, from 1 to the number of cars. |
| `--seed` | 0 | Only with `--attack`. Seeds the environment and the choice of attacked cars. |

### Observation attack

| Argument | Default | Meaning |
|---|---|---|
| `--attack` | None | Name of the attack: `minbest`. |
| `--epsilon` (or `--attack-eps`) | 1x and 3x the checkpoint's `eps_final` | One or more attack radii, each at least 0 and evaluated in turn. Required if the checkpoint has no `eps_final`, unless no car is attacked. |
| `--attacked-agents` | `all` | Number of attacked cars, from 0 to the number of cars, or `all`. |

With `--attacked-agents 0`, `--epsilon` is ignored and there is one result, with epsilon 0.

### Output

| Argument | Default | Meaning |
|---|---|---|
| `--out` | None | Only with `--attack`. Path of the JSON results file. The directory must exist; an existing file is overwritten. |
| `--logdir` | `runs` | Only with `--attack`. TensorBoard root directory. |
| `--run-name` | Timestamp | Only with `--attack`. Name of this evaluation run. Use a new name for each run. |

## Targeting

- `--attacked-agents` sets how many cars are attacked. Which ones is drawn at random for each episode.
- The cars stay fixed for the whole episode and are attacked at every step.
- The draw depends only on `--seed` and the episode number, so every epsilon attacks the same cars. A larger count keeps the cars of a smaller one.
- Episode `i` (from 0) resets the environment with seed `--seed + i`, so every epsilon and count starts from the same scenes.

## Rendering

Without `--attack`, windows are always shown, with or without `--render`:

```bash
# Three windows, two episodes each
uv run --locked python scripts/evaluate.py --checkpoint woca_dqn.pt --eval-windows 3 --episodes 2
```

Each window runs its own environment with all cars and prints the steps, return and crash of each episode. The episodes are not seeded.

With `--attack`, no window is opened by default and `--eval-windows` is ignored. `--render` opens one window, for all epsilons and episodes in turn:

```bash
# One attacked car per episode, with a view of each of the 3 cars
uv run --locked python scripts/evaluate.py --checkpoint woca_dqn.pt --attack minbest --attacked-agents 1 --render --render-agents 3 --episodes 2
```

- Attacked cars are outlined in red. No car is outlined with `--attacked-agents 0` or at epsilon 0. The very first frame of an episode has no outlines yet.
- `--render-agents N` shows one view per car for the first N cars. An attacked car outside these views is not visible, so use the number of cars to see all of them.
- A crashed car is filled in light red, which is not an attack outline.
- It only works for highway checkpoints; others give an error.
- Rendering does not change the results. Pass a small `--episodes`, since the default with `--attack` is 1000.

## Outputs

With `--attack`, one line is printed per epsilon. Its values are:

| Value | Meaning |
|---|---|
| `attacked` | Attacked cars / all cars. |
| `eps` | Attack radius. |
| `return` | Mean and standard deviation over episodes. An episode's return is the sum over steps of the reward averaged over all cars. |
| `crash_rate` | Fraction of episodes in which any controlled car crashed. |
| `changed_actions` | Fraction of the attacked cars' decisions where the attack changed the chosen action. `n/a` if no car is attacked. |

The JSON file has two parts:

- `settings`: `checkpoint`, `env`, `n_agents`, `attack`, `attacked_agents`, `epsilons`, `episodes`, `seed`.
- `results`: one entry per epsilon, with `epsilon`, `return_mean`, `return_std`, `crash_rate`, `changed_action_rate` (`null` if no car is attacked), and per episode `episode_returns`, `episode_crashes` and `episode_attacked_cars` (car indices from 0).

With `--attack`, the results are also logged to TensorBoard under `runs/attack_evaluation/<run-name>`:

| Scalar | Meaning |
|---|---|
| `episode_reward` | Return of each episode. |
| `episode_crashed` | 1 if any controlled car crashed in the episode, otherwise 0. |
| `reward_mean`, `reward_std` | Mean and standard deviation of the return. |
| `crash_rate`, `changed_action_rate` | As above. `changed_action_rate` is left out if no car is attacked. |

The names start with the setting, e.g. `minbest/attacked_1/epsilon_0.075/episode_reward`, or `clean/` if no car is attacked. Each epsilon is logged when it is finished.

```bash
uv run --locked tensorboard --logdir=runs
```

Open <http://localhost:6006> to compare evaluation runs.

## Adding an attack

1. Add one file in `src/marl/attacks/` with the attack. It must be callable like `ObservationAttack` in `registry.py`: it returns the observations of all cars, and changes only the first `obs_dim` features of the cars in `attacked_mask`.
2. Add one entry to `ATTACK_BUILDERS` in `src/marl/attacks/registry.py`: the name and a function that builds the attack.

The name is then accepted by `--attack`.
