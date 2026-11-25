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
| `minbest` | MinBest (FGSM): one gradient step that makes the car's best action less likely. |
| `pgd` | PGD: ten smaller gradient steps that make the car's best action less likely, as in WocaR-RL Sec. 5. |
| `random` | Uniform random noise, independent of the policy. A baseline for the two gradient attacks. |

Every attack runs without windows unless `--render` is given.

All attacks change only the observation features of the attacked cars, never the one-hot agent ID, move each feature by at most ε, and clip it to [-1, 1]. At ε = 0 every attack returns the clean observation, so the result equals the clean result.

- MinBest takes one step of size ε along the sign of the gradient of the cross-entropy loss against the car's clean greedy action.
- PGD starts from a random point, uniform in [-ε, ε] around the clean observation, and then takes 10 steps of size ε/10 along the sign of the gradient of the same loss. After each step the observation is projected back to the ε-ball around the clean observation (L-inf) and clipped to [-1, 1]. The result is the observation after the last step.
- Random adds noise drawn uniformly from [-ε, ε] to each feature.

PGD's random start and the random noise come from a generator seeded with `--seed`. The generator is reset for every ε, so every ε uses the same random numbers and a run is reproducible from its seed.

Run commands from the repository root.

## Examples

```bash
# Exp-1: no car is attacked (natural performance, without windows; --epsilon is required but ignored)
uv run scripts/evaluate.py --checkpoint woca_dqn.pt --attack minbest --epsilon 0 --attacked-agents 0 --episodes 100

# Exp-2: one random car is attacked per episode
uv run scripts/evaluate.py --checkpoint woca_dqn.pt --attack minbest --epsilon 0.05 0.15 --attacked-agents 1 --episodes 100

# Exp-3: all cars are attacked
uv run scripts/evaluate.py --checkpoint woca_dqn.pt --attack minbest --epsilon 0.05 0.15 --attacked-agents all --episodes 100

# PGD and random noise, at the same radii
uv run scripts/evaluate.py --checkpoint woca_dqn.pt --attack pgd --epsilon 0.05 0.15 --attacked-agents 1 --episodes 100
uv run scripts/evaluate.py --checkpoint woca_dqn.pt --attack random --epsilon 0.05 0.15 --attacked-agents 1 --episodes 100

# Watch the attack: one window, the attacked car outlined in red
uv run scripts/evaluate.py --checkpoint woca_dqn.pt --attack pgd --epsilon 0.15 --attacked-agents 1 --episodes 3 --render

# Also copy results.json to another path
uv run scripts/evaluate.py --checkpoint woca_dqn.pt --attack minbest --epsilon 0.05 --attacked-agents 1 --episodes 100 --out results.json
```

## Arguments

### General evaluation

| Argument | Default | Meaning |
|---|---|---|
| `--checkpoint` (or `--eval`) | Required | Path to a WocaR-DQN or MAPPO checkpoint (see the table above). |
| `--episodes` (or `--eval-episodes`) | 1000 with `--attack`, otherwise 5 | Number of evaluation episodes: per epsilon with `--attack`, otherwise per window. |
| `--eval-windows` | 3 | Number of windows. Only without `--attack`. |
| `--render` | off | With `--attack`: show the evaluation in a window and outline the attacked cars. Without `--attack` it has no effect (see Rendering). |
| `--render-agents` | 1 | Only with `--attack --render`, highway only. Number of camera views in the window, one per car (the first cars), from 1 to the number of cars. |
| `--seed` | 0 | Only with `--attack`. Seeds the environment, the choice of attacked cars, and the random numbers of `pgd` and `random`. |

### Observation attack

| Argument | Default | Meaning |
|---|---|---|
| `--attack` | None | Name of the attack: `minbest`, `pgd` or `random`. |
| `--epsilon` (or `--attack-eps`) | Required with `--attack` | One or more attack radii, each at least 0 and evaluated in turn. There is no default, so every model is compared at the same radii (the checkpoint's `eps_final` is not used). |
| `--attacked-agents` | `all` | Number of attacked cars, from 0 to the number of cars, or `all`. |

With `--attacked-agents 0`, `--epsilon` is ignored and there is one result, with epsilon 0.

### Output

| Argument | Default | Meaning |
|---|---|---|
| `--out` | None | Only with `--attack`. Extra copy of `results.json` at this path. The directory must exist; an existing file is overwritten. |
| `--logdir` | `runs` | Only with `--attack`. Root directory; each run gets its own folder `<logdir>/attack_evaluation/<run-name>`. |
| `--run-name` | `<attack>_<k>of<n>_<day>_<HH-MM-SS>` | Only with `--attack`. Name of the run folder. `k` is the number of attacked cars, `n` the number of cars, `day` the day of the month, e.g. `minbest_1of3_25_22-10-05`. |

Each run reserves a new folder when it starts and prints its path. If the folder already exists, `_2`, `_3`, ... is appended to the name, so two runs never share a folder, even with the same `--run-name` or when started at the same second.

## Targeting

- `--attacked-agents` sets how many cars are attacked. Which ones is drawn at random for each episode.
- The cars stay fixed for the whole episode and are attacked at every step.
- The draw depends only on `--seed` and the episode number, so every epsilon attacks the same cars. A larger count keeps the cars of a smaller one.
- Episode `i` (from 0) resets the environment with seed `--seed + i`, so every epsilon and count starts from the same scenes.

## Rendering

Without `--attack`, rendering is always on, with or without `--render`:

```bash
# Three windows, two episodes each
uv run scripts/evaluate.py --checkpoint woca_dqn.pt --eval-windows 3 --episodes 2
```

Each window runs its own environment with all cars and prints the steps, return and crash of each episode. The episodes are not seeded.

With `--attack`, no window is opened unless `--render` is given; `--eval-windows` is ignored. With `--render`, the attacked evaluation runs in one window, at the speed of the simulation, and gives the same results as without it. On highway, the attacked cars of each episode are outlined in red, and `--render-agents` sets how many cars get their own camera view. The intersection environment shows its usual window without outlines.

```bash
# Two camera views, every car attacked with random noise
uv run scripts/evaluate.py --checkpoint woca_dqn.pt --attack random --epsilon 0.1 --attacked-agents all --episodes 2 --render --render-agents 2
```

## Outputs

With `--attack`, one line is printed per epsilon. Its values are:

| Value | Meaning |
|---|---|
| `attacked` | Attacked cars / all cars. |
| `eps` | Attack radius. |
| `return` | Mean and standard deviation over episodes. An episode's return is the sum over steps of the reward averaged over all cars. |
| `crash_rate` | Fraction of episodes in which any controlled car crashed. |
| `changed_actions` | Fraction of the attacked cars' decisions where the attack changed the chosen action. `n/a` if no car is attacked. |

The results are always saved as `results.json` in the run folder (and copied to `--out` if given). The file has two parts:

- `settings`: `checkpoint`, `env`, `n_agents`, `attack`, `attacked_agents`, `epsilons`, `episodes`, `seed`.
- `results`: one entry per epsilon, with `epsilon`, `return_mean`, `return_std`, `crash_rate`, `changed_action_rate` (`null` if no car is attacked), and per episode `episode_returns`, `episode_crashes` and `episode_attacked_cars` (car indices from 0).

With `--attack`, the results are also logged to TensorBoard in the same run folder, `<logdir>/attack_evaluation/<run-name>`:

| Scalar | Meaning |
|---|---|
| `episode_reward` | Return of each episode. |
| `episode_crashed` | 1 if any controlled car crashed in the episode, otherwise 0. |
| `reward_mean`, `reward_std` | Mean and standard deviation of the return. |
| `crash_rate`, `changed_action_rate` | As above. `changed_action_rate` is left out if no car is attacked. |

The names start with the setting, e.g. `minbest/attacked_1/epsilon_0.075/episode_reward`, or `clean/` if no car is attacked. Each epsilon is logged when it is finished.

```bash
uv run tensorboard --logdir=runs
```

Open <http://localhost:6006> to compare evaluation runs.

## Adding an attack

1. Add one file in `src/marl/attacks/` with the attack. It must be callable like `ObservationAttack` in `registry.py`: it returns the observations of all cars, and changes only the first `obs_dim` features of the cars in `attacked_mask`.
2. Add one entry to `ATTACK_BUILDERS` in `src/marl/attacks/registry.py`: the name and a function that builds the attack from the evaluation seed (`--seed`). An attack that draws random numbers takes them from its own generator seeded with it, never from the global random state.

The name is then accepted by `--attack`.
