# Attack Evaluation CLI

One entry point for evaluating a trained WocaR-DQN model under clean or attacked observations.

The evaluation mode is selected using `--mode`:

| Value      | Description                                      |
| ---------- | ------------------------------------------------ |
| `clean`    | Evaluates without attacks.                       |
| `attacked` | Evaluates with random observation perturbations. |

## Examples

```bash
# Clean evaluation
uv run python scripts/evaluate_attack.py --checkpoint wocar_dqn_test.pt --mode clean --episodes 100

# Attack Agent 0
uv run python scripts/evaluate_attack.py --checkpoint wocar_dqn_test.pt --mode attacked --target-mode fixed --target-agents 0 --epsilon 0.05 --episodes 100

# Attack Agents 0 and 2
uv run python scripts/evaluate_attack.py --checkpoint wocar_dqn_test.pt --mode attacked --target-mode fixed --target-agents 0 2 --epsilon 0.05 --episodes 100

# Randomly select agents to attack
uv run python scripts/evaluate_attack.py --checkpoint wocar_dqn_test.pt --mode attacked --target-mode random --epsilon 0.05 --episodes 100

# Random targeting with 50% attack probability
uv run python scripts/evaluate_attack.py --checkpoint wocar_dqn_test.pt --mode attacked --target-mode random --epsilon 0.05 --attack-probability 0.5 --episodes 100
```

## Arguments

| Argument               | Default   | Meaning                                                                             |
| ---------------------- | --------- | ----------------------------------------------------------------------------------- |
| `--checkpoint`         | Required  | Trained model checkpoint.                                                           |
| `--agents`             | `3`       | Number of agents. Must match the trained model.                                     |
| `--mode`               | `clean`   | Evaluation mode: `clean` or `attacked`.                                             |
| `--episodes`           | `10`      | Number of evaluation episodes.                                                      |
| `--seed`               | `42`      | Random seed.                                                                        |
| `--epsilon`            | `0.05`    | Maximum magnitude of observation noise (perturbation budget).                       |
| `--attack-probability` | `1.0`     | Probability of applying an attack at each timestep.                                 |
| `--target-mode`        | `fixed`   | Targeting strategy: `fixed` or `random`.                                            |
| `--target-agents`      | `None`    | Agent indices to attack in fixed mode. Required when `--target-mode fixed` is used. |
| `--logdir`             | `runs`    | TensorBoard root directory.                                                         |
| `--run-name`           | Timestamp | Evaluation run name.                                                                |

## Targeting

- `fixed`: Attacks the agents specified with `--target-agents`.
- `random`: Randomly selects a non-empty group of agents at the beginning of each attack cycle. Both group size and agent identities can change between cycles.

The selected agents remain the same throughout each attack cycle.

## Model Configuration

The evaluator also accepts the model configuration arguments used during training.
These should match the configuration of the trained checkpoint.

| Argument        | Default |
| --------------- | ------- |
| `--hidden`      | 64      |
| `--gamma`       | 0.99    |
| `--tau`         | 0.005   |
| `--lr`          | 0.0001  |
| `--warmup`      | 5000    |
| `--total-steps` | 200000  |
| `--eps-final`   | 0.075   |
| `--kappa-reg`   | 0.1     |

## Rendering

The evaluator supports rendering multiple agent perspectives simultaneously in a single window.

Each rendered agent gets its own camera view, following that agent's vehicle. The views are arranged automatically in a grid layout based on the number of rendered agents.

Rendering is enabled using `--render`.

| Argument | Default | Meaning |
|---|---|---|
| `--render` | Disabled | Enables visual rendering during evaluation. |
| `--render-agents` | Number of agents | Number of agent perspectives displayed in the rendering window. |

The total number of agents and the number of rendered agents are independent settings. For example, an environment can contain three agents while displaying only two camera views.

Example:

```bash
# Render two agent perspectives in an environment with three agents
uv run python scripts/evaluate_attack.py --checkpoint wocar_dqn_test.pt --mode attacked --episodes 5 --render --agents 3 --render-agents 2 --target-mode random --attack-probability 0.8
## Outputs

The evaluator prints the average reward and reward standard deviation.

Episode rewards are logged to TensorBoard under:

```text
runs/attack_evaluation/<run-name>
```

Start TensorBoard with:

```bash
uv run tensorboard --logdir=runs
```

Open http://localhost:6006 to compare evaluation runs.
