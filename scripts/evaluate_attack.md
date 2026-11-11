# Attack Evaluation CLI

One entry point for evaluating a trained WocaR-DQN model under clean or attacked observations.

The evaluation mode is selected using `--mode`:

| Value | Description |
|---|---|
| `clean` | Evaluates without attacks. |
| `attacked` | Evaluates with random observation perturbations. |

## Examples

```bash
# Clean evaluation
uv run python scripts/evaluate_attack.py \
    --checkpoint wocar_dqn_test.pt \
    --mode clean --episodes 100

# Attack Agent 0
uv run python scripts/evaluate_attack.py \
    --checkpoint wocar_dqn_test.pt \
    --mode attacked \
    --target-mode fixed --target-agents 0 \
    --episodes 100

# Attack Agents 0 and 2
uv run python scripts/evaluate_attack.py \
    --checkpoint wocar_dqn_test.pt \
    --mode attacked \
    --target-mode fixed --target-agents 0 2 \
    --episodes 100

# Randomly select agents to attack
uv run python scripts/evaluate_attack.py \
    --checkpoint wocar_dqn_test.pt \
    --mode attacked \
    --target-mode random --episodes 100
```

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--checkpoint` | Required | Trained model checkpoint. |
| `--agents` | 3 | Number of agents. Must match the trained model. |
| `--mode` | `clean` | Evaluation mode: `clean` or `attacked`. |
| `--episodes` | 10 | Number of evaluation episodes. |
| `--seed` | 42 | Random seed. |
| `--attack-strength` | 0.05 | Maximum magnitude of observation noise. |
| `--target-mode` | `fixed` | Targeting strategy: `fixed` or `random`. |
| `--target-agents` | None | Agent indices to attack in fixed mode. |
| `--attack-interval` | 10 | Steps between attack cycles. |
| `--attack-duration` | 3 | Number of active attack steps per cycle. |
| `--logdir` | `runs` | TensorBoard root directory. |
| `--run-name` | Timestamp | Evaluation run name. |

## Targeting

- `fixed`: Attacks the agents specified with `--target-agents`.
- `random`: Randomly selects a non-empty group of agents at the beginning of each attack cycle. Both group size and agent identities can change between cycles.

The selected agents remain the same throughout each attack cycle.

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