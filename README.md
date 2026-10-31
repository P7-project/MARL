# MAPPO (Multi-Agent PPO) for multi-agent highway-env

## Design

- **Parameter-shared actor:** one policy network used by all agents. A one-hot agent ID is appended to each agent's observation so they can specialise.
- **Centralised critic:** V(s) takes the concatenation of all agents' observations.
- **Team reward:** the code assumes highway-env returns one shared scalar reward, so GAE is computed at team level and the same advantage is used for every agent's policy loss.

## Install

```bash
uv sync
```

## Train

```bash
uv run main.py --total-steps XXX --save ABC.pt
```

Each iteration collects `--rollout` steps (default 512) before updating, so `--total-steps` should be much larger than the rollout. For a quick smoke test:

```bash
uv run main.py --total-steps 2000 --rollout 256 --save test.pt
```

## Evaluate

```bash
uv run main.py --eval ABC.pt --render
```

## TensorBoard

To view training curves, run the following command in a separate terminal:

```bash
uv run tensorboard --logdir runs
```

TensorBoard is served at http://localhost:6006/.

Logged metrics: episode return, crash rate, actor/critic loss, and policy entropy. Each training run is saved in its own subfolder of `runs/` (named by timestamp, or by `--run-name`), so runs can be compared side by side.