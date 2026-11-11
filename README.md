# Multi-agent highway-env: MAPPO and WocaR-RL baselines

Three training modes share one entry point (`scripts/train.py`, selected with `--update-function`). All arguments are documented in [`scripts/train.md`](scripts/train.md).

| Mode | Algorithm | Environment |
|---|---|---|
| `mappo` (default) | MAPPO | multi-agent highway, discrete actions |
| `wocar_dqn` | WocaR-DQN | multi-agent highway, discrete actions |
| `wocar_ppo` | WocaR-PPO | single-agent MuJoCo, continuous actions |

## MAPPO design

- **Parameter-shared actor:** one policy network used by all agents. A one-hot agent ID is appended to each agent's observation so they can specialise.
- **Centralised critic:** V(s) takes the concatenation of all agents' observations.
- **Team reward:** `TeamHighwayEnv` (`src/marl/env.py`) averages the per-vehicle highway-env reward over all controlled vehicles and ends the episode when any vehicle crashes. GAE is computed at team level and the same advantage is used for every agent's policy loss.

## Install

```bash
uv sync
```

## Train

```bash
uv run scripts/train.py --total-steps XXX --save ABC.pt
```

Each iteration collects `--rollout` steps (default 512) before updating, so `--total-steps` should be much larger than the rollout. For a quick test:

```bash
uv run scripts/train.py --total-steps 2000 --rollout 256 --save test.pt
```

Other modes:

```bash
uv run scripts/train.py --update-function wocar_dqn --total-steps 200000 --save woca_dqn.pt
uv run scripts/train.py --update-function wocar_ppo --env-id Hopper-v5 --total-steps 2000000 --save woca_ppo.pt
```

## Evaluate

```bash
uv run scripts/evaluate.py --eval ABC.pt --render
```

## Tests

```bash
uv run pytest
```

## TensorBoard

To view training curves, run the following command in a separate terminal:

```bash
uv run tensorboard --logdir runs
```

TensorBoard is served at <http://localhost:6006/>.

Logged metrics: episode return, crash rate, actor/critic loss, and policy entropy. Each training run is saved in its own subfolder of `runs/` (named by timestamp, or by `--run-name`), so runs can be compared side by side.
