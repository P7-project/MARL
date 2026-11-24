# Cooperative MARL with MAPPO and WocaR-RL

Three training modes share one entry point (`scripts/train.py`, selected with `--update-function`). All arguments are documented in [`scripts/train.md`](scripts/train.md).

| Mode | Algorithm | Environment |
|---|---|---|
| `mappo` (default) | MAPPO | MPE2 Simple Spread (default) or highway, discrete actions |
| `wocar_dqn` | WocaR-DQN | MPE2 Simple Spread (default), highway or intersection, discrete actions |
| `wocar_ppo` | WocaR-PPO | single-agent MuJoCo, continuous actions |

`--env` selects the team environment (`simple_spread`, `highway`, `intersection`). Simple Spread uses `local_ratio=0`, so all agents share the landmark coverage reward. See [`scripts/train.md`](scripts/train.md#environments---env) for the reward and episode settings.

## MAPPO design

- **Parameter-shared actor:** one policy network used by all agents. A one-hot agent ID is appended to each agent's observation so they can specialise.
- **Centralised critic:** V(s) takes the concatenation of all agents' observations.
- **Team reward:** GAE is computed from the shared environment reward, and the same advantage is used for every agent's policy loss. On highway, `TeamHighwayEnv` averages the per-vehicle rewards and ends the episode when a controlled vehicle crashes.

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
uv run scripts/train.py --update-function mappo --env highway --total-steps 200000 --save mappo_highway.pt
uv run scripts/train.py --update-function wocar_dqn --env intersection --agents 2 \
    --total-steps 200000 --save woca_dqn_intersection.pt
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
