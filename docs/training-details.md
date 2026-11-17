# Training details

This file records implementation choices behind the overview in [architecture.md](architecture.md). CLI commands and argument defaults are in [`scripts/train.md`](../scripts/train.md).

## Observations and actions

| Environment | Per-agent observation | DQN input | Discrete actions |
| --- | --- | --- | --- |
| Highway | 5 nearby vehicles × 5 kinematics features (`presence`, `x`, `y`, `vx`, `vy`); relative coordinates | `25 + N` values | 5 meta-actions: lane left, idle, lane right, faster, slower |
| Intersection | 15 vehicles × 7 features (`presence`, `x`, `y`, `vx`, `vy`, `cos_h`, `sin_h`); absolute coordinates | `105 + N` values | 3 longitudinal speed choices |

`N` is the number of controlled vehicles. `src/marl/env.py:build_actor_obs` appends an `N`-element one-hot agent ID to each flattened observation. The highway observation setup is in `make_highway_env`; `make_intersection_env` selects HighwayEnv's `intersection-multi-agent-v2` defaults. Both DQN observations are normalized and clipped to `[-1, 1]`, which the robustness bounds assume.

## Rewards and episode ends

| Environment | Reward design | Returned reward and termination |
| --- | --- | --- |
| Highway | Default weights: collision `-1`, right lane `0.1`, high speed `0.4`; forward speed maps from `[20, 30]` to `[0, 1]`. Normalization is enabled, then reward is multiplied by the on-road flag. | `TeamHighwayEnv` averages vehicle rewards into one scalar and ends the team episode on a crash or configured off-road event. DQN copies the scalar reward and terminal flag to each agent's replay transition. |
| Intersection | Default terms: collision `-5`, high speed `1`, arrived `1`; speed maps from `[7, 9]` to `[0, 1]`. Reward normalization is disabled. | The multi-agent environment returns rewards and terminal flags per agent. Arrival or crash terminates that agent; the default episode duration is 13 seconds. |

The highway settings come from HighwayEnv defaults plus `TeamHighwayEnv._agent_reward` in `src/marl/env.py`. Intersection settings come from the registered environment's defaults; see [HighwayEnv intersection docs](https://highway-env.farama.org/environments/intersection/).


## Design choices and current limits

- **Parameter sharing:** one policy/Q-network serves all controlled vehicles; one-hot IDs distinguish their inputs. This avoids maintaining a separate network per vehicle.
- **Reward handling:** Highway supplies a shared team reward, while Intersection supplies per-agent rewards. DQN preserves these environment semantics rather than forcing both tasks to use the same reward aggregation.
- **Finished agents:** Intersection agents can terminate at different times. DQN continues selecting actions and storing transitions for a finished agent while others remain active; it resets only when all agents terminate or the time limit is reached.
- **Observation model:** WocaR-DQN uses an MLP for normalized vector observations. The current environment selector supports Highway and Intersection, not image-based Atari observations.
