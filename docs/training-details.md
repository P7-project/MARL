# Training details

This file records implementation choices behind the overview in [architecture.md](architecture.md). CLI commands and argument defaults are in [`scripts/train.md`](../scripts/train.md).

## Observations and actions

| Environment | Per-agent observation | DQN input | Discrete actions |
| --- | --- | --- | --- |
| Highway | 5 nearby vehicles × 5 kinematics features (`presence`, `x`, `y`, `vx`, `vy`); relative coordinates | `25 + N` values | 5 meta-actions: lane left, idle, lane right, faster, slower |
| Intersection | 15 vehicles × 7 features (`presence`, `x`, `y`, `vx`, `vy`, `cos_h`, `sin_h`); absolute coordinates | `105 + N` values | 3 longitudinal speed choices |

`N` is the number of controlled vehicles. `src/marl/env.py:build_actor_obs` appends an `N`-element one-hot agent ID to each flattened observation. The highway observation setup is in `make_highway_env`; `make_intersection_env` selects `team-intersection-v0`, which keeps the `intersection-multi-agent-v2` observation and action defaults. Both DQN observations are normalized and clipped to `[-1, 1]`, which the robustness bounds assume.

## Rewards and episode ends

Both environments return one reward per controlled vehicle, computed by the reward function selected with `--reward` (`src/marl/reward_funcs.py`):

- `mean`: every vehicle gets the mean of the per-vehicle driving rewards. This is the stock cooperative behaviour.
- `team`: each vehicle's own reward is mixed with the mean reward of its teammates,

  `r_i = (1 - lambda_env) * r_i^env + lambda_env * (1 / (N - 1)) * sum_{j != i} r_j^env`

  where `lambda_env` is `team_reward_weight` in the environment config (`--team-weight` on the CLI), which must be in `[0, 1]`. Larger weights push the normalized Highway reward outside `[0, 1]`, which the hinge margin `c = 1` and the `w(s)` minibatch normalization assume, and a negative weight would penalize a teammate's success. The teammate mean runs over every other controlled vehicle, including crashed or arrived ones. With `N = 1` the reward is unchanged, `r_i = r_i^env`. This keeps the game general-sum: `R_i` still depends on the joint state and joint action, and is not a common reward. `lambda_env = 0` reproduces the stock environment rewards.

Default weights: `0.25` for Highway and `0.1` for Intersection. A Highway episode ends as soon as any controlled vehicle crashes, so a teammate's crash is already terminal for everyone and the mixed term mostly shapes the driving reward, which needs a larger weight to be visible. An Intersection episode ends per agent, so a crashed teammate's negative reward is mixed into the survivors again on every remaining step; those steps accumulate, so a smaller per-step weight already makes a crash costly.

| Environment | Reward design | Returned reward and termination |
| --- | --- | --- |
| Highway | Default weights: collision `-1`, right lane `0.1`, high speed `0.4`; forward speed maps from `[20, 30]` to `[0, 1]`. Normalization is enabled, then reward is multiplied by the on-road flag. | `TeamHighwayEnv` runs the reward function once per step, keeps the per-vehicle rewards and returns them from `step`; `_reward` returns their mean for highway-env's own bookkeeping. The team episode ends on a crash or configured off-road event, and DQN stores each vehicle's reward and the shared terminal flag in that agent's transition. |
| Intersection | Default terms: collision `-5`, high speed `1`, arrived `1`; speed maps from `[7, 9]` to `[0, 1]`. Reward normalization is disabled. | `TeamIntersectionEnv` overrides `step` to return the per-agent rewards and `info["agents_terminated"]`, replacing `MultiAgentWrapper` from `intersection-multi-agent-v2`. Arrival or crash terminates that agent; the default episode duration is 13 seconds. |

`info["agents_rewards"]` stays raw in both environments, so it is not the training signal when `--reward team` is selected. MAPPO trains on a team-level return, so `src/marl/training/trainer.py` averages the returned reward vector before building its advantage estimates. The mean of the mixed rewards equals the mean of the raw rewards, so `--reward team` does not change MAPPO's objective numerically.

The highway settings come from HighwayEnv defaults plus `TeamHighwayEnv._agent_reward` in `src/marl/env.py`. Intersection settings come from the registered environment's defaults; see [HighwayEnv intersection docs](https://highway-env.farama.org/environments/intersection/). The mixing formula is `TeamReward` in `src/marl/reward_funcs.py`.


## Design choices and current limits

- **Parameter sharing:** one policy/Q-network serves all controlled vehicles; one-hot IDs distinguish their inputs. This avoids maintaining a separate network per vehicle.
- **Reward handling:** both environments return one reward per controlled vehicle from a swappable reward function (`--reward mean` or `team`), and DQN stores each vehicle's own reward in its transition. The highway environment ends the episode for all vehicles when any of them crashes, while Intersection terminates agents individually.
- **Finished agents:** Intersection agents can terminate at different times. DQN continues selecting actions and storing transitions for a finished agent while others remain active; it resets only when all agents terminate or the time limit is reached.
- **Observation model:** WocaR-DQN uses an MLP for normalized vector observations. The current environment selector supports Highway and Intersection, not image-based Atari observations.
