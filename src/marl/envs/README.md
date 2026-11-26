# Environments (`marl.envs`)

The multi-agent driving environments used by MAPPO and WocaR-DQN, the reward
functions that turn the driving into one reward per car, and small helpers the
trainers use on the observations. Both environments are built on
[highway-env](https://highway-env.farama.org/).

## Files

| File | What it holds |
|---|---|
| `__init__.py` | The public interface (see below). Importing the package registers both gym ids. |
| `highway.py` | `TeamHighwayEnv`, its default team weight, the `team-highway-v0` gym id and `make_highway_env`. |
| `intersection.py` | `TeamIntersectionEnv`, its default team weight, the `team-intersection-v0` gym id and `make_intersection_env`. |
| `factory.py` | `make_env`: checks `team_weight` and builds an environment by name. |
| `rewards.py` | The reward functions (`MeanReward`, `TeamReward`) and the `REWARD_FUNCTIONS` registry. |
| `helpers.py` | `flatten_obs`, `build_actor_obs`, `any_vehicle_crashed`. |
| `highway_viewer.py` | The pygame window for highway in human render mode: `render_agents` camera views, each centred on one controlled car, attacked cars outlined in red. Loaded only when a highway env renders in human mode, so the package itself does not need pygame. |

## Public interface

```python
from marl.envs import make_env, flatten_obs, build_actor_obs
```

| Name | What it does |
|---|---|
| `make_env(n_agents, render_mode, name, reward_function, team_weight, render_agents)` | Builds `"highway"` or `"intersection"` with `n_agents` controlled cars. Raises `ValueError` if `team_weight` is outside `[0, 1]` or the name is unknown. `render_agents` (number of camera views) only applies to highway. |
| `make_highway_env`, `make_intersection_env` | The per-environment builders `make_env` calls. |
| `TeamHighwayEnv`, `TeamIntersectionEnv` | The environment classes. |
| `flatten_obs(obs)` | Turns the per-car observations into a float32 array of shape `(agents, features)`. |
| `build_actor_obs(obs_flat, n_agents)` | Appends a one-hot agent ID to each car's flattened observation, so one shared network can tell the cars apart. |
| `any_vehicle_crashed(env)` | True if any controlled car has crashed. |
| `REWARD_FUNCTIONS` | Reward function name to class (`"mean"`, `"team"`). The `--reward` choices come from here. |
| `get_reward_function(name)` | Builds the reward function with that name; raises `ValueError` on an unknown name. |

Both environments return from `step` a tuple with one reward per controlled car,
in controlled-car order. `info["agents_rewards"]` always holds the stock
per-car highway-env rewards, before any mixing, so it is not the training
signal when `--reward team` is used.

## Highway (`team-highway-v0`)

**Observations.** One per car: 5 rows (the car itself, then up to 4 nearby
vehicles; missing rows are zero), each with 5 features `presence`, `x`, `y`,
`vx`, `vy`. The other vehicles are given in coordinates relative to the car.
highway-env normalizes and clips the values to `[-1, 1]` (presence is 0 or 1). With the one-hot ID the network input is `25 + N` values, `N` being the
number of controlled cars. These settings are in `make_highway_env`.

**Actions.** One per car, 5 discrete meta-actions: lane left, idle, lane
right, faster, slower.

**Road.** 4 lanes, 30 other vehicles, 1 decision per second (15 simulation
steps per decision).

**Raw reward per car.** highway-env's highway reward applied to each car
(`TeamHighwayEnv._agent_reward`): collision `-1`, right lane `0.1`, high speed
`0.4`, where the forward speed is mapped from `[20, 30]` m/s to `[0, 1]`. The
sum is normalized to `[0, 1]` and multiplied by the on-road flag.

**Episode end.** The episode ends for everyone as soon as any controlled car
crashes (or leaves the road, if `offroad_terminal` is set; it is off by
default). Otherwise it is truncated after 40 seconds, which is 40 steps.
`terminated` is one shared flag.

## Intersection (`team-intersection-v0`)

`TeamIntersectionEnv` is highway-env's `ConnectedLaneMultiAgentIntersectionEnv`
(the class behind `intersection-multi-agent-v2`) and keeps its observation and
action defaults. It does not use the `MultiAgentWrapper` of that gym id: its own
`step` returns the per-car rewards of the chosen reward function and the
per-car termination flags.

**Observations.** One per car: 15 rows (the car itself, then up to 14 nearby
vehicles), each with 7 features `presence`,
`x`, `y`, `vx`, `vy`, `cos_h`, `sin_h`, in absolute coordinates, normalized and
clipped to `[-1, 1]`. Network input: `105 + N` values.

**Actions.** One per car, 3 discrete target speeds (0, 4.5 and 9 m/s); no lane
changes.

**Raw reward per car.** highway-env's intersection reward: collision `-5`,
high speed `1` (speed mapped from `[7, 9]` m/s to `[0, 1]`), and a car that has
arrived gets `1` instead. The result is multiplied by the on-road flag and not
normalized.

**Episode end.** Per car: `terminated` is a tuple with one flag per car, and a
car is terminated once it has crashed or arrived. A terminated car stays in the
scene and keeps getting a reward every step. The trainers reset when all cars
are terminated or the episode is truncated after 13 seconds (13 steps).

## How the trainers use the rewards

WocaR-DQN stores one transition per car, with that car's own reward and that
car's own terminal flag (on highway the shared flag). A car that has finished
on the intersection still selects actions and stores transitions until the
whole episode resets.

MAPPO (highway only) trains on one team reward: `src/marl/training/trainer.py`
averages the returned per-car rewards. The `team` mixing does not change that
mean, so `--reward team` does not change MAPPO's objective.

## Reward functions (`--reward`)

Both environments run the reward function once per step on the stock per-car
rewards `r_i^env` (`raw_vehicle_rewards`).

- `mean`: every car gets the mean of all cars' raw rewards. This is
  highway-env's stock cooperative reward.
- `team` (the default): each car's own reward is mixed with the mean of its
  teammates' rewards,

  `r_i = (1 - lambda_env) * r_i^env + lambda_env * (1 / (N - 1)) * sum_{j != i} r_j^env`

  The teammate mean runs over every other controlled car, including crashed or
  arrived ones. With `N = 1` the raw reward is returned unchanged, and
  `lambda_env = 0` gives back the stock per-car rewards. Each car still gets its
  own reward, so the game stays general-sum rather than a common reward.

`lambda_env` is `team_reward_weight` in the environment config, set with
`--team-weight` on the CLI. `make_env` rejects values outside `[0, 1]`: a
weight above 1 can push the normalized highway reward outside `[0, 1]`, which
the WocaR-DQN hinge margin `c = 1` and the `w(s)` minibatch normalization
assume, and a negative weight would punish a teammate's success.

Without `--team-weight` each environment uses its own default:

| Environment | Default `lambda_env` | Why |
|---|---|---|
| Highway | `0.25` (`DEFAULT_HIGHWAY_TEAM_WEIGHT`) | A crash ends the episode for everyone, so a teammate's crash is already costly; the mixed term mostly shapes the driving reward and needs a larger weight to be visible. |
| Intersection | `0.1` (`DEFAULT_INTERSECTION_TEAM_WEIGHT`) | Episodes end per car, so a crashed teammate's negative reward is mixed into the survivors again on every remaining step. Those steps add up, so a smaller weight already makes a crash costly. |

## Adding a new environment

1. Create `src/marl/envs/<name>.py` with an env class that returns one reward
   per controlled car from `step` and has a `raw_vehicle_rewards()` method (the
   `TeamEnv` protocol in `rewards.py`), so every reward function works on it.
   Give it `reward_function` and `team_reward_weight` in `default_config`, and
   build the reward function in `configure` with `get_reward_function`.
2. In the same file, register a gym id with
   `gym.register(id="team-<name>-v0", entry_point=...)` and add a
   `make_<name>_env` builder that calls `gym.make` with that id.
3. In `factory.py`, add a branch for the new name to `make_env`.
4. Export the class and builder in `__init__.py`.
5. To make it selectable for WocaR-DQN, add the name to the `--dqn-env` choices
   in `scripts/train.py`.

## Adding a new reward function

1. In `rewards.py`, subclass `RewardFunction` and implement `__call__(env)`,
   returning a tuple with one reward per controlled car. Read only what the
   `TeamEnv` protocol offers (`config`, `controlled_vehicles`,
   `raw_vehicle_rewards()`).
2. Add it to `REWARD_FUNCTIONS` under a new name. It then becomes a choice of
   `--reward` automatically.
