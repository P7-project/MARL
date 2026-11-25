import numpy as np
import pytest

from src.marl import reward_funcs
from src.marl.env import make_env
from src.marl.reward_funcs import (
    MeanReward,
    RewardFunction,
    TeamReward,
    get_reward_function,
)


@pytest.fixture
def team_env():
    env = make_env(3, reward_function="mean")
    env.reset(seed=0, options={"config": {"vehicles_count": 0, "initial_lane_id": 1}})
    yield env
    env.close()


def _spread_vehicles(team) -> None:
    """Put each controlled vehicle in its own lane at its own speed, so their rewards differ."""
    for index, vehicle in enumerate(team.controlled_vehicles):
        lane_index = ("0", "1", index)
        lane = team.road.network.get_lane(lane_index)
        vehicle.position = lane.position(100 * (index + 1), 0)
        vehicle.on_state_update()
        vehicle.target_lane_index = lane_index
        vehicle.speed = 20 + 5 * index


def test_mean_reward_gives_every_vehicle_the_mean(team_env):
    team = team_env.unwrapped
    _spread_vehicles(team)

    rewards = MeanReward()(team)
    info = team._info(team.observation_type.observe(), (1, 1, 1))

    assert len(set(info["agents_rewards"])) > 1
    assert len(rewards) == 3
    assert len(set(rewards)) == 1
    assert rewards[0] == pytest.approx(np.mean(info["agents_rewards"]))


def test_step_returns_mean_reward_per_vehicle(team_env):
    _, reward, _, _, info = team_env.step((1, 1, 1))

    assert isinstance(reward, tuple)
    assert len(reward) == 3
    assert reward == pytest.approx((np.mean(info["agents_rewards"]),) * 3)


def test_unknown_reward_function_raises():
    with pytest.raises(ValueError, match="mean"):
        get_reward_function("foo")


def test_reward_function_runs_once_per_step(monkeypatch):
    class CountingReward(RewardFunction):
        calls = 0

        def __call__(self, env):
            CountingReward.calls += 1
            return MeanReward()(env)

    monkeypatch.setitem(reward_funcs.REWARD_FUNCTIONS, "counting", CountingReward)
    env = make_env(3, reward_function="counting")
    env.reset(seed=0)
    for steps in range(1, 4):
        env.step((1, 1, 1))
        assert CountingReward.calls == steps
    env.close()


def test_unknown_reward_function_fails_at_env_creation():
    with pytest.raises(ValueError, match="foo"):
        make_env(3, reward_function="foo")


@pytest.mark.parametrize("team_weight", [0.0, 0.25, 1.0])
def test_team_reward_mixes_own_and_teammate_rewards(team_env, team_weight):
    team = team_env.unwrapped
    _spread_vehicles(team)
    team.configure({"team_reward_weight": team_weight})
    raw_rewards = team.raw_vehicle_rewards()
    assert len(set(raw_rewards)) > 1

    rewards = TeamReward()(team)

    assert rewards == pytest.approx(_mixed_rewards(raw_rewards, team_weight))
    assert np.mean(rewards) == pytest.approx(np.mean(raw_rewards))


def test_team_reward_with_zero_weight_returns_raw_rewards(team_env):
    team = team_env.unwrapped
    _spread_vehicles(team)
    team.configure({"team_reward_weight": 0.0})

    assert TeamReward()(team) == pytest.approx(team.raw_vehicle_rewards())


def test_team_reward_with_one_vehicle_returns_raw_reward():
    env = make_env(1, reward_function="team")
    try:
        env.reset(seed=0, options={"config": {"vehicles_count": 0}})
        team = env.unwrapped

        rewards = TeamReward()(team)

        assert rewards == pytest.approx(team.raw_vehicle_rewards())
        assert len(rewards) == 1
    finally:
        env.close()


def test_team_reward_reads_weight_from_env_config(team_env):
    team = team_env.unwrapped
    _spread_vehicles(team)

    team.configure({"team_reward_weight": 1.0})
    teammate_only_rewards = TeamReward()(team)
    team.configure({"team_reward_weight": 0.0})
    own_only_rewards = TeamReward()(team)

    raw_rewards = team.raw_vehicle_rewards()
    assert teammate_only_rewards != pytest.approx(own_only_rewards)
    for index, reward in enumerate(teammate_only_rewards):
        teammate_rewards = [
            value for other, value in enumerate(raw_rewards) if other != index
        ]
        assert reward == pytest.approx(np.mean(teammate_rewards))


def test_get_reward_function_returns_team_reward():
    assert isinstance(get_reward_function("team"), TeamReward)


def _mixed_rewards(raw_rewards: tuple[float, ...], team_weight: float) -> list[float]:
    """Reference implementation of the team mixing formula."""
    total = sum(raw_rewards)
    teammate_count = len(raw_rewards) - 1
    return [
        (1 - team_weight) * own_reward
        + team_weight * (total - own_reward) / teammate_count
        for own_reward in raw_rewards
    ]
