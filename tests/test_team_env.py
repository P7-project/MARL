from collections.abc import Iterator
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from highway_env.envs.highway_env import HighwayEnv
from highway_env.envs.intersection_env import ConnectedLaneMultiAgentIntersectionEnv

from src.marl.env import (
    DEFAULT_HIGHWAY_TEAM_WEIGHT,
    DEFAULT_INTERSECTION_TEAM_WEIGHT,
    make_env,
)

TEAM_WEIGHTS: list[float] = [0.0, 0.25, 1.0]
VEHICLE_STATES: list[str] = ["driving", "crashed", "offroad"]


@pytest.fixture
def team_env() -> Iterator[gym.Env]:
    env = make_env(3)
    env.reset(seed=0, options={"config": {"vehicles_count": 0, "initial_lane_id": 1}})
    yield env
    env.close()


@pytest.fixture
def intersection_env() -> Iterator[gym.Env]:
    env = make_env(2, name="intersection")
    env.reset(seed=0)
    yield env
    env.close()


@pytest.mark.parametrize("normalize_reward", [True, False])
@pytest.mark.parametrize("vehicle_state", VEHICLE_STATES)
def test_highway_reward_mean_matches_stock_vehicle_rewards(
    team_env: gym.Env, normalize_reward: bool, vehicle_state: str
) -> None:
    team = team_env.unwrapped
    team.configure({"normalize_reward": normalize_reward})
    _place_highway_vehicles(team)
    _apply_vehicle_state(team, vehicle_state)

    stock = HighwayEnv(config={**team.config, "controlled_vehicles": 1})
    try:
        stock.road = team.road
        expected_rewards = []
        for vehicle in team.controlled_vehicles:
            stock.controlled_vehicles = [vehicle]
            expected_rewards.append(stock._reward(1))

        assert len(set(expected_rewards)) > 1
        assert team._reward((1, 1, 1)) == pytest.approx(np.mean(expected_rewards))
        info = team._info(team.observation_type.observe(), (1, 1, 1))
        assert info["agents_rewards"] == pytest.approx(expected_rewards)
    finally:
        stock.close()


@pytest.mark.parametrize("team_weight", TEAM_WEIGHTS)
def test_highway_step_returns_mixed_rewards_per_vehicle(
    team_env: gym.Env, team_weight: float
) -> None:
    team = team_env.unwrapped
    team.configure({"team_reward_weight": team_weight})
    _place_highway_vehicles(team)
    raw_rewards = team.raw_vehicle_rewards()
    assert len(set(raw_rewards)) > 1
    expected_rewards = _mixed_rewards(raw_rewards, team_weight)

    mean_reward = team._reward((1, 1, 1))

    assert isinstance(mean_reward, float)
    assert mean_reward == pytest.approx(np.mean(expected_rewards))
    assert team._last_vehicle_rewards == pytest.approx(expected_rewards)

    _, rewards, _, _, info = team_env.step((1, 1, 1))

    assert isinstance(rewards, tuple)
    assert rewards == pytest.approx(
        _mixed_rewards(team.raw_vehicle_rewards(), team_weight)
    )
    assert info["agents_rewards"] == pytest.approx(team.raw_vehicle_rewards())


@pytest.mark.parametrize("normalize_reward", [True, False])
@pytest.mark.parametrize("vehicle_state", VEHICLE_STATES)
def test_highway_zero_team_weight_matches_stock(
    team_env: gym.Env, normalize_reward: bool, vehicle_state: str
) -> None:
    team = team_env.unwrapped
    team.configure({"normalize_reward": normalize_reward, "team_reward_weight": 0.0})
    _place_highway_vehicles(team)
    _apply_vehicle_state(team, vehicle_state)

    stock = HighwayEnv(config={**team.config, "controlled_vehicles": 1})
    try:
        stock.road = team.road
        expected_rewards = []
        for vehicle in team.controlled_vehicles:
            stock.controlled_vehicles = [vehicle]
            expected_rewards.append(stock._reward(1))

        assert len(set(expected_rewards)) > 1
        assert team.raw_vehicle_rewards() == pytest.approx(expected_rewards)
        assert team._reward((1, 1, 1)) == pytest.approx(np.mean(expected_rewards))
    finally:
        stock.close()


@pytest.mark.parametrize("team_weight", TEAM_WEIGHTS)
def test_intersection_reward_mixes_own_and_teammate_rewards(
    intersection_env: gym.Env, team_weight: float
) -> None:
    team = intersection_env.unwrapped
    team.configure({"team_reward_weight": team_weight})
    _set_intersection_speeds(team)
    mean_reward = team._reward((1, 1))
    raw_rewards = team.raw_vehicle_rewards()
    assert len(set(raw_rewards)) > 1
    expected_rewards = _mixed_rewards(raw_rewards, team_weight)

    assert mean_reward == pytest.approx(np.mean(expected_rewards))
    assert team._last_vehicle_rewards == pytest.approx(expected_rewards)
    assert np.mean(team._last_vehicle_rewards) == pytest.approx(np.mean(raw_rewards))


def test_intersection_step_returns_per_agent_rewards_and_termination(
    intersection_env: gym.Env,
) -> None:
    team = intersection_env.unwrapped
    team.controlled_vehicles[0].crashed = True
    team_weight = float(team.config["team_reward_weight"])

    _, rewards, terminated, truncated, info = intersection_env.step((1, 1))

    raw_rewards = team.raw_vehicle_rewards()
    assert info["agents_rewards"] == pytest.approx(raw_rewards)
    assert rewards == pytest.approx(_mixed_rewards(raw_rewards, team_weight))
    assert tuple(terminated) == info["agents_terminated"]
    assert terminated[0]
    assert isinstance(truncated, bool)


def test_intersection_zero_team_weight_matches_registered_v2() -> None:
    team = make_env(2, name="intersection", team_weight=0.0)
    stock = gym.make("intersection-multi-agent-v2", config={"controlled_vehicles": 2})
    try:
        team_obs, _ = team.reset(seed=3)
        stock_obs, _ = stock.reset(seed=3)
        np.testing.assert_array_equal(team_obs, stock_obs)

        for _ in range(5):
            action = (1, 1)
            team_obs, team_rewards, team_terminated, team_truncated, _ = team.step(
                action
            )
            (
                stock_obs,
                stock_rewards,
                stock_terminated,
                stock_truncated,
                _,
            ) = stock.step(action)

            np.testing.assert_array_equal(team_obs, stock_obs)
            assert team_rewards == pytest.approx(stock_rewards)
            assert tuple(team_terminated) == tuple(stock_terminated)
            assert team_truncated == stock_truncated
            if any(team_terminated) or team_truncated:
                break
    finally:
        team.close()
        stock.close()


def test_intersection_env_matches_registered_v2_defaults() -> None:
    env = make_env(2, name="intersection")
    try:
        expected = ConnectedLaneMultiAgentIntersectionEnv.default_config()
        expected["controlled_vehicles"] = 2
        expected["reward_function"] = "team"
        expected["team_reward_weight"] = DEFAULT_INTERSECTION_TEAM_WEIGHT
        # AbstractEnv.configure resolves offscreen_rendering from render_mode,
        # and make_env passes render_mode=None.
        expected["offscreen_rendering"] = True

        _assert_configs_match(env.unwrapped.config, expected)
    finally:
        env.close()


@pytest.mark.parametrize(
    "name, expected_weight",
    [
        ("highway", DEFAULT_HIGHWAY_TEAM_WEIGHT),
        ("intersection", DEFAULT_INTERSECTION_TEAM_WEIGHT),
    ],
)
def test_make_env_resolves_team_weight_defaults(
    name: str, expected_weight: float
) -> None:
    env = make_env(2, name=name)
    try:
        assert env.unwrapped.config["team_reward_weight"] == pytest.approx(
            expected_weight
        )
    finally:
        env.close()


@pytest.mark.parametrize("name", ["highway", "intersection"])
def test_make_env_keeps_explicit_team_weight(name: str) -> None:
    env = make_env(2, name=name, team_weight=0.6)
    try:
        assert env.unwrapped.config["team_reward_weight"] == pytest.approx(0.6)
    finally:
        env.close()


def test_registered_highway_id_resets_with_default_team_weight() -> None:
    env = gym.make("team-highway-v0", config={"controlled_vehicles": 3})
    try:
        env.reset(seed=0)
        assert env.unwrapped.config["team_reward_weight"] == pytest.approx(
            DEFAULT_HIGHWAY_TEAM_WEIGHT
        )
    finally:
        env.close()


def test_registered_intersection_id_resets_with_default_team_weight() -> None:
    env = gym.make("team-intersection-v0")
    try:
        env.reset(seed=0)
        assert env.unwrapped.config["team_reward_weight"] == pytest.approx(
            DEFAULT_INTERSECTION_TEAM_WEIGHT
        )
    finally:
        env.close()


@pytest.mark.parametrize("name", ["highway", "intersection"])
@pytest.mark.parametrize("team_weight", [-0.1, 1.5])
def test_make_env_rejects_team_weight_outside_unit_interval(
    monkeypatch: pytest.MonkeyPatch, name: str, team_weight: float
) -> None:
    def fail(*args: Any, **kwargs: Any) -> gym.Env:
        raise AssertionError("make_env built an env before validating team_weight")

    monkeypatch.setattr("src.marl.env.make_highway_env", fail)
    monkeypatch.setattr("src.marl.env.make_intersection_env", fail)

    with pytest.raises(ValueError, match=r"team_weight must be in \[0, 1\]"):
        make_env(2, name=name, team_weight=team_weight)


def test_crashed_teammate_keeps_reducing_surviving_reward(
    intersection_env: gym.Env,
) -> None:
    team = intersection_env.unwrapped
    team.controlled_vehicles[0].crashed = True
    team_weight = float(team.config["team_reward_weight"])

    for _ in range(2):
        _, rewards, _, _, _ = intersection_env.step((1, 1))
        raw_rewards = team.raw_vehicle_rewards()
        assert team.controlled_vehicles[0].crashed
        assert raw_rewards[0] < 0.0
        assert rewards == pytest.approx(_mixed_rewards(raw_rewards, team_weight))
        # The survivor's own reward is reduced by the crashed teammate's reward,
        # on the crash step and on the step after it.
        assert rewards[1] < raw_rewards[1]


@pytest.mark.parametrize("agent_index", [0, 1, 2])
def test_any_controlled_car_crash_ends_episode(team_env, agent_index):
    team_env.unwrapped.controlled_vehicles[agent_index].crashed = True

    _, _, terminated, truncated, info = team_env.step((1, 1, 1))

    assert terminated
    assert not truncated
    assert info["crashed"]
    expected_crashes = tuple(index == agent_index for index in range(3))
    assert info["agents_crashed"] == expected_crashes


@pytest.mark.parametrize("normalize_reward", [True, False])
def test_single_agent_matches_stock_highway(normalize_reward):
    team = make_env(1)
    stock = HighwayEnv(config=team.unwrapped.config)
    config = {
        "vehicles_count": 0,
        "initial_lane_id": 1,
        "duration": 3,
        "normalize_reward": normalize_reward,
    }
    try:
        team_obs, _ = team.reset(seed=7, options={"config": config})
        stock_obs, _ = stock.reset(seed=7, options={"config": config})
        np.testing.assert_array_equal(team_obs, stock_obs)

        for action in [(3,), (0,), (1,)]:
            team_obs, reward, terminated, truncated, info = team.step(action)
            (
                stock_obs,
                stock_reward,
                stock_terminated,
                stock_truncated,
                stock_info,
            ) = stock.step(action)

            np.testing.assert_array_equal(team_obs, stock_obs)
            assert reward == pytest.approx((stock_reward,))
            assert terminated == stock_terminated
            assert truncated == stock_truncated
            assert info["crashed"] == stock_info["crashed"]
            if terminated or truncated:
                break
    finally:
        team.close()
        stock.close()


def test_duration_truncates_without_crash(team_env):
    team_env.unwrapped.configure({"duration": 2})

    for step in range(2):
        _, _, terminated, truncated, info = team_env.step((1, 1, 1))
        assert not terminated
        assert truncated == (step == 1)
        assert not info["crashed"]
        assert not any(info["agents_crashed"])


def _mixed_rewards(raw_rewards: tuple[float, ...], team_weight: float) -> list[float]:
    """Reference implementation of the team mixing formula."""
    total = sum(raw_rewards)
    teammate_count = len(raw_rewards) - 1
    return [
        (1 - team_weight) * own_reward
        + team_weight * (total - own_reward) / teammate_count
        for own_reward in raw_rewards
    ]


def _place_highway_vehicles(team: HighwayEnv) -> None:
    for index, vehicle in enumerate(team.controlled_vehicles):
        lane_index = ("0", "1", index)
        lane = team.road.network.get_lane(lane_index)
        vehicle.position = lane.position(100 * (index + 1), 0)
        vehicle.on_state_update()
        vehicle.target_lane_index = lane_index
        vehicle.speed = 20 + 5 * index


def _apply_vehicle_state(team: HighwayEnv, vehicle_state: str) -> None:
    last_vehicle = team.controlled_vehicles[-1]
    if vehicle_state == "crashed":
        last_vehicle.crashed = True
    elif vehicle_state == "offroad":
        last_vehicle.position[1] = 100
        last_vehicle.on_state_update()
        assert not last_vehicle.on_road


def _set_intersection_speeds(team: ConnectedLaneMultiAgentIntersectionEnv) -> None:
    for index, vehicle in enumerate(team.controlled_vehicles):
        vehicle.speed = 7.5 + 1.5 * index


def _assert_configs_match(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Compare two env configs key by key, naming the first key that differs."""
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    assert missing == [], f"config keys missing from the environment: {missing}"
    assert unexpected == [], f"unexpected config keys: {unexpected}"
    for key in expected:
        assert actual[key] == expected[key], (
            f"config key {key!r} differs: environment has {actual[key]!r}, "
            f"v2 defaults give {expected[key]!r}"
        )
