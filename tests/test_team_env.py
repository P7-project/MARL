import numpy as np
import pytest
from highway_env.envs.highway_env import HighwayEnv

from src.marl.env import make_env


@pytest.fixture
def team_env():
    env = make_env(3)
    env.reset(seed=0, options={"config": {"vehicles_count": 0, "initial_lane_id": 1}})
    yield env
    env.close()


@pytest.mark.parametrize("normalize_reward", [True, False])
@pytest.mark.parametrize("vehicle_state", ["driving", "crashed", "offroad"])
def test_team_reward_averages_stock_vehicle_rewards(
    team_env, normalize_reward, vehicle_state
):
    team = team_env.unwrapped
    team.configure({"normalize_reward": normalize_reward})
    for index, vehicle in enumerate(team.controlled_vehicles):
        lane_index = ("0", "1", index)
        lane = team.road.network.get_lane(lane_index)
        vehicle.position = lane.position(100 * (index + 1), 0)
        vehicle.on_state_update()
        vehicle.target_lane_index = lane_index
        vehicle.speed = 20 + 5 * index

    last_vehicle = team.controlled_vehicles[-1]
    if vehicle_state == "crashed":
        last_vehicle.crashed = True
    elif vehicle_state == "offroad":
        last_vehicle.position[1] = 100
        last_vehicle.on_state_update()
        assert not last_vehicle.on_road

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
            assert reward == pytest.approx(stock_reward)
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
