from types import SimpleNamespace

import pytest

from marl import env as env_module
from marl.config import load_config


def test_load_config_resolves_repo_relative_paths():
    config = load_config("configs/intersection.yaml")

    assert config["gym_id"] == "intersection-multi-agent-v2"


def test_load_config_none_returns_empty_config():
    assert load_config(None) == {}


def test_load_config_rejects_non_mapping_yaml(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a YAML mapping"):
        load_config(config_path)


def test_make_intersection_env_wraps_single_agent_configs(monkeypatch):
    captured = {}

    def fake_make(gym_id, *, config, render_mode):
        captured.update(gym_id=gym_id, config=config, render_mode=render_mode)
        return SimpleNamespace()

    monkeypatch.setattr(env_module.gym, "make", fake_make)
    env_module.make_env(
        n_agents=4,
        render_mode="human",
        env_cfg={
            "gym_id": "intersection-multi-agent-v2",
            "observation": {"type": "Kinematics", "vehicles_count": 15},
            "action": {"type": "DiscreteMetaAction"},
            "overrides": {"duration": 13},
        },
    )

    assert captured["gym_id"] == "intersection-multi-agent-v2"
    assert captured["render_mode"] == "human"
    assert captured["config"]["controlled_vehicles"] == 4
    assert captured["config"]["duration"] == 13
    assert captured["config"]["observation"] == {
        "type": "MultiAgentObservation",
        "observation_config": {"type": "Kinematics", "vehicles_count": 15},
    }
    assert captured["config"]["action"] == {
        "type": "MultiAgentAction",
        "action_config": {"type": "DiscreteMetaAction"},
    }


def test_make_highway_env_preserves_multi_agent_wrappers(monkeypatch):
    captured = {}
    observation = {
        "type": "MultiAgentObservation",
        "observation_config": {"type": "Kinematics"},
    }
    action = {
        "type": "MultiAgentAction",
        "action_config": {"type": "DiscreteMetaAction"},
    }

    def fake_make(gym_id, *, config, render_mode):
        captured.update(gym_id=gym_id, config=config, render_mode=render_mode)
        return SimpleNamespace()

    monkeypatch.setattr(env_module.gym, "make", fake_make)
    env_module.make_env(
        n_agents=3,
        env_cfg={"gym_id": "highway-v0", "observation": observation, "action": action},
    )

    assert captured["config"]["observation"] is observation
    assert captured["config"]["action"] is action
    assert captured["config"]["controlled_vehicles"] == 3
