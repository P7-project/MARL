import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from marl.training import trainer


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
SPEC = importlib.util.spec_from_file_location("marl_train_script", SCRIPT_PATH)
train_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_script)


def test_train_cli_uses_yaml_defaults_and_render_flag(monkeypatch, tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "gym_id: highway-v0\ntrain:\n  agents: 5\n  rollout: 32\n  total_steps: 96\n",
        encoding="utf-8",
    )
    received = {}
    monkeypatch.setattr(train_script, "train", lambda args: received.setdefault("args", args))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--agents",
            "2",
            "--total-steps",
            "10",
            "--render",
        ],
    )

    train_script.main()

    args = received["args"]
    assert args.agents == 2
    assert args.rollout == 32
    assert args.total_steps == 10
    assert args.render is True
    assert args.env_cfg["gym_id"] == "highway-v0"


def test_training_passes_human_render_mode_to_environment(monkeypatch, tmp_path):
    captured = {}

    class DummyEnv:
        observation_space = SimpleNamespace(
            spaces=[SimpleNamespace(shape=(2,)), SimpleNamespace(shape=(2,))]
        )
        action_space = SimpleNamespace(spaces=[SimpleNamespace(n=2)])

        def reset(self, seed=None):
            return [[0.0, 0.0], [0.0, 1.0]], {}

        def close(self):
            pass

    class DummyWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_text(self, *args, **kwargs):
            pass

        def close(self):
            pass

    def fake_make_env(n_agents, *, render_mode, env_cfg):
        captured.update(n_agents=n_agents, render_mode=render_mode, env_cfg=env_cfg)
        return DummyEnv()

    monkeypatch.setattr(trainer, "make_env", fake_make_env)
    monkeypatch.setattr(trainer, "SummaryWriter", DummyWriter)
    args = SimpleNamespace(
        agents=2,
        seed=0,
        lr_actor=3e-4,
        lr_critic=1e-3,
        render=True,
        env_cfg={"gym_id": "highway-v0"},
        run_name="unit-test",
        logdir=str(tmp_path),
        total_steps=0,
        save=None,
    )

    trainer.train(args)

    assert captured == {
        "n_agents": 2,
        "render_mode": "human",
        "env_cfg": {"gym_id": "highway-v0"},
    }
