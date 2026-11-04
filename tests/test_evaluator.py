from types import SimpleNamespace

import numpy as np
import torch

from marl.models import Actor
from marl.training import evaluator


class DummyEnv:
    def __init__(self):
        space = SimpleNamespace(shape=(2,))
        self.observation_space = SimpleNamespace(spaces=[space, space])
        self.action_space = SimpleNamespace(spaces=[SimpleNamespace(n=2)])
        self.unwrapped = self
        self.controlled_vehicles = [SimpleNamespace(crashed=False) for _ in range(2)]
        self.closed = False

    def reset(self):
        return (np.array([0.0, 0.0]), np.array([0.0, 1.0])), {}

    def step(self, actions):
        assert len(actions) == 2
        obs = (np.array([0.0, 0.0]), np.array([0.0, 1.0]))
        return obs, np.array([1.0, 3.0]), (True, False), False, {}

    def close(self):
        self.closed = True


def make_checkpoint():
    actor = Actor(in_dim=4, n_actions=2)
    return {
        "actor": actor.state_dict(),
        "n_agents": 2,
        "obs_dim": 2,
        "n_actions": 2,
        "env_cfg": {"gym_id": "checkpoint-env"},
    }


def test_evaluate_env_averages_agent_rewards_and_uses_joint_done(monkeypatch):
    env = DummyEnv()
    printed = []
    monkeypatch.setattr(evaluator, "make_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(
        evaluator.console, "print", lambda message: printed.append(str(message))
    )

    evaluator._evaluate_in_env(
        make_checkpoint(),
        {"gym_id": "test-env"},
        n_agents=2,
        eval_episodes=1,
        render=True,
    )

    assert env.closed
    assert any("return=2.00" in message for message in printed)
    assert any("steps=1" in message for message in printed)


def test_evaluate_uses_checkpoint_config_when_no_cli_config(monkeypatch):
    checkpoint = make_checkpoint()
    received = {}
    monkeypatch.setattr(evaluator.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(
        evaluator,
        "_evaluate_in_env",
        lambda *args, **kwargs: received.update(args=args, kwargs=kwargs),
    )

    evaluator.evaluate(
        SimpleNamespace(eval="saved.pt", env_cfg={}, render=False, eval_episodes=2)
    )

    assert received["args"][1] == checkpoint["env_cfg"]
    assert received["kwargs"]["render"] is False


def test_render_evaluation_spawns_one_view_per_agent(monkeypatch):
    checkpoint = make_checkpoint()
    workers = []

    class FakeWorker:
        exitcode = 0

        def __init__(self, *, target, args):
            self.target = target
            self.args = args
            self.started = False
            workers.append(self)

        def start(self):
            self.started = True

        def join(self):
            pass

    class FakeContext:
        @staticmethod
        def Process(*, target, args):
            return FakeWorker(target=target, args=args)

    monkeypatch.setattr(evaluator.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(torch.multiprocessing, "get_context", lambda method: FakeContext())

    evaluator.evaluate(
        SimpleNamespace(eval="saved.pt", env_cfg=None, render=True, eval_episodes=1)
    )

    assert len(workers) == checkpoint["n_agents"]
    assert all(worker.started for worker in workers)
    assert all(worker.args[1] == checkpoint["env_cfg"] for worker in workers)
