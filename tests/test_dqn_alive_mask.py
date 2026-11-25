import numpy as np
import torch

from src.marl.woca_r_rl.dqn import ReplayBuffer, _add_alive_transitions

N_AGENTS: int = 3
INPUT_DIM: int = 2 + N_AGENTS


def _step(
    buffer: ReplayBuffer, alive: np.ndarray, terminated: list[bool]
) -> np.ndarray:
    """One env step of transitions; agent i's state is (step, i, one-hot id)."""
    step = len(buffer)
    states = torch.cat(
        [
            torch.tensor([[float(step), float(agent)] for agent in range(N_AGENTS)]),
            torch.eye(N_AGENTS),
        ],
        dim=1,
    )
    return _add_alive_transitions(
        buffer,
        alive,
        states,
        [0] * N_AGENTS,
        np.arange(N_AGENTS, dtype=np.float32),
        states,
        np.array(terminated, dtype=np.bool_),
    )


def _stored_agents(buffer: ReplayBuffer) -> list[int]:
    return buffer._states[: len(buffer), 2:].argmax(dim=1).tolist()


def test_alive_mask_stores_the_terminal_step_then_nothing_from_the_dead_car() -> None:
    buffer = ReplayBuffer(16, INPUT_DIM, torch.device("cpu"))
    alive = np.ones(N_AGENTS, dtype=np.bool_)

    alive = _step(buffer, alive, [True, False, False])
    assert alive.tolist() == [False, True, True]
    assert _stored_agents(buffer) == [0, 1, 2]
    assert buffer._terminated[:3].tolist() == [1.0, 0.0, 0.0]

    # The env keeps reporting the crashed car as terminated; it stores nothing more.
    alive = _step(buffer, alive, [True, False, False])
    alive = _step(buffer, alive, [True, False, True])
    assert alive.tolist() == [False, True, False]
    assert _stored_agents(buffer) == [0, 1, 2, 1, 2, 1, 2]
    assert buffer._terminated[:7].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    alive = _step(buffer, alive, [True, True, True])
    assert not alive.any()
    assert _stored_agents(buffer) == [0, 1, 2, 1, 2, 1, 2, 1]
    assert buffer._terminated[7].item() == 1.0


def test_alive_mask_with_shared_termination_stores_every_car() -> None:
    buffer = ReplayBuffer(16, INPUT_DIM, torch.device("cpu"))
    alive = np.ones(N_AGENTS, dtype=np.bool_)

    alive = _step(buffer, alive, [False, False, False])
    alive = _step(buffer, alive, [True, True, True])

    assert not alive.any()
    assert _stored_agents(buffer) == [0, 1, 2, 0, 1, 2]
    assert buffer._terminated[:6].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
