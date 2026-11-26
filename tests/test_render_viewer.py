import numpy as np
import pytest

from src.marl.envs import make_env

N_AGENTS: int = 5


@pytest.fixture(autouse=True)
def offscreen_driver(monkeypatch) -> None:
    """Draw like human mode without a window; highway-env draws nothing under the dummy driver."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "offscreen")


def test_rgb_array_render_on_highway_returns_a_real_frame() -> None:
    env = make_env(3, render_mode="rgb_array", name="highway")
    env.reset(seed=0)
    frame = env.render()
    env.close()
    assert frame is not None
    assert frame.ndim == 3
    assert frame.std() > 0


def test_every_view_is_centred_on_its_own_car() -> None:
    env = make_env(N_AGENTS, render_mode="human", name="highway", render_agents=N_AGENTS)
    env.reset(seed=0)
    env.render()
    unwrapped = env.unwrapped
    viewer = unwrapped.viewer
    vehicles = unwrapped.controlled_vehicles
    assert type(viewer).__name__ == "MultiAgentEnvViewer"
    assert len(viewer.render_surfaces) == N_AGENTS
    assert len({float(vehicle.position[1]) for vehicle in vehicles}) > 1  # cars in different lanes
    for vehicle, surface in zip(vehicles, viewer.render_surfaces):
        centre = np.array(surface.get_size()) / 2
        car = np.array(surface.pos2pix(*vehicle.position))
        assert np.abs(car - centre).max() <= 1
    env.close()
