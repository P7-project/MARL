"""Trainer-independent policy rendering in a separate process."""

import math
import os
import time
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from queue import Empty, Full
from typing import Callable

import torch
import torch.multiprocessing as mp
from torch import Tensor
from typing import NamedTuple


Weights = dict[str, Tensor]


class StepResult(NamedTuple):
    finished: bool
    reward: float
    crashed: bool | None = None  # None when the env has no crash notion


class PolicyPlayer(ABC):
    """What a trainer implements. Built inside the render process, so everything it holds
    (env, networks) lives there. Keep its constructor arguments plain and picklable."""

    @abstractmethod
    def load_weights(self, weights: Weights) -> None:
        """Copy the newest weights into the player's policy."""

    @abstractmethod
    def start_episode(self) -> None:
        """Reset the env."""

    @abstractmethod
    def step(self) -> StepResult:
        """Act once, render one frame, and return the result."""

    @abstractmethod
    def close(self) -> None:
        """Release the env."""


PlayerFactory = Callable[[], PolicyPlayer]  # must be picklable: a top-level class or a functools.partial of one


class AsyncPolicyRenderer:
    """Runs a PolicyPlayer at a fixed frame rate in its own process."""

    def __init__(
        self,
        player_factory: PlayerFactory,
        fps: float,
        refresh_seconds: float,
        publish_every: int,
    ) -> None:
        self._publish_every: int = publish_every
        self._last_published_step: int | None = None
        context = mp.get_context("spawn")
        self._weights_queue = context.Queue(maxsize=1)
        self._stop_event = context.Event()
        self._process = context.Process(
            target=_render_loop,
            args=(player_factory, fps, refresh_seconds, self._weights_queue, self._stop_event),
            daemon=True,
        )
        self._process.start()

    def publish_if_due(self, step: int, get_weights: Callable[[], Weights]) -> None:
        """Call once per training iteration. get_weights is only called when a publish happens."""
        if self._last_published_step is not None:
            if step - self._last_published_step < self._publish_every:
                return
        self._last_published_step = step
        self._publish(get_weights())

    def close(self) -> None:
        self._stop_event.set()
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()

    def _publish(self, weights: Weights) -> None:
        # CPU copies, so the render process never touches GPU memory or live training tensors.
        snapshot: Weights = {name: tensor.detach().cpu().clone() for name, tensor in weights.items()}
        self._drop_stale_weights()
        try:
            self._weights_queue.put_nowait(snapshot)
        except Full:
            pass  # the next publish replaces it

    def _drop_stale_weights(self) -> None:
        try:
            self._weights_queue.get_nowait()
        except Empty:
            pass


def add_render_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--render", action="store_true")  # show the current policy in a separate window while training
    parser.add_argument("--render-fps", type=float, default=15.0)  # frames per second of the render process
    parser.add_argument("--render-publish-every", type=int, default=100)  # training steps between weight publishes
    parser.add_argument("--render-refresh-seconds", type=float, default=0)  # swap in the newest weights mid-episode every N seconds (0 = only at episode start)


def create_renderer(args: Namespace, player_factory: PlayerFactory) -> AsyncPolicyRenderer | None:
    if not args.render:
        return None
    return AsyncPolicyRenderer(
        player_factory, args.render_fps, args.render_refresh_seconds, args.render_publish_every
    )


def _render_loop(
    player_factory: PlayerFactory,
    fps: float,
    refresh_seconds: float,
    weights_queue,
    stop_event,
) -> None:
    torch.set_num_threads(1)  # do not compete with training for cores
    player: PolicyPlayer = player_factory()
    frame_seconds: float = 1.0 / fps
    latest: Weights | None = None

    while not stop_event.is_set():
        latest = _newest_weights(weights_queue, latest)
        if latest is None:
            time.sleep(0.1)  # nothing published yet
            continue
        player.load_weights(latest)
        last_refresh: float = time.monotonic()
        player.start_episode()

        finished: bool = False
        while not finished and not stop_event.is_set():
            frame_start: float = time.monotonic()
            if refresh_seconds > 0.0 and frame_start - last_refresh >= refresh_seconds:
                newest: Weights | None = _newest_weights(weights_queue, latest)
                if newest is not None and newest is not latest:
                    latest = newest
                    player.load_weights(latest)
                last_refresh = frame_start

            step_result = player.step()
            finished = step_result.finished

            remaining: float = frame_seconds - (time.monotonic() - frame_start)
            if remaining > 0.0:
                time.sleep(remaining)
    player.close()


def _newest_weights(weights_queue, current: Weights | None) -> Weights | None:
    try:
        return weights_queue.get_nowait()
    except Empty:
        return current


def evaluate_in_windows(player_factory: PlayerFactory, episodes: int, windows: int) -> None:
    """Runs `windows` independent copies of the player, each in its own process and window."""
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_evaluation_worker, args=(player_factory, index, episodes))
        for index in range(windows)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()


@torch.no_grad()
def _evaluation_worker(player_factory: PlayerFactory, index: int, episodes: int) -> None:
    # Must be set before the env creates its window.
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{25 + index % 3 * 625},{25 + 175 * math.floor(index / 3)}"
    player: PolicyPlayer = player_factory()

    for episode in range(episodes):
        player.start_episode()
        steps: int = 0
        total_return: float = 0.0
        result: StepResult = StepResult(finished=False, reward=0.0)
        while not result.finished:
            result = player.step()
            total_return += result.reward
            steps += 1
        print(
            f"(Window {index + 1}) episode {episode + 1}: "
            f"steps={steps}, return={total_return:.2f}, crashed={result.crashed}"
        )
    player.close()