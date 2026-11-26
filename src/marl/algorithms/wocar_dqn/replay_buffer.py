"""Uniform replay buffer of per-car transitions."""

from __future__ import annotations

import torch
from torch import Tensor

from .trainer import Batch


class ReplayBuffer:
    def __init__(self, capacity: int, input_dim: int, device: torch.device) -> None:
        self._capacity: int = capacity
        self._device: torch.device = device
        self._size: int = 0
        self._cursor: int = 0
        self._states: Tensor = torch.zeros(capacity, input_dim, device=device)
        self._actions: Tensor = torch.zeros(capacity, dtype=torch.long, device=device)
        self._rewards: Tensor = torch.zeros(capacity, device=device)
        self._next_states: Tensor = torch.zeros(capacity, input_dim, device=device)
        self._terminated: Tensor = torch.zeros(capacity, device=device)

    def __len__(self) -> int:
        return self._size

    def add(self, state: Tensor, action: int, reward: float, next_state: Tensor, terminated: float) -> None:
        self._states[self._cursor] = state
        self._actions[self._cursor] = action
        self._rewards[self._cursor] = reward
        self._next_states[self._cursor] = next_state
        self._terminated[self._cursor] = terminated
        self._cursor = (self._cursor + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int) -> Batch:
        indices: Tensor = torch.randint(self._size, (batch_size,), device=self._device)
        return Batch(
            states=self._states[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_states=self._next_states[indices],
            terminated=self._terminated[indices],
        )
