"""Generalized advantage estimation (GAE) shared by the PPO-style trainers."""

import numpy as np


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Compute team-level generalized advantages and value targets."""
    horizon = len(rewards)
    advantages = np.zeros(horizon, dtype=np.float32)
    last_advantage = 0.0

    for t in reversed(range(horizon)):
        next_value = last_value if t == horizon - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_advantage = delta + gamma * lam * nonterminal * last_advantage
        advantages[t] = last_advantage

    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns
