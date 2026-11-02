"""GAE and PPO update for the shared-policy MAPPO baseline."""

import numpy as np
import torch
import torch.nn as nn


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


def ppo_update(actor, critic, opt_actor, opt_critic, buf, last_value, args, device):
    """Run PPO minibatch updates from one collected rollout."""
    advantages, returns = compute_gae(
        buf["rew"], buf["val"], buf["done"], last_value, args.gamma, args.lam
    )

    actor_obs = torch.as_tensor(np.stack(buf["aobs"]), device=device)
    states = torch.as_tensor(np.stack(buf["state"]), device=device)
    actions = torch.as_tensor(np.stack(buf["act"]), device=device)
    old_log_probs = torch.as_tensor(np.stack(buf["logp"]), device=device)
    advantages = torch.as_tensor(advantages, device=device)
    returns = torch.as_tensor(returns, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    horizon = actor_obs.shape[0]
    stats = {"actor": 0.0, "critic": 0.0, "entropy": 0.0}
    update_count = 0

    for _ in range(args.epochs):
        permutation = torch.randperm(horizon, device=device)
        for start in range(0, horizon, args.minibatch):
            indices = permutation[start : start + args.minibatch]

            dist = actor.dist(actor_obs[indices])
            new_log_probs = dist.log_prob(actions[indices])
            ratio = torch.exp(new_log_probs - old_log_probs[indices])
            advantage_mb = advantages[indices].unsqueeze(-1)
            surrogate_1 = ratio * advantage_mb
            surrogate_2 = torch.clamp(
                ratio, 1 - args.clip, 1 + args.clip
            ) * advantage_mb
            entropy = dist.entropy().mean()
            actor_loss = -torch.min(surrogate_1, surrogate_2).mean() - args.ent * entropy

            opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            opt_actor.step()

            value = critic(states[indices])
            critic_loss = 0.5 * (value - returns[indices]).pow(2).mean()

            opt_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            opt_critic.step()

            stats["actor"] += actor_loss.item()
            stats["critic"] += critic_loss.item()
            stats["entropy"] += entropy.item()
            update_count += 1

    return {key: value / max(update_count, 1) for key, value in stats.items()}
