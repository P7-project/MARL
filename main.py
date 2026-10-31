import argparse

import gymnasium as gym
import highway_env  # noqa: F401  (registers the highway-* environments)

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rich.console import Console
from rich.live import Live

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


console = Console()

if hasattr(gym, "register_envs"):
    gym.register_envs(highway_env)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def make_env(n_agents: int = 3, render_mode=None) -> gym.Env:
    with console.status(f"Creating highway-v0 env with {n_agents} agents..."):
        config = {
            "controlled_vehicles": n_agents,
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "absolute": False,
                },
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {"type": "DiscreteMetaAction"},
            },
            "lanes_count": 4,
            "vehicles_count": 30,
            "duration": 40,
            "initial_lane_id": None,
            "policy_frequency": 1,
            "simulation_frequency": 15,
        }
        return gym.make("highway-v0", config=config, render_mode=render_mode)


def flatten_obs(obs) -> np.ndarray:
    """Tuple of N arrays -> (N, obs_dim) float32 array."""
    return np.stack([np.asarray(o, dtype=np.float32).reshape(-1) for o in obs])


def build_actor_obs(obs_flat: np.ndarray, n_agents: int) -> np.ndarray:
    """Append a one-hot agent id to each agent's observation -> (N, obs_dim + N)."""
    return np.concatenate([obs_flat, np.eye(n_agents, dtype=np.float32)], axis=-1)


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
def layer_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def mlp(in_dim: int, out_dim: int, hidden: int, out_gain: float) -> nn.Sequential:
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden)), nn.Tanh(),
        layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        layer_init(nn.Linear(hidden, out_dim), gain=out_gain),
    )


class Actor(nn.Module):
    """Shared policy: (…, obs_dim + n_agents) -> categorical over actions."""

    def __init__(self, in_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = mlp(in_dim, n_actions, hidden, out_gain=0.01)

    def dist(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(x))


class Critic(nn.Module):
    """Centralised value function: joint observation -> scalar V(s)."""

    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(state_dim, 1, hidden, out_gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# GAE
# --------------------------------------------------------------------------- #
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_adv = delta + gamma * lam * nonterminal * last_adv
        adv[t] = last_adv
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


# --------------------------------------------------------------------------- #
# PPO update
# --------------------------------------------------------------------------- #
def ppo_update(actor, critic, opt_actor, opt_critic, buf, last_value, args, device):
    adv, ret = compute_gae(buf["rew"], buf["val"], buf["done"],
                           last_value, args.gamma, args.lam)

    aobs = torch.as_tensor(np.stack(buf["aobs"]), device=device)    # (T, N, in)
    state = torch.as_tensor(np.stack(buf["state"]), device=device)  # (T, N*obs)
    act = torch.as_tensor(np.stack(buf["act"]), device=device)      # (T, N)
    old_logp = torch.as_tensor(np.stack(buf["logp"]), device=device)  # (T, N)
    adv = torch.as_tensor(adv, device=device)                       # (T,)
    ret = torch.as_tensor(ret, device=device)                       # (T,)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    T = aobs.shape[0]
    stats = {"actor": 0.0, "critic": 0.0, "entropy": 0.0}
    n_updates = 0

    for _ in range(args.epochs):
        perm = torch.randperm(T, device=device)
        for start in range(0, T, args.minibatch):
            idx = perm[start:start + args.minibatch]

            # ---- actor (shared across agents, team advantage broadcast) ----
            dist = actor.dist(aobs[idx])                    # (B, N, A)
            new_logp = dist.log_prob(act[idx])              # (B, N)
            ratio = torch.exp(new_logp - old_logp[idx])
            adv_mb = adv[idx].unsqueeze(-1)                 # (B, 1)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_mb
            entropy = dist.entropy().mean()
            actor_loss = -torch.min(surr1, surr2).mean() - args.ent * entropy

            opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            opt_actor.step()

            # ---- centralised critic ----
            value = critic(state[idx])                      # (B,)
            critic_loss = 0.5 * (value - ret[idx]).pow(2).mean()

            opt_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            opt_critic.step()

            stats["actor"] += actor_loss.item()
            stats["critic"] += critic_loss.item()
            stats["entropy"] += entropy.item()
            n_updates += 1

    return {k: v / max(n_updates, 1) for k, v in stats.items()}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args):
    with console.status(f"Setting up training with {args.agents} agents..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        n_agents = args.agents
        env = make_env(n_agents)
        obs_dim = int(np.prod(env.observation_space.spaces[0].shape))
        n_actions = int(env.action_space.spaces[0].n)

        actor = Actor(obs_dim + n_agents, n_actions).to(device)
        critic = Critic(obs_dim * n_agents).to(device)
        opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor, eps=1e-5)
        opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic, eps=1e-5)

        obs, _ = env.reset(seed=args.seed)
        obs = flatten_obs(obs)
        ep_ret, finished = 0.0, []
        steps_done, iteration = 0, 0

        # TensorBoard
        run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")  
        writer = SummaryWriter(f"{args.logdir}/{run_name}")  
        writer.add_text("hparams", "\n".join(f"{k}: {v}" for k, v in vars(args).items()))  

    
    while steps_done < args.total_steps:
        with console.status(f"Training ({steps_done}/{args.total_steps} steps, {len(finished)} episodes, {iteration} iterations)..."):
            buf = {k: [] for k in ("aobs", "state", "act", "logp", "rew", "done", "val")}

            rollout = 0
            #with console.status(f"Rollout ({rollout}/{args.rollout})..."):
            with Live(console=console, refresh_per_second=0.1) as rollout_live:
                for rollout in range(args.rollout):
                    rollout_live.update(f"Rollout ({rollout}/{args.rollout})...")
                    aobs = build_actor_obs(obs, n_agents)
                    state = obs.reshape(-1)

                    with torch.no_grad():
                        dist = actor.dist(torch.as_tensor(aobs, device=device))
                        a = dist.sample()
                        logp = dist.log_prob(a)
                        v = critic(torch.as_tensor(state, device=device)).item()

                    actions = tuple(int(x) for x in a.cpu().numpy())
                    next_obs, r, terminated, truncated, info = env.step(actions)  
                    next_obs = flatten_obs(next_obs)
                    r = float(r)
                    done = terminated or truncated

                    # Time-limit truncation is not a real terminal state: bootstrap from V(s').
                    r_train = r
                    if truncated and not terminated:
                        with torch.no_grad():
                            nv = critic(torch.as_tensor(next_obs.reshape(-1), device=device)).item()
                        r_train += args.gamma * nv

                    buf["aobs"].append(aobs)
                    buf["state"].append(state)
                    buf["act"].append(a.cpu().numpy())
                    buf["logp"].append(logp.cpu().numpy())
                    buf["rew"].append(r_train)
                    buf["done"].append(float(done))
                    buf["val"].append(v)

                    ep_ret += r
                    steps_done += 1

                    if done:
                        finished.append(ep_ret)
                        writer.add_scalar("episode/return", ep_ret, steps_done)  
                        writer.add_scalar("episode/crashed", float(info.get("crashed", False)), steps_done)  
                        ep_ret = 0.0
                        next_obs, _ = env.reset()
                        next_obs = flatten_obs(next_obs)
                    obs = next_obs

            with torch.no_grad():
                last_value = critic(torch.as_tensor(obs.reshape(-1), device=device)).item()

            stats = ppo_update(actor, critic, opt_actor, opt_critic,
                            buf, last_value, args, device)

            iteration += 1
            recent = finished[-10:]
            mean_ret = float(np.mean(recent)) if recent else float("nan")
            console.print(f"iter {iteration:4d} | steps {steps_done:7d} | "
                f"mean ep return (last {len(recent)}) {mean_ret:7.2f} | "
                f"actor {stats['actor']:.3f} critic {stats['critic']:.3f} "
                f"entropy {stats['entropy']:.3f}")

            # TensorBoard
            writer.add_scalar("loss/actor", stats["actor"], steps_done)  
            writer.add_scalar("loss/critic", stats["critic"], steps_done)  
            writer.add_scalar("policy/entropy", stats["entropy"], steps_done)  
            if recent:  
                writer.add_scalar("episode/mean_return_last10", mean_ret, steps_done)  

    if args.save:
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                    "n_agents": n_agents, "obs_dim": obs_dim,
                    "n_actions": n_actions}, args.save)
        console.print(f"Saved checkpoint to {args.save}")
    writer.close()  
    env.close()

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.eval, map_location=device)
    n_agents = ckpt["n_agents"]

    actor = Actor(ckpt["obs_dim"] + n_agents, ckpt["n_actions"]).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    env = make_env(n_agents, render_mode="human" if args.render else None)
    for ep in range(args.eval_episodes):
        obs, _ = env.reset()
        obs = flatten_obs(obs)
        done, total, steps, info = False, 0.0, 0, {}
        while not done:
            aobs = torch.as_tensor(build_actor_obs(obs, n_agents), device=device)
            actions = tuple(int(x) for x in actor.net(aobs).argmax(-1).cpu().numpy())  # greedy
            obs, r, terminated, truncated, info = env.step(actions)
            obs = flatten_obs(obs)
            total += float(r)
            steps += 1
            done = terminated or truncated
        console.print(f"Eval episode {ep + 1}: steps={steps}, return={total:.2f}, "
              f"crashed={info.get('crashed')}")
    env.close()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--rollout", type=int, default=512, help="joint env steps per update")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=128)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="mappo.pt")
    p.add_argument("--eval", type=str, default=None, help="path to checkpoint to evaluate")
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--render", action="store_true")

    # TensorBoard
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default=None, help="subfolder name in logdir (default: timestamp)")
    
    args = p.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)
