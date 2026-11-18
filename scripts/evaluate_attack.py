
import argparse
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from enum import Enum
from gymnasium.spaces import Tuple
from gymnasium.spaces import Discrete


from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marl.env import make_env, flatten_obs, build_actor_obs
from marl.woca_r_rl.dqn import (
    _build_trainer,
    _to_tensor,
    _select_actions,
)

from rich.progress import Progress

from marl.attacks.random_attack import RandomObservationAttacker

class EvaluationMode(Enum):
    clean = "clean"
    attacked = "attacked"

class TargetMode(Enum):
    fixed = "fixed"
    random = "random"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate WocaR-DQN with random observation attacks."
    )

    # Model configuration
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--eps-final", type=float, default=0.075)
    parser.add_argument("--kappa-reg", type=float, default=0.1)

    # Evaluation configuration
    parser.add_argument("--mode", choices=[mode.value for mode in EvaluationMode], default=EvaluationMode.clean.value)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Attack configuration
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--attack-probability", type=float, default=1.0)
    parser.add_argument("--target-mode", choices=[mode.value for mode in TargetMode], default=TargetMode.fixed.value)
    parser.add_argument("--target-agents",type=int,nargs="+",help="Agent indices to attack in fixed targeting mode.")

    # Logging configuration
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--run-name",type=str,default=datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Rendering
    parser.add_argument("--render", action="store_true", help="Display the environment during evaluation.")
    parser.add_argument("--render-agents",type=int,default=1,help="Number of agent camera views to display.")

    args = parser.parse_args()

    if args.epsilon < 0:
        parser.error("--epsilon cannot be negative.")

    if not 0.0 <= args.attack_probability <= 1.0:
        parser.error("--attack-probability must be between 0 and 1.")

    if args.agents < 1:
        parser.error("--agents must be at least 1.")

    if args.episodes < 1:
        parser.error("--episodes must be at least 1.")

    if args.target_mode == TargetMode.fixed.value and args.target_agents is not None:
        if any(agent < 0 or agent >= args.agents for agent in args.target_agents):
            parser.error("Target agent indices must be between 0 and agents - 1.")

    if args.mode == EvaluationMode.attacked.value and args.target_mode == TargetMode.fixed.value:
        if args.target_agents is None:
            parser.error("--target-agents is required when using fixed targeting.")


    if args.render_agents < 1:
        parser.error("--render-agents must be at least 1.")

    if args.render_agents > args.agents:
        parser.error("--render-agents cannot exceed --agents.")

    return args


def load_model(args, device):

    render_mode = "human" if args.render else None
    env = env = make_env(
        args.agents,
        render_mode=render_mode,
        render_agents=args.render_agents,
    )

    assert isinstance(env.observation_space, Tuple)
    assert isinstance(env.action_space, Tuple)

    observation_space = env.observation_space.spaces[0]
    action_space = env.action_space.spaces[0]

    assert observation_space.shape is not None
    assert isinstance(action_space, Discrete)

    
    obs_dim = int(np.prod(observation_space.shape))
    n_actions = int(action_space.n)

    trainer = _build_trainer(
        obs_dim,
        args.agents,
        n_actions,
        args,
        device
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    trainer.load_robust_q_state_dict(checkpoint["robust_q"])

    return trainer, env


def evaluate(args, trainer, env, device):
    attacker = None

    if args.mode == EvaluationMode.attacked.value:
        attacker = RandomObservationAttacker(
            epsilon=args.epsilon,
            attack_probability=args.attack_probability,
            target_agents=args.target_agents,
            target_mode=args.target_mode,
            seed=args.seed
        )

    episode_rewards = []

    with Progress() as progress:
        task = progress.add_task(
            "Evaluating episodes...",
            total=args.episodes
        )

        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            obs, info = env.reset(seed=episode_seed)
            obs = flatten_obs(obs)

            done = False
            total_reward = 0

            while not done:
                if attacker is not None:
                    obs = attacker.attack(obs)
                    # Tell the environment which vehicles should be highlighted during rendering.
                    env.unwrapped.set_attacked_agents(attacker.last_attacked_agents)

                actor_obs = build_actor_obs(obs, args.agents)
                actor_obs = _to_tensor(actor_obs, device)

                actions = _select_actions(
                    trainer,
                    actor_obs,
                    exploration_probability=0.0
                )

                next_obs, reward, terminated, truncated, info = env.step(
                    tuple(actions)
                )

                if args.render:
                    env.render()

                total_reward += float(np.mean(reward))
                obs = flatten_obs(next_obs)
                done = terminated or truncated

            episode_rewards.append(total_reward)
            progress.update(task, advance=1)

    return episode_rewards


def main():
    args = parse_args()

    log_dir = Path(args.logdir) / "attack_evaluation" / args.run_name
    writer = SummaryWriter(log_dir=str(log_dir))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    trainer, env = load_model(args, device)

    rewards = evaluate(args, trainer, env, device)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    if args.mode == EvaluationMode.clean.value:
        tag = EvaluationMode.clean.value
    else:
        tag = (
            f"attacked/{args.target_mode}/"
            f"epsilon_{args.epsilon}/"
            f"probability_{args.attack_probability}"
        )

    for episode, reward in enumerate(rewards):
        writer.add_scalar(
            f"{tag}/episode_reward",
            reward,
            global_step=episode
        )
    writer.close()

    print(f"Mode: {args.mode}")

    if args.mode == EvaluationMode.attacked.value:
        print(f"Target mode: {args.target_mode}")
        print(f"Epsilon: {args.epsilon}")
        print(f"Attack probability: {args.attack_probability}")

        if args.target_mode == TargetMode.fixed.value:
            print(f"Target agents: {args.target_agents}")

    print(f"Average reward: {mean_reward:.2f}")
    print(f"Reward standard deviation: {std_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
