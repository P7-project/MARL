import argparse
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from marl.env import make_env, flatten_obs, build_actor_obs
from marl.woca_r_rl.dqn import (
    _build_trainer,
    _to_tensor,
    _select_actions,
)

from rich.progress import Progress

from marl.experiments.random_attack import RandomObservationAttacker

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate WocaR-DQN with random observation attacks."
    )

    # Model configuration
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained WocaR-DQN checkpoint.")
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
    parser.add_argument("--mode", choices=["clean", "attacked"], default="clean")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Attack configuration
    parser.add_argument("--attack-strength", type=float, default=0.05)
    parser.add_argument("--attack-mode", choices=["single", "multiple", "random"], default="single")
    parser.add_argument("--target-agents", type=int, nargs="+", default=[0], help="Agent indices to attack in single or multiple mode.")
    parser.add_argument("--attack-interval", type=int, default=10)
    parser.add_argument("--attack-duration", type=int, default=3)

    # Logging configuration
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=datetime.now().strftime("%Y%m%d-%H%M%S"))

    return parser.parse_args()


def load_model(args, device):

    # Create the MARL environment
    env = make_env(args.agents)

    # Get the observation and action dimensions
    observation_space = env.observation_space.spaces[0]
    action_space = env.action_space.spaces[0]

    obs_dim = int(np.prod(observation_space.shape))
    n_actions = int(action_space.n)

    # Recreate the WocaR-DQN trainer
    trainer = _build_trainer(
        obs_dim,
        args.agents,
        n_actions,
        args,
        device
    )

    # Load the saved checkpoint
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    trainer.load_robust_q_state_dict(checkpoint)

    return trainer, env


def evaluate(args, trainer, env, device):

    # Create the random observation attacker
    attacker = RandomObservationAttacker(
        attack_strength=args.attack_strength,
        target_agents=args.target_agents,
        attack_mode=args.attack_mode,
        attack_interval=args.attack_interval,
        attack_duration=args.attack_duration,
        seed=args.seed
    )

    episode_rewards = []

    # Display progress while running evaluation episodes
    with Progress() as progress:

        task = progress.add_task(
            "Evaluating episodes...",
            total=args.episodes
        )

        # Run the requested number of episodes
        for episode in range(args.episodes):

            # Give each episode a predictable seed
            episode_seed = args.seed + episode

            # Reset the environment using that seed
            obs, info = env.reset(seed=episode_seed)

            obs = flatten_obs(obs)

            attacker.reset()

            done = False
            total_reward = 0

            while not done:

                # Apply attack only when requested
                if args.mode == "attacked":
                    obs = attacker.attack(obs)

                # Prepare observations for WocaR-DQN
                actor_obs = build_actor_obs(obs, args.agents)
                actor_obs = _to_tensor(actor_obs, device)

                # Select actions without exploration
                actions = _select_actions(
                    trainer,
                    actor_obs,
                    exploration_probability=0.0
                )

                # Execute actions in the environment
                next_obs, reward, terminated, truncated, info = env.step(
                    tuple(actions)
                )

                total_reward += float(np.mean(reward))

                obs = flatten_obs(next_obs)

                done = terminated or truncated

            episode_rewards.append(total_reward)

            # Mark one episode as completed in the progress bar
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

    if args.mode == "clean":
        tag = "clean"
    else:
        tag = f"attacked/{args.attack_mode}/strength_{args.attack_strength}"

    for episode, reward in enumerate(rewards):
        writer.add_scalar(
            f"{tag}/episode_reward",
            reward,
            global_step=episode
        )

    writer.close()

    print(f"Mode: {args.mode}")
    if args.mode == "attacked":
        print(f"Attack mode: {args.attack_mode}")
        if args.attack_mode in ["single", "multiple"]:
            print(f"Target agents: {args.target_agents}")


    print(f"Average reward: {mean_reward:.2f}")
    print(f"Reward standard deviation: {std_reward:.2f}")


    env.close()


if __name__ == "__main__":
    main()