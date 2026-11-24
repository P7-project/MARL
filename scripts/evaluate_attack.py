
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

from marl.attacks.observation_attacks import TargetMode
from marl.attacks.pgd_attack import   PGDObservationAttacker
from marl.attacks.random_attack import  RandomObservationAttacker


class EvaluationMode(Enum):
    """Available evaluation settings."""
    CLEAN = "clean"
    RANDOM = "random"
    PGD = "pgd"


def parse_args():
    """
    Parse command-line arguments and validate the evaluation configuration.

    Returns:
        Parsed arguments used to configure the model, environment,
        attack, and evaluation process.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate WocaR-DQN with random observation attacks."
    )

    # ---------------------------------------------------------
    # Model configuration
    # ---------------------------------------------------------
    # These arguments describe the model architecture and training
    # configuration needed to reconstruct the trainer before loading
    # the trained model weights from the checkpoint.
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=5_000)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--eps-final", type=float, default=0.075)
    parser.add_argument("--kappa-reg", type=float, default=0.1)

    # ---------------------------------------------------------
    # WocaR-DQN configuration
    # ---------------------------------------------------------
    # These parameters must match the configuration used during
    # training so that the trainer and its networks are reconstructed
    # correctly when loading the checkpoint.
    parser.add_argument("--adam-eps", type=float, default=1.5e-4)
    parser.add_argument("--regularizer-margin", type=float, default=1)
    parser.add_argument("--target-update-every", type=int, default=2000)
    parser.add_argument("--worst-ratio", type=float, default=0.1)
    parser.add_argument("--worst-rate", type=int, default=100)
    parser.add_argument("--worst-eps-final", type=float, default=0.005)
    parser.add_argument("--worst-schedule", type=int, default=0)
    parser.add_argument("--worst-kappa-final", type=float, default=0.1)
    parser.add_argument("--reachable-eps", type=float, default=1e-7)
    parser.add_argument("--reachable-eps-scheduled", action="store_true")
    parser.add_argument("--mse-loss", action="store_true")

    # DQN replay and training configuration
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--explore-steps", type=int, default=50_000)
    parser.add_argument("--explore-final", type=float, default=0.05)

    # Evaluation configuration
    parser.add_argument("--mode", choices=[mode.value for mode in EvaluationMode], default=EvaluationMode.CLEAN.value)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    # Observation attack configuration
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--attack-probability", type=float, default=1.0)
    parser.add_argument("--target-mode", choices=[mode.value for mode in TargetMode], default=TargetMode.FIXED.value)
    parser.add_argument("--target-agents",type=int,nargs="+",help="Agent indices to attack in fixed targeting mode.")

    # PGD-specific configuration
    parser.add_argument("--pgd-iters", type=int, default=10)
    parser.add_argument("--pgd-step-size", type=float, default=None)
    parser.add_argument("--pgd-random-start",action=argparse.BooleanOptionalAction,default=True,)

    # Logging configuration
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--run-name",type=str,default=datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Rendering configuration
    parser.add_argument("--render", action="store_true", help="Display the environment during evaluation.")
    parser.add_argument("--render-agents",type=int,default=1,help="Number of agent camera views to display.")

    args = parser.parse_args()

    # Validate the evaluation configuration

    if args.epsilon < 0:
        parser.error("--epsilon cannot be negative.")

    if not 0.0 <= args.attack_probability <= 1.0:
        parser.error("--attack-probability must be between 0 and 1.")

    if args.agents < 1:
        parser.error("--agents must be at least 1.")

    if args.episodes < 1:
        parser.error("--episodes must be at least 1.")

    if args.target_mode == TargetMode.FIXED.value and args.target_agents is not None:
        if any(agent < 0 or agent >= args.agents for agent in args.target_agents):
            parser.error("Target agent indices must be between 0 and agents - 1.")

    if (
        args.mode != EvaluationMode.CLEAN.value
        and args.target_mode == TargetMode.FIXED.value
        and args.target_agents is None
    ):
        parser.error("--target-agents is required when using fixed targeting.")

    if args.pgd_iters < 1:
        parser.error("--pgd-iters must be at least 1.")

    if args.pgd_step_size is not None and args.pgd_step_size <= 0:
        parser.error("--pgd-step-size must be positive.")

    if args.render_agents < 1:
        parser.error("--render-agents must be at least 1.")

    if args.render_agents > args.agents:
        parser.error("--render-agents cannot exceed --agents.")

    return args


def load_model(args, device):
    """
    Create the environment, reconstruct the WocaR-DQN trainer,
    and load the trained network weights.

    Args:
        args: Parsed command-line arguments.
        device: CPU or GPU device used for model evaluation.

    Returns:
        The loaded trainer and initialized environment.
    """

    # Enable human rendering only if requested by the user.
    render_mode = "human" if args.render else None

    # Create the multi-agent environment with the requested number of agents.
    env = make_env(
        args.agents,
        render_mode=render_mode,
        render_agents=args.render_agents,
    )

    # Verify that the environment uses tuple spaces for its agents.
    assert isinstance(env.observation_space, Tuple)
    assert isinstance(env.action_space, Tuple)

    # All agents share the same observation and action space.
    observation_space = env.observation_space.spaces[0]
    action_space = env.action_space.spaces[0]

    # Extract the observation dimension and number of available actions.
    assert observation_space.shape is not None
    assert isinstance(action_space, Discrete)

    
    obs_dim = int(np.prod(observation_space.shape))
    n_actions = int(action_space.n)

    # Reconstruct the trainer using the same configuration as training.
    trainer = _build_trainer(
        obs_dim,
        args.agents,
        n_actions,
        args,
        device
    )

    # Load the checkpoint and restore the trained robust Q-network.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    trainer._robust_q.load_state_dict(checkpoint["robust_q"])

    return trainer, env


def evaluate(args, trainer, env, device):
    """
    Evaluate the trained model over multiple episodes.

    Depending on the selected mode, observations are either left unchanged
    or perturbed by a random or PGD observation attacker.

    Args:
        args: Parsed evaluation and attack configuration.
        trainer: WocaR-DQN trainer containing the trained Q-network.
        env: Multi-agent environment used for evaluation.
        device: CPU or GPU device used for inference.

    Returns:
        List containing the cumulative reward for each episode.
    """

    # No attacker is used during clean evaluation.
    attacker = None

    # Shared configuration passed to either attack implementation.
    common_args = {
        "epsilon": args.epsilon,
        "attack_probability": args.attack_probability,
        "target_agents": args.target_agents,
        "target_mode": args.target_mode,
        "seed": args.seed,
    }

    # Create the random observation attacker if random mode is selected.
    if args.mode == EvaluationMode.RANDOM.value:
        attacker = RandomObservationAttacker(**common_args)

    # Create the PGD attacker using the trained robust Q-network.
    elif args.mode == EvaluationMode.PGD.value:
        attacker = PGDObservationAttacker(
            model=trainer._robust_q,
            n_agents=args.agents,
            niters=args.pgd_iters,
            step_size=args.pgd_step_size,
            random_start=args.pgd_random_start,
            **common_args,
        )

    # Store the cumulative reward obtained in each episode.
    episode_rewards = []

    # Display evaluation progress in the terminal.
    with Progress() as progress:
        task = progress.add_task(
            "Evaluating episodes...",
            total=args.episodes
        )

        # Evaluate the model over the requested number of episodes.
        for episode in range(args.episodes):

            # Give each episode a different but reproducible environment seed.
            episode_seed = args.seed + episode

            # Reset the environment and prepare the initial observations.
            obs, info = env.reset(seed=episode_seed)
            obs = flatten_obs(obs)

            done = False
            total_reward = 0

            # Continue until all agents have terminated or the episode is truncated.
            while not done:
                if attacker is not None:
                    obs = attacker.attack(obs)
                    # Tell the environment which vehicles should be highlighted during rendering.
                    env.unwrapped.set_attacked_agents(attacker.last_attacked_agents)

                # Add agent identity information to each observation.
                actor_obs = build_actor_obs(obs, args.agents)

                # Convert the observations into tensors on the selected device.
                actor_obs = _to_tensor(actor_obs, device)

                # Select actions using the trained Q-network.
                actions = _select_actions(
                    trainer,
                    actor_obs,
                    exploration_probability=0.0
                )

                # Execute the selected joint action in the environment.
                next_obs, reward, terminated, truncated, info = env.step(
                    tuple(actions)
                )

                if args.render:
                    env.render()

                # Accumulate the mean reward across agents for this timestep.
                total_reward += float(np.mean(reward))
                # Prepare the next timestep's observations.
                obs = flatten_obs(next_obs)
                done = terminated or truncated

            # Store the total reward for the completed episode.
            episode_rewards.append(total_reward)

             # Update the terminal progress bar.
            progress.update(task, advance=1)

    return episode_rewards


def main():
    """Run the complete model evaluation and save the results."""

    # Parse and validate the command-line arguments.
    args = parse_args()

    # Create a separate TensorBoard directory for this evaluation run.
    log_dir = Path(args.logdir) / "attack_evaluation" / args.run_name
    writer = SummaryWriter(log_dir=str(log_dir))

    # Use a GPU when available; otherwise fall back to the CPU.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Reconstruct the trainer and load the trained model.
    trainer, env = load_model(args, device)

    # Run the evaluation episodes and collect their rewards.
    rewards = evaluate(args, trainer, env, device)

    # Calculate the average episode reward and its standard deviation.
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # Construct a TensorBoard tag describing the evaluation configuration.
    if args.mode == EvaluationMode.CLEAN.value:
        tag = EvaluationMode.CLEAN.value
    else:
        tag = (
            f"{args.mode}/{args.target_mode}/"
            f"epsilon_{args.epsilon}/"
            f"probability_{args.attack_probability}"
        )

    # Log each episode's reward to TensorBoard.
    for episode, reward in enumerate(rewards):
        writer.add_scalar(
            f"{tag}/episode_reward",
            reward,
            global_step=episode
        )
    writer.close()

    # Print the evaluation configuration and summary statistics.
    print(f"Mode: {args.mode}")

    if args.mode in [EvaluationMode.random.value,EvaluationMode.pgd.value]:
        print(f"Target mode: {args.target_mode}")
        print(f"Epsilon: {args.epsilon}")
        print(f"Attack probability: {args.attack_probability}")

        if args.target_mode == TargetMode.FIXED.value:
            print(f"Target agents: {args.target_agents}")

    print(f"Average reward: {mean_reward:.2f}")
    print(f"Reward standard deviation: {std_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
