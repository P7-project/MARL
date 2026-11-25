import argparse
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path

from rich.console import Console
from config_argument_parser import ConfigArgumentParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marl.rendering import add_render_arguments
from marl.envs import REWARD_FUNCTIONS
from marl.training.trainer import train
from marl.woca_r_rl.ppo import train_ppo
from marl.woca_r_rl.dqn import train_dqn

console = Console()

class UpdateFunction(Enum):
    mappo = "mappo"
    wocar_ppo = "wocar_ppo"
    wocar_dqn = "wocar_dqn"


def main() -> None:
    parser: ConfigArgumentParser = build_parser()
    args: argparse.Namespace = parser.parse()
    args.run_name = args.run_name or build_default_run_name(args.update_function)
    args.run_name = reserve_run_folder(Path(args.logdir), args.run_name)
    console.print(f"Run name: {args.run_name}")
    resolve_save_path(args)

    if args.update_function == UpdateFunction.wocar_ppo.value:
        train_ppo(args)
    elif args.update_function == UpdateFunction.wocar_dqn.value:
        train_dqn(args, console)
    elif args.update_function == UpdateFunction.mappo.value:
        train(args)
    else:
        raise NotImplementedError(f"Update function {args.update_function} is not implemented!")


def build_parser() -> ConfigArgumentParser:
    parser = argparse.ArgumentParser()
    add_render_arguments(parser)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout", type=int, default=512, help="joint env steps per update")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch", type=int, default=128)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ent", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--update-function",
        choices=[f.value for f in UpdateFunction],
        default=UpdateFunction.mappo.value,
    )
    parser.add_argument(
        "--reward",
        choices=list(REWARD_FUNCTIONS),
        default="team",
        help="Reward function for the multi-agent environments (marl.envs.rewards)",
    )
    parser.add_argument(
        "--team-weight",
        type=float,
        default=None,
        help=(
            "Weight of the teammates' mean reward in each agent's reward, in [0, 1] "
            "(default: 0.25 for highway, 0.1 for intersection)"
        ),
    )
    parser.add_argument("--kappa-reg", type=float, default=0.1) # Weight of the regularizer term in the actor loss. The paper grid-searches {0.01, 0.03, 0.1, 0.3, 1.0}. "where κ_wst and κ_reg are hyperparameters balancing between natural performance and robustness"
    parser.add_argument("--eps-final", type=float, default=0.075) # Final attack radius ε, the value epsilon_schedule ramps to.    

    # WocaR-PPO only (read by train_ppo)
    parser.add_argument("--env-id", type=str, default="Hopper-v5")

    parser.add_argument("--kappa-wst-final", type=float, default=0.8) # Final weight of the Q_φ advantage shift. "where κ_wst and κ_reg are hyperparameters balancing between natural performance and robustness"
    parser.add_argument("--kappa-wst-shape", choices=["linear", "exponential"], default="linear") # linear or exponential, the ramp shape. Exponential only on Ant.

    # WocaR-DQN only (read by train_dqn)
    parser.add_argument("--regularizer-margin", type=float, default=1)
    parser.add_argument(
        "--dqn-env",
        choices=["highway", "intersection"],
        default="highway",
        help="Environment for WocaR-DQN",
    )
    parser.add_argument("--hidden", type=int, default=64)  # width of the three Q-networks
    parser.add_argument("--buffer-size", type=int, default=100_000)  # replay buffer capacity, in transitions (one per agent per env step)
    parser.add_argument("--warmup", type=int, default=5_000)  # transitions in the buffer before updates start; also when eps_t starts ramping
    parser.add_argument("--batch-size", type=int, default=64)  # minibatch size per update
    parser.add_argument("--explore-steps", type=int, default=50_000)  # env steps over which the exploration probability decays
    parser.add_argument("--explore-final", type=float, default=0.05)  # final epsilon-greedy exploration probability
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--tau", type=float, default=0.005)  # soft-update coefficient for Q_v'
    parser.add_argument("--dqn-kappa-wst-final", type=float, default=0.5)  # paper: kappa_wst ramps exponentially 0 -> 0.5
    parser.add_argument("--kappa-wst-start-fraction", type=float, default=1 / 3)  # paper: 0 for the first 2M of 6M steps
    parser.add_argument("--double-dqn", action="store_true")  # not in the algorithm; the original paper's D.1.2 says its baselines use it

    return ConfigArgumentParser(parser)


def reserve_run_folder(logdir: Path, run_name: str) -> str:
    """Create a new <logdir>/<run_name> folder and return the name that was used.

    If the folder exists, try <run_name>_2, <run_name>_3, ... The folder is created
    with exist_ok=False, so two runs started at the same time never share it.
    """
    logdir.mkdir(parents=True, exist_ok=True)
    number: int = 1
    while True:
        name: str = run_name if number == 1 else f"{run_name}_{number}"
        try:
            (logdir / name).mkdir(exist_ok=False)
        except FileExistsError:
            number += 1
            continue
        return name


def resolve_save_path(args: argparse.Namespace) -> None:
    """Set args.save to a reserved file in the run folder when --save was not given.

    For an explicit --save, create its parent folder so torch.save does not fail
    after training.
    """
    if args.save is None:
        args.save = reserve_default_save_path(Path(args.logdir) / args.run_name)
    else:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)


def reserve_default_save_path(save_directory: Path) -> str:
    """Create and return the first free model.pt, model.2.pt, ... in save_directory.

    The file is created with exclusive create, so two runs started at the same time
    never get the same path.
    """
    save_directory.mkdir(parents=True, exist_ok=True)
    number: int = 1
    while True:
        name: str = "model.pt" if number == 1 else f"model.{number}.pt"
        path: Path = save_directory / name
        try:
            with path.open("x"):
                pass
        except FileExistsError:
            number += 1
            continue
        return str(path)


def build_default_run_name(update_function: str) -> str:
    timestamp: str = datetime.now().strftime("%d_%H-%M-%S")
    return f"{update_function}_{timestamp}"


if __name__ == "__main__":
    main()
