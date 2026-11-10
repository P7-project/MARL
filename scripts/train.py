import argparse
import re
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marl.training.trainer import train
from marl.woca_r_rl.ppo import train_ppo
from marl.woca_r_rl.dqn import train_dqn

DEFAULT_SAVE_STEM: str = "default_save_name"

console = Console()

class UpdateFunction(Enum):
    mappo = "mappo"
    wocar_ppo = "wocar_ppo"
    wocar_dqn = "wocar_dqn"


def main() -> None:
    parser: argparse.ArgumentParser = build_parser()
    args: argparse.Namespace = parser.parse_args()
    
    if args.update_function == UpdateFunction.wocar_ppo.value:
        train_ppo(args)
    elif args.update_function == UpdateFunction.wocar_dqn.value:
        train_dqn(args, console)
    elif args.update_function == UpdateFunction.mappo.value:
        train(args)
    else:
        raise NotImplementedError(f"Update function {args.update_function} is not implemented!")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--save", type=str, default=get_default_save_name())
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument(
        "--update-function",
        choices=[f.value for f in UpdateFunction],
        default=UpdateFunction.mappo,
    )
    parser.add_argument("--kappa-reg", type=float, default=0.1) # Weight of the regularizer term in the actor loss. The paper grid-searches {0.01, 0.03, 0.1, 0.3, 1.0}. "where κ_wst and κ_reg are hyperparameters balancing between natural performance and robustness"
    parser.add_argument("--eps-final", type=float, default=0.075) # Final attack radius ε, the value epsilon_schedule ramps to.    

    # WocaR-PPO only (read by train_ppo / ppo_update_wocar)
    parser.add_argument("--env-id", type=str, default="Hopper-v4")

    parser.add_argument("--kappa-wst-final", type=float, default=0.8) # Final weight of the Q_φ advantage shift. "where κ_wst and κ_reg are hyperparameters balancing between natural performance and robustness"
    parser.add_argument("--kappa-wst-shape", choices=["linear", "exponential"], default="linear") # linear or exponential, the ramp shape. Exponential only on Ant.

    # WocaR-DQN (read by train_dqn / dqn_update_wocar)
        # WocaR-DQN only (read by train_dqn)
    parser.add_argument("--hidden", type=int, default=64)  # width of the three Q-networks
    parser.add_argument("--tau", type=float, default=0.005)  # soft-update coefficient for Q_v'
    parser.add_argument("--lr", type=float, default=1e-4)  # learning rate for Q_v, Q_phi and Q_r
    parser.add_argument("--buffer-size", type=int, default=100_000)  # replay buffer capacity, in transitions (one per agent per env step)
    parser.add_argument("--warmup", type=int, default=5_000)  # transitions in the buffer before updates start; also when eps_t starts ramping
    parser.add_argument("--batch-size", type=int, default=64)  # minibatch size per update
    parser.add_argument("--explore-steps", type=int, default=50_000)  # env steps over which the exploration probability decays
    parser.add_argument("--explore-final", type=float, default=0.05)  # final epsilon-greedy exploration probability
    return parser


def get_default_save_name(save_directory: Path = Path(".")) -> str:
    next_number: int = _highest_save_number(save_directory) + 1
    return str(save_directory / f"{DEFAULT_SAVE_STEM}.{next_number}.pt")


def _highest_save_number(save_directory: Path) -> int:
    pattern: re.Pattern[str] = re.compile(rf"^{re.escape(DEFAULT_SAVE_STEM)}\.(\d+)\.pt$")
    numbers: list[int] = []
    for path in save_directory.glob(f"{DEFAULT_SAVE_STEM}.*.pt"):
        match: re.Match[str] | None = pattern.match(path.name)
        if match is not None:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0)


if __name__ == "__main__":
    main()