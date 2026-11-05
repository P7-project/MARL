import argparse
import re
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marl.training.trainer import train
from marl.woca_r_rl.ppo import train_ppo  # adjust to your real module path

DEFAULT_SAVE_STEM: str = "default_save_name"

class UpdateFunction(Enum):
    mappo = "mappo"
    wocar = "wocar"


def main() -> None:
    parser: argparse.ArgumentParser = build_parser()
    args: argparse.Namespace = parser.parse_args()

    if args.update_function is UpdateFunction.wocar.value:
        train_ppo(args)
    elif args.update_function is UpdateFunction.mappo.value:
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

    # WocaR-PPO only (read by train_ppo / ppo_update_wocar)
    parser.add_argument("--env-id", type=str, default="Hopper-v4")
    parser.add_argument("--eps-final", type=float, default=0.075)  # final attack radius ε
    parser.add_argument("--kappa-wst-final", type=float, default=0.8)  # final weight of the Q_φ advantage shift
    parser.add_argument("--kappa-wst-shape", choices=["linear", "exponential"], default="linear")
    parser.add_argument("--kappa-reg", type=float, default=0.1)  # paper grid: {0.01, 0.03, 0.1, 0.3, 1.0}
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