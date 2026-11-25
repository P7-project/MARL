import argparse
import math
import sys
from pathlib import Path

import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marl.attacks.registry import ATTACK_BUILDERS
from marl.training.evaluator import evaluate, evaluate_attacked


def _epsilon(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"must be a finite number >= 0, got {text}")
    return value


def _attacked_agents(text: str) -> int | str:
    if text == "all":
        return text
    if not text.isdecimal():
        raise argparse.ArgumentTypeError(f"must be a count >= 0 or 'all', got {text}")
    return int(text)


def build_parser() -> argparse.ArgumentParser:
    """The command-line options; aliases store under the first name of their option."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", "--checkpoint", dest="eval", type=str, required=True, help="path to checkpoint")
    parser.add_argument("--eval-episodes", "--episodes", dest="eval_episodes", type=int, default=None)  # default 5, or 1000 with --attack
    parser.add_argument("--eval-windows", type=int, default=3)  # parallel windows, each running all agents
    parser.add_argument("--render", action="store_true")  # with --attack: show the window and highlight the attacked cars
    parser.add_argument("--render-agents", type=int, default=None)  # camera views with --render --attack, default 1, highway only
    parser.add_argument("--attack", choices=list(ATTACK_BUILDERS), default=None)  # attacked evaluation, headless unless --render
    parser.add_argument("--attack-eps", "--epsilon", dest="attack_eps", type=_epsilon, nargs="+", default=None)  # required with --attack
    parser.add_argument("--attacked-agents", type=_attacked_agents, default="all")  # a count or "all"
    parser.add_argument("--seed", type=int, default=0)  # only with --attack
    parser.add_argument("--out", type=str, default=None)  # extra copy of results.json, only with --attack
    parser.add_argument("--logdir", type=str, default="runs")  # TensorBoard root, only with --attack
    parser.add_argument("--run-name", type=str, default=None)  # default <attack>_<k>of<n>_<day>_<time>, only with --attack
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.attack is not None and args.attack_eps is None:
        parser.error("--attack-eps (or --epsilon) is required with --attack")

    if args.eval_episodes is None:
        args.eval_episodes = 1000 if args.attack else 5
    if args.attack is None:
        evaluate(args)
        return
    evaluate_attacked(args)


if __name__ == "__main__":
    main()
