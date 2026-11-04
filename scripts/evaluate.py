import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marl.config import load_config
from marl.training.evaluator import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, required=True, help="path to checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="environment YAML (needed for older checkpoints without saved config)",
    )
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.env_cfg = load_config(args.config)
    evaluate(args)


if __name__ == "__main__":
    main()
