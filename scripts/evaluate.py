import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marl.training.evaluator import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, required=True, help="path to checkpoint")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="show the policy in a window")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
