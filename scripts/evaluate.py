import argparse
import sys
from pathlib import Path

import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marl.training.evaluator import evaluate



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, required=True, help="path to checkpoint")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-windows", type=int, default=3)  # parallel windows, each running all agents
    parser.add_argument("--render", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
