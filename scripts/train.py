import argparse
from datetime import datetime
import sys
from pathlib import Path
from enum import Enum

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marl.training.trainer import train


class UpdateFunction(Enum):
    mappo = "mappo"
    wocar = "wocar"

class KappaWorstShape(Enum):
    linear = "linear"
    exponential = "exponential"

def main():
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
    parser.add_argument("--save", type=str, default="mappo.pt")
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--update-function", choices=UpdateFunction.__dict__.values(), default=UpdateFunction.mappo)
    parser.add_argument("--epsilon_final", type=float, default=0.075) # Final attack radius ε, the value epsilon_schedule ramps to.
    parser.add_argument("--kappa_wst_final", type=float, default=0.8) # Final weight of the Q_φ advantage shift. "where κ_wst and κ_reg are hyperparameters balancing between natural performance and robustness"
    parser.add_argument("--kappa_wst_shape", choices=KappaWorstShape.__dict__.values(), default="linear") # linear or exponential, the ramp shape. Exponential only on Ant.
    parser.add_argument("--kappa_reg", type=float, default=0.1) # Weight of the regularizer term in the actor loss. The paper grid-searches {0.01, 0.03, 0.1, 0.3, 1.0}. "where κ_wst and κ_reg are hyperparameters balancing between natural performance and robustness"

    train(parser.parse_args())


if __name__ == "__main__":
    main()
