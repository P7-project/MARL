"""Train MAPPO using an optional YAML environment configuration."""

import argparse

from marl.config import load_config
from marl.training.trainer import train


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="configs/highway.yaml")
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config(pre_args.config)
    train_cfg = config.get("train", {})

    parser = argparse.ArgumentParser(description="Train MAPPO")
    parser.add_argument("--config", default=pre_args.config, help="YAML config path")
    parser.add_argument("--agents", type=int, default=train_cfg.get("agents", 3))
    parser.add_argument("--total-steps", type=int, default=train_cfg.get("total_steps", 200_000))
    parser.add_argument("--rollout", type=int, default=train_cfg.get("rollout", 512))
    parser.add_argument("--n-attacked", type=int, default=0)
    parser.add_argument("--eps", type=float, default=config.get("perturbation", {}).get("eps"))
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
    parser.add_argument(
        "--save",
        type=str,
        default="mappo.pt",
        help="checkpoint path; a bare filename is saved under checkpoints/",
    )
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--render", action="store_true", help="show the training environment")
    args = parser.parse_args()

    # Reload the selected path in case --config appeared after other options.
    args.env_cfg = load_config(args.config)
    train(args)


if __name__ == "__main__":
    main()
