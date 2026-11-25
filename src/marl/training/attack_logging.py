"""TensorBoard logging of the attacked evaluation."""

from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

LOG_FOLDER: str = "attack_evaluation"


def create_attack_writer(logdir: str, run_name: str | None) -> SummaryWriter:
    """Writer for one evaluation run."""
    name: str = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    return SummaryWriter(str(Path(logdir) / LOG_FOLDER / name))


def log_attack_result(writer: SummaryWriter, settings: dict, result: dict) -> None:
    """Log the episodes and the averages of one epsilon."""
    tag: str = _attack_tag(settings, result["epsilon"])
    _log_episodes(writer, tag, result)
    _log_averages(writer, tag, result)
    writer.flush()


def _attack_tag(settings: dict, epsilon: float) -> str:
    """Name of the setting, as a prefix of its scalars."""
    if settings["attacked_agents"] == 0:
        return "clean"
    return f"{settings['attack']}/attacked_{settings['attacked_agents']}/epsilon_{epsilon:g}"


def _log_episodes(writer: SummaryWriter, tag: str, result: dict) -> None:
    for episode, reward in enumerate(result["episode_returns"]):
        writer.add_scalar(f"{tag}/episode_reward", reward, episode)
    for episode, crashed in enumerate(result["episode_crashes"]):
        writer.add_scalar(f"{tag}/episode_crashed", float(crashed), episode)


def _log_averages(writer: SummaryWriter, tag: str, result: dict) -> None:
    writer.add_scalar(f"{tag}/reward_mean", result["return_mean"], 0)
    writer.add_scalar(f"{tag}/reward_std", result["return_std"], 0)
    writer.add_scalar(f"{tag}/crash_rate", result["crash_rate"], 0)
    if result["changed_action_rate"] is not None:
        writer.add_scalar(f"{tag}/changed_action_rate", result["changed_action_rate"], 0)
