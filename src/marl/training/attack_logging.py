"""TensorBoard logging of the attacked evaluation."""

from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

LOG_FOLDER: str = "attack_evaluation"


RESULTS_FILE: str = "results.json"


def default_run_name(attack: str, attacked_agents: int, n_agents: int) -> str:
    """Run name from the setting and the start time, e.g. minbest_1of3_25_22-10-05."""
    return f"{attack}_{attacked_agents}of{n_agents}_{datetime.now().strftime('%d_%H-%M-%S')}"


def reserve_attack_folder(logdir: str, run_name: str) -> Path:
    """Create a new <logdir>/attack_evaluation/<run_name> folder and return it.

    If the folder exists, try <run_name>_2, <run_name>_3, ... The folder is created
    with exist_ok=False, so two runs started at the same time never share it.
    Same logic as reserve_run_folder in scripts/train.py.
    """
    root: Path = Path(logdir) / LOG_FOLDER
    root.mkdir(parents=True, exist_ok=True)
    number: int = 1
    while True:
        name: str = run_name if number == 1 else f"{run_name}_{number}"
        try:
            (root / name).mkdir(exist_ok=False)
        except FileExistsError:
            number += 1
            continue
        return root / name


def create_attack_writer(folder: Path) -> SummaryWriter:
    """Writer for one evaluation run, in its reserved folder."""
    return SummaryWriter(str(folder))


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
