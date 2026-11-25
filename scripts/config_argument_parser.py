import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

NON_CONFIG_KEYS: frozenset[str] = frozenset({"help", "config"})


class ConfigArgumentParser:
    """Wrap an ArgumentParser so a JSON file passed with --config sets its defaults.

    Keys in the file are argparse dest names (e.g. "kappa_reg"). Flags given on the
    command line override values from the file.
    """

    def __init__(self, parser: argparse.ArgumentParser) -> None:
        self._parser: argparse.ArgumentParser = parser

    def parse(self, argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
        """Apply the --config file (if any) as defaults, then parse the command line."""
        config_path: Optional[Path] = self._parse_config_path(argv)
        if config_path is not None:
            self._apply_config(config_path)
        return self._parser.parse_args(argv)

    def _parse_config_path(self, argv: Optional[Sequence[str]]) -> Optional[Path]:
        pre_parser = argparse.ArgumentParser(add_help=False)
        pre_parser.add_argument("--config", type=Path, default=None)
        known, _ = pre_parser.parse_known_args(argv)
        return known.config

    def _apply_config(self, config_path: Path) -> None:
        config: dict[str, Any] = self._load_config(config_path)
        self._validate_keys(config, config_path)
        self._validate_choices(config, config_path)
        self._parser.set_defaults(**config)

    def _load_config(self, config_path: Path) -> dict[str, Any]:
        try:
            with config_path.open("r", encoding="utf-8") as file:
                config: Any = json.load(file)
        except FileNotFoundError:
            self._parser.error(f"config file not found: {config_path}")
        except OSError as error:
            self._parser.error(f"cannot read config file {config_path}: {error}")
        except json.JSONDecodeError as error:
            self._parser.error(f"invalid JSON in config file {config_path}: {error}")
        if not isinstance(config, dict):
            self._parser.error(f"config file {config_path} must contain a JSON object")
        return config

    def _validate_keys(self, config: dict[str, Any], config_path: Path) -> None:
        valid_keys: set[str] = {action.dest for action in self._parser._actions} - NON_CONFIG_KEYS
        unknown_keys: list[str] = sorted(set(config) - valid_keys)
        if unknown_keys:
            self._parser.error(
                f"unknown key(s) in config file {config_path}: {', '.join(unknown_keys)} "
                "(keys are argparse dest names with underscores, e.g. kappa_reg)"
            )

    def _validate_choices(self, config: dict[str, Any], config_path: Path) -> None:
        for action in self._parser._actions:
            if action.choices is None or action.dest not in config:
                continue
            value: Any = config[action.dest]
            if value not in action.choices:
                allowed: str = ", ".join(repr(choice) for choice in action.choices)
                self._parser.error(
                    f"invalid value for {action.dest} in config file {config_path}: "
                    f"{value!r} (choose from {allowed})"
                )
