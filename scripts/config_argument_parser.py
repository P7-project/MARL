

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence


class ConfigArgumentParser:
    def __init__(self, parser: Optional[argparse.ArgumentParser] = None) -> None:
        self._parser: argparse.ArgumentParser = parser or self._build_parser()

    def parse(self, argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
        config_path: Optional[Path] = self._parse_config_path(argv)
        if config_path is not None:
            self._apply_config(config_path)
        return self._parser.parse_args(argv)

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=Path, default=None)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--name", type=str, default="run")
        return parser

    def _parse_config_path(self, argv: Optional[Sequence[str]]) -> Optional[Path]:
        pre_parser = argparse.ArgumentParser(add_help=False)
        pre_parser.add_argument("--config", type=Path, default=None)
        known, _ = pre_parser.parse_known_args(argv)
        return known.config

    def _apply_config(self, config_path: Path) -> None:
        config: dict[str, Any] = self._load_config(config_path)
        self._validate_keys(config)
        self._parser.set_defaults(**config)

    def _load_config(self, config_path: Path) -> dict[str, Any]:
        with config_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _validate_keys(self, config: dict[str, Any]) -> None:
        valid_keys: set[str] = {action.dest for action in self._parser._actions}
        unknown_keys: set[str] = set(config) - valid_keys
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {sorted(unknown_keys)}")


if __name__ == "__main__":
    args: argparse.Namespace = ConfigArgumentParser().parse()
    print(args)