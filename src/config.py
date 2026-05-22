from __future__ import annotations

from pathlib import Path
import json
from typing import Any


CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class ConfigError(RuntimeError):
    pass


def load_config(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{name}.json"
    try:
        with path.open("r", encoding="utf-8") as config_file:
            data = json.load(config_file)
    except FileNotFoundError as error:
        raise ConfigError(f"Missing config file: {path}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSON in config file: {path}") from error

    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a JSON object: {path}")
    return data


def config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)
