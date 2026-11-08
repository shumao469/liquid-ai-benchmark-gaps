"""Configuration loading with deterministic hashing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .io_utils import canonical_json_hash


def load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return config, canonical_json_hash(config)


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid mapping: {key}")
    return value

