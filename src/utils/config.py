"""YAML config loading and persistence helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.utils.io import resolve_path


def load_config(path_like: str | Path) -> dict[str, Any]:
    path = resolve_path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must define a mapping.")
    return config


def dump_config(config: dict[str, Any], path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def save_config_snapshot(config: dict[str, Any], output_dir: str | Path, filename: str = "config_used.yaml") -> Path:
    output_path = resolve_path(output_dir) / filename
    return dump_config(config, output_path)


def build_config_argparser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file.")
    return parser
