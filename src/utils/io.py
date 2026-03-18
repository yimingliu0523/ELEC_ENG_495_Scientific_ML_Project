"""Project-wide file and path helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root() / path


def ensure_dir(path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path_like: str | Path, indent: int = 2) -> Path:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, sort_keys=True)
    return path


def load_json(path_like: str | Path) -> dict[str, Any]:
    path = resolve_path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(text: str, path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    columns = [str(column) for column in dataframe.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for row in dataframe.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join([header, separator, *rows]) + "\n"


def write_markdown_table(dataframe: pd.DataFrame, path_like: str | Path) -> Path:
    return write_text(dataframe_to_markdown(dataframe), path_like)
