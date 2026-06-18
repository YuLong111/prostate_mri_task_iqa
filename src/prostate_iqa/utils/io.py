"""Small helpers for common file input and output operations."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PathType = str | PathLike[str]


def ensure_dir(path: PathType) -> Path:
    """Create *path* and its parents if needed, then return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: PathType) -> Any:
    """Read a UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: Any, path: PathType, *, indent: int = 2) -> Path:
    """Write data to a UTF-8 JSON file, creating its parent directory."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)
        file.write("\n")
    return output_path


def read_csv(path: PathType, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame."""
    return pd.read_csv(path, **kwargs)


def write_csv(
    data: pd.DataFrame | Iterable[Mapping[str, Any]],
    path: PathType,
    *,
    index: bool = False,
    **kwargs: Any,
) -> Path:
    """Write a DataFrame or iterable of row mappings to a CSV file."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    frame.to_csv(output_path, index=index, **kwargs)
    return output_path


def read_yaml(path: PathType) -> Any:
    """Read a YAML file using PyYAML's safe loader."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_yaml(data: Any, path: PathType) -> Path:
    """Write data to a UTF-8 YAML file, creating its parent directory."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
    return output_path


def list_files_recursive(root: PathType, suffixes: str | Iterable[str]) -> list[Path]:
    """Return sorted files below *root* whose names match *suffixes*.

    Matching is case-insensitive. Suffixes may include or omit the leading dot.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_path}")

    values = (suffixes,) if isinstance(suffixes, str) else tuple(suffixes)
    normalized = tuple(
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in values
    )
    if not normalized:
        return []

    return sorted(
        path for path in root_path.rglob("*")
        if path.is_file() and path.name.lower().endswith(normalized)
    )
