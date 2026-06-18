"""Configuration loading and path expansion."""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
from typing import Any

import yaml


PathType = str | PathLike[str]


def load_yaml(path: PathType) -> Any:
    """Load a YAML file using PyYAML's safe loader."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _expand_path_values(value: Any) -> Any:
    """Recursively expand environment variables and user-home markers."""
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {key: _expand_path_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_path_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_path_values(item) for item in value)
    return value


def load_paths_config(
    path: PathType = "configs/paths.local.yaml",
) -> dict[str, Any]:
    """Load a paths YAML file and safely expand Windows-compatible paths.

    Both ``%NAME%`` and ``$NAME`` environment-variable forms are supported by
    :func:`os.path.expandvars`. Backslashes are preserved and paths are not
    resolved, so network paths and paths that do not yet exist remain valid.
    """
    config = load_yaml(path)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Paths config must contain a YAML mapping: {path}")
    return _expand_path_values(config)
