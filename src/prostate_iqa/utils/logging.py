"""Project logging helpers."""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

from .io import ensure_dir


PathType = str | PathLike[str]
_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str, log_file: PathType | None = None) -> logging.Logger:
    """Return a configured logger with console and optional file output.

    Repeated calls do not add duplicate handlers for the same destination.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT)

    if not any(getattr(handler, "_prostate_iqa_console", False) for handler in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler._prostate_iqa_console = True  # type: ignore[attr-defined]
        logger.addHandler(console_handler)

    if log_file is not None:
        file_path = Path(log_file).expanduser().resolve()
        ensure_dir(file_path.parent)
        existing_files = {
            Path(handler.baseFilename).resolve()
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        }
        if file_path not in existing_files:
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
