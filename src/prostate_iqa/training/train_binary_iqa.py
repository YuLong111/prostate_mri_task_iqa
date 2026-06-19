"""Train a binary task-derived prostate MRI image-quality baseline."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

from prostate_iqa.training.train_binary_task import (
    calculate_binary_metrics,
    evaluate,
    train,
)


def _positive_int(value: str) -> int:
    """Parse a positive integer command-line value."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    """Parse a finite positive floating-point command-line value."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse binary IQA training arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a binary prostate MRI IQA baseline using task-derived or "
            "direct binary quality labels."
        )
    )
    parser.add_argument("--train_json", type=Path, required=True)
    parser.add_argument("--val_json", type=Path, required=True)
    parser.add_argument(
        "--image_keys",
        nargs="+",
        required=True,
        help=(
            "Ordered image inputs, for example: dwi adc t2 prostate_mask. "
            "Channels are concatenated in the supplied order."
        ),
    )
    parser.add_argument("--target_key", default="task_quality_bin")
    parser.add_argument(
        "--roi_size",
        nargs=3,
        type=_positive_int,
        default=(160, 160, 64),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=_positive_int, default=50)
    parser.add_argument("--batch_size", type=_positive_int, default=1)
    parser.add_argument("--lr", type=_positive_float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run binary IQA training from the command line."""
    args = parse_args(argv)
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    train(args)
    return 0


__all__ = [
    "calculate_binary_metrics",
    "evaluate",
    "main",
    "parse_args",
    "train",
]


if __name__ == "__main__":
    raise SystemExit(main())
