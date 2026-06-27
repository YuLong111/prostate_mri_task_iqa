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


def _nonnegative_float(value: str) -> float:
    """Parse a finite non-negative floating-point command-line value."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def _less_than_one_float(value: str) -> float:
    """Parse a finite value in the half-open interval [0, 1)."""
    parsed = _nonnegative_float(value)
    if parsed >= 1:
        raise argparse.ArgumentTypeError("must be less than 1")
    return parsed


def _unit_interval(value: str) -> float:
    """Parse a finite value in [0, 1]."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
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
    parser.add_argument(
        "--crop_margin",
        nargs=3,
        type=int,
        default=(16, 16, 8),
        metavar=("X", "Y", "Z"),
        help=(
            "Margin for prostate-mask foreground crop before resize. "
            "Try 24 24 12 or 32 32 16 if the crop is too tight."
        ),
    )
    parser.add_argument(
        "--no_mask_crop",
        action="store_false",
        dest="mask_crop",
        help="Disable prostate-mask foreground cropping.",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=_positive_int, default=50)
    parser.add_argument("--batch_size", type=_positive_int, default=1)
    parser.add_argument("--lr", type=_positive_float, default=1e-4)
    parser.add_argument(
        "--weight_decay",
        type=_nonnegative_float,
        default=0.0,
        help="AdamW weight decay. Try 1e-4 when validation AUC overfits.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=_less_than_one_float,
        default=0.0,
        help="Cross-entropy label smoothing in [0, 1). Try 0.03-0.05.",
    )
    parser.add_argument(
        "--dropout_prob",
        type=_less_than_one_float,
        default=0.0,
        help="DenseNet dropout probability. Try 0.1 for weak generalization.",
    )
    parser.add_argument(
        "--imbalance_strategy",
        choices=("sampler", "class_weight", "none"),
        default="sampler",
        help=(
            "How to handle binary class imbalance. 'class_weight' often gives "
            "better calibration than oversampling when imbalance is mild."
        ),
    )
    parser.add_argument(
        "--loss",
        choices=("ce", "focal"),
        default="ce",
        help="Classification loss. Focal loss can improve ranking for hard cases.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=_nonnegative_float,
        default=2.0,
        help="Focal-loss gamma when --loss focal is used.",
    )
    parser.add_argument(
        "--grad_clip",
        type=_nonnegative_float,
        default=0.0,
        help="Optional gradient norm clipping. Try 1.0 for unstable training.",
    )
    parser.add_argument(
        "--ema_decay",
        type=_less_than_one_float,
        default=0.0,
        help="EMA decay for validation/checkpointing. Try 0.99.",
    )
    parser.add_argument(
        "--positive_threshold",
        type=_unit_interval,
        default=0.5,
        help=(
            "Threshold for converting prob_1 to pred_label. Does not affect AUC. "
            "Use validation-tuned values such as 0.90 for PI-RADS if needed."
        ),
    )
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
