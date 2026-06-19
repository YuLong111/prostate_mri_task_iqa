"""Evaluate a trained generic 3D prostate MRI segmentation model."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from monai.data import DataLoader, Dataset

from prostate_iqa.data.segmentation_transforms import get_segmentation_val_transforms
from prostate_iqa.evaluation.eval_model import (
    _checkpoint_roi_size,
    _load_checkpoint,
    _load_datalist,
)
from prostate_iqa.models.unet_segmentation import build_unet3d
from prostate_iqa.training.train_segmentation_task import (
    evaluate_segmentation,
    prepare_segmentation_items,
)
from prostate_iqa.utils.io import ensure_dir, write_csv, write_json


def _extract_unet_state_dict(
    checkpoint: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Extract UNet weights without stripping MONAI's real ``model.`` prefix."""
    candidate: Mapping[str, Any] = checkpoint
    for key in ("model_state_dict", "state_dict", "model_state", "network", "net"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            candidate = value
            break
    state_dict = {
        str(key): value
        for key, value in candidate.items()
        if isinstance(value, torch.Tensor)
    }
    if not state_dict:
        raise ValueError("Checkpoint does not contain a tensor model state dictionary.")
    for prefix in ("module.", "_orig_mod."):
        if all(key.startswith(prefix) for key in state_dict):
            state_dict = {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    """Load a checkpoint, evaluate a datalist, and save metrics/predictions."""
    checkpoint = _load_checkpoint(Path(args.ckpt))
    checkpoint_task = str(checkpoint.get("task_type") or "segmentation")
    if checkpoint_task != "segmentation":
        raise ValueError(f"Checkpoint task_type is {checkpoint_task!r}, not segmentation.")
    checkpoint_keys = tuple(checkpoint.get("image_keys") or ())
    if checkpoint_keys and checkpoint_keys != tuple(args.image_keys):
        raise ValueError(
            f"Checkpoint image_keys={checkpoint_keys} do not match CLI={tuple(args.image_keys)}."
        )
    checkpoint_label = str(checkpoint.get("label_key") or args.label_key)
    if checkpoint_label != args.label_key:
        raise ValueError(
            f"Checkpoint label_key={checkpoint_label!r} does not match {args.label_key!r}."
        )
    roi_size = tuple(args.roi_size) if args.roi_size is not None else _checkpoint_roi_size(checkpoint)
    task_name = args.task_name or str(checkpoint.get("task_name") or args.label_key)
    items = prepare_segmentation_items(
        _load_datalist(Path(args.datalist_json)),
        args.image_keys,
        args.label_key,
        "evaluation datalist",
    )
    transform = get_segmentation_val_transforms(
        args.image_keys, args.label_key, roi_size
    )
    loader = DataLoader(
        Dataset(items, transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = build_unet3d(len(args.image_keys), 2)
    model.load_state_dict(_extract_unet_state_dict(checkpoint), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prediction_dir = ensure_dir(args.prediction_dir) if args.prediction_dir else None
    print(f"Evaluating {len(items)} segmentation cases on {device}.")
    metrics, frame = evaluate_segmentation(
        model,
        loader,
        device,
        task_name,
        args.label_key,
        prediction_dir,
    )
    output_csv = write_csv(frame, args.out_csv)
    output_json = write_json(metrics, args.out_metrics_json)
    print(f"Mean Dice={float(metrics.get('mean_dice') or 0):.4f}")
    print(f"Saved metrics: {output_csv}; summary: {output_json}")
    return metrics


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained 3D prostate or lesion segmentation model."
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--datalist_json", type=Path, required=True)
    parser.add_argument("--image_keys", nargs="+", required=True)
    parser.add_argument("--label_key", default="prostate_mask")
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--roi_size", nargs=3, type=_positive_int, default=None)
    parser.add_argument("--batch_size", type=_positive_int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prediction_dir", type=Path, default=None)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--out_metrics_json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    evaluate(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
