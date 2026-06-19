"""Train a generic 3D prostate MRI segmentation downstream task."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from prostate_iqa.data.segmentation_transforms import (
    get_segmentation_train_transforms,
    get_segmentation_val_transforms,
)
from prostate_iqa.models.unet_segmentation import build_unet3d
from prostate_iqa.training.train_binary_task import (
    _assert_patient_disjoint,
    _is_present,
    _load_datalist,
    _reject_test_input,
)
from prostate_iqa.utils.io import ensure_dir, write_csv, write_json
from prostate_iqa.utils.seed import set_global_seed
from prostate_iqa.utils.segmentation import (
    binary_segmentation_metrics,
    mean_finite_metrics,
)


IDENTITY_COLUMNS = (
    "patient_id",
    "scan_id",
    "distortion_status",
    "acquisition_id",
)
METRIC_COLUMNS = (
    *IDENTITY_COLUMNS,
    "task_name",
    "task_type",
    "label_key",
    "dice",
    "iou",
    "asd",
    "hd95",
    "foreground_confidence",
    "predicted_voxels",
    "target_voxels",
)


def prepare_segmentation_items(
    items: Sequence[dict[str, Any]],
    image_keys: Sequence[str],
    label_key: str,
    source_name: str,
) -> list[dict[str, Any]]:
    """Validate paths and retain only collatable segmentation fields."""
    prepared: list[dict[str, Any]] = []
    skipped: list[str] = []
    if label_key in image_keys:
        raise ValueError("label_key cannot also appear in image_keys (target leakage).")
    for index, source in enumerate(items):
        required = (*image_keys, label_key)
        missing = [key for key in required if not _is_present(source.get(key))]
        ambiguous = [key for key in required if ";" in str(source.get(key) or "")]
        if missing or ambiguous:
            reasons = []
            if missing:
                reasons.append("missing " + ", ".join(missing))
            if ambiguous:
                reasons.append("multiple acquisitions in " + ", ".join(ambiguous))
            skipped.append(f"row {index}: {'; '.join(reasons)}")
            continue
        row = {key: source[key] for key in required}
        row["label_path"] = str(source[label_key])
        for key in IDENTITY_COLUMNS:
            row[key] = str(source[key]) if _is_present(source.get(key)) else ""
        prepared.append(row)
    if skipped:
        print(
            f"WARNING: skipped {len(skipped)} unusable {source_name} rows. "
            f"First entries: {' | '.join(skipped[:5])}"
        )
    if not prepared:
        raise ValueError(f"No usable segmentation rows remain in {source_name}.")
    return prepared


def _batch_values(batch: dict[str, Any], key: str, count: int) -> list[Any]:
    values = batch.get(key)
    if values is None:
        return [""] * count
    if isinstance(values, torch.Tensor):
        result = values.detach().cpu().tolist()
        return result if isinstance(result, list) else [result] * count
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values] * count


def _spacing_from_path(path: str) -> tuple[float, float, float]:
    """Read source NIfTI voxel spacing, falling back to voxel units."""
    try:
        zooms = nib.load(path).header.get_zooms()[:3]
        spacing = tuple(float(value) for value in zooms)
        if len(spacing) == 3 and all(math.isfinite(v) and v > 0 for v in spacing):
            return spacing
    except Exception:
        pass
    return (1.0, 1.0, 1.0)


def _save_prediction(
    mask: np.ndarray,
    label_path: str,
    output_path: Path,
) -> None:
    """Save a cropped prediction using the processed label affine."""
    reference = nib.load(label_path)
    image = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, reference.header)
    image.set_data_dtype(np.uint8)
    nib.save(image, str(output_path))


@torch.inference_mode()
def evaluate_segmentation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_name: str,
    label_key: str,
    prediction_dir: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate one model and return aggregate plus acquisition-level metrics."""
    model.eval()
    if prediction_dir is not None:
        ensure_dir(prediction_dir)
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
        probabilities = torch.softmax(model(images), dim=1)
        predictions = probabilities.argmax(dim=1)
        labels_binary = labels[:, 0] > 0
        batch_size = int(images.shape[0])
        metadata = {
            key: _batch_values(batch, key, batch_size)
            for key in (*IDENTITY_COLUMNS, "label_path")
        }
        for index in range(batch_size):
            prediction = predictions[index].cpu().numpy().astype(bool)
            target = labels_binary[index].cpu().numpy().astype(bool)
            label_path = str(metadata["label_path"][index])
            metrics = binary_segmentation_metrics(
                prediction,
                target,
                _spacing_from_path(label_path),
            )
            foreground = probabilities[index, 1]
            confidence = (
                float(foreground[predictions[index] > 0].mean().cpu())
                if bool((predictions[index] > 0).any())
                else 0.0
            )
            row = {
                key: str(metadata[key][index]) for key in IDENTITY_COLUMNS
            }
            row.update(
                {
                    "task_name": task_name,
                    "task_type": "segmentation",
                    "label_key": label_key,
                    **metrics,
                    "foreground_confidence": confidence,
                    "predicted_voxels": int(prediction.sum()),
                    "target_voxels": int(target.sum()),
                }
            )
            rows.append(row)
            if prediction_dir is not None:
                identity = row["acquisition_id"] or row["scan_id"] or f"case_{len(rows):05d}"
                safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in identity)
                _save_prediction(
                    prediction,
                    label_path,
                    prediction_dir / f"{safe}_prediction.nii.gz",
                )
    frame = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    metrics = mean_finite_metrics(rows)
    metrics["task_name"] = task_name
    metrics["task_type"] = "segmentation"
    metrics["label_key"] = label_key
    metrics["num_cases"] = len(rows)
    return metrics, frame


def _checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": metrics,
        "task_type": "segmentation",
        "task_name": args.task_name,
        "image_keys": list(args.image_keys),
        "label_key": args.label_key,
        "roi_size": list(args.roi_size),
        "in_channels": len(args.image_keys),
        "out_channels": 2,
        "seed": args.seed,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Train a 3D segmentation model selected by validation Dice."""
    set_global_seed(args.seed)
    set_determinism(seed=args.seed)
    output_dir = ensure_dir(args.out_dir)
    train_path = Path(args.train_json)
    val_path = Path(args.val_json)
    train_raw = _load_datalist(train_path)
    val_raw = _load_datalist(val_path)
    _reject_test_input(train_path, train_raw)
    _reject_test_input(val_path, val_raw)
    train_items = prepare_segmentation_items(
        train_raw, args.image_keys, args.label_key, "training datalist"
    )
    val_items = prepare_segmentation_items(
        val_raw, args.image_keys, args.label_key, "validation datalist"
    )
    _assert_patient_disjoint(train_items, val_items)
    print(f"Segmentation rows: train={len(train_items)}, val={len(val_items)}")

    train_transform = get_segmentation_train_transforms(
        args.image_keys, args.label_key, args.roi_size
    )
    train_transform.set_random_state(seed=args.seed)
    val_transform = get_segmentation_val_transforms(
        args.image_keys, args.label_key, args.roi_size
    )
    train_loader = DataLoader(
        Dataset(train_items, transform=train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        Dataset(val_items, transform=val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training segmentation on device: {device}")
    model = build_unet3d(len(args.image_keys), 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    best_dice = -math.inf
    best_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        train_loss = total_loss / max(batches, 1)
        metrics, predictions = evaluate_segmentation(
            model, val_loader, device, args.task_name, args.label_key
        )
        mean_dice = float(metrics.get("mean_dice") or 0.0)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        torch.save(
            _checkpoint(model, optimizer, epoch, metrics, args),
            output_dir / "last.pt",
        )
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_metrics = {"epoch": epoch, **metrics}
            torch.save(
                _checkpoint(model, optimizer, epoch, metrics, args),
                output_dir / "best.pt",
            )
            write_csv(predictions, output_dir / "val_segmentation_metrics.csv")
            write_json(best_metrics, output_dir / "best_val_metrics.json")
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"dice={mean_dice:.4f} | iou={float(metrics.get('mean_iou') or 0):.4f}"
        )
    write_csv(pd.DataFrame(history), output_dir / "training_history.csv")
    print(f"Best validation Dice: {best_dice:.4f}; outputs: {output_dir}")
    return best_metrics


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a generic 3D prostate MRI downstream segmentation task."
    )
    parser.add_argument("--train_json", type=Path, required=True)
    parser.add_argument("--val_json", type=Path, required=True)
    parser.add_argument("--image_keys", nargs="+", required=True)
    parser.add_argument(
        "--label_key",
        default="prostate_mask",
        help="Segmentation target, e.g. prostate_mask or lesion_mask.",
    )
    parser.add_argument("--task_name", default="prostate_segmentation")
    parser.add_argument("--roi_size", nargs=3, type=_positive_int, default=(160, 160, 64))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=_positive_int, default=100)
    parser.add_argument("--batch_size", type=_positive_int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive.")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
