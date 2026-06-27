"""Train a binary 3D DenseNet downstream task classifier."""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import WeightedRandomSampler

from prostate_iqa.data.transforms import get_train_transforms, get_val_transforms
from prostate_iqa.models.densenet_quality import build_densenet121
from prostate_iqa.utils.io import ensure_dir, read_json, write_csv, write_json
from prostate_iqa.utils.seed import set_global_seed


PREDICTION_COLUMNS = (
    "patient_id",
    "scan_id",
    "distortion_status",
    "acquisition_id",
    "true_label",
    "pred_label",
    "prob_0",
    "prob_1",
    "confidence",
    "correct",
)


def _is_present(value: Any) -> bool:
    """Return whether a datalist scalar contains a non-empty value."""
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def _binary_label(value: Any, target_key: str) -> int:
    """Parse a strict binary label."""
    if not _is_present(value):
        raise ValueError(f"Missing target {target_key!r}.")
    text = str(value).strip().lower()
    if text in {"0", "0.0", "false", "no", "negative"}:
        return 0
    if text in {"1", "1.0", "true", "yes", "positive"}:
        return 1
    raise ValueError(f"Target {target_key!r} must be binary 0/1, received {value!r}.")


def _load_datalist(path: Path) -> list[dict[str, Any]]:
    """Load a list-style or common MONAI-style datalist."""
    payload = read_json(path)
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = next(
            (
                payload[key]
                for key in ("data", "training", "validation")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
        if items is None:
            raise ValueError(f"No training/validation list found in {path}.")
    else:
        raise ValueError(f"Datalist must contain a list or mapping: {path}")
    if not all(isinstance(item, dict) for item in items):
        raise ValueError(f"Every datalist item must be an object: {path}")
    return items


def _reject_test_input(path: Path, items: Sequence[dict[str, Any]]) -> None:
    """Reject accidental use of locked-test data for training or selection."""
    name = path.name.lower()
    test_name = bool(
        "test_locked" in name
        or re.search(r"(?:^|[_-])test(?:[_\-.]|$)", name)
    )
    test_rows = [
        item
        for item in items
        if str(item.get("split", "")).strip().lower() in {"test", "test_locked"}
    ]
    if test_name or test_rows:
        raise ValueError(
            f"Locked test data cannot be used by this training script: {path}. "
            "Use train and validation datalists only."
        )


def _prepare_items(
    items: Sequence[dict[str, Any]],
    image_keys: Sequence[str],
    target_key: str,
    source_name: str,
) -> list[dict[str, Any]]:
    """Validate paths and targets, filtering unusable unlabeled records."""
    prepared: list[dict[str, Any]] = []
    skipped: list[str] = []
    for index, source in enumerate(items):
        quality_target = str(source.get("quality_target_key") or "").strip()
        if quality_target and quality_target in image_keys:
            raise ValueError(
                f"Target leakage in {source_name} row {index}: segmentation target "
                f"{quality_target!r} cannot also be an IQA image input."
            )
        missing_keys = [key for key in image_keys if not _is_present(source.get(key))]
        ambiguous_keys = [
            key for key in image_keys if ";" in str(source.get(key) or "")
        ]
        if missing_keys or ambiguous_keys or not _is_present(source.get(target_key)):
            reasons = []
            if missing_keys:
                reasons.append("missing " + ", ".join(missing_keys))
            if ambiguous_keys:
                reasons.append(
                    "multiple acquisitions in " + ", ".join(ambiguous_keys)
                )
            if not _is_present(source.get(target_key)):
                reasons.append(f"missing {target_key}")
            skipped.append(f"row {index}: {'; '.join(reasons)}")
            continue

        item = {key: source[key] for key in image_keys}
        for key in (
            "patient_id",
            "scan_id",
            "distortion_status",
            "acquisition_id",
        ):
            item[key] = str(source[key]) if _is_present(source.get(key)) else ""
        item["label"] = _binary_label(source[target_key], target_key)
        prepared.append(item)

    if skipped:
        preview = " | ".join(skipped[:5])
        print(
            f"WARNING: skipped {len(skipped)} unusable {source_name} rows. "
            f"First entries: {preview}"
        )
    if not prepared:
        raise ValueError(f"No usable labeled rows remain in {source_name}.")
    return prepared


def _assert_patient_disjoint(
    train_items: Sequence[dict[str, Any]],
    val_items: Sequence[dict[str, Any]],
) -> None:
    """Prevent patient leakage between train and validation sets."""
    train_patients = {
        str(item["patient_id"]).strip()
        for item in train_items
        if _is_present(item.get("patient_id"))
    }
    val_patients = {
        str(item["patient_id"]).strip()
        for item in val_items
        if _is_present(item.get("patient_id"))
    }
    overlap = train_patients & val_patients
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"Patient leakage between train and validation: {examples}")


def _class_counts(items: Sequence[dict[str, Any]]) -> dict[int, int]:
    """Count binary labels and require both classes."""
    counts = pd.Series([int(item["label"]) for item in items]).value_counts()
    result = {label: int(counts.get(label, 0)) for label in (0, 1)}
    if min(result.values()) == 0:
        raise ValueError(f"Both binary classes are required, received counts {result}.")
    return result


def _weighted_sampler(
    items: Sequence[dict[str, Any]],
    seed: int,
) -> WeightedRandomSampler | None:
    """Return an inverse-frequency sampler only when classes are imbalanced."""
    counts = _class_counts(items)
    if counts[0] == counts[1]:
        return None
    labels = [int(item["label"]) for item in items]
    weights = torch.as_tensor(
        [1.0 / counts[label] for label in labels],
        dtype=torch.double,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def _class_weight_tensor(
    items: Sequence[dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    """Return inverse-frequency class weights normalized around one."""
    counts = _class_counts(items)
    total = float(sum(counts.values()))
    weights = [
        total / (2.0 * max(float(counts[label]), 1.0))
        for label in (0, 1)
    ]
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


class FocalCrossEntropyLoss(torch.nn.Module):
    """Cross-entropy focal loss for hard-example emphasis."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("focal gamma must be non-negative.")
        if label_smoothing < 0 or label_smoothing >= 1:
            raise ValueError("label_smoothing must be in [0, 1).")
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        if weight is not None:
            self.register_buffer("weight", weight.detach().clone().float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate mean focal cross-entropy."""
        ce_loss = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        true_class_probability = torch.softmax(logits, dim=1).gather(
            1,
            target.unsqueeze(1),
        ).squeeze(1)
        focal_factor = (1.0 - true_class_probability).clamp_min(0.0).pow(self.gamma)
        return (focal_factor * ce_loss).mean()


def _make_binary_criterion(
    items: Sequence[dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
) -> torch.nn.Module:
    """Create a binary classification loss with optional label smoothing/weights."""
    imbalance_strategy = getattr(args, "imbalance_strategy", "sampler")
    class_weights = (
        _class_weight_tensor(items, device)
        if imbalance_strategy == "class_weight"
        else None
    )
    label_smoothing = float(getattr(args, "label_smoothing", 0.0))
    loss_name = str(getattr(args, "loss", "ce")).strip().lower()
    if loss_name == "focal":
        return FocalCrossEntropyLoss(
            gamma=float(getattr(args, "focal_gamma", 2.0)),
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    return torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


def _batch_strings(batch: dict[str, Any], key: str, count: int) -> list[str]:
    """Extract string identifiers from a collated MONAI batch."""
    values = batch.get(key)
    if values is None:
        return [""] * count
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values]
    if isinstance(values, torch.Tensor):
        return [str(value) for value in values.detach().cpu().tolist()]
    return [str(values)] * count


def _transform_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Return optional transform tuning settings."""
    return {
        "crop_margin": tuple(getattr(args, "crop_margin", (16, 16, 8))),
        "mask_crop": bool(getattr(args, "mask_crop", True)),
    }


def _build_binary_model(args: argparse.Namespace) -> torch.nn.Module:
    """Build the configured binary DenseNet classifier."""
    return build_densenet121(
        len(args.image_keys),
        2,
        dropout_prob=float(getattr(args, "dropout_prob", 0.0)),
    )


def _make_ema_model(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module | None:
    """Create an EMA copy of the model when requested."""
    ema_decay = float(getattr(args, "ema_decay", 0.0))
    if ema_decay <= 0:
        return None
    if ema_decay >= 1:
        raise ValueError("ema_decay must be in [0, 1).")
    ema_model = _build_binary_model(args).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


@torch.no_grad()
def _update_ema_model(
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    ema_decay: float,
) -> None:
    """Update an exponential moving average model in-place."""
    if ema_model is None:
        return
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()
    for key, value in model_state.items():
        if torch.is_floating_point(value):
            ema_state[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
        else:
            ema_state[key].copy_(value)


def calculate_binary_metrics(
    true_labels: Sequence[int],
    probabilities_1: Sequence[float],
    predicted_labels: Sequence[int] | None = None,
) -> dict[str, float]:
    """Calculate discrimination and thresholded binary classification metrics."""
    truth = np.asarray(true_labels, dtype=int)
    probability = np.asarray(probabilities_1, dtype=float)
    prediction = (
        np.asarray(predicted_labels, dtype=int)
        if predicted_labels is not None
        else (probability >= 0.5).astype(int)
    )
    if np.unique(truth).size != 2:
        raise ValueError("AUC and PR-AUC require both classes in validation data.")

    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "auc": float(roc_auc_score(truth, probability)),
        "pr_auc": float(average_precision_score(truth, probability)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    positive_threshold: float = 0.5,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate a model and return metrics plus per-scan predictions."""
    if not 0.0 <= positive_threshold <= 1.0:
        raise ValueError("positive_threshold must be in [0, 1].")
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = (probabilities[:, 1] >= positive_threshold).long()

        labels_cpu = labels.detach().cpu().numpy()
        probabilities_cpu = probabilities.detach().cpu().numpy()
        predictions_cpu = predictions.detach().cpu().numpy()
        patient_ids = _batch_strings(batch, "patient_id", len(labels_cpu))
        scan_ids = _batch_strings(batch, "scan_id", len(labels_cpu))
        distortion_statuses = _batch_strings(
            batch, "distortion_status", len(labels_cpu)
        )
        acquisition_ids = _batch_strings(batch, "acquisition_id", len(labels_cpu))
        for index, true_label in enumerate(labels_cpu):
            predicted = int(predictions_cpu[index])
            prob_0 = float(probabilities_cpu[index, 0])
            prob_1 = float(probabilities_cpu[index, 1])
            rows.append(
                {
                    "patient_id": patient_ids[index],
                    "scan_id": scan_ids[index],
                    "distortion_status": distortion_statuses[index],
                    "acquisition_id": acquisition_ids[index],
                    "true_label": int(true_label),
                    "pred_label": predicted,
                    "prob_0": prob_0,
                    "prob_1": prob_1,
                    "confidence": max(prob_0, prob_1),
                    "correct": int(predicted == int(true_label)),
                }
            )

    predictions_frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    metrics = calculate_binary_metrics(
        predictions_frame["true_label"].tolist(),
        predictions_frame["prob_1"].tolist(),
        predictions_frame["pred_label"].tolist(),
    )
    return metrics, predictions_frame


def _checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build a self-describing checkpoint payload."""
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": metrics,
        "image_keys": list(args.image_keys),
        "target_key": args.target_key,
        "roi_size": list(args.roi_size),
        "seed": args.seed,
        "dropout_prob": float(getattr(args, "dropout_prob", 0.0)),
        "weight_decay": float(getattr(args, "weight_decay", 0.0)),
        "label_smoothing": float(getattr(args, "label_smoothing", 0.0)),
        "imbalance_strategy": getattr(args, "imbalance_strategy", "sampler"),
        "loss": getattr(args, "loss", "ce"),
        "focal_gamma": float(getattr(args, "focal_gamma", 2.0)),
        "ema_decay": float(getattr(args, "ema_decay", 0.0)),
        "grad_clip": float(getattr(args, "grad_clip", 0.0)),
        "positive_threshold": float(getattr(args, "positive_threshold", 0.5)),
        "crop_margin": list(getattr(args, "crop_margin", (16, 16, 8))),
        "mask_crop": bool(getattr(args, "mask_crop", True)),
        "scheduler": "cosine_annealing_lr",
    }


def _print_metrics(epoch: int, train_loss: float, metrics: dict[str, float]) -> None:
    """Print all requested validation metrics for one epoch."""
    values = " | ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | {values}")


def train(args: argparse.Namespace) -> dict[str, float]:
    """Run binary model training and return best validation metrics."""
    set_global_seed(args.seed)
    set_determinism(seed=args.seed)
    output_dir = ensure_dir(args.out_dir)

    train_path = Path(args.train_json)
    val_path = Path(args.val_json)
    raw_train = _load_datalist(train_path)
    raw_val = _load_datalist(val_path)
    _reject_test_input(train_path, raw_train)
    _reject_test_input(val_path, raw_val)

    train_items = _prepare_items(
        raw_train, args.image_keys, args.target_key, "training datalist"
    )
    val_items = _prepare_items(
        raw_val, args.image_keys, args.target_key, "validation datalist"
    )
    _assert_patient_disjoint(train_items, val_items)
    train_counts = _class_counts(train_items)
    val_counts = _class_counts(val_items)
    print(f"Training class counts: {train_counts}")
    print(f"Validation class counts: {val_counts}")

    transform_kwargs = _transform_kwargs(args)
    train_transforms = get_train_transforms(
        args.image_keys,
        args.roi_size,
        **transform_kwargs,
    )
    train_transforms.set_random_state(seed=args.seed)
    val_transforms = get_val_transforms(
        args.image_keys,
        args.roi_size,
        **transform_kwargs,
    )
    train_dataset = Dataset(train_items, transform=train_transforms)
    val_dataset = Dataset(val_items, transform=val_transforms)

    imbalance_strategy = getattr(args, "imbalance_strategy", "sampler")
    sampler = (
        _weighted_sampler(train_items, args.seed)
        if imbalance_strategy == "sampler"
        else None
    )
    if sampler is not None:
        print("Using inverse-frequency WeightedRandomSampler for class imbalance.")
    elif imbalance_strategy == "class_weight":
        print("Using inverse-frequency class weights for class imbalance.")
    elif imbalance_strategy == "none":
        print("Using unweighted random sampling/loss.")
    drop_singleton = args.batch_size > 1 and len(train_items) % args.batch_size == 1
    if drop_singleton:
        print("Dropping the final singleton training batch for BatchNorm stability.")
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
        drop_last=drop_singleton,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model = _build_binary_model(args).to(device)
    ema_model = _make_ema_model(model, args, device)
    if ema_model is not None:
        print(f"Using EMA model for validation/checkpointing: decay={args.ema_decay}")
    criterion = _make_binary_criterion(train_items, device, args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    best_auc = -math.inf
    best_metrics: dict[str, float] = {}
    history: list[dict[str, Any]] = []
    ema_decay = float(getattr(args, "ema_decay", 0.0))
    grad_clip = float(getattr(args, "grad_clip", 0.0))
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        observed = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            _update_ema_model(model, ema_model, ema_decay)
            batch_count = int(labels.shape[0])
            running_loss += float(loss.detach()) * batch_count
            observed += batch_count

        train_loss = running_loss / max(observed, 1)
        validation_model = ema_model if ema_model is not None else model
        val_metrics, val_predictions = evaluate(
            validation_model,
            val_loader,
            device,
            positive_threshold=float(getattr(args, "positive_threshold", 0.5)),
        )
        _print_metrics(epoch, train_loss, val_metrics)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        scheduler.step()

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_metrics = dict(val_metrics)
            torch.save(
                _checkpoint(validation_model, optimizer, epoch, val_metrics, args),
                output_dir / "best.pt",
            )
            write_csv(val_predictions, output_dir / "val_predictions.csv")
            write_json(
                {"epoch": epoch, **val_metrics},
                output_dir / "best_val_metrics.json",
            )

    final_model = ema_model if ema_model is not None else model
    final_metrics, _ = evaluate(
        final_model,
        val_loader,
        device,
        positive_threshold=float(getattr(args, "positive_threshold", 0.5)),
    )
    torch.save(
        _checkpoint(final_model, optimizer, args.epochs, final_metrics, args),
        output_dir / "last.pt",
    )
    write_csv(pd.DataFrame(history), output_dir / "training_history.csv")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Saved outputs to: {output_dir}")
    return best_metrics


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def _dropout_probability(value: str) -> float:
    parsed = _nonnegative_float(value)
    if parsed >= 1:
        raise argparse.ArgumentTypeError("must be less than 1")
    return parsed


def _label_smoothing_value(value: str) -> float:
    parsed = _nonnegative_float(value)
    if parsed >= 1:
        raise argparse.ArgumentTypeError("must be less than 1")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a binary prostate MRI downstream task classifier."
    )
    parser.add_argument("--train_json", type=Path, required=True)
    parser.add_argument("--val_json", type=Path, required=True)
    parser.add_argument("--image_keys", nargs="+", required=True)
    parser.add_argument("--target_key", required=True)
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
        type=_label_smoothing_value,
        default=0.0,
        help="Cross-entropy label smoothing in [0, 1). Try 0.03-0.05.",
    )
    parser.add_argument(
        "--dropout_prob",
        type=_dropout_probability,
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
        type=_dropout_probability,
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
    """CLI entry point."""
    args = parse_args(argv)
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
