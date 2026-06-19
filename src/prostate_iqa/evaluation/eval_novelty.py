"""Estimate feature-space novelty for prostate MRI cases."""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset

from prostate_iqa.data.transforms import get_val_transforms
from prostate_iqa.evaluation.eval_model import (
    _batch_values,
    _build_model,
    _checkpoint_roi_size,
    _is_present,
    _load_checkpoint,
    _load_datalist,
    _parse_label,
)
from prostate_iqa.utils.io import write_csv


OUTPUT_COLUMNS = (
    "patient_id",
    "scan_id",
    "novelty_distance",
    "entropy",
    "confidence",
    "predicted_class",
)


def _prepare_items(
    items: Sequence[dict[str, Any]],
    image_keys: Sequence[str],
    target_key: str,
    num_classes: int,
    require_label: bool,
    source_name: str,
) -> list[dict[str, Any]]:
    """Retain model inputs and identifiers, filtering unusable cases."""
    prepared: list[dict[str, Any]] = []
    skipped: list[str] = []
    for index, source in enumerate(items):
        missing_images = [key for key in image_keys if not _is_present(source.get(key))]
        label_present = _is_present(source.get(target_key))
        if missing_images or (require_label and not label_present):
            reasons = []
            if missing_images:
                reasons.append("missing " + ", ".join(missing_images))
            if require_label and not label_present:
                reasons.append(f"missing {target_key}")
            skipped.append(f"row {index}: {'; '.join(reasons)}")
            continue

        row = {key: source[key] for key in image_keys}
        row["patient_id"] = (
            str(source["patient_id"]) if _is_present(source.get("patient_id")) else ""
        )
        row["scan_id"] = (
            str(source["scan_id"]) if _is_present(source.get("scan_id")) else ""
        )
        row["has_label"] = int(label_present)
        row["true_label"] = (
            _parse_label(source[target_key], target_key, num_classes)
            if label_present
            else -1
        )
        prepared.append(row)

    if skipped:
        print(
            f"WARNING: skipped {len(skipped)} unusable {source_name} rows. "
            f"First entries: {' | '.join(skipped[:5])}"
        )
    if not prepared:
        raise ValueError(f"No usable cases remain in {source_name}.")
    return prepared


def _patient_key(value: Any) -> str:
    """Normalize patient identifiers for leakage detection."""
    text = str(value).strip().lower()
    if not text:
        return ""
    match = re.fullmatch(r"patient[-_ ]?0*(\d+)(?:\.0+)?", text)
    if match is None:
        match = re.fullmatch(r"0*(\d+)(?:\.0+)?", text)
    if match:
        return f"number:{int(match.group(1))}"
    return re.sub(r"[^a-z0-9]+", "", text)


def _assert_patient_disjoint(
    train_items: Sequence[dict[str, Any]],
    eval_items: Sequence[dict[str, Any]],
) -> None:
    """Reject reference/evaluation patient overlap that biases novelty scores."""
    train_patients = {
        _patient_key(item["patient_id"])
        for item in train_items
        if _patient_key(item["patient_id"])
    }
    eval_patients = {
        _patient_key(item["patient_id"])
        for item in eval_items
        if _patient_key(item["patient_id"])
    }
    overlap = train_patients & eval_patients
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(
            "Patient overlap between novelty reference and evaluation sets: " + examples
        )


def _classifier_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Locate MONAI DenseNet's final linear classifier layer."""
    class_layers = getattr(model, "class_layers", None)
    classifier = getattr(class_layers, "out", None)
    if not isinstance(classifier, torch.nn.Linear):
        raise ValueError("Could not locate DenseNet penultimate feature interface.")
    return classifier


@torch.inference_mode()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    include_predictions: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Capture classifier inputs as penultimate features during inference."""
    model.eval()
    captured: dict[str, torch.Tensor] = {}

    def capture_input(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        if not inputs:
            raise RuntimeError("Classifier hook received no feature tensor.")
        captured["features"] = inputs[0].detach()

    handle = _classifier_layer(model).register_forward_pre_hook(capture_input)
    feature_batches: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    try:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            captured.clear()
            logits = model(images)
            features = captured.get("features")
            if features is None:
                raise RuntimeError("DenseNet classifier hook did not capture features.")
            if features.ndim != 2 or features.shape[0] != images.shape[0]:
                raise ValueError(
                    "Expected penultimate features with shape [batch, features], "
                    f"received {tuple(features.shape)}."
                )
            if logits.ndim != 2 or logits.shape[1] != num_classes:
                raise ValueError(
                    f"Expected logits [batch, {num_classes}], "
                    f"received {tuple(logits.shape)}."
                )
            feature_batches.append(features.cpu().numpy().astype(np.float64, copy=False))

            if not include_predictions:
                continue
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            confidence = probabilities.max(dim=1).values
            entropy = -(
                probabilities
                * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
            ).sum(dim=1)

            batch_size = int(images.shape[0])
            patient_ids = _batch_values(batch, "patient_id", batch_size)
            scan_ids = _batch_values(batch, "scan_id", batch_size)
            has_labels = torch.as_tensor(batch["has_label"]).cpu().tolist()
            true_labels = torch.as_tensor(batch["true_label"]).cpu().tolist()
            predictions_cpu = predictions.cpu().tolist()
            confidence_cpu = confidence.cpu().tolist()
            entropy_cpu = entropy.cpu().tolist()
            for index in range(batch_size):
                rows.append(
                    {
                        "patient_id": str(patient_ids[index]),
                        "scan_id": str(scan_ids[index]),
                        "entropy": float(entropy_cpu[index]),
                        "confidence": float(confidence_cpu[index]),
                        "predicted_class": int(predictions_cpu[index]),
                        "true_label": (
                            int(true_labels[index]) if int(has_labels[index]) else None
                        ),
                    }
                )
    finally:
        handle.remove()

    if not feature_batches:
        raise ValueError("Feature loader produced no batches.")
    return np.concatenate(feature_batches, axis=0), rows


def fit_regularized_covariance(
    train_features: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit feature mean/covariance and return a stable Cholesky factor."""
    features = np.asarray(train_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("At least two two-dimensional training feature rows are required.")
    if not np.isfinite(features).all():
        raise ValueError("Training features contain NaN or infinite values.")
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("Covariance regularization must be finite and positive.")

    mean = features.mean(axis=0)
    centered = features - mean
    covariance = centered.T @ centered / float(features.shape[0] - 1)
    covariance = 0.5 * (covariance + covariance.T)
    variance_scale = float(np.trace(covariance) / covariance.shape[0])
    if not math.isfinite(variance_scale) or variance_scale <= 0:
        variance_scale = 1.0

    applied = regularization * variance_scale
    identity = np.eye(covariance.shape[0], dtype=np.float64)
    for _ in range(8):
        try:
            cholesky = np.linalg.cholesky(covariance + applied * identity)
            return mean, cholesky, applied
        except np.linalg.LinAlgError:
            applied *= 10.0
    raise ValueError("Could not regularize feature covariance to positive definiteness.")


def mahalanobis_distances(
    features: np.ndarray,
    mean: np.ndarray,
    covariance_cholesky: np.ndarray,
) -> np.ndarray:
    """Compute Mahalanobis distances using a Cholesky solve."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != mean.shape[0]:
        raise ValueError("Evaluation and reference feature dimensions do not match.")
    if not np.isfinite(values).all():
        raise ValueError("Evaluation features contain NaN or infinite values.")
    centered = values - mean
    whitened = np.linalg.solve(covariance_cholesky, centered.T).T
    return np.sqrt(np.sum(whitened * whitened, axis=1))


def run_novelty(args: argparse.Namespace) -> pd.DataFrame:
    """Extract features, fit the reference distribution, and score eval cases."""
    checkpoint = _load_checkpoint(Path(args.ckpt))
    model = _build_model(checkpoint, args.image_keys, args.num_classes)
    roi_size = _checkpoint_roi_size(checkpoint)

    train_items = _prepare_items(
        _load_datalist(Path(args.train_json)),
        args.image_keys,
        args.target_key,
        args.num_classes,
        require_label=True,
        source_name="training datalist",
    )
    eval_items = _prepare_items(
        _load_datalist(Path(args.eval_json)),
        args.image_keys,
        args.target_key,
        args.num_classes,
        require_label=False,
        source_name="evaluation datalist",
    )
    _assert_patient_disjoint(train_items, eval_items)

    transform = get_val_transforms(args.image_keys, roi_size)
    train_loader = DataLoader(
        Dataset(train_items, transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        Dataset(eval_items, transform=get_val_transforms(args.image_keys, roi_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(
        f"Extracting features on {device}: "
        f"train={len(train_items):,}, eval={len(eval_items):,}"
    )
    train_features, _ = extract_features(
        model, train_loader, device, args.num_classes, include_predictions=False
    )
    eval_features, rows = extract_features(
        model, eval_loader, device, args.num_classes, include_predictions=True
    )
    mean, cholesky, applied_regularization = fit_regularized_covariance(
        train_features,
        args.covariance_regularization,
    )
    distances = mahalanobis_distances(eval_features, mean, cholesky)
    for row, distance in zip(rows, distances, strict=True):
        row["novelty_distance"] = float(distance)

    include_true_label = any(row["true_label"] is not None for row in rows)
    columns = list(OUTPUT_COLUMNS)
    if include_true_label:
        columns.append("true_label")
    output = pd.DataFrame(rows, columns=columns)
    output_path = write_csv(output, args.out_csv)
    print(f"Applied covariance diagonal regularization: {applied_regularization:.6g}")
    print(
        "Novelty distance: "
        f"median={np.median(distances):.4f}, "
        f"p95={np.percentile(distances, 95):.4f}, "
        f"max={np.max(distances):.4f}"
    )
    print(f"Saved novelty scores to: {output_path}")
    return output


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    """Parse a finite positive floating-point CLI value."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse novelty-evaluation command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Estimate DenseNet feature-space novelty for prostate MRI cases."
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--train_json", type=Path, required=True)
    parser.add_argument("--eval_json", type=Path, required=True)
    parser.add_argument("--image_keys", nargs="+", required=True)
    parser.add_argument("--target_key", required=True)
    parser.add_argument("--num_classes", type=int, choices=(2, 3), required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--batch_size", type=_positive_int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--covariance_regularization",
        type=_positive_float,
        default=1e-3,
        help="Diagonal covariance regularization relative to mean feature variance.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    run_novelty(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
