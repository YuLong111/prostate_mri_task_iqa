"""Evaluate a trained binary or ternary prostate MRI classifier."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from prostate_iqa.data.transforms import get_val_transforms
from prostate_iqa.models.densenet_quality import build_densenet121
from prostate_iqa.utils.io import read_json, write_csv, write_json


METADATA_COLUMNS = (
    "patient_id",
    "scan_id",
    "distortion_status",
    "acquisition_id",
    "acquisition_index",
    "site",
    "vendor",
    "b_value",
    "field_strength",
)
STATE_DICT_KEYS = (
    "model_state_dict",
    "state_dict",
    "model_state",
    "model",
    "network",
    "net",
)


def _is_present(value: Any) -> bool:
    """Return whether a datalist value is non-missing and non-empty."""
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def _load_datalist(path: Path) -> list[dict[str, Any]]:
    """Read list-style and common MONAI-style JSON datalists."""
    payload = read_json(path)
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = next(
            (
                payload[key]
                for key in ("data", "training", "validation", "test")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
        if items is None:
            raise ValueError(f"No case list found in datalist: {path}")
    else:
        raise ValueError(f"Datalist must contain a list or mapping: {path}")
    if not all(isinstance(item, dict) for item in items):
        raise ValueError(f"Every datalist item must be an object: {path}")
    return items


def _parse_label(value: Any, target_key: str, num_classes: int) -> int:
    """Parse binary or reject/caution/accept labels."""
    if not _is_present(value):
        raise ValueError(f"Missing target {target_key!r}.")
    text = str(value).strip().lower()
    aliases: dict[str, int] = {
        "0": 0,
        "0.0": 0,
        "false": 0,
        "no": 0,
        "negative": 0,
        "reject": 0,
        "1": 1,
        "1.0": 1,
        "true": 1,
        "yes": 1,
        "positive": 1,
    }
    if num_classes == 3:
        aliases.update({"caution": 1, "2": 2, "2.0": 2, "accept": 2})
    if text not in aliases or aliases[text] >= num_classes:
        expected = "0/1" if num_classes == 2 else "0/1/2"
        raise ValueError(
            f"Target {target_key!r} must be {expected}, received {value!r}."
        )
    return aliases[text]


def _prepare_items(
    items: Sequence[dict[str, Any]],
    image_keys: Sequence[str],
    target_key: str,
    num_classes: int,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Filter unusable rows and retain only model inputs plus requested metadata."""
    metadata_columns = tuple(
        column
        for column in METADATA_COLUMNS
        if any(_is_present(item.get(column)) for item in items)
    )
    prepared: list[dict[str, Any]] = []
    skipped: list[str] = []
    for index, source in enumerate(items):
        missing_images = [key for key in image_keys if not _is_present(source.get(key))]
        ambiguous_images = [
            key for key in image_keys if ";" in str(source.get(key) or "")
        ]
        if missing_images or ambiguous_images or not _is_present(source.get(target_key)):
            reasons = []
            if missing_images:
                reasons.append("missing " + ", ".join(missing_images))
            if ambiguous_images:
                reasons.append(
                    "multiple acquisitions in " + ", ".join(ambiguous_images)
                )
            if not _is_present(source.get(target_key)):
                reasons.append(f"missing {target_key}")
            skipped.append(f"row {index}: {'; '.join(reasons)}")
            continue

        row = {key: source[key] for key in image_keys}
        row["label"] = _parse_label(source[target_key], target_key, num_classes)
        for column in metadata_columns:
            value = source.get(column)
            # A consistent string representation keeps optional numeric
            # metadata collatable when other cases have missing values.
            row[column] = str(value) if _is_present(value) else ""
        prepared.append(row)

    if skipped:
        print(
            f"WARNING: skipped {len(skipped)} unevaluable rows. "
            f"First entries: {' | '.join(skipped[:5])}"
        )
    if not prepared:
        raise ValueError("No evaluable labeled cases remain after input validation.")
    return prepared, metadata_columns


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    """Load a local PyTorch checkpoint on CPU."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint must contain a mapping: {path}")
    return payload


def _strip_prefixes(state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Extract tensor weights and remove common training-wrapper prefixes."""
    prefixes = ("module.", "model.", "network.", "net.", "_orig_mod.")
    result: dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
                    break
        result[key] = value
    return result


def _extract_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Extract a DenseNet state dictionary from common checkpoint layouts."""
    candidate: Mapping[str, Any] = checkpoint
    for key in STATE_DICT_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            candidate = value
            break
    state_dict = _strip_prefixes(candidate)
    if not state_dict:
        raise ValueError("Checkpoint does not contain a tensor model state dictionary.")
    return state_dict


def _find_weight(
    state_dict: Mapping[str, torch.Tensor],
    suffix: str,
) -> torch.Tensor:
    """Find exactly one checkpoint tensor ending with a known model-key suffix."""
    matches = [value for key, value in state_dict.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one checkpoint tensor ending in {suffix!r}, found {len(matches)}."
        )
    return matches[0]


def _build_model(
    checkpoint: Mapping[str, Any],
    image_keys: Sequence[str],
    num_classes: int,
) -> torch.nn.Module:
    """Build DenseNet121 and load an exactly compatible evaluation state."""
    checkpoint_keys = checkpoint.get("image_keys")
    if checkpoint_keys is None and isinstance(checkpoint.get("config"), Mapping):
        checkpoint_keys = checkpoint["config"].get("image_keys")
    if checkpoint_keys is not None and list(checkpoint_keys) != list(image_keys):
        raise ValueError(
            "--image_keys must match checkpoint channel order exactly: "
            f"checkpoint={list(checkpoint_keys)}, supplied={list(image_keys)}."
        )

    state_dict = _extract_state_dict(checkpoint)
    first_conv = _find_weight(state_dict, "features.conv0.weight")
    classifier = _find_weight(state_dict, "class_layers.out.weight")
    if first_conv.ndim != 5 or int(first_conv.shape[1]) != len(image_keys):
        raise ValueError(
            f"Checkpoint expects {int(first_conv.shape[1])} input channels, "
            f"but {len(image_keys)} image keys were supplied."
        )
    if classifier.ndim != 2 or int(classifier.shape[0]) != num_classes:
        raise ValueError(
            f"Checkpoint classifier has {int(classifier.shape[0])} outputs, "
            f"but --num_classes={num_classes}."
        )

    model = build_densenet121(len(image_keys), num_classes)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Checkpoint is incompatible with DenseNet121: {error}") from error
    return model


def _checkpoint_roi_size(checkpoint: Mapping[str, Any]) -> tuple[int, int, int]:
    """Read preprocessing ROI size from current or legacy checkpoint metadata."""
    value = checkpoint.get("roi_size")
    if value is None and isinstance(checkpoint.get("config"), Mapping):
        value = checkpoint["config"].get("roi_size")
    if value is None:
        value = (160, 160, 64)
        print("WARNING: checkpoint has no roi_size; using (160, 160, 64).")
    try:
        roi_size = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid roi_size in checkpoint: {value!r}") from error
    if len(roi_size) != 3 or any(item <= 0 for item in roi_size):
        raise ValueError(f"Checkpoint roi_size must contain three positive values: {value}")
    return roi_size


def calculate_binary_metrics(
    true_labels: Sequence[int],
    probabilities_1: Sequence[float],
    predicted_labels: Sequence[int],
) -> dict[str, Any]:
    """Calculate requested binary discrimination and classification metrics."""
    truth = np.asarray(true_labels, dtype=int)
    probability = np.asarray(probabilities_1, dtype=float)
    prediction = np.asarray(predicted_labels, dtype=int)
    if truth.size == 0 or probability.shape != truth.shape or prediction.shape != truth.shape:
        raise ValueError("Binary metric inputs must be non-empty arrays of equal shape.")
    if not np.isin(truth, (0, 1)).all() or not np.isin(prediction, (0, 1)).all():
        raise ValueError("Binary metric labels must be 0 or 1.")

    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    both_classes = np.unique(truth).size == 2
    return {
        "auc": float(roc_auc_score(truth, probability)) if both_classes else None,
        "pr_auc": (
            float(average_precision_score(truth, probability))
            if both_classes
            else None
        ),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def calculate_ternary_metrics(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
) -> dict[str, Any]:
    """Calculate requested ternary categorical and ordinal metrics."""
    truth = np.asarray(true_labels, dtype=int)
    prediction = np.asarray(predicted_labels, dtype=int)
    if truth.size == 0 or prediction.shape != truth.shape:
        raise ValueError("Ternary metric inputs must be non-empty arrays of equal shape.")
    if not np.isin(truth, (0, 1, 2)).all() or not np.isin(
        prediction, (0, 1, 2)
    ).all():
        raise ValueError("Ternary metric labels must be 0, 1, or 2.")

    matrix = confusion_matrix(truth, prediction, labels=[0, 1, 2])
    variable = np.unique(np.concatenate((truth, prediction))).size >= 2
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                truth,
                prediction,
                labels=[0, 1, 2],
                average="weighted",
                zero_division=0,
            )
        ),
        "quadratic_weighted_kappa": (
            float(
                cohen_kappa_score(
                    truth,
                    prediction,
                    labels=[0, 1, 2],
                    weights="quadratic",
                )
            )
            if variable
            else None
        ),
        "ordinal_mae": float(np.mean(np.abs(truth - prediction))),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _batch_values(batch: Mapping[str, Any], key: str, count: int) -> list[Any]:
    """Convert collated metadata into one scalar value per batch row."""
    values = batch.get(key)
    if values is None:
        return [""] * count
    if isinstance(values, torch.Tensor):
        converted = values.detach().cpu().tolist()
        return converted if isinstance(converted, list) else [converted] * count
    if isinstance(values, np.ndarray):
        converted = values.tolist()
        return converted if isinstance(converted, list) else [converted] * count
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values] * count


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    metadata_columns: Sequence[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run inference and return aggregate metrics plus per-case predictions."""
    model.eval()
    rows: list[dict[str, Any]] = []
    ordinal_scores = torch.arange(num_classes, dtype=torch.float32, device=device)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
        logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != num_classes:
            raise ValueError(
                f"Model returned logits with shape {tuple(logits.shape)}; expected "
                f"[batch, {num_classes}]."
            )
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        confidence = probabilities.max(dim=1).values
        entropy = -(
            probabilities
            * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        ).sum(dim=1)
        expected = (probabilities * ordinal_scores.unsqueeze(0)).sum(dim=1)

        labels_cpu = labels.cpu().numpy()
        probabilities_cpu = probabilities.cpu().numpy()
        predictions_cpu = predictions.cpu().numpy()
        confidence_cpu = confidence.cpu().numpy()
        entropy_cpu = entropy.cpu().numpy()
        expected_cpu = expected.cpu().numpy()
        metadata = {
            column: _batch_values(batch, column, len(labels_cpu))
            for column in metadata_columns
        }
        for index, true_label in enumerate(labels_cpu):
            row = {column: metadata[column][index] for column in metadata_columns}
            row.update(
                {
                    "true_label": int(true_label),
                    "pred_label": int(predictions_cpu[index]),
                }
            )
            if num_classes == 2:
                row["prob_0"] = float(probabilities_cpu[index, 0])
                row["prob_1"] = float(probabilities_cpu[index, 1])
            else:
                row["prob_reject"] = float(probabilities_cpu[index, 0])
                row["prob_caution"] = float(probabilities_cpu[index, 1])
                row["prob_accept"] = float(probabilities_cpu[index, 2])
            row["confidence"] = float(confidence_cpu[index])
            row["entropy"] = float(entropy_cpu[index])
            if num_classes == 3:
                row["expected_ordinal_score"] = float(expected_cpu[index])
            rows.append(row)

    predictions = pd.DataFrame(rows)
    if num_classes == 2:
        metrics = calculate_binary_metrics(
            predictions["true_label"].tolist(),
            predictions["prob_1"].tolist(),
            predictions["pred_label"].tolist(),
        )
    else:
        metrics = calculate_ternary_metrics(
            predictions["true_label"].tolist(),
            predictions["pred_label"].tolist(),
        )
    metrics["num_cases"] = len(predictions)
    metrics["num_classes"] = num_classes
    return metrics, predictions


def _print_metrics(metrics: Mapping[str, Any]) -> None:
    """Print scalar metrics followed by a labeled confusion matrix."""
    print("Evaluation metrics:")
    for key, value in metrics.items():
        if key == "confusion_matrix":
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("  confusion_matrix (rows=true, columns=pred):")
    for row in metrics["confusion_matrix"]:
        print("    " + " ".join(f"{int(value):6d}" for value in row))


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Load checkpoint and datalist, run inference, and save requested outputs."""
    checkpoint = _load_checkpoint(Path(args.ckpt))
    model = _build_model(checkpoint, args.image_keys, args.num_classes)
    roi_size = _checkpoint_roi_size(checkpoint)

    raw_items = _load_datalist(Path(args.datalist_json))
    items, metadata_columns = _prepare_items(
        raw_items,
        args.image_keys,
        args.target_key,
        args.num_classes,
    )
    dataset = Dataset(
        items,
        transform=get_val_transforms(args.image_keys, roi_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {len(items):,} cases on device: {device}")
    model = model.to(device)
    metrics, predictions = evaluate_model(
        model,
        loader,
        device,
        args.num_classes,
        metadata_columns,
    )
    prediction_path = write_csv(predictions, args.out_csv)
    metrics_path = write_json(metrics, args.out_metrics_json)
    _print_metrics(metrics)
    print(f"Saved predictions to: {prediction_path}")
    print(f"Saved metrics to: {metrics_path}")
    return metrics


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse evaluation command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a binary or ternary prostate MRI model."
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--datalist_json", type=Path, required=True)
    parser.add_argument("--image_keys", nargs="+", required=True)
    parser.add_argument("--target_key", required=True)
    parser.add_argument("--num_classes", type=int, choices=(2, 3), required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--out_metrics_json", type=Path, required=True)
    parser.add_argument("--batch_size", type=_positive_int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
