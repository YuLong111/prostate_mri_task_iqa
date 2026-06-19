"""Train a ternary task-derived prostate MRI image-quality classifier."""

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
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import WeightedRandomSampler

from prostate_iqa.data.transforms import get_train_transforms, get_val_transforms
from prostate_iqa.models.densenet_quality import build_densenet121
from prostate_iqa.utils.io import ensure_dir, read_json, write_csv, write_json
from prostate_iqa.utils.losses import ordinal_ce_mae_loss
from prostate_iqa.utils.seed import set_global_seed


QUALITY_CLASSES = (0, 1, 2)
PREDICTION_COLUMNS = (
    "patient_id",
    "scan_id",
    "distortion_status",
    "acquisition_id",
    "true_quality",
    "pred_quality",
    "prob_reject",
    "prob_caution",
    "prob_accept",
    "expected_quality_score",
    "confidence",
    "entropy",
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


def _ternary_label(value: Any, target_key: str) -> int:
    """Parse reject/caution/accept labels into ordered integers."""
    if not _is_present(value):
        raise ValueError(f"Missing target {target_key!r}.")
    text = str(value).strip().lower()
    aliases = {
        "0": 0,
        "0.0": 0,
        "reject": 0,
        "1": 1,
        "1.0": 1,
        "caution": 1,
        "2": 2,
        "2.0": 2,
        "accept": 2,
    }
    if text not in aliases:
        raise ValueError(
            f"Target {target_key!r} must be 0/1/2 or "
            f"reject/caution/accept, received {value!r}."
        )
    return aliases[text]


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
        "test_locked" in name or re.search(r"(?:^|[_-])test(?:[_\-.]|$)", name)
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
    """Validate targets and filter records missing required inputs."""
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
        missing_target = not _is_present(source.get(target_key))
        if missing_keys or ambiguous_keys or missing_target:
            reasons = []
            if missing_keys:
                reasons.append("missing " + ", ".join(missing_keys))
            if ambiguous_keys:
                reasons.append(
                    "multiple acquisitions in " + ", ".join(ambiguous_keys)
                )
            if missing_target:
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
        item["label"] = _ternary_label(source[target_key], target_key)
        prepared.append(item)

    if skipped:
        print(
            f"WARNING: skipped {len(skipped)} unusable {source_name} rows. "
            f"First entries: {' | '.join(skipped[:5])}"
        )
    if not prepared:
        raise ValueError(f"No usable labeled rows remain in {source_name}.")
    return prepared


def _patient_key(value: Any) -> str:
    """Normalize patient identifiers before leakage checks."""
    text = str(value).strip().lower()
    match = re.fullmatch(r"patient[-_ ]?0*(\d+)(?:\.0+)?", text)
    if match is None:
        match = re.fullmatch(r"0*(\d+)(?:\.0+)?", text)
    if match:
        return f"number:{int(match.group(1))}"
    return re.sub(r"[^a-z0-9]+", "", text)


def _assert_patient_disjoint(
    train_items: Sequence[dict[str, Any]],
    val_items: Sequence[dict[str, Any]],
) -> None:
    """Prevent patient leakage between training and validation."""
    train_patients = {
        _patient_key(item["patient_id"])
        for item in train_items
        if _is_present(item.get("patient_id"))
    }
    val_patients = {
        _patient_key(item["patient_id"])
        for item in val_items
        if _is_present(item.get("patient_id"))
    }
    overlap = train_patients & val_patients
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"Patient leakage between train and validation: {examples}")


def _class_counts(
    items: Sequence[dict[str, Any]],
    require_all: bool,
) -> dict[int, int]:
    """Count ordered labels and optionally require all three classes."""
    counts = pd.Series([int(item["label"]) for item in items]).value_counts()
    result = {label: int(counts.get(label, 0)) for label in QUALITY_CLASSES}
    if require_all and min(result.values()) == 0:
        raise ValueError(
            "Training requires reject, caution, and accept examples; "
            f"received counts {result}."
        )
    return result


def _weighted_sampler(
    items: Sequence[dict[str, Any]],
    seed: int,
) -> WeightedRandomSampler | None:
    """Balance unequal training classes using inverse-frequency sampling."""
    counts = _class_counts(items, require_all=True)
    if len(set(counts.values())) == 1:
        return None
    labels = [int(item["label"]) for item in items]
    weights = torch.as_tensor(
        [1.0 / counts[label] for label in labels],
        dtype=torch.double,
    )
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def _batch_strings(batch: dict[str, Any], key: str, count: int) -> list[str]:
    """Extract identifiers from a collated MONAI batch."""
    values = batch.get(key)
    if values is None:
        return [""] * count
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values]
    if isinstance(values, torch.Tensor):
        return [str(value) for value in values.detach().cpu().tolist()]
    return [str(values)] * count


def calculate_ternary_metrics(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
) -> dict[str, Any]:
    """Calculate categorical and ordinal validation metrics."""
    truth = np.asarray(true_labels, dtype=int)
    prediction = np.asarray(predicted_labels, dtype=int)
    if truth.size == 0 or prediction.shape != truth.shape:
        raise ValueError("Metric inputs must be non-empty arrays of equal shape.")
    if not np.isin(truth, QUALITY_CLASSES).all() or not np.isin(
        prediction, QUALITY_CLASSES
    ).all():
        raise ValueError("Metric labels must be ternary values 0, 1, or 2.")

    matrix = confusion_matrix(truth, prediction, labels=QUALITY_CLASSES)
    if np.unique(np.concatenate((truth, prediction))).size < 2:
        quadratic_kappa = float("nan")
    else:
        quadratic_kappa = float(
            cohen_kappa_score(
                truth,
                prediction,
                labels=list(QUALITY_CLASSES),
                weights="quadratic",
            )
        )
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                labels=list(QUALITY_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "quadratic_weighted_kappa": quadratic_kappa,
        "ordinal_mae": float(np.mean(np.abs(truth - prediction))),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate the model and create one prediction row per scan."""
    model.eval()
    rows: list[dict[str, Any]] = []
    class_scores = torch.arange(3, dtype=torch.float32, device=device)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
        probabilities = torch.softmax(model(images), dim=1)
        predictions = probabilities.argmax(dim=1)
        expected_scores = (probabilities * class_scores.unsqueeze(0)).sum(dim=1)
        entropies = -(
            probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        ).sum(dim=1)

        labels_cpu = labels.detach().cpu().numpy()
        probabilities_cpu = probabilities.detach().cpu().numpy()
        predictions_cpu = predictions.detach().cpu().numpy()
        expected_cpu = expected_scores.detach().cpu().numpy()
        entropy_cpu = entropies.detach().cpu().numpy()
        patient_ids = _batch_strings(batch, "patient_id", len(labels_cpu))
        scan_ids = _batch_strings(batch, "scan_id", len(labels_cpu))
        distortion_statuses = _batch_strings(
            batch, "distortion_status", len(labels_cpu)
        )
        acquisition_ids = _batch_strings(batch, "acquisition_id", len(labels_cpu))
        for index, true_quality in enumerate(labels_cpu):
            probabilities_row = probabilities_cpu[index]
            rows.append(
                {
                    "patient_id": patient_ids[index],
                    "scan_id": scan_ids[index],
                    "distortion_status": distortion_statuses[index],
                    "acquisition_id": acquisition_ids[index],
                    "true_quality": int(true_quality),
                    "pred_quality": int(predictions_cpu[index]),
                    "prob_reject": float(probabilities_row[0]),
                    "prob_caution": float(probabilities_row[1]),
                    "prob_accept": float(probabilities_row[2]),
                    "expected_quality_score": float(expected_cpu[index]),
                    "confidence": float(probabilities_row.max()),
                    "entropy": float(entropy_cpu[index]),
                }
            )

    predictions_frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    metrics = calculate_ternary_metrics(
        predictions_frame["true_quality"].tolist(),
        predictions_frame["pred_quality"].tolist(),
    )
    return metrics, predictions_frame


def _checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build a self-describing training checkpoint."""
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": metrics,
        "image_keys": list(args.image_keys),
        "target_key": args.target_key,
        "roi_size": list(args.roi_size),
        "lambda_ord": args.lambda_ord,
        "selection_metric": args.selection_metric,
        "seed": args.seed,
    }


def _print_metrics(epoch: int, train_loss: float, metrics: dict[str, Any]) -> None:
    """Print requested validation metrics and the fixed-order confusion matrix."""
    scalar_keys = (
        "accuracy",
        "macro_f1",
        "quadratic_weighted_kappa",
        "ordinal_mae",
    )
    values = " | ".join(
        f"{key}={float(metrics[key]):.4f}" for key in scalar_keys
    )
    print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | {values}")
    print("Confusion matrix (rows=true, columns=pred; reject/caution/accept):")
    for row in metrics["confusion_matrix"]:
        print("  " + " ".join(f"{int(value):6d}" for value in row))


def _history_row(
    epoch: int,
    train_loss: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Flatten validation metrics for a tidy training-history CSV."""
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
        "ordinal_mae": metrics["ordinal_mae"],
    }
    matrix = metrics["confusion_matrix"]
    for true_label in QUALITY_CLASSES:
        for predicted_label in QUALITY_CLASSES:
            row[f"cm_{true_label}_{predicted_label}"] = matrix[true_label][predicted_label]
    return row


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Train a ternary IQA model and return its best validation metrics."""
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
    train_counts = _class_counts(train_items, require_all=True)
    val_counts = _class_counts(val_items, require_all=False)
    print(f"Training class counts: {train_counts}")
    print(f"Validation class counts: {val_counts}")
    if min(val_counts.values()) == 0:
        print("WARNING: validation does not contain every ternary quality class.")

    train_transforms = get_train_transforms(args.image_keys, args.roi_size)
    train_transforms.set_random_state(seed=args.seed)
    val_transforms = get_val_transforms(args.image_keys, args.roi_size)
    train_dataset = Dataset(train_items, transform=train_transforms)
    val_dataset = Dataset(val_items, transform=val_transforms)

    sampler = _weighted_sampler(train_items, args.seed)
    if sampler is not None:
        print("Using inverse-frequency WeightedRandomSampler for class imbalance.")
    drop_singleton = args.batch_size > 1 and len(train_items) % args.batch_size == 1
    if drop_singleton:
        print("Dropping the final singleton training batch for BatchNorm stability.")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(args.seed),
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
    model = build_densenet121(len(args.image_keys), out_channels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_score = -math.inf
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        observed = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = ordinal_ce_mae_loss(logits, labels, lambda_ord=args.lambda_ord)
            loss.backward()
            optimizer.step()
            batch_count = int(labels.shape[0])
            running_loss += float(loss.detach()) * batch_count
            observed += batch_count

        train_loss = running_loss / max(observed, 1)
        val_metrics, val_predictions = evaluate(model, val_loader, device)
        _print_metrics(epoch, train_loss, val_metrics)
        history.append(_history_row(epoch, train_loss, val_metrics))

        selection_score = float(val_metrics[args.selection_metric])
        improved = best_epoch == 0 or (
            math.isfinite(selection_score) and selection_score > best_score
        )
        if improved:
            best_epoch = epoch
            if math.isfinite(selection_score):
                best_score = selection_score
            best_metrics = dict(val_metrics)
            torch.save(
                _checkpoint(model, optimizer, epoch, val_metrics, args),
                output_dir / "best.pt",
            )
            write_csv(val_predictions, output_dir / "val_predictions.csv")
            write_json(
                {"epoch": epoch, **val_metrics},
                output_dir / "best_val_metrics.json",
            )

    final_metrics, _ = evaluate(model, val_loader, device)
    torch.save(
        _checkpoint(model, optimizer, args.epochs, final_metrics, args),
        output_dir / "last.pt",
    )
    write_csv(pd.DataFrame(history), output_dir / "training_history.csv")
    score_text = f"{best_score:.4f}" if math.isfinite(best_score) else "undefined"
    print(f"Best validation {args.selection_metric}: {score_text} (epoch {best_epoch})")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a ternary task-derived prostate MRI IQA classifier."
    )
    parser.add_argument("--train_json", type=Path, required=True)
    parser.add_argument("--val_json", type=Path, required=True)
    parser.add_argument("--image_keys", nargs="+", required=True)
    parser.add_argument("--target_key", default="quality_ternary")
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
    parser.add_argument("--lambda_ord", type=_nonnegative_float, default=0.5)
    parser.add_argument(
        "--selection_metric",
        choices=("macro_f1", "quadratic_weighted_kappa"),
        default="macro_f1",
    )
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
