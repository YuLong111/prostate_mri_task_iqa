"""Generate patient-level out-of-fold downstream task predictions."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from prostate_iqa.data.transforms import get_train_transforms, get_val_transforms
from prostate_iqa.models.densenet_quality import build_densenet121
from prostate_iqa.training.train_binary_task import (
    _batch_strings,
    _class_counts,
    _load_datalist,
    _make_binary_criterion,
    _prepare_items,
    _weighted_sampler,
)
from prostate_iqa.utils.io import ensure_dir, write_csv, write_json
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
    "fold",
    "source_index",
)


def _is_present(value: Any) -> bool:
    """Return whether a datalist scalar is non-missing and non-empty."""
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def _patient_key(value: Any) -> str:
    """Normalize patient IDs so formatting differences cannot leak across folds."""
    if not _is_present(value):
        return ""
    text = str(value).strip().lower()
    match = re.fullmatch(r"patient[-_ ]?0*(\d+)(?:\.0+)?", text)
    if match is None:
        match = re.fullmatch(r"0*(\d+)(?:\.0+)?", text)
    if match:
        return f"number:{int(match.group(1))}"
    return "text:" + re.sub(r"[^a-z0-9]+", "", text)


def _reject_nontraining_input(
    path: Path,
    items: Sequence[dict[str, Any]],
) -> None:
    """Prevent validation or locked-test rows from entering OOF generation."""
    name = path.name.lower()
    test_name = bool(
        "test_locked" in name
        or re.search(r"(?:^|[_-])test(?:[_\-.]|$)", name)
    )
    validation_name = bool(re.search(r"(?:^|[_-])val(?:[_\-.]|$)", name))
    if test_name or validation_name:
        raise ValueError(f"OOF input must be a training datalist, received: {path}")
    nontraining = [
        str(item.get("split", "")).strip().lower()
        for item in items
        if _is_present(item.get("split"))
        and str(item.get("split", "")).strip().lower() not in {"train", "training"}
    ]
    if nontraining:
        counts = dict(Counter(nontraining))
        raise ValueError(
            "OOF datalist contains non-training split rows: " + str(counts)
        )


def _patient_groups(
    items: Sequence[dict[str, Any]],
) -> tuple[list[str], np.ndarray, dict[str, list[int]], dict[str, str]]:
    """Group row indices by patient and derive labels for stratification."""
    rows_by_patient: dict[str, list[int]] = defaultdict(list)
    display_ids: dict[str, str] = {}
    for index, item in enumerate(items):
        key = _patient_key(item.get("patient_id"))
        if not key:
            raise ValueError(f"Training row {index} is missing patient_id.")
        rows_by_patient[key].append(index)
        display_ids.setdefault(key, str(item["patient_id"]))

    patient_keys = sorted(rows_by_patient)
    patient_labels: list[int] = []
    mixed_patients: list[str] = []
    for key in patient_keys:
        labels = [int(items[index]["label"]) for index in rows_by_patient[key]]
        counts = Counter(labels)
        if len(counts) > 1:
            mixed_patients.append(display_ids[key])
        # Majority label is used only to balance fold assignment. Every row
        # retains its own target for training and evaluation.
        patient_labels.append(max(counts, key=lambda label: (counts[label], label)))
    if mixed_patients:
        print(
            f"WARNING: {len(mixed_patients)} patients have mixed row labels; "
            "using majority patient labels for fold stratification only."
        )
    return patient_keys, np.asarray(patient_labels), rows_by_patient, display_ids


def _fold_indices(
    patient_keys: Sequence[str],
    patient_labels: np.ndarray,
    num_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic patient-level folds, stratifying when feasible."""
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    if num_folds > len(patient_keys):
        raise ValueError(
            f"num_folds={num_folds} exceeds {len(patient_keys)} unique patients."
        )
    class_counts = Counter(patient_labels.tolist())
    can_stratify = len(class_counts) == 2 and min(class_counts.values()) >= num_folds
    indices = np.arange(len(patient_keys))
    if can_stratify:
        print(f"Using stratified {num_folds}-fold patient splitting: {dict(class_counts)}")
        splitter = StratifiedKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(indices, patient_labels))

    print(
        "WARNING: patient-level stratification is not feasible for class counts "
        f"{dict(class_counts)}; using shuffled K-fold splitting."
    )
    splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    return list(splitter.split(indices))


def _expand_patient_indices(
    patient_positions: Sequence[int],
    patient_keys: Sequence[str],
    rows_by_patient: dict[str, list[int]],
) -> list[int]:
    """Expand patient positions into source-row indices."""
    return [
        row_index
        for patient_position in patient_positions
        for row_index in rows_by_patient[patient_keys[int(patient_position)]]
    ]


def _make_loaders(
    train_items: Sequence[dict[str, Any]],
    holdout_items: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    fold_seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Create shared-transform training and deterministic holdout loaders."""
    train_transform = get_train_transforms(args.image_keys, args.roi_size)
    train_transform.set_random_state(seed=fold_seed)
    holdout_transform = get_val_transforms(args.image_keys, args.roi_size)
    imbalance_strategy = getattr(args, "imbalance_strategy", "sampler")
    sampler = (
        _weighted_sampler(train_items, fold_seed)
        if imbalance_strategy == "sampler"
        else None
    )
    if sampler is not None:
        print("  Using inverse-frequency WeightedRandomSampler.")
    elif imbalance_strategy == "class_weight":
        print("  Using inverse-frequency class weights.")
    elif imbalance_strategy == "none":
        print("  Using unweighted random sampling/loss.")
    drop_singleton = args.batch_size > 1 and len(train_items) % args.batch_size == 1
    if drop_singleton:
        print("  Dropping the final singleton training batch for BatchNorm stability.")
    train_loader = DataLoader(
        Dataset(list(train_items), transform=train_transform),
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(fold_seed),
        drop_last=drop_singleton,
    )
    holdout_loader = DataLoader(
        Dataset(list(holdout_items), transform=holdout_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, holdout_loader


def _fold_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    fold: int,
    train_loss: float,
    train_patients: Sequence[str],
    holdout_patients: Sequence[str],
) -> dict[str, Any]:
    """Build a self-describing fixed-budget fold checkpoint."""
    return {
        "epoch": args.epochs,
        "fold": fold,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "image_keys": list(args.image_keys),
        "target_key": args.target_key,
        "roi_size": list(args.roi_size),
        "seed": args.seed + fold,
        "dropout_prob": float(getattr(args, "dropout_prob", 0.0)),
        "weight_decay": float(getattr(args, "weight_decay", 0.0)),
        "label_smoothing": float(getattr(args, "label_smoothing", 0.0)),
        "imbalance_strategy": getattr(args, "imbalance_strategy", "sampler"),
        "scheduler": "cosine_annealing_lr",
        "train_patient_ids": list(train_patients),
        "holdout_patient_ids": list(holdout_patients),
        "selection_basis": "fixed_epoch_budget_without_holdout_tuning",
    }


def _train_fold(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    fold: int,
    train_items: Sequence[dict[str, Any]],
) -> tuple[torch.optim.Optimizer, float, list[dict[str, Any]]]:
    """Train one fold without observing holdout labels or predictions."""
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
    history: list[dict[str, Any]] = []
    final_loss = float("nan")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        observed = 0
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            count = int(labels.shape[0])
            running_loss += float(loss.detach()) * count
            observed += count
        final_loss = running_loss / max(observed, 1)
        history.append({"fold": fold, "epoch": epoch, "train_loss": final_loss})
        print(
            f"  Fold {fold} | epoch {epoch:03d}/{args.epochs:03d} "
            f"| train_loss={final_loss:.4f}"
        )
        scheduler.step()
    return optimizer, final_loss, history


@torch.inference_mode()
def _predict_holdout(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    fold: int,
) -> pd.DataFrame:
    """Predict a held-out fold exactly once after training is complete."""
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
        probabilities = torch.softmax(model(images), dim=1)
        predictions = probabilities.argmax(dim=1)

        labels_cpu = labels.cpu().tolist()
        probabilities_cpu = probabilities.cpu().tolist()
        predictions_cpu = predictions.cpu().tolist()
        patient_ids = _batch_strings(batch, "patient_id", len(labels_cpu))
        scan_ids = _batch_strings(batch, "scan_id", len(labels_cpu))
        distortion_statuses = _batch_strings(
            batch, "distortion_status", len(labels_cpu)
        )
        acquisition_ids = _batch_strings(batch, "acquisition_id", len(labels_cpu))
        source_indices = torch.as_tensor(batch["source_index"]).cpu().tolist()
        for index, true_label in enumerate(labels_cpu):
            prob_0, prob_1 = probabilities_cpu[index]
            predicted = int(predictions_cpu[index])
            rows.append(
                {
                    "patient_id": patient_ids[index],
                    "scan_id": scan_ids[index],
                    "distortion_status": distortion_statuses[index],
                    "acquisition_id": acquisition_ids[index],
                    "true_label": int(true_label),
                    "pred_label": predicted,
                    "prob_0": float(prob_0),
                    "prob_1": float(prob_1),
                    "confidence": float(max(prob_0, prob_1)),
                    "correct": int(predicted == int(true_label)),
                    "fold": fold,
                    "source_index": int(source_indices[index]),
                }
            )
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def _fold_metrics(predictions: pd.DataFrame, fold: int) -> dict[str, Any]:
    """Calculate fold accuracy and AUC when the holdout contains both classes."""
    truth = predictions["true_label"].astype(int)
    auc = (
        float(roc_auc_score(truth, predictions["prob_1"]))
        if truth.nunique() == 2
        else None
    )
    return {
        "fold": fold,
        "n_cases": len(predictions),
        "n_patients": predictions["patient_id"].nunique(),
        "auc": auc,
        "accuracy": float(accuracy_score(truth, predictions["pred_label"])),
    }


def run_oof(args: argparse.Namespace) -> dict[str, Any]:
    """Train all patient folds and save one prediction per training row."""
    set_global_seed(args.seed)
    set_determinism(seed=args.seed)
    output_dir = ensure_dir(args.out_dir)
    train_path = Path(args.train_json)
    raw_items = _load_datalist(train_path)
    _reject_nontraining_input(train_path, raw_items)
    items = _prepare_items(
        raw_items,
        args.image_keys,
        args.target_key,
        "OOF training datalist",
    )
    for source_index, item in enumerate(items):
        item["source_index"] = source_index
    global_counts = _class_counts(items)
    print(f"Usable training rows: {len(items):,}; class counts: {global_counts}")

    patient_keys, patient_labels, rows_by_patient, display_ids = _patient_groups(items)
    splits = _fold_indices(patient_keys, patient_labels, args.num_folds, args.seed)
    assignment_rows: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    all_history: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    seen_holdout_patients: set[str] = set()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"OOF training device: {device}")

    for fold, (train_positions, holdout_positions) in enumerate(splits):
        fold_seed = args.seed + fold
        set_global_seed(fold_seed)
        train_patient_keys = [patient_keys[int(index)] for index in train_positions]
        holdout_patient_keys = [patient_keys[int(index)] for index in holdout_positions]
        if set(train_patient_keys) & set(holdout_patient_keys):
            raise AssertionError(f"Patient leakage detected within fold {fold}.")
        if seen_holdout_patients & set(holdout_patient_keys):
            raise AssertionError("A patient appears in multiple held-out folds.")
        seen_holdout_patients.update(holdout_patient_keys)

        train_indices = _expand_patient_indices(
            train_positions, patient_keys, rows_by_patient
        )
        holdout_indices = _expand_patient_indices(
            holdout_positions, patient_keys, rows_by_patient
        )
        fold_train = [items[index] for index in train_indices]
        fold_holdout = [items[index] for index in holdout_indices]
        train_counts = _class_counts(fold_train)
        holdout_counts = {
            label: sum(int(item["label"]) == label for item in fold_holdout)
            for label in (0, 1)
        }
        print(
            f"Fold {fold}: train patients={len(train_patient_keys)}, "
            f"holdout patients={len(holdout_patient_keys)}, "
            f"train counts={train_counts}, holdout counts={holdout_counts}"
        )
        for key in holdout_patient_keys:
            assignment_rows.append(
                {"patient_id": display_ids[key], "patient_key": key, "fold": fold}
            )

        train_loader, holdout_loader = _make_loaders(
            fold_train, fold_holdout, args, fold_seed
        )
        model = build_densenet121(
            len(args.image_keys),
            out_channels=2,
            dropout_prob=float(getattr(args, "dropout_prob", 0.0)),
        ).to(device)
        optimizer, train_loss, history = _train_fold(
            model, train_loader, device, args, fold, fold_train
        )
        all_history.extend(history)
        checkpoint_path = output_dir / f"fold_{fold}_best.pt"
        torch.save(
            _fold_checkpoint(
                model,
                optimizer,
                args,
                fold,
                train_loss,
                [display_ids[key] for key in train_patient_keys],
                [display_ids[key] for key in holdout_patient_keys],
            ),
            checkpoint_path,
        )

        predictions = _predict_holdout(model, holdout_loader, device, fold)
        write_csv(predictions, output_dir / f"fold_{fold}_predictions.csv")
        fold_metrics = _fold_metrics(predictions, fold)
        fold_metric_rows.append(fold_metrics)
        all_predictions.append(predictions)
        auc_text = (
            f"{fold_metrics['auc']:.4f}"
            if fold_metrics["auc"] is not None
            else "undefined (one class)"
        )
        print(
            f"  Held-out fold {fold}: AUC={auc_text}, "
            f"accuracy={fold_metrics['accuracy']:.4f}"
        )
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if seen_holdout_patients != set(patient_keys):
        raise AssertionError("Not every patient appeared in exactly one held-out fold.")
    oof = pd.concat(all_predictions, ignore_index=True).sort_values("source_index")
    if len(oof) != len(items) or oof["source_index"].duplicated().any():
        raise AssertionError("OOF predictions do not map one-to-one to training rows.")
    if oof["true_label"].nunique() != 2:
        raise ValueError("Aggregate OOF AUC requires both binary classes.")

    aggregate = {
        "auc": float(roc_auc_score(oof["true_label"], oof["prob_1"])),
        "accuracy": float(accuracy_score(oof["true_label"], oof["pred_label"])),
        "num_cases": int(len(oof)),
        "num_patients": int(len(patient_keys)),
        "num_folds": int(args.num_folds),
    }
    write_csv(oof, output_dir / "oof_predictions.csv")
    write_csv(pd.DataFrame(fold_metric_rows), output_dir / "fold_metrics.csv")
    write_csv(pd.DataFrame(assignment_rows), output_dir / "fold_assignments.csv")
    write_csv(pd.DataFrame(all_history), output_dir / "training_history.csv")
    write_json(aggregate, output_dir / "oof_metrics.json")
    print(
        f"Aggregate OOF AUC={aggregate['auc']:.4f}, "
        f"accuracy={aggregate['accuracy']:.4f}"
    )
    print(f"Saved OOF outputs to: {output_dir}")
    return aggregate


def _positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _at_least_two(value: str) -> int:
    """Parse a fold count of at least two."""
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return parsed


def _positive_float(value: str) -> float:
    """Parse a finite positive floating-point CLI value."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _nonnegative_float(value: str) -> float:
    """Parse a finite non-negative floating-point CLI value."""
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse OOF training command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate leakage-safe patient-level out-of-fold predictions for a "
            "binary downstream task."
        )
    )
    parser.add_argument("--train_json", type=Path, required=True)
    parser.add_argument("--image_keys", nargs="+", required=True)
    parser.add_argument("--target_key", required=True)
    parser.add_argument("--num_folds", type=_at_least_two, default=5)
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
    parser.add_argument(
        "--weight_decay",
        type=_nonnegative_float,
        default=0.0,
        help="AdamW weight decay. Try 1e-4 when fold losses overfit.",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    run_oof(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
