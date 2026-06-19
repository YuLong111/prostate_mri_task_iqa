"""Generate task-derived image-quality labels from downstream performance."""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prostate_iqa.utils.io import read_csv, write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_CSV = (
    PROJECT_ROOT / "data" / "manifests" / "task_quality_label_summary.csv"
)
QUALITY_NAMES = {0: "reject", 1: "caution", 2: "accept"}


def _column_key(name: object) -> str:
    """Normalize a spreadsheet header for alias matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _is_present(value: Any) -> bool:
    """Return whether a scalar is non-missing and non-empty."""
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def _clean_id(value: Any) -> str:
    """Convert spreadsheet identifiers to stable strings."""
    if not _is_present(value):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _patient_key(value: Any) -> str:
    """Match Patient-prefixed IDs and numeric IDs despite leading zeros."""
    text = _clean_id(value)
    if not text:
        return ""
    match = re.fullmatch(r"(?i)patient[-_ ]?0*(\d+)(?:\.0+)?", text)
    if match is None:
        match = re.fullmatch(r"0*(\d+)(?:\.0+)?", text)
    if match:
        return f"number:{int(match.group(1))}"
    return "text:" + re.sub(r"[^a-z0-9]+", "", text.lower())


def _scan_key(value: Any) -> str:
    """Create a case-insensitive scan key without image suffixes."""
    text = _clean_id(value).lower()
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".dcm"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return re.sub(r"\s+", "", text)


def _find_column(
    frame: pd.DataFrame,
    aliases: Sequence[str],
    required: bool = False,
) -> str | None:
    """Find one source column using normalized aliases."""
    keyed = {_column_key(column): column for column in frame.columns}
    for alias in aliases:
        match = keyed.get(_column_key(alias))
        if match is not None:
            return match
    if required:
        raise ValueError(
            "Required column not found; expected one of: " + ", ".join(aliases)
        )
    return None


def _parse_binary(value: Any, column: str) -> int | None:
    """Parse common boolean and binary encodings."""
    if not _is_present(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y", "correct", "positive"}:
        return 1
    if text in {"0", "0.0", "false", "no", "n", "incorrect", "negative"}:
        return 0
    raise ValueError(f"Column {column!r} contains a non-binary value: {value!r}")


def _numeric(value: Any) -> float | None:
    """Parse a finite numeric value or return None."""
    if not _is_present(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _quality_name(value: Any) -> str | None:
    """Return the standard name for a numeric ternary quality value."""
    numeric = _numeric(value)
    if numeric is None or not numeric.is_integer():
        return None
    return QUALITY_NAMES.get(int(numeric))


def _identifier_columns(frame: pd.DataFrame) -> tuple[str, str]:
    """Locate required patient and scan identifier columns."""
    patient_column = _find_column(
        frame,
        ("patient_id", "patientid", "patient", "case_id", "subject_id"),
        required=True,
    )
    scan_column = _find_column(
        frame,
        ("scan_id", "scanid", "scan", "volume_id", "study_id"),
        required=True,
    )
    assert patient_column is not None and scan_column is not None
    return patient_column, scan_column


def _classification_records(
    predictions: pd.DataFrame,
    task_name: str,
    accept_confidence: float,
    reject_confidence: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Convert classification predictions to task-quality records."""
    patient_column, scan_column = _identifier_columns(predictions)
    correct_column = _find_column(predictions, ("correct", "task_correct"))
    true_column = _find_column(
        predictions,
        ("true_label", "target", "label", "y_true"),
    )
    predicted_column = _find_column(
        predictions,
        ("pred_label", "prediction", "predicted_label", "y_pred"),
    )
    confidence_column = _find_column(
        predictions,
        ("confidence", "task_confidence", "max_probability"),
    )
    prob_0_column = _find_column(predictions, ("prob_0", "probability_0"))
    prob_1_column = _find_column(predictions, ("prob_1", "probability_1"))

    if correct_column is None and (true_column is None or predicted_column is None):
        raise ValueError(
            "Classification predictions need either correct, or both true_label "
            "and pred_label columns."
        )
    if confidence_column is None and (prob_0_column is None or prob_1_column is None):
        raise ValueError(
            "Classification predictions need confidence, or prob_0 and prob_1."
        )

    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row_index, row in predictions.iterrows():
        key = (_patient_key(row[patient_column]), _scan_key(row[scan_column]))
        if not all(key):
            raise ValueError(f"Prediction row {row_index} has missing patient/scan ID.")

        if correct_column is not None:
            correct = _parse_binary(row[correct_column], correct_column)
        else:
            correct = None
        if (
            correct is None
            and true_column is not None
            and predicted_column is not None
            and _is_present(row[true_column])
            and _is_present(row[predicted_column])
        ):
            correct = int(_values_equal(row[true_column], row[predicted_column]))

        if confidence_column is not None:
            confidence = _numeric(row[confidence_column])
        else:
            probabilities = (_numeric(row[prob_0_column]), _numeric(row[prob_1_column]))
            confidence = (
                max(value for value in probabilities if value is not None)
                if any(value is not None for value in probabilities)
                else None
            )
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"Prediction row {row_index} confidence must be in [0, 1], "
                f"received {confidence}."
            )

        if correct == 1 and confidence is not None and confidence >= accept_confidence:
            quality = 2
        elif correct == 0 and confidence is not None and confidence >= reject_confidence:
            quality = 0
        else:
            quality = 1
        record = {
            "task_quality_ternary": quality,
            "task_quality_label_name": QUALITY_NAMES[quality],
            "task_quality_bin": 1 if quality == 2 else (0 if quality == 0 else None),
            "task_correct": correct,
            "task_confidence": confidence,
            "quality_source": task_name,
            "task_quality_source": task_name,
            "task_dice": None,
            "task_hd95": None,
        }
        _store_unique(records, key, record, row_index)
    return records


def _segmentation_records(
    metrics: pd.DataFrame,
    task_name: str,
    dice_accept: float,
    dice_reject: float,
    hd95_accept: float,
    hd95_reject: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Convert per-scan Dice/95HD results to task-quality records."""
    patient_column, scan_column = _identifier_columns(metrics)
    dice_column = _find_column(
        metrics,
        ("dice", "dice_score", "mean_dice", "val_dice"),
        required=True,
    )
    hd95_column = _find_column(
        metrics,
        ("hd95", "95hd", "hausdorff_95", "hausdorff95", "hausdorff_distance_95"),
    )
    assert dice_column is not None

    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row_index, row in metrics.iterrows():
        key = (_patient_key(row[patient_column]), _scan_key(row[scan_column]))
        if not all(key):
            raise ValueError(f"Metric row {row_index} has missing patient/scan ID.")
        dice = _numeric(row[dice_column])
        hd95 = _numeric(row[hd95_column]) if hd95_column is not None else None

        if dice is not None and not 0.0 <= dice <= 1.0:
            raise ValueError(
                f"Metric row {row_index} Dice must be in [0, 1], received {dice}."
            )
        if hd95 is not None and hd95 < 0:
            raise ValueError(f"Metric row {row_index} 95HD cannot be negative.")

        accept = dice is not None and dice >= dice_accept
        if hd95_column is not None:
            accept = accept and hd95 is not None and hd95 <= hd95_accept
        reject = (dice is not None and dice < dice_reject) or (
            hd95 is not None and hd95 >= hd95_reject
        )
        quality = 0 if reject else (2 if accept else 1)
        record = {
            "task_quality_ternary": quality,
            "task_quality_label_name": QUALITY_NAMES[quality],
            "task_quality_bin": 1 if quality == 2 else (0 if quality == 0 else None),
            "task_correct": None,
            "task_confidence": None,
            "quality_source": task_name,
            "task_quality_source": task_name,
            "task_dice": dice,
            "task_hd95": hd95,
        }
        _store_unique(records, key, record, row_index)
    return records


def _values_equal(left: Any, right: Any) -> bool:
    """Compare duplicate metric values while tolerating numeric formatting."""
    if not _is_present(left) and not _is_present(right):
        return True
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left).strip().lower() == str(right).strip().lower()


def _store_unique(
    records: dict[tuple[str, str], dict[str, Any]],
    key: tuple[str, str],
    record: dict[str, Any],
    row_index: Any,
) -> None:
    """Store one task result per patient/scan and reject conflicts."""
    existing = records.get(key)
    if existing is not None and any(
        not _values_equal(existing[column], value)
        for column, value in record.items()
    ):
        raise ValueError(
            f"Conflicting task results for patient/scan {key} at row {row_index}."
        )
    records[key] = record


def _combine_without_overwrite(existing: pd.Series, derived: pd.Series) -> pd.Series:
    """Fill only missing/empty existing values with task-derived values."""
    # Object dtype permits a conservative mix of pre-existing spreadsheet
    # annotations and numeric task labels (including with pandas StringDtype).
    result = existing.astype("object").copy()
    missing = ~result.map(_is_present)
    result.loc[missing] = derived.loc[missing]
    return result


def apply_quality_records(
    manifest: pd.DataFrame,
    records: Mapping[tuple[str, str], Mapping[str, Any]],
) -> pd.DataFrame:
    """Add task labels while preserving all original manifest values."""
    patient_column, scan_column = _identifier_columns(manifest)
    result = manifest.copy()
    original_quality = result.get(
        "quality_ternary",
        pd.Series(None, index=result.index, dtype="object"),
    ).copy()
    original_quality_present = original_quality.map(_is_present)
    derived_rows: list[dict[str, Any]] = []
    empty_record = {
        "task_quality_ternary": None,
        "task_quality_label_name": None,
        "task_quality_bin": None,
        "task_correct": None,
        "task_confidence": None,
        "quality_source": None,
        "task_quality_source": None,
        "task_dice": None,
        "task_hd95": None,
    }
    for _, row in result.iterrows():
        key = (_patient_key(row[patient_column]), _scan_key(row[scan_column]))
        derived_rows.append(dict(records.get(key, empty_record)))
    derived = pd.DataFrame(derived_rows, index=result.index)

    for column in derived.columns:
        if column != "quality_source":
            result[column] = derived[column]

    existing_source = result.get(
        "quality_source",
        pd.Series(None, index=result.index, dtype="object"),
    ).copy()
    source_missing = ~existing_source.map(_is_present)
    use_task_source = source_missing & ~original_quality_present
    existing_source.loc[use_task_source] = derived.loc[use_task_source, "quality_source"]
    result["quality_source"] = existing_source

    if "quality_ternary" in result.columns:
        result["quality_ternary"] = _combine_without_overwrite(
            result["quality_ternary"],
            derived["task_quality_ternary"],
        )
    else:
        result["quality_ternary"] = derived["task_quality_ternary"]

    existing_names = result.get(
        "quality_label_name",
        pd.Series(None, index=result.index, dtype="object"),
    ).copy()
    clinical_names = original_quality.map(_quality_name)
    existing_names = _combine_without_overwrite(existing_names, clinical_names)
    name_missing = ~existing_names.map(_is_present)
    use_task_name = name_missing & ~original_quality_present
    existing_names.loc[use_task_name] = derived.loc[
        use_task_name, "task_quality_label_name"
    ]
    result["quality_label_name"] = existing_names

    integer_columns = ("task_quality_ternary", "task_quality_bin", "task_correct")
    for column in integer_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    return result


def build_label_summary(
    labeled_manifest: pd.DataFrame,
    task_name: str,
    task_type: str,
) -> pd.DataFrame:
    """Summarize task-derived labels, including unmatched manifest rows."""
    task_labels = labeled_manifest["task_quality_ternary"]
    matched = int(task_labels.notna().sum())
    total = len(labeled_manifest)
    rows = []
    for quality in (0, 1, 2):
        count = int(task_labels.eq(quality).sum())
        rows.append(
            {
                "task_name": task_name,
                "task_type": task_type,
                "quality_ternary": quality,
                "quality_label_name": QUALITY_NAMES[quality],
                "count": count,
                "percent_of_labeled": 100.0 * count / matched if matched else 0.0,
                "matched_manifest_rows": matched,
                "unmatched_manifest_rows": total - matched,
                "total_manifest_rows": total,
            }
        )
    return pd.DataFrame(rows)


def _unit_interval(value: str) -> float:
    """Argparse type for thresholds in [0, 1]."""
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _nonnegative(value: str) -> float:
    """Argparse type for non-negative distances."""
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create task-derived image-quality labels."
    )
    parser.add_argument("--manifest_csv", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task_predictions_csv", type=Path)
    source.add_argument("--segmentation_metrics_csv", type=Path)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--accept_confidence", type=_unit_interval, default=0.70)
    parser.add_argument("--reject_confidence", type=_unit_interval, default=0.70)
    parser.add_argument("--dice_accept", type=_unit_interval, default=0.75)
    parser.add_argument("--dice_reject", type=_unit_interval, default=0.50)
    parser.add_argument("--hd95_accept", type=_nonnegative, default=10.0)
    parser.add_argument("--hd95_reject", type=_nonnegative, default=20.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate labels, save the augmented manifest, and summarize labels."""
    args = parse_args(argv)
    if args.dice_reject >= args.dice_accept:
        raise ValueError("dice_reject must be lower than dice_accept.")
    if args.hd95_accept >= args.hd95_reject:
        raise ValueError("hd95_accept must be lower than hd95_reject.")

    manifest = read_csv(args.manifest_csv)
    if args.segmentation_metrics_csv is not None:
        task_type = "segmentation"
        metrics = read_csv(args.segmentation_metrics_csv)
        records = _segmentation_records(
            metrics,
            args.task_name,
            args.dice_accept,
            args.dice_reject,
            args.hd95_accept,
            args.hd95_reject,
        )
    else:
        task_type = "classification"
        predictions = read_csv(args.task_predictions_csv)
        records = _classification_records(
            predictions,
            args.task_name,
            args.accept_confidence,
            args.reject_confidence,
        )

    labeled = apply_quality_records(manifest, records)
    output_path = write_csv(labeled, args.out_csv)
    summary = build_label_summary(labeled, args.task_name, task_type)
    summary_path = write_csv(summary, SUMMARY_CSV)

    matched = int(labeled["task_quality_ternary"].notna().sum())
    distribution = (
        labeled["task_quality_label_name"].value_counts().sort_index().to_dict()
    )
    print(f"Saved task-labeled manifest to: {output_path}")
    print(f"Saved task-quality summary to: {summary_path}")
    print(f"Matched {matched:,} of {len(labeled):,} manifest rows.")
    print(f"Task-quality distribution: {distribution}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
