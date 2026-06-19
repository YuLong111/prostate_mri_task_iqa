"""Analyze whether predicted image quality explains downstream performance."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from prostate_iqa.utils.io import ensure_dir, read_csv, write_csv


QUALITY_NAMES = {0: "reject", 1: "caution", 2: "accept"}
SEGMENTATION_ALIASES = {
    "dice": ("dice", "dice_score", "mean_dice"),
    "iou": ("iou", "iou_score", "jaccard", "jaccard_score"),
    "asd": (
        "asd",
        "average_surface_distance",
        "avg_surface_distance",
        "mean_surface_distance",
    ),
    "hd95": (
        "hd95",
        "95hd",
        "hausdorff_95",
        "hausdorff95",
        "hausdorff_distance_95",
    ),
}


def _column_key(name: object) -> str:
    """Normalize spreadsheet headers for alias matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _find_column(
    frame: pd.DataFrame,
    aliases: Sequence[str],
    required: bool = False,
) -> str | None:
    """Find a source column using normalized aliases."""
    keyed = {_column_key(column): column for column in frame.columns}
    for alias in aliases:
        match = keyed.get(_column_key(alias))
        if match is not None:
            return match
    if required:
        raise ValueError("Expected one of these columns: " + ", ".join(aliases))
    return None


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
    """Convert spreadsheet identifiers to stable display strings."""
    if not _is_present(value):
        return ""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _patient_key(value: Any) -> str:
    """Match Patient-prefixed and numeric identifiers despite leading zeros."""
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
    """Create a case-insensitive scan identifier key."""
    text = _clean_id(value).lower()
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".dcm"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return re.sub(r"\s+", "", text)


def _identifier_columns(frame: pd.DataFrame) -> tuple[str, str]:
    """Locate patient and scan ID columns."""
    patient = _find_column(
        frame,
        ("patient_id", "patientid", "patient", "case_id", "subject_id"),
        required=True,
    )
    scan = _find_column(
        frame,
        ("scan_id", "scanid", "scan", "volume_id", "study_id"),
        required=True,
    )
    assert patient is not None and scan is not None
    return patient, scan


def _add_identifiers(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str, str | None]:
    """Add normalized merge keys and canonical display identifiers."""
    patient_column, scan_column = _identifier_columns(frame)
    result = frame.copy()
    result["_patient_key"] = result[patient_column].map(_patient_key)
    result["_scan_key"] = result[scan_column].map(_scan_key)
    acquisition_column = _find_column(
        result,
        ("acquisition_id", "acquisitionid", "volume_id", "distortion_status"),
    )
    result["_acquisition_key"] = (
        result[acquisition_column].map(_scan_key)
        if acquisition_column is not None
        else ""
    )
    if (result["_patient_key"] == "").any() or (result["_scan_key"] == "").any():
        raise ValueError("Patient and scan identifiers cannot be missing.")
    return result, patient_column, scan_column, acquisition_column


def _assert_unique(frame: pd.DataFrame, source_name: str) -> None:
    """Reject duplicate case rows before one-to-one merging."""
    keys = ["_patient_key", "_scan_key", "_acquisition_key"]
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, keys].head(5)
        raise ValueError(
            f"{source_name} contains duplicate acquisition rows: "
            + examples.to_dict("records").__repr__()
        )


def _numeric_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    """Return a float series, or an all-missing series when unavailable."""
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _binary_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    """Return a binary float series supporting numeric and boolean text."""
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    numeric = pd.to_numeric(frame[column], errors="coerce")
    parsed = frame[column].map(_parse_binary)
    return numeric.fillna(parsed).astype(float)


def _parse_quality(value: Any) -> int | None:
    """Parse reject/caution/accept predictions."""
    if not _is_present(value):
        return None
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
        raise ValueError(f"Unrecognized predicted quality value: {value!r}")
    return aliases[text]


def _parse_binary(value: Any) -> float:
    """Parse a binary/boolean scalar, returning NaN when missing."""
    if not _is_present(value):
        return float("nan")
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y", "correct", "positive"}:
        return 1.0
    if text in {"0", "0.0", "false", "no", "n", "incorrect", "negative"}:
        return 0.0
    return float("nan")


def _values_equal(left: Any, right: Any) -> bool:
    """Compare prediction labels while tolerating numeric formatting."""
    if not _is_present(left) or not _is_present(right):
        return False
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left).strip().lower() == str(right).strip().lower()


def _prefixed_source_columns(
    frame: pd.DataFrame,
    patient_column: str,
    scan_column: str,
    acquisition_column: str | None,
    prefix: str,
) -> pd.DataFrame:
    """Preserve source columns with a namespace to prevent merge collisions."""
    result = pd.DataFrame(index=frame.index)
    result["_patient_key"] = frame["_patient_key"]
    result["_scan_key"] = frame["_scan_key"]
    result["_acquisition_key"] = frame["_acquisition_key"]
    for column in frame.columns:
        if column not in {
            patient_column,
            scan_column,
            acquisition_column,
            "_patient_key",
            "_scan_key",
            "_acquisition_key",
        }:
            result[f"{prefix}_{column}"] = frame[column]
    return result


def _standardize_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Create canonical quality fields while retaining original columns."""
    source, patient_column, scan_column, acquisition_column = _add_identifiers(frame)
    _assert_unique(source, "quality predictions")
    predicted_column = _find_column(
        source,
        (
            "pred_quality",
            "predicted_quality",
            "quality_ternary_pred",
            "task_quality_ternary",
            "quality_ternary",
            "pred_label",
            "quality_label_name",
        ),
        required=True,
    )
    expected_column = _find_column(
        source,
        ("expected_quality_score", "expected_ordinal_score", "quality_score"),
    )
    confidence_column = _find_column(
        source,
        ("confidence", "quality_confidence", "max_probability"),
    )
    entropy_column = _find_column(source, ("entropy", "quality_entropy"))

    result = _prefixed_source_columns(
        source, patient_column, scan_column, acquisition_column, "quality_source"
    )
    result["patient_id"] = source[patient_column].map(_clean_id)
    result["scan_id"] = source[scan_column].map(_clean_id)
    if acquisition_column is not None:
        result["acquisition_id"] = source[acquisition_column].map(_clean_id)
    assert predicted_column is not None
    result["predicted_quality"] = source[predicted_column].map(_parse_quality).astype(
        "Int64"
    )
    result["quality_group"] = result["predicted_quality"].map(QUALITY_NAMES)
    result["expected_quality_score"] = _numeric_series(source, expected_column)

    probability_columns = [
        _find_column(source, ("prob_reject", "prob_0")),
        _find_column(source, ("prob_caution",)),
        _find_column(source, ("prob_accept", "prob_2")),
    ]
    if all(column is not None for column in probability_columns):
        probabilities = np.column_stack(
            [_numeric_series(source, column) for column in probability_columns]
        )
        probability_score = pd.Series(probabilities @ np.arange(3), index=source.index)
        result["expected_quality_score"] = result["expected_quality_score"].fillna(
            probability_score
        )

    result["quality_confidence"] = _numeric_series(source, confidence_column)
    result["quality_entropy"] = _numeric_series(source, entropy_column)
    return result


def _standardize_task(frame: pd.DataFrame) -> pd.DataFrame:
    """Create canonical downstream binary-classification fields."""
    source, patient_column, scan_column, acquisition_column = _add_identifiers(frame)
    _assert_unique(source, "task predictions")
    true_column = _find_column(
        source,
        ("true_label", "task_true_label", "target", "label", "y_true"),
    )
    predicted_column = _find_column(
        source,
        ("pred_label", "task_pred_label", "prediction", "predicted_label", "y_pred"),
    )
    correct_column = _find_column(source, ("correct", "task_correct"))
    probability_column = _find_column(
        source,
        ("prob_1", "probability_1", "positive_probability", "task_probability"),
    )
    probability_0_column = _find_column(source, ("prob_0", "probability_0"))
    confidence_column = _find_column(
        source,
        ("confidence", "task_confidence", "max_probability"),
    )

    result = _prefixed_source_columns(
        source, patient_column, scan_column, acquisition_column, "task_source"
    )
    result["task_true_label"] = _binary_series(source, true_column)
    result["task_pred_label"] = _binary_series(source, predicted_column)
    result["task_probability"] = _numeric_series(source, probability_column)
    if probability_column is not None:
        missing_prediction = result["task_pred_label"].isna() & result[
            "task_probability"
        ].notna()
        result.loc[missing_prediction, "task_pred_label"] = (
            result.loc[missing_prediction, "task_probability"] >= 0.5
        ).astype(float)

    if correct_column is not None:
        result["task_correct"] = source[correct_column].map(_parse_binary)
    else:
        result["task_correct"] = pd.Series(np.nan, index=source.index, dtype=float)
    can_derive = (
        result["task_correct"].isna()
        & result["task_true_label"].notna()
        & result["task_pred_label"].notna()
    )
    result.loc[can_derive, "task_correct"] = [
        float(_values_equal(true, predicted))
        for true, predicted in zip(
            result.loc[can_derive, "task_true_label"],
            result.loc[can_derive, "task_pred_label"],
            strict=True,
        )
    ]
    if result["task_correct"].isna().all():
        raise ValueError(
            "Task predictions require either correct, or true_label plus pred_label/prob_1."
        )

    result["task_confidence"] = _numeric_series(source, confidence_column)
    if probability_column is not None:
        if probability_0_column is not None:
            probability_0 = _numeric_series(source, probability_0_column)
            derived_confidence = pd.concat(
                [probability_0, result["task_probability"]], axis=1
            ).max(axis=1)
        else:
            derived_confidence = pd.Series(
                np.maximum(
                    result["task_probability"], 1.0 - result["task_probability"]
                ),
                index=source.index,
            )
        result["task_confidence"] = result["task_confidence"].fillna(
            derived_confidence
        )
    result["task_error"] = 1.0 - result["task_correct"]
    return result


def _standardize_novelty(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a canonical novelty-distance table."""
    source, patient_column, scan_column, acquisition_column = _add_identifiers(frame)
    _assert_unique(source, "novelty scores")
    distance_column = _find_column(
        source,
        ("novelty_distance", "mahalanobis_distance", "novelty_score"),
        required=True,
    )
    result = _prefixed_source_columns(
        source, patient_column, scan_column, acquisition_column, "novelty_source"
    )
    result["novelty_distance"] = _numeric_series(source, distance_column)
    return result


def _standardize_segmentation(frame: pd.DataFrame) -> pd.DataFrame:
    """Create canonical Dice/IoU/ASD/95HD columns."""
    source, patient_column, scan_column, acquisition_column = _add_identifiers(frame)
    _assert_unique(source, "segmentation metrics")
    result = _prefixed_source_columns(
        source,
        patient_column,
        scan_column,
        acquisition_column,
        "segmentation_source",
    )
    found = 0
    for metric, aliases in SEGMENTATION_ALIASES.items():
        column = _find_column(source, aliases)
        result[f"seg_{metric}"] = _numeric_series(source, column)
        found += int(column is not None)
    if found == 0:
        raise ValueError("No Dice, IoU, ASD, or 95HD column found in segmentation CSV.")
    return result


def _safe_auc(labels: pd.Series, scores: pd.Series) -> float | None:
    """Compute ROC AUC when paired finite values contain both classes."""
    valid = labels.notna() & scores.notna()
    truth = labels.loc[valid].astype(int)
    probability = scores.loc[valid].astype(float)
    if len(truth) == 0 or truth.nunique() != 2 or not truth.isin([0, 1]).all():
        return None
    return float(roc_auc_score(truth, probability))


def _safe_spearman(
    left: pd.Series,
    right: pd.Series,
) -> tuple[float | None, float | None, int]:
    """Compute Spearman rho/p only for non-constant paired values."""
    valid = left.notna() & right.notna()
    left_valid = left.loc[valid].astype(float)
    right_valid = right.loc[valid].astype(float)
    count = int(valid.sum())
    if count < 3 or left_valid.nunique() < 2 or right_valid.nunique() < 2:
        return None, None, count
    result = spearmanr(left_valid, right_valid)
    rho = float(result.statistic) if np.isfinite(result.statistic) else None
    pvalue = float(result.pvalue) if np.isfinite(result.pvalue) else None
    return rho, pvalue, count


def _group_summary_row(
    subset: pd.DataFrame,
    scope: str,
    quality_value: int | None,
) -> dict[str, Any]:
    """Summarize classification and segmentation results for one subset."""
    valid_task = subset["task_correct"].notna()
    task_count = int(valid_task.sum())
    accuracy = (
        float(subset.loc[valid_task, "task_correct"].mean()) if task_count else None
    )
    row: dict[str, Any] = {
        "scope": scope,
        "predicted_quality": quality_value,
        "quality_group": QUALITY_NAMES.get(quality_value, "overall"),
        "n_merged_cases": len(subset),
        "n_task_cases": task_count,
        "task_accuracy": accuracy,
        "task_error_rate": 1.0 - accuracy if accuracy is not None else None,
        "task_auc": _safe_auc(
            subset["task_true_label"], subset["task_probability"]
        ),
        "mean_task_confidence": (
            float(subset["task_confidence"].mean())
            if subset["task_confidence"].notna().any()
            else None
        ),
    }
    for metric in SEGMENTATION_ALIASES:
        column = f"seg_{metric}"
        if column in subset:
            values = subset[column].dropna()
        else:
            values = pd.Series(dtype=float)
        row[f"n_{metric}"] = len(values)
        row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
        row[f"{metric}_median"] = float(values.median()) if len(values) else None
    return row


def build_quality_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """Build reject/caution/accept and overall performance summaries."""
    correlations: dict[str, Any] = {}
    rho, pvalue, count = _safe_spearman(
        merged["expected_quality_score"], merged["task_correct"]
    )
    correlations.update(
        {
            "spearman_quality_vs_task_success": rho,
            "spearman_quality_vs_task_success_pvalue": pvalue,
            "n_quality_task_correlation": count,
        }
    )
    for metric in SEGMENTATION_ALIASES:
        column = f"seg_{metric}"
        if column in merged:
            rho, pvalue, count = _safe_spearman(
                merged["expected_quality_score"], merged[column]
            )
        else:
            rho, pvalue, count = None, None, 0
        correlations[f"spearman_quality_vs_{metric}"] = rho
        correlations[f"spearman_quality_vs_{metric}_pvalue"] = pvalue
        correlations[f"n_quality_{metric}_correlation"] = count

    rows = []
    for quality in (0, 1, 2):
        rows.append(
            _group_summary_row(
                merged.loc[merged["predicted_quality"].eq(quality)],
                "quality_group",
                quality,
            )
        )
    overall = _group_summary_row(merged, "overall", None)
    overall.update(correlations)
    rows.append(overall)
    return pd.DataFrame(rows)


def build_novelty_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """Summarize novelty as a predictor of downstream classification failure."""
    columns = (
        "novelty_group",
        "n_cases",
        "novelty_distance_mean",
        "novelty_distance_median",
        "downstream_failure_rate",
        "failure_prediction_auc",
        "spearman_novelty_vs_failure",
        "spearman_novelty_vs_failure_pvalue",
    )
    if "novelty_distance" not in merged:
        return pd.DataFrame(
            [
                {
                    "novelty_group": "overall",
                    "n_cases": 0,
                    "novelty_distance_mean": None,
                    "novelty_distance_median": None,
                    "downstream_failure_rate": None,
                    "failure_prediction_auc": None,
                    "spearman_novelty_vs_failure": None,
                    "spearman_novelty_vs_failure_pvalue": None,
                }
            ],
            columns=columns,
        )

    valid = merged.loc[
        merged["novelty_distance"].notna() & merged["task_error"].notna()
    ].copy()
    auc = _safe_auc(valid["task_error"], valid["novelty_distance"])
    rho, pvalue, _ = _safe_spearman(
        valid["novelty_distance"], valid["task_error"]
    )
    rows: list[dict[str, Any]] = [
        {
            "novelty_group": "overall",
            "n_cases": len(valid),
            "novelty_distance_mean": (
                float(valid["novelty_distance"].mean()) if len(valid) else None
            ),
            "novelty_distance_median": (
                float(valid["novelty_distance"].median()) if len(valid) else None
            ),
            "downstream_failure_rate": (
                float(valid["task_error"].mean()) if len(valid) else None
            ),
            "failure_prediction_auc": auc,
            "spearman_novelty_vs_failure": rho,
            "spearman_novelty_vs_failure_pvalue": pvalue,
        }
    ]
    if len(valid):
        quartile_count = min(4, len(valid))
        valid["_novelty_bin"] = pd.qcut(
            valid["novelty_distance"].rank(method="first"),
            q=quartile_count,
            labels=False,
        )
        for bin_index, subset in valid.groupby("_novelty_bin", sort=True):
            rows.append(
                {
                    "novelty_group": f"Q{int(bin_index) + 1}",
                    "n_cases": len(subset),
                    "novelty_distance_mean": float(
                        subset["novelty_distance"].mean()
                    ),
                    "novelty_distance_median": float(
                        subset["novelty_distance"].median()
                    ),
                    "downstream_failure_rate": float(subset["task_error"].mean()),
                    "failure_prediction_auc": None,
                    "spearman_novelty_vs_failure": None,
                    "spearman_novelty_vs_failure_pvalue": None,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _placeholder(axis: plt.Axes, message: str) -> None:
    """Render a clear placeholder when a requested plot lacks data."""
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def _save_plots(
    merged: pd.DataFrame,
    quality_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate the three requested diagnostic plots."""
    colors = ["#b23a48", "#e0a458", "#3a7d44"]
    groups = quality_summary.loc[quality_summary["scope"].eq("quality_group")]

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    valid_groups = groups.loc[groups["task_accuracy"].notna()]
    if len(valid_groups):
        axis.bar(
            valid_groups["quality_group"],
            valid_groups["task_accuracy"],
            color=[colors[int(value)] for value in valid_groups["predicted_quality"]],
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("Downstream accuracy")
        axis.set_xlabel("Predicted IQA quality")
        axis.set_title("Downstream accuracy by predicted quality")
    else:
        _placeholder(axis, "No matched downstream correctness data")
    figure.tight_layout()
    figure.savefig(output_dir / "quality_group_vs_downstream_accuracy.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    valid = merged.loc[
        merged["expected_quality_score"].notna()
        & merged["task_confidence"].notna()
    ]
    if len(valid):
        axis.scatter(
            valid["expected_quality_score"],
            valid["task_confidence"],
            alpha=0.65,
            color="#35618f",
            edgecolors="none",
        )
        axis.set_xlabel("Expected IQA quality score")
        axis.set_ylabel("Downstream confidence")
        axis.set_title("Quality score vs downstream confidence")
    else:
        _placeholder(axis, "Quality score or downstream confidence unavailable")
    figure.tight_layout()
    figure.savefig(output_dir / "quality_score_vs_downstream_confidence.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    if "novelty_distance" in merged:
        valid = merged.loc[
            merged["novelty_distance"].notna() & merged["task_error"].notna()
        ]
    else:
        valid = pd.DataFrame()
    if len(valid):
        jitter = np.random.default_rng(42).normal(0.0, 0.025, len(valid))
        axis.scatter(
            valid["novelty_distance"],
            valid["task_error"] + jitter,
            alpha=0.65,
            color="#6f4a8e",
            edgecolors="none",
        )
        axis.set_yticks([0, 1], labels=["correct", "failure"])
        axis.set_ylim(-0.15, 1.15)
        axis.set_xlabel("Novelty distance")
        axis.set_ylabel("Downstream outcome")
        axis.set_title("Novelty distance vs downstream error")
    else:
        _placeholder(axis, "Novelty or downstream error data unavailable")
    figure.tight_layout()
    figure.savefig(output_dir / "novelty_distance_vs_downstream_error.png", dpi=150)
    plt.close(figure)


def run_analysis(args: argparse.Namespace) -> dict[str, Path]:
    """Merge input tables, compute summaries, and save tables and plots."""
    output_dir = ensure_dir(args.out_dir)
    quality = _standardize_quality(read_csv(args.quality_predictions_csv))
    task = _standardize_task(read_csv(args.task_predictions_csv))
    merged = quality.merge(
        task,
        on=["_patient_key", "_scan_key", "_acquisition_key"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Quality and task predictions have no matching patient/scan rows.")
    print(
        f"Matched {len(merged):,} cases from quality={len(quality):,} "
        f"and task={len(task):,} rows."
    )

    if args.novelty_csv is not None:
        novelty = _standardize_novelty(read_csv(args.novelty_csv))
        merged = merged.merge(
            novelty,
            on=["_patient_key", "_scan_key", "_acquisition_key"],
            how="left",
            validate="one_to_one",
        )
    if args.segmentation_metrics_csv is not None:
        segmentation = _standardize_segmentation(
            read_csv(args.segmentation_metrics_csv)
        )
        merged = merged.merge(
            segmentation,
            on=["_patient_key", "_scan_key", "_acquisition_key"],
            how="left",
            validate="one_to_one",
        )

    quality_summary = build_quality_summary(merged)
    novelty_summary = build_novelty_summary(merged)
    merged = merged.drop(columns=["_patient_key", "_scan_key", "_acquisition_key"])
    leading_columns = [
        column
        for column in ("patient_id", "scan_id", "acquisition_id")
        if column in merged.columns
    ]
    merged = merged[
        leading_columns
        + [column for column in merged.columns if column not in leading_columns]
    ]

    paths = {
        "quality_summary": write_csv(
            quality_summary, output_dir / "quality_group_task_summary.csv"
        ),
        "novelty_summary": write_csv(
            novelty_summary, output_dir / "novelty_failure_summary.csv"
        ),
        "merged_table": write_csv(
            merged, output_dir / "merged_quality_task_table.csv"
        ),
    }
    _save_plots(merged, quality_summary, output_dir)
    for name, path in paths.items():
        print(f"Saved {name}: {path}")
    print(f"Saved plots to: {output_dir}")
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse analysis command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze predicted IQA quality against downstream performance."
    )
    parser.add_argument("--quality_predictions_csv", type=Path, required=True)
    parser.add_argument("--task_predictions_csv", type=Path, required=True)
    parser.add_argument("--novelty_csv", type=Path)
    parser.add_argument("--segmentation_metrics_csv", type=Path)
    parser.add_argument("--out_dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    run_analysis(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
