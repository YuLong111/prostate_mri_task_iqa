"""Generate report-ready figures for prostate MRI image-quality analysis."""

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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from prostate_iqa.utils.io import ensure_dir, read_csv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "figures"
QUALITY_NAMES = {0: "reject", 1: "caution", 2: "accept"}
QUALITY_COLORS = {0: "#b23a48", 1: "#e0a458", 2: "#3a7d44"}


def _set_report_style() -> None:
    """Apply a restrained, consistent style to all report figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.titleweight": "bold",
            "font.size": 10,
            "savefig.bbox": "tight",
        }
    )


def _column_key(name: object) -> str:
    """Normalize spreadsheet headers for alias matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _find_column(
    frame: pd.DataFrame,
    aliases: Sequence[str],
) -> str | None:
    """Find a source column using normalized aliases."""
    keyed = {_column_key(column): column for column in frame.columns}
    for alias in aliases:
        match = keyed.get(_column_key(alias))
        if match is not None:
            return match
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


def _parse_quality(value: Any) -> float:
    """Parse numeric or named reject/caution/accept labels."""
    if not _is_present(value):
        return float("nan")
    text = str(value).strip().lower()
    aliases = {
        "0": 0,
        "0.0": 0,
        "reject": 0,
        "poor": 0,
        "bad": 0,
        "1": 1,
        "1.0": 1,
        "caution": 1,
        "intermediate": 1,
        "2": 2,
        "2.0": 2,
        "accept": 2,
        "good": 2,
    }
    return float(aliases[text]) if text in aliases else float("nan")


def _parse_binary(value: Any) -> float:
    """Parse common binary task encodings."""
    if not _is_present(value):
        return float("nan")
    text = str(value).strip().lower()
    if text in {"0", "0.0", "false", "no", "negative", "reject", "bad"}:
        return 0.0
    if text in {"1", "1.0", "true", "yes", "positive", "accept", "good"}:
        return 1.0
    return float("nan")


def _quality_labels(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, int]:
    """Find true/predicted IQA labels and infer binary versus ternary."""
    true_column = _find_column(
        frame,
        (
            "true_quality",
            "quality_true_label",
            "true_label",
            "target_quality",
            "target",
            "label",
        ),
    )
    predicted_column = _find_column(
        frame,
        (
            "pred_quality",
            "predicted_quality",
            "quality_pred_label",
            "pred_label",
            "quality_ternary",
            "quality_label_name",
        ),
    )
    truth = (
        frame[true_column].map(_parse_quality)
        if true_column is not None
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    prediction = (
        frame[predicted_column].map(_parse_quality)
        if predicted_column is not None
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    ternary_probabilities = all(
        _find_column(frame, aliases) is not None
        for aliases in (
            ("prob_reject", "prob_0"),
            ("prob_caution",),
            ("prob_accept", "prob_2"),
        )
    )
    observed = pd.concat([truth, prediction]).dropna()
    num_classes = 3 if ternary_probabilities or (len(observed) and observed.max() >= 2) else 2
    return truth, prediction, num_classes


def _probability_matrix(
    frame: pd.DataFrame,
    num_classes: int,
) -> np.ndarray | None:
    """Extract binary or ternary softmax probabilities when available."""
    if num_classes == 3:
        aliases = (
            ("prob_reject", "prob_0"),
            ("prob_caution", "prob_1"),
            ("prob_accept", "prob_2"),
        )
    else:
        aliases = (
            ("prob_0", "prob_reject", "probability_0"),
            ("prob_1", "prob_accept", "probability_1"),
        )
    columns = [_find_column(frame, values) for values in aliases]
    if any(column is None for column in columns):
        return None
    probabilities = np.column_stack(
        [pd.to_numeric(frame[column], errors="coerce") for column in columns]
    )
    return probabilities


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    """Save and close one figure."""
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _confusion_matrix_figure(
    truth: pd.Series,
    prediction: pd.Series,
    num_classes: int,
    path: Path,
) -> Path:
    """Render an annotated IQA confusion matrix."""
    valid = truth.notna() & prediction.notna()
    figure, axis = plt.subplots(figsize=(5.5, 4.8))
    if not valid.any():
        axis.text(0.5, 0.5, "True/predicted IQA labels unavailable", ha="center", va="center")
        axis.set_axis_off()
        return _save_figure(figure, path)

    labels = list(range(num_classes))
    matrix = confusion_matrix(
        truth.loc[valid].astype(int),
        prediction.loc[valid].astype(int),
        labels=labels,
    )
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Cases")
    names = (
        [QUALITY_NAMES[index] for index in labels]
        if num_classes == 3
        else ["reject", "accept"]
    )
    axis.set_xticks(labels, labels=names)
    axis.set_yticks(labels, labels=names)
    axis.set_xlabel("Predicted quality")
    axis.set_ylabel("True quality")
    axis.set_title("IQA confusion matrix")
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in labels:
        for column in labels:
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontweight="bold",
            )
    return _save_figure(figure, path)


def _binary_roc_data(
    frame: pd.DataFrame,
    quality_truth: pd.Series,
    num_classes: int,
    probabilities: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Prefer downstream-task ROC fields, then fall back to binary IQA fields."""
    task_true_column = _find_column(frame, ("task_true_label", "task_source_true_label"))
    task_probability_column = _find_column(
        frame,
        ("task_probability", "task_source_prob_1", "task_prob_1"),
    )
    if task_true_column is not None and task_probability_column is not None:
        labels = frame[task_true_column].map(_parse_binary)
        scores = pd.to_numeric(frame[task_probability_column], errors="coerce")
    elif num_classes == 2 and probabilities is not None:
        labels = quality_truth.map(_parse_binary)
        scores = pd.Series(probabilities[:, 1], index=frame.index)
    else:
        return None
    valid = labels.notna() & scores.notna()
    labels_array = labels.loc[valid].astype(int).to_numpy()
    scores_array = scores.loc[valid].astype(float).to_numpy()
    if len(labels_array) == 0 or np.unique(labels_array).size != 2:
        return None
    return labels_array, scores_array


def _roc_figure(labels: np.ndarray, scores: np.ndarray, path: Path) -> Path:
    """Render a binary receiver operating characteristic curve."""
    false_positive, true_positive, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    figure, axis = plt.subplots(figsize=(5.8, 5.0))
    axis.plot(false_positive, true_positive, color="#35618f", linewidth=2.2, label=f"AUC = {auc:.3f}")
    axis.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("Binary classification ROC curve")
    axis.legend(loc="lower right", frameon=False)
    return _save_figure(figure, path)


def _class_distribution_figure(prediction: pd.Series, path: Path) -> Path:
    """Render ternary predicted-class counts."""
    counts = prediction.dropna().astype(int).value_counts().reindex([0, 1, 2], fill_value=0)
    figure, axis = plt.subplots(figsize=(6.0, 4.5))
    bars = axis.bar(
        [QUALITY_NAMES[index] for index in counts.index],
        counts.values,
        color=[QUALITY_COLORS[index] for index in counts.index],
    )
    axis.bar_label(bars, padding=3)
    axis.set_ylabel("Cases")
    axis.set_xlabel("Predicted IQA quality")
    axis.set_title("Ternary IQA class distribution")
    axis.set_ylim(0, max(counts.max() * 1.15, 1))
    return _save_figure(figure, path)


def _histogram_figure(
    values: pd.Series,
    title: str,
    xlabel: str,
    color: str,
    path: Path,
) -> Path:
    """Render one clean univariate histogram."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    figure, axis = plt.subplots(figsize=(6.2, 4.5))
    if len(clean):
        bins = min(30, max(5, int(np.sqrt(len(clean)))))
        axis.hist(clean, bins=bins, color=color, alpha=0.85, edgecolor="white")
        axis.axvline(clean.median(), color="#222222", linestyle="--", linewidth=1.2, label="Median")
        axis.legend(frameon=False)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Cases")
        axis.set_title(title)
    else:
        axis.text(0.5, 0.5, f"{xlabel} unavailable", ha="center", va="center")
        axis.set_axis_off()
    return _save_figure(figure, path)


def _quality_accuracy_figure(summary: pd.DataFrame, path: Path) -> Path | None:
    """Render downstream accuracy by predicted quality from a summary table."""
    group_column = _find_column(summary, ("quality_group", "quality_label_name"))
    accuracy_column = _find_column(
        summary,
        ("task_accuracy", "downstream_accuracy", "accuracy"),
    )
    if group_column is None or accuracy_column is None:
        return None
    values = summary[[group_column, accuracy_column]].copy()
    values["_quality"] = values[group_column].map(_parse_quality)
    values["_accuracy"] = pd.to_numeric(values[accuracy_column], errors="coerce")
    values = values.dropna(subset=["_quality", "_accuracy"]).drop_duplicates("_quality")
    values = values.sort_values("_quality")
    if values.empty:
        return None

    figure, axis = plt.subplots(figsize=(6.2, 4.5))
    bars = axis.bar(
        [QUALITY_NAMES[int(value)] for value in values["_quality"]],
        values["_accuracy"],
        color=[QUALITY_COLORS[int(value)] for value in values["_quality"]],
    )
    axis.bar_label(bars, labels=[f"{value:.2f}" for value in values["_accuracy"]], padding=3)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Downstream accuracy")
    axis.set_xlabel("Predicted IQA quality")
    axis.set_title("Downstream accuracy by predicted quality")
    return _save_figure(figure, path)


def _dice_boxplot_figure(
    frame: pd.DataFrame,
    prediction: pd.Series,
    path: Path,
) -> Path | None:
    """Render raw segmentation Dice distributions by predicted quality."""
    dice_column = _find_column(
        frame,
        ("seg_dice", "dice", "dice_score", "segmentation_dice"),
    )
    if dice_column is None:
        return None
    dice = pd.to_numeric(frame[dice_column], errors="coerce")
    groups = []
    labels = []
    colors = []
    for quality in (0, 1, 2):
        values = dice.loc[prediction.eq(quality)].dropna().to_numpy()
        if len(values):
            groups.append(values)
            labels.append(QUALITY_NAMES[quality])
            colors.append(QUALITY_COLORS[quality])
    if not groups:
        return None

    figure, axis = plt.subplots(figsize=(6.2, 4.7))
    boxes = axis.boxplot(
        groups,
        tick_labels=labels,
        patch_artist=True,
        showfliers=True,
    )
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in boxes["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Segmentation Dice")
    axis.set_xlabel("Predicted IQA quality")
    axis.set_title("Segmentation Dice by predicted quality")
    return _save_figure(figure, path)


def _merge_segmentation_metrics(
    quality: pd.DataFrame,
    segmentation: pd.DataFrame,
) -> pd.DataFrame:
    """Attach raw segmentation metrics using acquisition-safe case identity."""
    left = quality.copy()
    right = segmentation.copy()
    existing_dice = _find_column(
        left, ("seg_dice", "dice", "dice_score", "segmentation_dice")
    )
    if existing_dice is not None:
        left = left.drop(columns=existing_dice)
    join_keys: list[str] = []
    for canonical, aliases in (
        ("patient_id", ("patient_id", "patient")),
        ("scan_id", ("scan_id", "scan", "study_id")),
    ):
        left_column = _find_column(left, aliases)
        right_column = _find_column(right, aliases)
        if left_column is None or right_column is None:
            raise ValueError(
                f"Both quality and segmentation tables require {canonical}."
            )
        key = f"_{canonical}_key"
        left[key] = left[left_column].astype(str).str.strip().str.casefold()
        right[key] = right[right_column].astype(str).str.strip().str.casefold()
        join_keys.append(key)

    left_acquisition = _find_column(left, ("acquisition_id", "acquisition"))
    right_acquisition = _find_column(right, ("acquisition_id", "acquisition"))
    if left_acquisition is not None and right_acquisition is not None:
        key = "_acquisition_id_key"
        left[key] = left[left_acquisition].astype(str).str.strip().str.casefold()
        right[key] = right[right_acquisition].astype(str).str.strip().str.casefold()
        join_keys.append(key)

    if left.duplicated(join_keys).any() or right.duplicated(join_keys).any():
        raise ValueError(
            "Quality/segmentation identity is not one-to-one. Preserve acquisition_id "
            "when a scan has multiple acquisitions."
        )
    dice_column = _find_column(
        right, ("seg_dice", "dice", "dice_score", "segmentation_dice")
    )
    if dice_column is None:
        raise ValueError("Segmentation metrics CSV does not contain a Dice column.")
    metrics = right[join_keys + [dice_column]].rename(columns={dice_column: "seg_dice"})
    merged = left.merge(metrics, on=join_keys, how="left", validate="one_to_one")
    if merged["seg_dice"].notna().sum() == 0:
        raise ValueError("No quality predictions matched segmentation metrics.")
    return merged


def generate_figures(args: argparse.Namespace) -> list[Path]:
    """Generate every applicable report figure and return saved paths."""
    _set_report_style()
    output_dir = ensure_dir(args.out_dir)
    quality = read_csv(args.quality_predictions_csv)
    truth, prediction, num_classes = _quality_labels(quality)
    probabilities = _probability_matrix(quality, num_classes)
    created: list[Path] = []

    created.append(
        _confusion_matrix_figure(
            truth,
            prediction,
            num_classes,
            output_dir / "iqa_confusion_matrix.png",
        )
    )
    roc_data = _binary_roc_data(quality, truth, num_classes, probabilities)
    if roc_data is not None:
        created.append(
            _roc_figure(*roc_data, output_dir / "binary_roc_curve.png")
        )
    else:
        print("Skipping ROC curve: paired binary labels/probabilities unavailable.")

    if num_classes == 3:
        created.append(
            _class_distribution_figure(
                prediction,
                output_dir / "ternary_class_distribution.png",
            )
        )
    else:
        print("Skipping ternary distribution: predictions are binary.")

    confidence_column = _find_column(
        quality,
        ("quality_confidence", "confidence", "max_probability"),
    )
    confidence = (
        pd.to_numeric(quality[confidence_column], errors="coerce")
        if confidence_column is not None
        else (
            pd.Series(np.nanmax(probabilities, axis=1), index=quality.index)
            if probabilities is not None
            else pd.Series(dtype=float)
        )
    )
    created.append(
        _histogram_figure(
            confidence,
            "IQA prediction confidence",
            "Confidence",
            "#35618f",
            output_dir / "confidence_histogram.png",
        )
    )

    entropy_column = _find_column(quality, ("quality_entropy", "entropy"))
    if entropy_column is not None:
        entropy = pd.to_numeric(quality[entropy_column], errors="coerce")
    elif probabilities is not None:
        clipped = np.clip(probabilities, np.finfo(float).tiny, 1.0)
        entropy = pd.Series(-np.sum(clipped * np.log(clipped), axis=1), index=quality.index)
    else:
        entropy = pd.Series(dtype=float)
    created.append(
        _histogram_figure(
            entropy,
            "IQA predictive entropy",
            "Entropy",
            "#7b5ea7",
            output_dir / "entropy_histogram.png",
        )
    )

    if args.novelty_csv is not None:
        novelty = read_csv(args.novelty_csv)
        novelty_column = _find_column(
            novelty,
            ("novelty_distance", "mahalanobis_distance", "novelty_score"),
        )
        if novelty_column is not None:
            created.append(
                _histogram_figure(
                    novelty[novelty_column],
                    "Feature-space novelty distribution",
                    "Mahalanobis novelty distance",
                    "#6f4a8e",
                    output_dir / "novelty_score_histogram.png",
                )
            )
        else:
            print("Skipping novelty histogram: novelty-distance column unavailable.")

    if args.quality_task_summary_csv is not None:
        task_summary = read_csv(args.quality_task_summary_csv)
        accuracy_path = _quality_accuracy_figure(
            task_summary,
            output_dir / "quality_group_vs_downstream_accuracy.png",
        )
        if accuracy_path is not None:
            created.append(accuracy_path)
        else:
            print("Skipping accuracy plot: quality groups or task accuracy unavailable.")

    dice_frame = quality
    dice_prediction = prediction
    if args.segmentation_metrics_csv is not None:
        dice_frame = _merge_segmentation_metrics(
            quality, read_csv(args.segmentation_metrics_csv)
        )
        _, dice_prediction, _ = _quality_labels(dice_frame)
    dice_path = _dice_boxplot_figure(
        dice_frame,
        dice_prediction,
        output_dir / "quality_group_vs_segmentation_dice_boxplot.png",
    )
    if dice_path is not None:
        created.append(dice_path)
    else:
        print("Skipping Dice boxplot: raw segmentation Dice unavailable.")

    for path in created:
        print(f"Saved figure: {path}")
    return created


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse report-figure command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate report-ready prostate MRI IQA figures."
    )
    parser.add_argument("--quality_predictions_csv", type=Path, required=True)
    parser.add_argument("--novelty_csv", type=Path)
    parser.add_argument("--quality_task_summary_csv", type=Path)
    parser.add_argument(
        "--segmentation_metrics_csv",
        type=Path,
        help="Optional raw Dice table for the quality-group segmentation boxplot.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    generate_figures(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
