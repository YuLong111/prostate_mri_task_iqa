"""Metrics for binary 3D segmentation tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def binary_segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> dict[str, float]:
    """Calculate Dice, IoU, ASD, and HD95 for one binary 3D mask.

    Surface distances are reported in the physical units represented by
    ``spacing``. If exactly one mask is empty, ASD and HD95 are returned as
    NaN because no finite surface-to-surface distance exists.
    """
    predicted = np.asarray(prediction).astype(bool, copy=False).squeeze()
    reference = np.asarray(target).astype(bool, copy=False).squeeze()
    if predicted.shape != reference.shape or predicted.ndim != 3:
        raise ValueError(
            "prediction and target must be matching 3D arrays, received "
            f"{predicted.shape} and {reference.shape}."
        )
    voxel_spacing = tuple(float(value) for value in spacing)
    if len(voxel_spacing) != 3 or any(value <= 0 for value in voxel_spacing):
        raise ValueError("spacing must contain three positive values.")

    predicted_count = int(predicted.sum())
    reference_count = int(reference.sum())
    intersection = int(np.logical_and(predicted, reference).sum())
    union = int(np.logical_or(predicted, reference).sum())
    if predicted_count == 0 and reference_count == 0:
        return {"dice": 1.0, "iou": 1.0, "asd": 0.0, "hd95": 0.0}

    dice = 2.0 * intersection / max(predicted_count + reference_count, 1)
    iou = intersection / max(union, 1)
    if predicted_count == 0 or reference_count == 0:
        return {"dice": float(dice), "iou": float(iou), "asd": np.nan, "hd95": np.nan}

    predicted_surface = predicted & ~binary_erosion(predicted, border_value=0)
    reference_surface = reference & ~binary_erosion(reference, border_value=0)
    distance_to_reference = distance_transform_edt(
        ~reference_surface, sampling=voxel_spacing
    )
    distance_to_prediction = distance_transform_edt(
        ~predicted_surface, sampling=voxel_spacing
    )
    distances = np.concatenate(
        [
            distance_to_reference[predicted_surface],
            distance_to_prediction[reference_surface],
        ]
    )
    return {
        "dice": float(dice),
        "iou": float(iou),
        "asd": float(distances.mean()),
        "hd95": float(np.percentile(distances, 95)),
    }


def mean_finite_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    """Aggregate finite segmentation metrics across case dictionaries."""
    result: dict[str, float | None] = {}
    for key in ("dice", "iou", "asd", "hd95"):
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        result[f"mean_{key}"] = float(finite.mean()) if finite.size else None
    result["num_cases"] = float(len(rows))
    return result
