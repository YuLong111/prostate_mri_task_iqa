"""Generate visual and metadata quality-control outputs for prostate MRI."""

from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep Matplotlib's cache out of restricted or OneDrive-managed home folders.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "prostate_iqa_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from prostate_iqa.utils.io import ensure_dir, read_json, write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QC_SUMMARY_CSV = PROJECT_ROOT / "reports" / "qc" / "qc_summary.csv"
MODALITIES = ("t2", "dwi", "adc", "prostate_mask", "lesion_mask")
IMAGE_MODALITIES = ("t2", "dwi", "adc")
MASK_MODALITIES = ("prostate_mask", "lesion_mask")
PERCENTILES = (1, 5, 50, 95, 99)

QC_SUMMARY_COLUMNS = (
    "patient_id",
    "scan_id",
    "distortion_status",
    "acquisition_id",
    "case_output_dir",
    "status",
    "missing_modalities",
    "warnings",
    "t2_shape",
    "dwi_shape",
    "adc_shape",
    "prostate_mask_shape",
    "lesion_mask_shape",
    "t2_spacing",
    "dwi_spacing",
    "adc_spacing",
    "prostate_mask_spacing",
    "lesion_mask_spacing",
    "t2_intensity_min",
    "t2_intensity_max",
    "dwi_intensity_min",
    "dwi_intensity_max",
    "adc_intensity_min",
    "adc_intensity_max",
    "prostate_mask_nonzero_voxels",
    "lesion_mask_nonzero_voxels",
)


@dataclass
class LoadedNifti:
    """A loaded NIfTI image and its derived metadata."""

    path: Path
    image: nib.Nifti1Image
    data: np.ndarray
    display_data: np.ndarray | None
    metadata: dict[str, Any]


def _safe_component(value: Any, fallback: str) -> str:
    """Create a filesystem-safe case identifier component."""
    if value is None or str(value).strip() == "":
        return fallback
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return safe.strip("._") or fallback


def _resolve_case_path(value: Any, modality: str, warnings_list: list[str]) -> Path | None:
    """Resolve an optional manifest path, warning about ambiguous path lists."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if ";" in text:
        paths = [item.strip() for item in text.split(";") if item.strip()]
        warnings_list.append(
            f"{modality}: {len(paths)} paths were listed in one field; skipped. "
            "Rebuild the manifest so each acquisition has its own row."
        )
        return None
    return Path(text).expanduser()


def _display_volume(
    data: np.ndarray,
    modality: str,
    warnings_list: list[str],
) -> np.ndarray | None:
    """Convert a NIfTI array to a displayable 3D volume without resampling."""
    volume = np.squeeze(data)
    if volume.ndim == 2:
        return volume[:, :, np.newaxis]
    if volume.ndim < 2:
        warnings_list.append(
            f"{modality}: array has {volume.ndim} dimensions and cannot be plotted."
        )
        return None
    if volume.ndim > 3:
        original_shape = tuple(int(value) for value in volume.shape)
        warnings_list.append(
            f"{modality}: {original_shape} is greater than 3D; no frame was "
            "selected automatically. Split the acquisition into explicit 3D volumes."
        )
        return None
    return volume


def _intensity_statistics(data: np.ndarray) -> dict[str, float | None]:
    """Calculate finite-value intensity statistics."""
    finite = np.asarray(data, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    statistics: dict[str, float | None] = {
        "min": None,
        "max": None,
        **{f"p{percentile:02d}": None for percentile in PERCENTILES},
    }
    if finite.size == 0:
        return statistics

    statistics["min"] = float(np.min(finite))
    statistics["max"] = float(np.max(finite))
    values = np.percentile(finite, PERCENTILES)
    for percentile, value in zip(PERCENTILES, values, strict=True):
        statistics[f"p{percentile:02d}"] = float(value)
    return statistics


def _metadata_for_image(
    image: nib.Nifti1Image,
    data: np.ndarray,
    path: Path,
    is_mask: bool,
) -> dict[str, Any]:
    """Build JSON-serializable NIfTI metadata."""
    metadata: dict[str, Any] = {
        "path": str(path),
        "shape": [int(value) for value in image.shape],
        "voxel_spacing": [
            float(value) for value in image.header.get_zooms()[: len(image.shape)]
        ],
        "affine": np.asarray(image.affine, dtype=float).tolist(),
        "dtype": str(image.get_data_dtype()),
        "intensity": _intensity_statistics(data),
    }
    if is_mask:
        finite = np.isfinite(data)
        metadata["nonzero_voxel_count"] = int(np.count_nonzero(data[finite]))
    return metadata


def load_nifti(
    value: Any,
    modality: str,
    warnings_list: list[str],
) -> LoadedNifti | None:
    """Load one optional NIfTI safely and return its metadata."""
    path = _resolve_case_path(value, modality, warnings_list)
    if path is None:
        return None
    if not path.is_file():
        warnings_list.append(f"{modality}: file does not exist: {path}")
        return None

    try:
        image = nib.load(str(path))
        data = image.get_fdata(dtype=np.float32)
    except Exception as error:  # nibabel raises several format/IO exception types
        warnings_list.append(f"{modality}: failed to load {path}: {error}")
        return None

    display_data = _display_volume(data, modality, warnings_list)
    metadata = _metadata_for_image(
        image,
        data,
        path,
        is_mask=modality in MASK_MODALITIES,
    )
    if not np.isfinite(data).all():
        invalid_count = int(data.size - np.count_nonzero(np.isfinite(data)))
        warnings_list.append(
            f"{modality}: contains {invalid_count} non-finite intensity values."
        )
    return LoadedNifti(path, image, data, display_data, metadata)


def _window(data: np.ndarray) -> tuple[float, float]:
    """Return a robust display window using finite non-background voxels."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    nonzero = finite[finite != 0]
    values = nonzero if nonzero.size >= 100 else finite
    low, high = np.percentile(values, (1, 99))
    if not np.isfinite(low) or not np.isfinite(high):
        return 0.0, 1.0
    if high <= low:
        low = float(np.min(values))
        high = float(np.max(values))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _slice_indices(volume: np.ndarray, count: int = 12) -> list[int]:
    """Choose evenly spaced axial slices while avoiding extreme edges."""
    depth = volume.shape[2]
    number = min(count, depth)
    if number <= 1:
        return [0]
    start = int(round(0.08 * (depth - 1)))
    stop = int(round(0.92 * (depth - 1)))
    return sorted(set(np.linspace(start, stop, number, dtype=int).tolist()))


def _mask_slice_indices(mask: np.ndarray, count: int = 12) -> list[int]:
    """Choose axial slices spanning a mask, or central slices when it is empty."""
    occupied = np.flatnonzero(np.any(mask > 0, axis=(0, 1)))
    if occupied.size == 0:
        return _slice_indices(mask, count)
    number = min(count, occupied.size)
    return sorted(
        set(
            np.linspace(occupied[0], occupied[-1], number, dtype=int).tolist()
        )
    )


def _display_slice(data: np.ndarray, index: int, low: float, high: float) -> np.ndarray:
    """Prepare one axial slice for imshow."""
    axial = np.rot90(data[:, :, index])
    return np.nan_to_num(axial, nan=low, posinf=high, neginf=low)


def save_contact_sheet(volume: np.ndarray, title: str, out_path: Path) -> None:
    """Save an axial contact sheet for one image volume."""
    indices = _slice_indices(volume)
    columns = 4
    rows = math.ceil(len(indices) / columns)
    low, high = _window(volume)
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for axis, index in zip(axes_array, indices, strict=False):
        axis.imshow(
            _display_slice(volume, index, low, high),
            cmap="gray",
            vmin=low,
            vmax=high,
        )
        axis.set_title(f"z={index}", fontsize=9)
        axis.axis("off")
    for axis in axes_array[len(indices) :]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _grids_match(image: LoadedNifti, mask: LoadedNifti) -> bool:
    """Check whether image and mask arrays share a directly overlayable grid."""
    if image.display_data is None or mask.display_data is None:
        return False
    return image.display_data.shape == mask.display_data.shape and np.allclose(
        image.image.affine,
        mask.image.affine,
        rtol=1e-4,
        atol=1e-3,
    )


def save_mask_overlay(
    image: LoadedNifti,
    mask: LoadedNifti,
    title: str,
    out_path: Path,
    warnings_list: list[str],
) -> bool:
    """Save a prostate-mask overlay when image and mask grids agree."""
    if image.display_data is None or mask.display_data is None:
        return False
    if not _grids_match(image, mask):
        warnings_list.append(
            f"{title}: overlay skipped because image and mask grids differ "
            f"({image.display_data.shape} vs {mask.display_data.shape})."
        )
        return False

    image_data = image.display_data
    mask_data = mask.display_data
    indices = _mask_slice_indices(mask_data)
    columns = 4
    rows = math.ceil(len(indices) / columns)
    low, high = _window(image_data)
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for axis, index in zip(axes_array, indices, strict=False):
        base = _display_slice(image_data, index, low, high)
        overlay = np.rot90(mask_data[:, :, index] > 0)
        axis.imshow(base, cmap="gray", vmin=low, vmax=high)
        axis.imshow(np.ma.masked_where(~overlay, overlay), cmap="autumn", alpha=0.4)
        axis.set_title(f"z={index}", fontsize=9)
        axis.axis("off")
    for axis in axes_array[len(indices) :]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return True


def save_t2_dwi_side_by_side(
    t2: LoadedNifti,
    dwi: LoadedNifti,
    out_path: Path,
    warnings_list: list[str],
) -> bool:
    """Save approximate native-grid T2/DWI comparisons at relative depths."""
    if t2.display_data is None or dwi.display_data is None:
        return False
    t2_data = t2.display_data
    dwi_data = dwi.display_data
    if t2_data.shape != dwi_data.shape or not np.allclose(
        t2.image.affine,
        dwi.image.affine,
        rtol=1e-4,
        atol=1e-3,
    ):
        warnings_list.append(
            "T2-DWI side-by-side uses relative slice positions because the "
            "native grids differ; it is not a registration assessment."
        )

    fractions = (0.3, 0.5, 0.7)
    t2_window = _window(t2_data)
    dwi_window = _window(dwi_data)
    figure, axes = plt.subplots(len(fractions), 2, figsize=(8, 11))
    for row, fraction in enumerate(fractions):
        t2_index = int(round(fraction * (t2_data.shape[2] - 1)))
        dwi_index = int(round(fraction * (dwi_data.shape[2] - 1)))
        axes[row, 0].imshow(
            _display_slice(t2_data, t2_index, *t2_window),
            cmap="gray",
            vmin=t2_window[0],
            vmax=t2_window[1],
        )
        axes[row, 1].imshow(
            _display_slice(dwi_data, dwi_index, *dwi_window),
            cmap="gray",
            vmin=dwi_window[0],
            vmax=dwi_window[1],
        )
        axes[row, 0].set_title(f"T2 z={t2_index}")
        axes[row, 1].set_title(f"DWI z={dwi_index}")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    figure.suptitle("Approximate T2-DWI side-by-side (native grids)")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return True


def _summary_row(case_summary: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    """Flatten selected case metadata into the global QC CSV schema."""
    modalities = case_summary["modalities"]
    row: dict[str, Any] = {
        "patient_id": case_summary["patient_id"],
        "scan_id": case_summary["scan_id"],
        "distortion_status": case_summary["distortion_status"],
        "acquisition_id": case_summary["acquisition_id"],
        "case_output_dir": str(case_dir),
        "status": "warning" if case_summary["warnings"] else "ok",
        "missing_modalities": ";".join(case_summary["missing_modalities"]),
        "warnings": " | ".join(case_summary["warnings"]),
    }
    for modality in MODALITIES:
        metadata = modalities.get(modality)
        row[f"{modality}_shape"] = (
            "x".join(map(str, metadata["shape"])) if metadata else ""
        )
        row[f"{modality}_spacing"] = (
            "x".join(f"{value:g}" for value in metadata["voxel_spacing"])
            if metadata
            else ""
        )
    for modality in IMAGE_MODALITIES:
        metadata = modalities.get(modality)
        row[f"{modality}_intensity_min"] = (
            metadata["intensity"]["min"] if metadata else None
        )
        row[f"{modality}_intensity_max"] = (
            metadata["intensity"]["max"] if metadata else None
        )
    for modality in MASK_MODALITIES:
        metadata = modalities.get(modality)
        row[f"{modality}_nonzero_voxels"] = (
            metadata["nonzero_voxel_count"] if metadata else None
        )
    return {column: row.get(column) for column in QC_SUMMARY_COLUMNS}


def process_case(
    case: dict[str, Any],
    out_dir: Path,
    case_index: int,
    used_names: set[str] | None = None,
) -> dict[str, Any]:
    """Load one case, save available QC artifacts, and return a CSV row."""
    patient_id = str(case.get("patient_id") or "")
    scan_id = str(case.get("scan_id") or "")
    distortion_status = str(case.get("distortion_status") or "")
    acquisition_id = str(case.get("acquisition_id") or "")
    identity = acquisition_id or "_".join(
        part for part in (scan_id or patient_id, distortion_status) if part
    )
    base_name = _safe_component(identity, f"case_{case_index:05d}")
    case_name = base_name
    if used_names is not None:
        suffix = 2
        while case_name in used_names:
            case_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(case_name)
    case_dir = ensure_dir(out_dir / case_name)

    warnings_list: list[str] = []
    loaded: dict[str, LoadedNifti | None] = {}
    for modality in MODALITIES:
        loaded[modality] = load_nifti(case.get(modality), modality, warnings_list)

    missing_modalities = [name for name in MODALITIES if loaded[name] is None]
    figures: list[str] = []
    for modality in IMAGE_MODALITIES:
        image = loaded[modality]
        if image is not None and image.display_data is not None:
            figure_path = case_dir / f"{modality}_contact_sheet.png"
            save_contact_sheet(
                image.display_data,
                f"{patient_id} | {scan_id} | {modality.upper()}",
                figure_path,
            )
            figures.append(figure_path.name)

    prostate_mask = loaded["prostate_mask"]
    if prostate_mask is not None:
        for modality in ("t2", "dwi"):
            image = loaded[modality]
            if image is None:
                continue
            figure_path = case_dir / f"{modality}_prostate_mask_overlay.png"
            if save_mask_overlay(
                image,
                prostate_mask,
                f"{patient_id} | {modality.upper()} + prostate mask",
                figure_path,
                warnings_list,
            ):
                figures.append(figure_path.name)

    t2 = loaded["t2"]
    dwi = loaded["dwi"]
    if t2 is not None and dwi is not None:
        figure_path = case_dir / "t2_dwi_side_by_side.png"
        if save_t2_dwi_side_by_side(t2, dwi, figure_path, warnings_list):
            figures.append(figure_path.name)

    case_summary = {
        "patient_id": patient_id,
        "scan_id": scan_id,
        "distortion_status": distortion_status,
        "acquisition_id": acquisition_id,
        "modalities": {
            modality: image.metadata if image is not None else None
            for modality, image in loaded.items()
        },
        "missing_modalities": missing_modalities,
        "mask_nonzero_voxel_count": {
            modality: (
                loaded[modality].metadata["nonzero_voxel_count"]
                if loaded[modality] is not None
                else None
            )
            for modality in MASK_MODALITIES
        },
        "warnings": warnings_list,
        "figures": figures,
    }
    write_json(case_summary, case_dir / "qc_summary.json")
    return _summary_row(case_summary, case_dir)


def _load_datalist(path: Path) -> list[dict[str, Any]]:
    """Load a list-style or common MONAI-style datalist JSON."""
    payload = read_json(path)
    if isinstance(payload, list):
        cases = payload
    elif isinstance(payload, dict):
        cases = next(
            (
                payload[key]
                for key in ("data", "training", "validation", "test")
                if isinstance(payload.get(key), list)
            ),
            None,
        )
        if cases is None:
            raise ValueError(f"No case list found in datalist JSON: {path}")
    else:
        raise ValueError(f"Datalist JSON must contain a list or mapping: {path}")

    if not all(isinstance(case, dict) for case in cases):
        raise ValueError(f"Every datalist item must be a JSON object: {path}")
    return cases


def _positive_integer(value: str) -> int:
    """Argparse type for positive integer limits."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate prostate MRI NIfTI visual QC and metadata summaries."
    )
    parser.add_argument(
        "--datalist_json",
        type=Path,
        required=True,
        help="Input JSON containing patient/scan records.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory for per-case QC figures and JSON summaries.",
    )
    parser.add_argument(
        "--max_cases",
        type=_positive_integer,
        default=None,
        help="Optional maximum number of cases to process.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate QC outputs for the requested datalist."""
    args = parse_args(argv)
    cases = _load_datalist(args.datalist_json)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    output_dir = ensure_dir(args.out_dir)
    rows: list[dict[str, Any]] = []
    used_names: set[str] = set()
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        row = process_case(case, output_dir, index, used_names)
        rows.append(row)
        print(
            f"[{index}/{total}] {row['patient_id']} | {row['scan_id']} | "
            f"status={row['status']}"
        )

    summary_frame = pd.DataFrame(rows, columns=QC_SUMMARY_COLUMNS)
    summary_path = write_csv(summary_frame, QC_SUMMARY_CSV)
    print(f"Saved {len(rows):,} case QC summaries to: {output_dir}")
    print(f"Saved global QC summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
