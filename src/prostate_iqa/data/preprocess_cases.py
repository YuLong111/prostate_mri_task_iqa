"""Preprocess prostate MRI cases into aligned, cropped training volumes."""

from __future__ import annotations

import argparse
import logging
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk

from prostate_iqa.utils.io import ensure_dir, read_csv, read_json, write_csv, write_json
from prostate_iqa.utils.logging import close_file_handlers, get_logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_MODALITIES = ("t2", "dwi", "adc")
MASK_MODALITIES = ("prostate_mask",)
MODALITIES = (*IMAGE_MODALITIES, *MASK_MODALITIES)
DEFAULT_SPACING = (0.5, 0.5, 1.0)
DEFAULT_ROI_SIZE = (160, 160, 64)
FAILURE_COLUMNS = (
    "patient_id",
    "scan_id",
    "distortion_status",
    "acquisition_id",
    "split",
    "error",
    "warnings",
)


def _safe_component(value: Any, fallback: str) -> str:
    """Create a filesystem-safe patient or scan identifier."""
    if value is None or str(value).strip() == "":
        return fallback
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return safe.strip("._") or fallback


def _is_present(value: Any) -> bool:
    """Return whether a manifest scalar contains a usable value."""
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _source_path(
    value: Any,
    modality: str,
    warnings_list: list[str],
) -> Path | None:
    """Resolve one source path, rejecting legacy collapsed path lists."""
    if not _is_present(value):
        return None
    text = str(value).strip()
    if ";" in text:
        raise ValueError(
            f"{modality}: multiple paths are stored in one manifest cell. "
            "Rebuild the inventory and master manifest so every acquisition "
            "has its own row."
        )
    return Path(text).expanduser()


def _ensure_3d(
    image: sitk.Image,
    modality: str,
    warnings_list: list[str],
) -> sitk.Image:
    """Require one physical 3D volume per manifest row."""
    if image.GetDimension() != 3:
        raise ValueError(
            f"{modality}: expected one 3D volume per manifest row, got "
            f"{image.GetDimension()}D with size {tuple(image.GetSize())}. "
            "Split 4D NIfTI files into separate 3D acquisitions first."
        )
    return image


def _load_images(
    case: dict[str, Any],
    warnings_list: list[str],
) -> tuple[dict[str, sitk.Image], list[str]]:
    """Load each available modality independently using SimpleITK."""
    images: dict[str, sitk.Image] = {}
    missing: list[str] = []
    for modality in MODALITIES:
        path = _source_path(case.get(modality), modality, warnings_list)
        if path is None:
            missing.append(modality)
            continue
        if not path.is_file():
            missing.append(modality)
            warnings_list.append(f"{modality}: source file does not exist: {path}")
            continue
        try:
            image = sitk.ReadImage(str(path))
            images[modality] = _ensure_3d(image, modality, warnings_list)
        except Exception as error:
            missing.append(modality)
            warnings_list.append(f"{modality}: failed to load {path}: {error}")
    return images, missing


def _geometry_matches(left: sitk.Image, right: sitk.Image) -> bool:
    """Check whether two SimpleITK images share the same physical grid."""
    return (
        left.GetSize() == right.GetSize()
        and np.allclose(left.GetSpacing(), right.GetSpacing(), rtol=1e-5, atol=1e-5)
        and np.allclose(left.GetOrigin(), right.GetOrigin(), rtol=1e-5, atol=1e-4)
        and np.allclose(left.GetDirection(), right.GetDirection(), rtol=1e-5, atol=1e-5)
    )


def _target_reference(
    reference: sitk.Image,
    spacing: tuple[float, float, float],
) -> sitk.Image:
    """Create a target grid at fixed spacing while preserving physical center."""
    old_size = np.asarray(reference.GetSize(), dtype=float)
    old_spacing = np.asarray(reference.GetSpacing(), dtype=float)
    new_spacing = np.asarray(spacing, dtype=float)
    new_size = np.maximum(
        1,
        np.rint(old_size * old_spacing / new_spacing).astype(int),
    )

    old_center_index = ((old_size - 1.0) / 2.0).tolist()
    old_center = np.asarray(
        reference.TransformContinuousIndexToPhysicalPoint(old_center_index),
        dtype=float,
    )
    direction = np.asarray(reference.GetDirection(), dtype=float).reshape(3, 3)
    new_center_offset = direction @ (((new_size - 1.0) * new_spacing) / 2.0)
    new_origin = old_center - new_center_offset

    target = sitk.Image([int(value) for value in new_size], sitk.sitkFloat32)
    target.SetSpacing(tuple(float(value) for value in new_spacing))
    target.SetOrigin(tuple(float(value) for value in new_origin))
    target.SetDirection(reference.GetDirection())
    return target


def _resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    modality: str,
    is_mask: bool,
    warnings_list: list[str],
) -> sitk.Image:
    """Resample an image to a shared reference grid."""
    if not _geometry_matches(image, reference):
        warnings_list.append(
            f"{modality}: resampled from size={image.GetSize()}, "
            f"spacing={tuple(round(v, 5) for v in image.GetSpacing())} "
            f"to size={reference.GetSize()}, "
            f"spacing={tuple(round(v, 5) for v in reference.GetSpacing())}."
        )

    if is_mask:
        source = sitk.Cast(image > 0, sitk.sitkUInt8)
        interpolator = sitk.sitkNearestNeighbor
        pixel_type = sitk.sitkUInt8
    else:
        source = sitk.Cast(image, sitk.sitkFloat32)
        interpolator = sitk.sitkLinear
        pixel_type = sitk.sitkFloat32

    source_nonzero = int(np.count_nonzero(sitk.GetArrayViewFromImage(source)))
    resampled = sitk.Resample(
        source,
        reference,
        sitk.Transform(),
        interpolator,
        0.0,
        pixel_type,
    )
    result_nonzero = int(np.count_nonzero(sitk.GetArrayViewFromImage(resampled)))
    if source_nonzero > 0 and result_nonzero == 0:
        warnings_list.append(
            f"{modality}: resampling produced an all-zero volume; check "
            "origin, direction, and affine alignment."
        )
    return resampled


def _mask_center(mask: sitk.Image) -> tuple[int, int, int] | None:
    """Return the bounding-box center of a non-empty mask in x/y/z order."""
    array = sitk.GetArrayViewFromImage(mask)
    coordinates = np.argwhere(array > 0)
    if coordinates.size == 0:
        return None
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    center_zyx = np.rint((lower + upper) / 2.0).astype(int)
    return tuple(int(value) for value in center_zyx[::-1])


def _image_center(image: sitk.Image) -> tuple[int, int, int]:
    """Return the discrete center index of an image."""
    return tuple(int(round((size - 1) / 2.0)) for size in image.GetSize())


def _crop_or_pad(
    image: sitk.Image,
    center: tuple[int, int, int],
    roi_size: tuple[int, int, int],
    modality: str,
    warnings_list: list[str],
) -> sitk.Image:
    """Crop a fixed ROI around a center, padding outside image bounds."""
    image_size = np.asarray(image.GetSize(), dtype=int)
    roi = np.asarray(roi_size, dtype=int)
    center_array = np.asarray(center, dtype=int)
    start = center_array - roi // 2
    end = start + roi
    lower_pad = np.maximum(0, -start)
    upper_pad = np.maximum(0, end - image_size)

    if np.any(lower_pad) or np.any(upper_pad):
        warnings_list.append(
            f"{modality}: ROI required padding "
            f"lower={tuple(int(v) for v in lower_pad)}, "
            f"upper={tuple(int(v) for v in upper_pad)}."
        )
        image = sitk.ConstantPad(
            image,
            [int(value) for value in lower_pad],
            [int(value) for value in upper_pad],
            0.0,
        )

    padded_start = start + lower_pad
    return sitk.RegionOfInterest(
        image,
        [int(value) for value in roi],
        [int(value) for value in padded_start],
    )


def _normalize_image(
    image: sitk.Image,
    modality: str,
    warnings_list: list[str],
) -> tuple[sitk.Image, dict[str, float | None]]:
    """Clip finite foreground intensities to p1/p99 and scale to [0, 1]."""
    array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    finite = np.isfinite(array)
    invalid_count = int(array.size - np.count_nonzero(finite))
    if invalid_count:
        warnings_list.append(
            f"{modality}: replaced {invalid_count} non-finite values with zero."
        )
    values = array[finite]
    foreground = values[values != 0]
    percentile_values = foreground if foreground.size >= 100 else values

    if percentile_values.size == 0:
        low = high = None
        normalized = np.zeros_like(array, dtype=np.float32)
        warnings_list.append(f"{modality}: image contains no finite intensities.")
    else:
        low_value, high_value = np.percentile(percentile_values, (1, 99))
        low = float(low_value)
        high = float(high_value)
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            normalized = np.zeros_like(array, dtype=np.float32)
            warnings_list.append(
                f"{modality}: robust intensity range is degenerate; saved zeros."
            )
        else:
            clean = np.nan_to_num(array, nan=low, posinf=high, neginf=low)
            normalized = np.clip(clean, low, high)
            normalized = ((normalized - low) / (high - low)).astype(np.float32)
            normalized[array == 0] = 0.0

    output = sitk.GetImageFromArray(normalized)
    output.CopyInformation(image)
    return output, {"p01": low, "p99": high}


def _infer_split(datalist_path: Path) -> str:
    """Infer a split name from the standard datalist filename."""
    name = datalist_path.name.lower()
    if "test_locked" in name or re.search(r"(?:^|[_-])test(?:[_\-.]|$)", name):
        return "test_locked"
    if re.search(r"(?:^|[_-])val(?:idation)?(?:[_\-.]|$)", name):
        return "val"
    if "train" in name:
        return "train"
    return "unspecified"


def _case_split(case: dict[str, Any], default_split: str) -> str:
    """Use an explicit row split or the datalist-derived split."""
    value = case.get("split")
    if _is_present(value):
        return _safe_component(value, default_split)
    return default_split


def _output_paths(case_dir: Path) -> dict[str, Path]:
    """Return canonical processed paths for all supported modalities."""
    return {modality: case_dir / f"{modality}.nii.gz" for modality in MODALITIES}


def _record_for_case(
    case: dict[str, Any],
    split: str,
    case_dir: Path,
    output_paths: dict[str, Path],
    missing_modalities: list[str],
    warnings_list: list[str],
    spacing: tuple[float, float, float],
    roi_size: tuple[int, int, int],
    status: str,
) -> dict[str, Any]:
    """Build a preprocessing-manifest row while preserving labels/metadata."""
    record = dict(case)
    for modality in MODALITIES:
        record[f"source_{modality}"] = case.get(modality)
        record[modality] = (
            str(output_paths[modality]) if output_paths[modality].is_file() else ""
        )
    record["split"] = split
    record["case_dir"] = str(case_dir)
    record["preprocessing_status"] = status
    record["missing_modalities"] = ";".join(missing_modalities)
    record["preprocessing_warnings"] = " | ".join(warnings_list)
    record["target_spacing"] = "x".join(f"{value:g}" for value in spacing)
    record["roi_size"] = "x".join(str(value) for value in roi_size)
    return record


def preprocess_case(
    case: dict[str, Any],
    out_root: Path,
    split: str,
    spacing: tuple[float, float, float],
    roi_size: tuple[int, int, int],
    overwrite: bool,
    logger: logging.Logger,
    case_index: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Preprocess one case and return either a manifest or failure row."""
    patient_id = str(case.get("patient_id") or "")
    scan_id = str(case.get("scan_id") or "")
    patient_name = _safe_component(patient_id, f"patient_{case_index:05d}")
    acquisition_value = case.get("acquisition_id")
    if not _is_present(acquisition_value):
        status = str(case.get("distortion_status") or "unknown")
        acquisition_value = f"{scan_id}__{status}"
    acquisition_name = _safe_component(
        acquisition_value, f"acquisition_{case_index:05d}"
    )
    case_dir = ensure_dir(out_root / split / f"{patient_name}_{acquisition_name}")
    output_paths = _output_paths(case_dir)
    warnings_list: list[str] = []

    expected_modalities = [name for name in MODALITIES if _is_present(case.get(name))]
    complete = (
        bool(expected_modalities)
        and all(output_paths[name].is_file() for name in expected_modalities)
        and (case_dir / "preprocessing_summary.json").is_file()
    )
    if complete and not overwrite:
        logger.info("Skipping existing case %s | %s", patient_id, scan_id)
        missing = [name for name in MODALITIES if name not in expected_modalities]
        return (
            _record_for_case(
                case,
                split,
                case_dir,
                output_paths,
                missing,
                warnings_list,
                spacing,
                roi_size,
                "skipped_existing",
            ),
            None,
        )

    try:
        images, missing = _load_images(case, warnings_list)
        if not images:
            raise ValueError("No readable T2, DWI, ADC, or prostate mask was found.")

        if "t2" in images:
            reference_name = "t2"
        elif "dwi" in images:
            reference_name = "dwi"
            warnings_list.append("T2 missing: DWI used as the reference grid.")
        elif "adc" in images:
            reference_name = "adc"
            warnings_list.append("T2 and DWI missing: ADC used as the reference grid.")
        else:
            reference_name = "prostate_mask"
            warnings_list.append(
                "No intensity image available: prostate mask used as reference grid."
            )

        target = _target_reference(images[reference_name], spacing)
        aligned: dict[str, sitk.Image] = {}
        for modality, image in images.items():
            aligned[modality] = _resample_to_reference(
                image,
                target,
                modality,
                is_mask=modality in MASK_MODALITIES,
                warnings_list=warnings_list,
            )

        mask = aligned.get("prostate_mask")
        center = _mask_center(mask) if mask is not None else None
        if center is None:
            center = _image_center(target)
            if mask is None:
                warnings_list.append(
                    "Prostate mask unavailable: used a center crop."
                )
            else:
                warnings_list.append(
                    "Prostate mask is empty after resampling: used a center crop."
                )

        normalization: dict[str, Any] = {}
        for modality, image in aligned.items():
            cropped = _crop_or_pad(
                image,
                center,
                roi_size,
                modality,
                warnings_list,
            )
            if modality in IMAGE_MODALITIES:
                processed, statistics = _normalize_image(
                    cropped,
                    modality,
                    warnings_list,
                )
                normalization[modality] = statistics
            else:
                processed = sitk.Cast(cropped > 0, sitk.sitkUInt8)
            sitk.WriteImage(processed, str(output_paths[modality]), True)

        case_summary = {
            "patient_id": patient_id,
            "scan_id": scan_id,
            "split": split,
            "reference_modality": reference_name,
            "target_spacing": list(spacing),
            "roi_size": list(roi_size),
            "crop_center_index": list(center),
            "normalization_percentiles": normalization,
            "missing_modalities": missing,
            "warnings": warnings_list,
            "outputs": {
                modality: (
                    str(path) if path.is_file() else None
                )
                for modality, path in output_paths.items()
            },
        }
        write_json(case_summary, case_dir / "preprocessing_summary.json")
        for warning in warnings_list:
            logger.warning("%s | %s | %s", patient_id, scan_id, warning)
        logger.info("Processed %s | %s", patient_id, scan_id)
        return (
            _record_for_case(
                case,
                split,
                case_dir,
                output_paths,
                missing,
                warnings_list,
                spacing,
                roi_size,
                "processed",
            ),
            None,
        )
    except Exception as error:
        logger.exception("Failed %s | %s", patient_id, scan_id)
        failure = {
            "patient_id": patient_id,
            "scan_id": scan_id,
            "distortion_status": str(case.get("distortion_status") or ""),
            "acquisition_id": str(case.get("acquisition_id") or ""),
            "split": split,
            "error": str(error),
            "warnings": " | ".join(warnings_list),
        }
        return None, failure


def _load_datalist(path: Path) -> list[dict[str, Any]]:
    """Load list-style or common MONAI-style datalist JSON."""
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


def _merge_rows(
    path: Path,
    new_rows: list[dict[str, Any]],
    keys: Sequence[str],
) -> pd.DataFrame:
    """Update a cumulative CSV without dropping rows from other splits."""
    frames = []
    if path.is_file():
        frames.append(read_csv(path))
    if new_rows:
        frames.append(pd.DataFrame(new_rows))
    if not frames:
        return pd.DataFrame(columns=list(keys))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    existing_keys = [key for key in keys if key in combined.columns]
    if len(existing_keys) == len(keys):
        combined = combined.drop_duplicates(list(keys), keep="last")
    return combined


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return JSON-safe records, converting pandas missing values to ``None``."""
    if frame.empty:
        return []
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _parse_bool(value: str | bool) -> bool:
    """Parse a user-friendly optional boolean argument."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess aligned and cropped prostate MRI training cases."
    )
    parser.add_argument("--datalist_json", type=Path, required=True)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Processed-data root (default: data/processed).",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=DEFAULT_SPACING,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--roi_size",
        type=int,
        nargs=3,
        default=DEFAULT_ROI_SIZE,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--overwrite",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=False,
        help="Overwrite completed cases (default: false).",
    )
    return parser.parse_args(argv)


def _validate_parameters(
    spacing: tuple[float, float, float],
    roi_size: tuple[int, int, int],
) -> None:
    """Validate physical spacing and ROI dimensions."""
    if any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("All spacing values must be finite and positive.")
    if any(value <= 0 for value in roi_size):
        raise ValueError("All roi_size values must be positive integers.")


def main(argv: Sequence[str] | None = None) -> int:
    """Preprocess all cases in a datalist and save cumulative manifests."""
    args = parse_args(argv)
    spacing = tuple(float(value) for value in args.spacing)
    roi_size = tuple(int(value) for value in args.roi_size)
    _validate_parameters(spacing, roi_size)

    out_root = ensure_dir(args.out_dir)
    logger = get_logger("prostate_iqa.preprocess_cases", out_root / "preprocessing.log")
    cases = _load_datalist(args.datalist_json)
    default_split = _infer_split(args.datalist_json)
    if default_split == "test_locked":
        logger.warning(
            "LOCKED TEST: applying fixed spacing=%s and roi_size=%s. Do not "
            "use these results to choose or tune preprocessing parameters.",
            spacing,
            roi_size,
        )

    manifest_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        split = _case_split(case, default_split)
        record, failure = preprocess_case(
            case,
            out_root,
            split,
            spacing,
            roi_size,
            args.overwrite,
            logger,
            index,
        )
        if record is not None:
            manifest_rows.append(record)
        if failure is not None:
            failure_rows.append(failure)

    manifest_path = out_root / "preprocessing_manifest.csv"
    manifest = _merge_rows(
        manifest_path,
        manifest_rows,
        keys=(
            "split",
            "patient_id",
            "scan_id",
            "distortion_status",
            "acquisition_id",
        ),
    )
    write_csv(manifest, manifest_path)
    if "split" in manifest.columns:
        for split_name, split_frame in manifest.groupby("split", dropna=False):
            normalized_split = str(split_name or "unknown")
            filename = (
                "datalist_test_locked.json"
                if normalized_split == "test_locked"
                else f"datalist_{normalized_split}.json"
            )
            write_json(_json_records(split_frame.reset_index(drop=True)), out_root / filename)

    failures_path = out_root / "preprocessing_failures.csv"
    failures = _merge_rows(
        failures_path,
        failure_rows,
        keys=(
            "split",
            "patient_id",
            "scan_id",
            "distortion_status",
            "acquisition_id",
        ),
    )
    if manifest_rows and not failures.empty:
        successful_keys = {
            (
                row.get("split"),
                row.get("patient_id"),
                row.get("scan_id"),
                row.get("distortion_status"),
                row.get("acquisition_id"),
            )
            for row in manifest_rows
        }
        keep = failures.apply(
            lambda row: (
                row.get("split"),
                row.get("patient_id"),
                row.get("scan_id"),
                row.get("distortion_status"),
                row.get("acquisition_id"),
            )
            not in successful_keys,
            axis=1,
        )
        failures = failures.loc[keep].reset_index(drop=True)
    for column in FAILURE_COLUMNS:
        if column not in failures.columns:
            failures[column] = pd.Series(dtype="object")
    failures = failures[list(FAILURE_COLUMNS)]
    write_csv(failures, failures_path)

    logger.info(
        "Finished: %d succeeded/skipped, %d failed. Manifest: %s. Failures: %s",
        len(manifest_rows),
        len(failure_rows),
        manifest_path,
        failures_path,
    )
    close_file_handlers(logger)
    return 0 if not failure_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
