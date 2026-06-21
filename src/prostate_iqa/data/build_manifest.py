"""Build a patient/scan-level master manifest from a file inventory."""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from prostate_iqa.utils.io import read_csv, write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INVENTORY_CSV = (
    PROJECT_ROOT / "data" / "manifests" / "file_inventory.csv"
)
DEFAULT_OUT_CSV = PROJECT_ROOT / "data" / "manifests" / "master_manifest.csv"

PATH_COLUMNS = ("t2", "dwi", "adc", "prostate_mask", "lesion_mask")
ACQUISITION_COLUMNS = (
    "distortion_status",
    "acquisition_index",
    "acquisition_id",
)
LABEL_COLUMNS = (
    "pirads",
    "pirads_ge4",
    "gleason_group",
    "gleason_ge2",
    "dwi_quality_bin",
    "dwi_quality_ord",
    "quality_ternary",
    "site",
    "vendor",
    "field_strength",
    "b_value",
    "notes",
)
MANIFEST_COLUMNS = (
    "patient_id",
    "scan_id",
    *ACQUISITION_COLUMNS,
    *PATH_COLUMNS,
    *LABEL_COLUMNS,
)

MEDICAL_VOLUME_SUFFIXES = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")

REQUIRED_INVENTORY_COLUMNS = {
    "file_path",
    "patient_id_guess",
    "scan_id_guess",
    "modality_guess",
}

_LABEL_ALIASES = {
    "patient_id": {
        "id",
        "patient_id",
        "patientid",
        "patient",
        "subject_id",
        "subjectid",
        "case_id",
        "caseid",
        "reference",
        "reference_number",
    },
    "scan_id": {
        "scan_id",
        "scanid",
        "scan",
        "study_id",
        "studyid",
        "volume_id",
        "volumeid",
    },
    "pirads": {"pirads", "pi_rads", "pi_rads_score", "pirads_score"},
    "pirads_ge4": {"pirads_ge4", "pi_rads_ge4", "pirads_4", "pi_rads_4"},
    "gleason_group": {
        "gleason_group",
        "gleason_grade_group",
        "grade_group",
        "isup",
        "isup_grade_group",
        "gleason",
    },
    "gleason_ge2": {"gleason_ge2", "gleason_group_ge2", "isup_ge2"},
    "dwi_quality_bin": {
        "dwi_quality_bin",
        "dwi_quality_binary",
        "quality_bin",
        "quality_binary",
    },
    "dwi_quality_ord": {
        "dwi_quality_ord",
        "dwi_quality_ordinal",
        "quality_ord",
        "quality_ordinal",
    },
    "quality_ternary": {"quality_ternary", "ternary_quality"},
    "site": {"site", "centre", "center"},
    "vendor": {"vendor", "manufacturer", "scanner_vendor"},
    "field_strength": {
        "field_strength",
        "field_strength_t",
        "magnetic_field_strength",
    },
    "b_value": {"b_value", "bvalue", "b_val", "high_b_value"},
    "notes": {"notes", "note", "comments", "comment"},
}


def _column_key(name: object) -> str:
    """Convert a spreadsheet header to a predictable snake-case key."""
    key = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return key.strip("_")


def _is_present(value: Any) -> bool:
    """Return whether a scalar contains a non-empty value."""
    return not pd.isna(value) and str(value).strip() != ""


def _clean_id(value: Any) -> str:
    """Convert spreadsheet identifiers to strings without a trailing .0."""
    if not _is_present(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _patient_key(value: Any) -> str:
    """Create a matching key that tolerates Patient prefixes and leading zeros."""
    text = _clean_id(value)
    if not text:
        return ""

    patient_match = re.fullmatch(r"(?i)patient[-_ ]?0*(\d+)", text)
    numeric_match = re.fullmatch(r"0*(\d+)", text)
    match = patient_match or numeric_match
    if match:
        return f"number:{int(match.group(1))}"
    return "text:" + re.sub(r"[^a-z0-9]+", "", text.lower())


def _scan_key(value: Any) -> str:
    """Create a case-insensitive scan/volume matching key."""
    text = _clean_id(value).lower()
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".dcm"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return re.sub(r"\s+", "", text)


def _study_key(value: Any) -> str:
    """Extract a study-level key shared by all volumes in that study."""
    scan_key = _scan_key(value)
    match = re.search(r"(?i)(?:^|_)study[_-]?(\d+)(?:_|$)", scan_key)
    if match:
        return f"study:{int(match.group(1))}"
    match = re.fullmatch(r"(?i)ses[-_]?([a-z0-9]+)", scan_key)
    if match:
        return f"session:{match.group(1).lower()}"
    return ""


def _is_study_level_scan(value: Any) -> bool:
    """Return whether an ID names a study rather than an individual volume."""
    scan_key = _scan_key(value)
    return bool(
        re.fullmatch(r"(?i)(?:patient\d+_)?study[_-]?\d+", scan_key)
        or re.fullmatch(r"(?i)ses[-_]?[a-z0-9]+", scan_key)
    )


def _distortion_status(value: Any, file_path: Any = "") -> str:
    """Normalize acquisition distortion groups, with a path-based fallback."""
    text = _clean_id(value).lower().replace(" ", "_")
    if text in {"distorted", "undistorted", "not_applicable"}:
        return text
    path_text = _clean_id(file_path).lower().replace("\\", "/")
    if "undistorted" in path_text:
        return "undistorted"
    if "distorted" in path_text:
        return "distorted"
    return "unknown"


def _is_medical_volume(row: pd.Series) -> bool:
    """Exclude spreadsheets and sidecars from modality path columns."""
    suffix = _clean_id(row.get("suffix")).lower()
    if suffix:
        return suffix in MEDICAL_VOLUME_SUFFIXES
    path = _clean_id(row.get("file_path")).lower()
    return path.endswith(MEDICAL_VOLUME_SUFFIXES)


def _unique_paths(values: pd.Series) -> list[str]:
    """Return every unique physical path in deterministic order."""
    return sorted({_clean_id(value) for value in values if _is_present(value)})


def _acquisition_id(scan_id: str, status: str, index: int) -> str:
    """Create a stable identifier for one physical acquisition row."""
    scan = re.sub(r"[^A-Za-z0-9._-]+", "_", scan_id).strip("._") or "scan"
    value = f"{scan}__{status}"
    return value if index == 0 else f"{value}__rep{index + 1}"


def _guess_b_value(scan_id: str) -> int | None:
    """Infer values such as 1k4, 1k6, or 2k from a volume identifier."""
    match = re.search(r"(?i)__(\d+)k(\d*)-", scan_id)
    if match:
        whole, fraction = match.groups()
        fraction_value = float(f"0.{fraction}") if fraction else 0.0
        return int(round((int(whole) + fraction_value) * 1000))

    match = re.search(r"(?i)(?:^|[_-])b(?:-?value)?[_-]?(\d{3,4})(?:$|[_-])", scan_id)
    if match:
        return int(match.group(1))
    match = re.search(r"(?i)(?:^|[/\\_-])dwi[_-]?(\d{3,4})(?:$|[/\\_.-])", scan_id)
    if match:
        return int(match.group(1))
    return None


def _validate_inventory(inventory: pd.DataFrame) -> None:
    missing = REQUIRED_INVENTORY_COLUMNS.difference(inventory.columns)
    if missing:
        raise ValueError(
            "Inventory is missing required columns: " + ", ".join(sorted(missing))
        )


def _unique_path_map(
    inventory: pd.DataFrame,
    modality: str,
    keys: Sequence[str],
) -> dict[Any, str]:
    """Return only unambiguous one-path modality mappings."""
    rows = inventory.loc[inventory["modality_guess"].eq(modality)]
    result: dict[Any, str] = {}
    for raw_key, group in rows.groupby(list(keys), sort=False, dropna=False):
        key = raw_key[0] if len(keys) == 1 and isinstance(raw_key, tuple) else raw_key
        paths = _unique_paths(group["file_path"])
        if len(paths) == 1:
            result[key] = paths[0]
    return result


def build_manifest(inventory: pd.DataFrame) -> pd.DataFrame:
    """Build one row per physical acquisition without collapsing path variants."""
    _validate_inventory(inventory)
    files = inventory.copy()
    files = files.loc[files.apply(_is_medical_volume, axis=1)].copy()
    files["patient_id_guess"] = files["patient_id_guess"].map(_clean_id)
    files["scan_id_guess"] = files["scan_id_guess"].map(_clean_id)
    files["_patient_key"] = files["patient_id_guess"].map(_patient_key)
    files["_scan_key"] = files["scan_id_guess"].map(_scan_key)
    files["_study_key"] = files["scan_id_guess"].map(_study_key)
    if "distortion_status" in files.columns:
        files["_distortion_status"] = [
            _distortion_status(status, path)
            for status, path in zip(
                files["distortion_status"], files["file_path"], strict=True
            )
        ]
    else:
        files["_distortion_status"] = files["file_path"].map(
            lambda path: _distortion_status("", path)
        )
    files = files.loc[files["_patient_key"].ne("")].copy()
    files.loc[files["_scan_key"].eq(""), "_scan_key"] = files["_patient_key"]

    image_modalities = {"t2", "dwi", "adc"}
    image_rows = files.loc[files["modality_guess"].isin(image_modalities)]

    display_patient = (
        files.drop_duplicates("_patient_key")
        .set_index("_patient_key")["patient_id_guess"]
        .to_dict()
    )
    exact_masks = {
        modality: _unique_path_map(
            files, modality, ("_patient_key", "_scan_key")
        )
        for modality in ("prostate_mask", "lesion_mask")
    }
    study_masks = {
        modality: _unique_path_map(
            files.loc[files["_study_key"].ne("")],
            modality,
            ("_patient_key", "_study_key"),
        )
        for modality in ("prostate_mask", "lesion_mask")
    }
    patient_masks = {
        modality: _unique_path_map(files, modality, ("_patient_key",))
        for modality in ("prostate_mask", "lesion_mask")
    }
    study_images = {
        modality: _unique_path_map(
            image_rows.loc[image_rows["_study_key"].ne("")],
            modality,
            ("_patient_key", "_study_key", "_distortion_status"),
        )
        for modality in image_modalities
    }

    records: list[dict[str, Any]] = []
    group_keys = ("_patient_key", "_scan_key", "_distortion_status")
    for (patient_key, scan_key, status), group in image_rows.groupby(
        list(group_keys), sort=True, dropna=False
    ):
        modality_paths = {
            modality: _unique_paths(
                group.loc[group["modality_guess"].eq(modality), "file_path"]
            )
            for modality in image_modalities
        }
        replicate_count = max((len(paths) for paths in modality_paths.values()), default=0)
        if replicate_count == 0:
            continue
        display_scan_values = [
            _clean_id(value) for value in group["scan_id_guess"] if _is_present(value)
        ]
        scan_id = (
            display_scan_values[0]
            if display_scan_values
            else display_patient[patient_key]
        )
        study_key = _study_key(scan_id)
        for acquisition_index in range(replicate_count):
            record: dict[str, Any] = {
                "patient_id": display_patient[patient_key],
                "scan_id": scan_id,
                "distortion_status": status,
                "acquisition_index": acquisition_index,
                "acquisition_id": _acquisition_id(
                    scan_id, status, acquisition_index
                ),
            }
            for modality in image_modalities:
                paths = modality_paths[modality]
                if len(paths) == 1:
                    path = paths[0]
                elif acquisition_index < len(paths):
                    path = paths[acquisition_index]
                else:
                    path = ""
                if not path and study_key:
                    path = study_images[modality].get(
                        (patient_key, study_key, status), ""
                    )
                record[modality] = path
            for modality in ("prostate_mask", "lesion_mask"):
                path = exact_masks[modality].get((patient_key, scan_key), "")
                if not path and study_key:
                    path = study_masks[modality].get(
                        (patient_key, study_key), ""
                    )
                if not path:
                    path = patient_masks[modality].get(patient_key, "")
                record[modality] = path
            for column in LABEL_COLUMNS:
                record[column] = pd.NA
            record["b_value"] = _guess_b_value(
                f"{scan_id} {record.get('dwi', '')}"
            )
            records.append(record)

    # Preserve mask-only patients for QC while keeping them explicitly separate
    # from distorted/undistorted image acquisitions.
    patients_with_images = {
        _patient_key(record["patient_id"]) for record in records
    }
    mask_rows = files.loc[
        files["modality_guess"].isin({"prostate_mask", "lesion_mask"})
        & ~files["_patient_key"].isin(patients_with_images)
    ]
    for (patient_key, scan_key), group in mask_rows.groupby(
        ["_patient_key", "_scan_key"], sort=True
    ):
        scan_values = [
            _clean_id(value) for value in group["scan_id_guess"] if _is_present(value)
        ]
        scan_id = scan_values[0] if scan_values else display_patient[patient_key]
        record = {
            "patient_id": display_patient[patient_key],
            "scan_id": scan_id,
            "distortion_status": "not_applicable",
            "acquisition_index": 0,
            "acquisition_id": _acquisition_id(scan_id, "not_applicable", 0),
            "t2": "",
            "dwi": "",
            "adc": "",
            "prostate_mask": exact_masks["prostate_mask"].get(
                (patient_key, scan_key), ""
            ),
            "lesion_mask": exact_masks["lesion_mask"].get(
                (patient_key, scan_key), ""
            ),
        }
        for column in LABEL_COLUMNS:
            record[column] = pd.NA
        records.append(record)

    return pd.DataFrame(records, columns=MANIFEST_COLUMNS)


def _read_label_sheet(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel label sheet."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Labels must be CSV or Excel, received: {path}")


def _coalesce_sources(frame: pd.DataFrame, source_columns: list[str]) -> pd.Series:
    """Coalesce synonymous source columns from left to right."""
    if not source_columns:
        return pd.Series(pd.NA, index=frame.index, dtype="object")
    result = frame[source_columns[0]].copy()
    for column in source_columns[1:]:
        result = result.combine_first(frame[column])
    return result


def standardize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Map common label-sheet header variants to manifest column names."""
    keyed_columns = {column: _column_key(column) for column in labels.columns}
    standardized = pd.DataFrame(index=labels.index)
    # Some cohorts use a column called simply "Gleason" for a precomputed
    # binary grade-group >= 2 target. Other cohorts use the same header for a
    # true score or grade group. Infer the meaning only when every populated
    # value is binary, preserving support for conventional Gleason notation.
    gleason_sources = [
        column for column, key in keyed_columns.items() if key == "gleason"
    ]
    gleason_is_binary = bool(gleason_sources) and all(
        all(_parse_binary(value) is not None for value in labels[column].dropna())
        for column in gleason_sources
    )

    for target, aliases in _LABEL_ALIASES.items():
        sources = [
            column for column, key in keyed_columns.items() if key in aliases
        ]
        if target == "gleason_group" and gleason_is_binary:
            sources = [
                column for column in sources if keyed_columns[column] != "gleason"
            ]
        if target == "gleason_ge2" and gleason_is_binary:
            sources.extend(
                column for column in gleason_sources if column not in sources
            )
        standardized[target] = _coalesce_sources(labels, sources)

    if standardized["patient_id"].isna().all():
        raise ValueError(
            "Labels sheet needs a patient identifier column such as patient_id."
        )

    standardized["patient_id"] = standardized["patient_id"].map(_clean_id)
    standardized["scan_id"] = standardized["scan_id"].map(_clean_id)
    standardized["_patient_key"] = standardized["patient_id"].map(_patient_key)
    standardized["_scan_key"] = standardized["scan_id"].map(_scan_key)
    standardized["_study_key"] = standardized["scan_id"].map(_study_key)
    standardized["_is_study_level"] = standardized["scan_id"].map(
        _is_study_level_scan
    )
    return standardized.loc[standardized["_patient_key"].ne("")].copy()


def _values_equal(left: Any, right: Any) -> bool:
    """Compare duplicate label values while tolerating numeric formatting."""
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left).strip().lower() == str(right).strip().lower()


def _label_records(
    labels: pd.DataFrame,
    keys: Sequence[str],
) -> dict[Any, dict[str, Any]]:
    """Collapse duplicate label rows and reject conflicting annotations."""
    records: dict[Any, dict[str, Any]] = {}
    grouped = labels.groupby(list(keys), dropna=False, sort=False)
    for raw_key, group in grouped:
        key = raw_key[0] if len(keys) == 1 and isinstance(raw_key, tuple) else raw_key
        record: dict[str, Any] = {}
        for column in LABEL_COLUMNS:
            values = [value for value in group[column] if _is_present(value)]
            if not values:
                continue
            first = values[0]
            if any(not _values_equal(first, value) for value in values[1:]):
                raise ValueError(f"Conflicting {column} labels for key {key!r}")
            record[column] = first
        records[key] = record
    return records


def _parse_score(value: Any, minimum: int, maximum: int) -> int | None:
    """Extract an integer ordinal score within an expected range."""
    if not _is_present(value):
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    score = float(match.group())
    if score.is_integer() and minimum <= score <= maximum:
        return int(score)
    return None


def _parse_gleason_group(value: Any) -> int | None:
    """Parse a grade group, including common Gleason pattern notation."""
    if not _is_present(value):
        return None
    text = str(value).strip().lower()
    pattern = re.search(r"([3-5])\s*\+\s*([3-5])", text)
    if pattern:
        primary, secondary = map(int, pattern.groups())
        mapping = {
            (3, 3): 1,
            (3, 4): 2,
            (4, 3): 3,
            (4, 4): 4,
            (3, 5): 4,
            (5, 3): 4,
            (4, 5): 5,
            (5, 4): 5,
            (5, 5): 5,
        }
        return mapping.get((primary, secondary))
    return _parse_score(value, 1, 5)


def _parse_binary(value: Any) -> int | None:
    """Parse common binary encodings while preserving missing values."""
    if not _is_present(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y", "positive"}:
        return 1
    if text in {"0", "0.0", "false", "no", "n", "negative"}:
        return 0
    return None


def merge_labels(manifest: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Merge patient-wide and exact-scan labels into a manifest."""
    result = manifest.copy()
    standardized = standardize_labels(labels)
    patient_labels = standardized.loc[standardized["_scan_key"].eq("")]
    scan_labels = standardized.loc[standardized["_scan_key"].ne("")]
    study_labels = standardized.loc[standardized["_is_study_level"]]
    patient_records = _label_records(patient_labels, ("_patient_key",))
    study_records = _label_records(study_labels, ("_patient_key", "_study_key"))
    scan_records = _label_records(scan_labels, ("_patient_key", "_scan_key"))

    for index, row in result.iterrows():
        patient_key = _patient_key(row["patient_id"])
        scan_key = _scan_key(row["scan_id"])
        study_key = _study_key(row["scan_id"])
        values: dict[str, Any] = {}
        values.update(patient_records.get(patient_key, {}))
        values.update(study_records.get((patient_key, study_key), {}))
        values.update(scan_records.get((patient_key, scan_key), {}))
        for column, value in values.items():
            result.at[index, column] = value

    result["pirads"] = result["pirads"].map(lambda value: _parse_score(value, 1, 5))
    direct_pirads = result["pirads_ge4"].map(_parse_binary)
    derived_pirads = result["pirads"].map(
        lambda value: pd.NA if pd.isna(value) else int(value >= 4)
    )
    result["pirads_ge4"] = direct_pirads.combine_first(derived_pirads).astype("Int64")

    result["gleason_group"] = result["gleason_group"].map(_parse_gleason_group)
    direct_gleason = result["gleason_ge2"].map(_parse_binary)
    derived_gleason = result["gleason_group"].map(
        lambda value: pd.NA if pd.isna(value) else int(value >= 2)
    )
    result["gleason_ge2"] = direct_gleason.combine_first(derived_gleason).astype("Int64")
    result["dwi_quality_bin"] = result["dwi_quality_bin"].map(_parse_binary).astype("Int64")

    numeric_columns = ("pirads", "gleason_group", "field_strength", "b_value")
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def apply_manifest_defaults(
    manifest: pd.DataFrame,
    site: str | None = None,
) -> pd.DataFrame:
    """Fill cohort metadata supplied outside an inventory or label sheet."""
    result = manifest.copy()
    if site is not None and site.strip():
        missing = ~result["site"].map(_is_present)
        result.loc[missing, "site"] = site.strip()
    return result


def build_missingness_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values, treating empty strings as missing."""
    rows = []
    total = len(manifest)
    for column in manifest.columns:
        present = manifest[column].map(_is_present)
        non_missing = int(present.sum())
        missing = total - non_missing
        rows.append(
            {
                "column": column,
                "non_missing_count": non_missing,
                "missing_count": missing,
                "missing_percent": (100.0 * missing / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_modality_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Summarize path availability for each imaging modality."""
    rows = []
    total = len(manifest)
    for modality in PATH_COLUMNS:
        present = manifest[modality].map(_is_present)
        available = int(present.sum())
        rows.append(
            {
                "modality": modality,
                "available_count": available,
                "missing_count": total - available,
                "availability_percent": (100.0 * available / total) if total else 0.0,
                "unique_patients_available": manifest.loc[
                    present, "patient_id"
                ].nunique(),
                "total_patient_scans": total,
            }
        )
    return pd.DataFrame(rows)


def _print_summary(manifest: pd.DataFrame) -> None:
    """Print modality and label availability counts."""
    print(f"Manifest contains {len(manifest):,} patient/scan rows.")
    print(f"Unique patients: {manifest['patient_id'].nunique():,}")
    for modality in PATH_COLUMNS:
        count = int(manifest[modality].map(_is_present).sum())
        print(f"Cases with {modality}: {count:,}")

    for label in ("pirads", "pirads_ge4", "gleason_group", "gleason_ge2"):
        count = int(manifest[label].map(_is_present).sum())
        print(f"Cases with {label} labels: {count:,}")
    any_label = manifest[list(LABEL_COLUMNS[:7])].apply(
        lambda row: any(_is_present(value) for value in row),
        axis=1,
    )
    print(f"Cases with any task/quality label: {int(any_label.sum()):,}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a patient/scan-level prostate MRI master manifest."
    )
    parser.add_argument(
        "--inventory_csv",
        type=Path,
        default=DEFAULT_INVENTORY_CSV,
        help="File inventory CSV (default: data/manifests/file_inventory.csv).",
    )
    parser.add_argument(
        "--labels_csv",
        type=Path,
        default=None,
        help="Optional CSV or Excel sheet containing labels and metadata.",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Master manifest output (default: data/manifests/master_manifest.csv).",
    )
    parser.add_argument(
        "--site",
        default=None,
        help=(
            "Optional cohort/site name used when labels do not provide one "
            "(for example Miami or PROMIS)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the manifest and its summary CSV files."""
    args = parse_args(argv)
    inventory = read_csv(args.inventory_csv)
    manifest = build_manifest(inventory)
    if args.labels_csv is not None:
        labels = _read_label_sheet(args.labels_csv)
        manifest = merge_labels(manifest, labels)
    manifest = apply_manifest_defaults(manifest, args.site)

    out_csv = write_csv(manifest, args.out_csv)
    if out_csv.name == DEFAULT_OUT_CSV.name:
        missingness_csv = out_csv.parent / "manifest_missingness.csv"
        modality_csv = out_csv.parent / "modality_summary.csv"
    else:
        missingness_csv = out_csv.with_name(f"{out_csv.stem}_missingness.csv")
        modality_csv = out_csv.with_name(f"{out_csv.stem}_modality_summary.csv")
    write_csv(build_missingness_summary(manifest), missingness_csv)
    write_csv(build_modality_summary(manifest), modality_csv)

    print(f"Saved master manifest to: {out_csv}")
    print(f"Saved missingness summary to: {missingness_csv}")
    print(f"Saved modality summary to: {modality_csv}")
    _print_summary(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
