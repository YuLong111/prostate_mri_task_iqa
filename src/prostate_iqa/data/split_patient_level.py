"""Create leakage-safe patient-level train, validation, and test splits."""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from prostate_iqa.utils.config import load_paths_config
from prostate_iqa.utils.io import ensure_dir, read_csv, write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.local.yaml"
MISSING_STRATUM = "<missing>"


def _validate_sizes(test_size: float, val_size: float) -> None:
    """Validate that requested fractions leave a non-empty training fraction."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.")


def _clean_patient_ids(manifest: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize patient identifiers used for grouping."""
    if "patient_id" not in manifest.columns:
        raise ValueError("Manifest is missing required column: patient_id")

    cleaned = manifest.copy()
    missing = cleaned["patient_id"].isna() | cleaned["patient_id"].astype(str).str.strip().eq("")
    if missing.any():
        raise ValueError(
            f"Manifest contains {int(missing.sum())} rows without patient_id; "
            "patient-level splitting would be unsafe."
        )
    cleaned["patient_id"] = cleaned["patient_id"].astype(str).str.strip()
    return cleaned


def _label_value_key(value: Any) -> str:
    """Convert a label value to a stable string used for stratification."""
    if pd.isna(value) or str(value).strip() == "":
        return MISSING_STRATUM
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def patient_level_labels(manifest: pd.DataFrame, stratify_key: str) -> pd.Series:
    """Reduce row labels to one deterministic majority label per patient.

    A patient may have multiple image rows and, for scan-level quality tasks,
    more than one label. The most frequent non-missing label represents that
    patient for stratification; ties are resolved lexicographically.
    """
    if stratify_key not in manifest.columns:
        raise KeyError(stratify_key)

    labels: dict[str, str] = {}
    mixed_patients = 0
    for patient_id, group in manifest.groupby("patient_id", sort=True):
        values = [
            _label_value_key(value)
            for value in group[stratify_key]
            if _label_value_key(value) != MISSING_STRATUM
        ]
        if not values:
            labels[patient_id] = MISSING_STRATUM
            continue

        counts = pd.Series(values, dtype="string").value_counts()
        if len(counts) > 1:
            mixed_patients += 1
        highest_count = counts.max()
        labels[patient_id] = sorted(counts[counts.eq(highest_count)].index)[0]

    if mixed_patients:
        warnings.warn(
            f"{mixed_patients} patients have multiple {stratify_key} values; "
            "their majority label is used for patient-level stratification.",
            stacklevel=2,
        )
    return pd.Series(labels, name=stratify_key, dtype="string").sort_index()


def _random_split(
    patient_ids: list[str],
    test_size: float,
    val_size: float,
    seed: int,
) -> dict[str, set[str]]:
    """Split patient IDs randomly with reproducible two-stage sampling."""
    train_val, test = train_test_split(
        patient_ids,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    val_fraction_of_train_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_train_val,
        random_state=seed,
        shuffle=True,
    )
    return {
        "train": set(train),
        "val": set(val),
        "test_locked": set(test),
    }


def _stratified_split(
    patient_ids: list[str],
    strata: pd.Series,
    test_size: float,
    val_size: float,
    seed: int,
) -> dict[str, set[str]]:
    """Perform a two-stage stratified split at patient level."""
    labels = strata.loc[patient_ids]
    train_val, test = train_test_split(
        patient_ids,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    val_fraction_of_train_val = val_size / (1.0 - test_size)
    train_val_labels = strata.loc[train_val]
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_train_val,
        random_state=seed,
        shuffle=True,
        stratify=train_val_labels,
    )
    return {
        "train": set(train),
        "val": set(val),
        "test_locked": set(test),
    }


def assert_no_patient_leakage(
    split_patients: Mapping[str, set[str]],
    expected_patients: set[str] | None = None,
) -> None:
    """Assert that patient sets are disjoint and optionally exhaustive."""
    names = list(split_patients)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = split_patients[left_name] & split_patients[right_name]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise AssertionError(
                    f"Patient leakage between {left_name} and {right_name}: {examples}"
                )

    if expected_patients is not None:
        assigned = set().union(*split_patients.values())
        if assigned != expected_patients:
            missing = expected_patients - assigned
            extra = assigned - expected_patients
            raise AssertionError(
                f"Split patient assignment is not exhaustive "
                f"(missing={len(missing)}, extra={len(extra)})."
            )


def split_patients(
    manifest: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    stratify_key: str | None = None,
) -> tuple[dict[str, set[str]], pd.Series | None, bool]:
    """Assign each patient to exactly one split.

    Returns the patient sets, patient-level labels (when available), and a flag
    indicating whether stratification was successfully used.
    """
    _validate_sizes(test_size, val_size)
    cleaned = _clean_patient_ids(manifest)
    patient_ids = sorted(cleaned["patient_id"].unique().tolist())
    if len(patient_ids) < 3:
        raise ValueError("At least three unique patients are required.")

    strata: pd.Series | None = None
    used_stratification = False
    if stratify_key:
        if stratify_key not in cleaned.columns:
            warnings.warn(
                f"Stratify key {stratify_key!r} is not in the manifest; "
                "falling back to a random patient-level split.",
                stacklevel=2,
            )
        else:
            strata = patient_level_labels(cleaned, stratify_key)
            non_missing_classes = strata[strata.ne(MISSING_STRATUM)].nunique()
            if non_missing_classes < 2:
                warnings.warn(
                    f"Stratify key {stratify_key!r} has fewer than two "
                    "non-missing classes; falling back to a random "
                    "patient-level split.",
                    stacklevel=2,
                )
            else:
                try:
                    split_sets = _stratified_split(
                        patient_ids,
                        strata,
                        test_size,
                        val_size,
                        seed,
                    )
                    used_stratification = True
                except ValueError as error:
                    warnings.warn(
                        f"Could not stratify by {stratify_key!r} ({error}); "
                        "falling back to a random patient-level split.",
                        stacklevel=2,
                    )

    if not used_stratification:
        try:
            split_sets = _random_split(patient_ids, test_size, val_size, seed)
        except ValueError as error:
            raise ValueError(
                "The requested split sizes cannot produce non-empty patient "
                f"splits for {len(patient_ids)} patients: {error}"
            ) from error

    assert_no_patient_leakage(split_sets, set(patient_ids))
    return split_sets, strata, used_stratification


def assign_split_column(
    manifest: pd.DataFrame,
    split_patients: Mapping[str, set[str]],
) -> pd.DataFrame:
    """Add a split column to every manifest row using its patient assignment."""
    result = _clean_patient_ids(manifest)
    patient_to_split = {
        patient_id: split_name
        for split_name, patient_ids in split_patients.items()
        for patient_id in patient_ids
    }
    result["split"] = result["patient_id"].map(patient_to_split)
    if result["split"].isna().any():
        raise AssertionError("Some manifest rows did not receive a split assignment.")
    return result


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to standards-compliant JSON records with nulls."""
    return json.loads(frame.to_json(orient="records"))


def _distribution_for_patients(
    patient_ids: set[str],
    strata: pd.Series | None,
) -> dict[str, int] | None:
    """Return patient-level label counts for a split."""
    if strata is None:
        return None
    counts = strata.loc[sorted(patient_ids)].value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _print_summary(
    split_manifest: pd.DataFrame,
    split_patients: Mapping[str, set[str]],
    stratify_key: str | None,
    strata: pd.Series | None,
    used_stratification: bool,
) -> None:
    """Print row counts, patient counts, and label distributions."""
    method = "stratified" if used_stratification else "random"
    print(f"Split method: {method} patient-level split")
    if stratify_key:
        print(f"Requested stratify key: {stratify_key}")

    for split_name in ("train", "val", "test_locked"):
        rows = int(split_manifest["split"].eq(split_name).sum())
        patients = split_patients[split_name]
        distribution = _distribution_for_patients(patients, strata)
        print(f"\n{split_name}: {len(patients):,} patients, {rows:,} rows")
        if distribution is None:
            print("  label distribution: not requested")
        else:
            print(f"  patient-level label distribution: {distribution}")

    print(
        "\nLOCKED TEST POLICY: datalist_test_locked.json is for final "
        "evaluation only. Do not use it for model selection, threshold "
        "tuning, preprocessing choices, or hyperparameter tuning."
    )


def save_splits(
    manifest: pd.DataFrame,
    split_patients: Mapping[str, set[str]],
    out_dir: Path,
    overwrite_locked_test: bool = False,
) -> pd.DataFrame:
    """Save split JSON files and the row-level split summary CSV."""
    output_dir = ensure_dir(out_dir)
    locked_path = output_dir / "datalist_test_locked.json"
    if locked_path.is_file() and not overwrite_locked_test:
        raise FileExistsError(
            f"Locked test split already exists: {locked_path}. Refusing to replace "
            "it. Use a new --out_dir, or pass --overwrite_locked_test only when "
            "intentionally establishing a new experimental split."
        )
    split_manifest = assign_split_column(manifest, split_patients)

    json_names = {
        "train": "datalist_train.json",
        "val": "datalist_val.json",
        "test_locked": "datalist_test_locked.json",
    }
    manifest_columns = list(manifest.columns)
    for split_name, file_name in json_names.items():
        rows = split_manifest.loc[
            split_manifest["split"].eq(split_name), manifest_columns
        ]
        write_json(_json_records(rows), output_dir / file_name)

    write_csv(split_manifest, output_dir / "split_summary.csv")
    return split_manifest


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve config-relative paths against the prostate_mri_task_iqa root."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def resolve_input_paths(
    manifest_csv: Path | None,
    out_dir: Path | None,
    paths_config: Path,
) -> tuple[Path, Path]:
    """Resolve CLI overrides or config-driven manifest and split paths."""
    if manifest_csv is not None and out_dir is not None:
        return manifest_csv, out_dir

    paths = load_paths_config(paths_config)
    if manifest_csv is None:
        manifest_dir = paths.get("manifest_dir")
        if not manifest_dir:
            raise ValueError(
                f"No manifest_dir is defined in paths config: {paths_config}"
            )
        manifest_csv = _resolve_project_path(manifest_dir) / "master_manifest.csv"

    if out_dir is None:
        configured_out_dir = paths.get("split_dir")
        if not configured_out_dir:
            raise ValueError(
                f"No split_dir is defined in paths config: {paths_config}"
            )
        out_dir = _resolve_project_path(configured_out_dir)
    return manifest_csv, out_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create leakage-safe patient-level dataset splits."
    )
    parser.add_argument(
        "--manifest_csv",
        type=Path,
        default=None,
        help=(
            "Master manifest CSV. Defaults to "
            "<manifest_dir>/master_manifest.csv from --paths_config."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Split output directory. Defaults to split_dir from --paths_config.",
    )
    parser.add_argument(
        "--paths_config",
        type=Path,
        default=DEFAULT_PATHS_CONFIG,
        help=(
            "Project paths YAML (default: "
            "<project_root>/configs/paths.local.yaml)."
        ),
    )
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify_key",
        default=None,
        help="Optional patient-level stratification column.",
    )
    parser.add_argument(
        "--overwrite_locked_test",
        action="store_true",
        help=(
            "Explicitly permit replacement of an existing locked test JSON. "
            "Do not use this after model development has started."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Create, validate, report, and save patient-level splits."""
    args = parse_args(argv)
    manifest_csv, out_dir = resolve_input_paths(
        args.manifest_csv,
        args.out_dir,
        args.paths_config,
    )
    manifest = read_csv(manifest_csv)
    manifest = _clean_patient_ids(manifest)
    split_sets, strata, used_stratification = split_patients(
        manifest,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        stratify_key=args.stratify_key,
    )
    split_manifest = save_splits(
        manifest,
        split_sets,
        out_dir,
        overwrite_locked_test=args.overwrite_locked_test,
    )
    assert_no_patient_leakage(
        {
            name: set(
                split_manifest.loc[
                    split_manifest["split"].eq(name), "patient_id"
                ]
            )
            for name in split_sets
        },
        set(manifest["patient_id"]),
    )
    _print_summary(
        split_manifest,
        split_sets,
        args.stratify_key,
        strata,
        used_stratification,
    )
    print(f"\nSaved split files to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
