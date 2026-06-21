"""Build a file inventory for a prostate MRI dataset.

This module only reads the dataset tree. It never moves, renames, or modifies
the files it discovers.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from prostate_iqa.utils.config import load_paths_config
from prostate_iqa.utils.io import list_files_recursive, write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.local.yaml"

DEFAULT_SUFFIXES = (
    ".nii",
    ".nii.gz",
    ".mha",
    ".mhd",
    ".nrrd",
    ".dcm",
    ".json",
    ".csv",
    ".xlsx",
)

INVENTORY_COLUMNS = (
    "file_path",
    "file_name",
    "suffix",
    "parent_dir",
    "relative_path",
    "patient_id_guess",
    "scan_id_guess",
    "modality_guess",
    "distortion_status",
    "is_nifti",
    "is_dicom",
)

# Requiring a recognisable prefix avoids treating dates, series numbers, and
# image dimensions as patient identifiers.
_PATIENT_PATTERNS = (
    # Miami and PROMIS use directory names such as P-1000 and P-10104751.
    # Keep the required hyphen so ordinary tokens beginning with "P" are not
    # mistaken for patient identifiers.
    re.compile(r"(?i)(?<![a-z0-9])(p-\d+)(?![a-z0-9])"),
    re.compile(r"(?i)(?<![a-z0-9])(patient\d+)(?![a-z0-9])"),
    re.compile(
        r"(?i)(?<![a-z0-9])"
        r"(prostatex[-_ ]?[a-z0-9]+|patient[-_ ]?[a-z0-9]+|"
        r"subject[-_ ]?[a-z0-9]+|case[-_ ]?[a-z0-9]+|pt[-_ ]?\d+)"
        r"(?![a-z0-9])"
    ),
    re.compile(r"(?i)(?<![a-z0-9])(sub-[a-z0-9]+)(?![a-z0-9])"),
)

_SCAN_PATTERNS = (
    # In this dataset the final acquisition and numeric tokens distinguish
    # multiple high-b-value volumes belonging to the same patient/study.
    re.compile(
        r"(?i)(?<![a-z0-9])"
        r"(patient\d+_study_\d+__[a-z0-9-]+__\d+)"
        r"(?![a-z0-9])"
    ),
    # A prostate mask has one patient/study reference and no volume token.
    re.compile(
        r"(?i)(?<![a-z0-9])(patient\d+_study_\d+)(?![a-z0-9])"
    ),
    re.compile(
        r"(?i)(?<![a-z0-9])"
        r"((?:scan|study|exam|series|accession)[-_ ]?[a-z0-9]+)"
        r"(?![a-z0-9])"
    ),
    re.compile(r"(?i)(?<![a-z0-9])(ses-[a-z0-9]+)(?![a-z0-9])"),
)

_KEYWORD_PATTERNS = {
    "t2": re.compile(r"(?i)(?<![a-z0-9])t2(?:w)?(?![a-z0-9])"),
    "dwi": re.compile(
        r"(?i)(?<![a-z0-9])(?:dwi|diff(?:usion)?|bval|b-value)(?![a-z0-9])"
    ),
    "adc": re.compile(r"(?i)(?<![a-z0-9])adc(?![a-z0-9])"),
    "mask": re.compile(r"(?i)(?<![a-z0-9])mask(?![a-z0-9])"),
    "prostate": re.compile(r"(?i)(?<![a-z0-9])prostate(?![a-z0-9])"),
    "lesion": re.compile(r"(?i)(?<![a-z0-9])lesion(?![a-z0-9])"),
    "seg": re.compile(r"(?i)(?<![a-z0-9])seg(?:mentation)?(?![a-z0-9])"),
    "label": re.compile(r"(?i)(?<![a-z0-9])labels?(?![a-z0-9])"),
}


def _normalize_suffixes(suffixes: Iterable[str]) -> tuple[str, ...]:
    """Normalize comma- or space-separated extensions, longest first."""
    normalized = set()
    for value in suffixes:
        for item in value.split(","):
            suffix = item.strip().lower()
            if suffix:
                normalized.add(suffix if suffix.startswith(".") else f".{suffix}")
    return tuple(sorted(normalized, key=lambda value: (-len(value), value)))


def _file_suffix(path: Path, suffixes: Sequence[str]) -> str:
    """Return the matching configured suffix, including compound suffixes."""
    name = path.name.lower()
    return next(
        (suffix for suffix in suffixes if name.endswith(suffix)),
        path.suffix.lower(),
    )


def _guess_prefixed_id(relative_path: Path, patterns: Sequence[re.Pattern[str]]) -> str:
    """Find a conservative prefixed identifier near the file first."""
    components = [relative_path.name, *reversed(relative_path.parent.parts)]
    for component in components:
        for pattern in patterns:
            match = pattern.search(component)
            if match:
                return match.group(1)
    return ""


def guess_patient_id(relative_path: Path) -> str:
    """Guess a patient identifier only when a known patient prefix is present."""
    return _guess_prefixed_id(relative_path, _PATIENT_PATTERNS)


def guess_scan_id(relative_path: Path) -> str:
    """Guess a scan identifier only when a known scan prefix is present."""
    return _guess_prefixed_id(relative_path, _SCAN_PATTERNS)


def guess_modality(relative_path: Path) -> str:
    """Infer a canonical modality category from path and filename keywords."""
    text = relative_path.as_posix()
    path_parts = {part.lower() for part in relative_path.parts}

    # The dataset's directory names are stronger evidence than filenames: T2
    # and DWI files deliberately share the same volume reference filenames.
    if any("gland_zone" in part for part in path_parts):
        return "gland_zone_mask"
    if any("prostate_mask" in part for part in path_parts):
        return "prostate_mask"
    if "dwi" in path_parts:
        return "dwi"
    if "t2" in path_parts or "t2w" in path_parts:
        return "t2"

    found = {name for name, pattern in _KEYWORD_PATTERNS.items() if pattern.search(text)}
    is_annotation = bool(found.intersection({"mask", "seg", "label"}))

    if "lesion" in found and is_annotation:
        return "lesion_mask"
    if "prostate" in found and is_annotation:
        return "prostate_mask"
    if is_annotation:
        return "mask"
    if "adc" in found:
        return "adc"
    if "dwi" in found:
        return "dwi"
    if "t2" in found:
        return "t2"
    if "lesion" in found:
        return "lesion"
    if "prostate" in found:
        return "prostate"
    return "unknown"


def guess_distortion_status(relative_path: Path, modality: str) -> str:
    """Read the image distortion group from the dataset folder name."""
    if modality.endswith("_mask") or modality == "mask":
        return "not_applicable"

    text = relative_path.as_posix().lower()
    # Check undistorted first because the word contains "distorted".
    if "undistorted" in text:
        return "undistorted"
    if "distorted" in text:
        return "distorted"
    return "unknown"


def build_inventory(dataset_root: Path, suffixes: Sequence[str]) -> pd.DataFrame:
    """Scan *dataset_root* and return one inventory row per matching file."""
    root = dataset_root.expanduser().resolve()
    normalized_suffixes = _normalize_suffixes(suffixes)
    files = list_files_recursive(root, normalized_suffixes)

    rows = []
    for file_path in files:
        relative_path = file_path.relative_to(root)
        suffix = _file_suffix(file_path, normalized_suffixes)
        modality = guess_modality(relative_path)
        rows.append(
            {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "suffix": suffix,
                "parent_dir": str(file_path.parent),
                "relative_path": relative_path.as_posix(),
                "patient_id_guess": guess_patient_id(relative_path),
                "scan_id_guess": guess_scan_id(relative_path),
                "modality_guess": modality,
                "distortion_status": guess_distortion_status(
                    relative_path,
                    modality,
                ),
                "is_nifti": suffix in {".nii", ".nii.gz"},
                "is_dicom": suffix == ".dcm",
            }
        )

    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


def _print_summary(inventory: pd.DataFrame) -> None:
    """Print file counts grouped by suffix and inferred modality."""
    print(f"Inventory contains {len(inventory):,} files.")

    if inventory.empty:
        unique_cases = 0
    else:
        patient_ids = inventory["patient_id_guess"].replace("", pd.NA).dropna()
        unique_cases = patient_ids.nunique()
    print(f"Identified {unique_cases:,} unique patient/case references.")

    print("\nCounts by suffix:")
    if inventory.empty:
        print("  (none)")
    else:
        for suffix, count in inventory["suffix"].value_counts().sort_index().items():
            print(f"  {suffix}: {count:,}")

    print("\nCounts by modality_guess:")
    if inventory.empty:
        print("  (none)")
    else:
        counts = inventory["modality_guess"].value_counts().sort_index()
        for modality, count in counts.items():
            print(f"  {modality}: {count:,}")

    print("\nCounts by distortion_status:")
    if inventory.empty:
        print("  (none)")
    else:
        counts = inventory["distortion_status"].value_counts().sort_index()
        for status, count in counts.items():
            print(f"  {status}: {count:,}")

    print("\nFiles and unique cases by modality_guess:")
    if inventory.empty:
        print("  (none)")
    else:
        grouped = inventory.groupby("modality_guess", dropna=False)
        for modality, group in grouped:
            case_count = group["patient_id_guess"].replace("", pd.NA).nunique()
            print(f"  {modality}: {len(group):,} files, {case_count:,} cases")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Recursively inventory files in a prostate MRI dataset."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=None,
        help=(
            "Root directory of the dataset to scan. Overrides dataset_root "
            "from --paths_config."
        ),
    )
    parser.add_argument(
        "--paths_config",
        type=Path,
        default=DEFAULT_PATHS_CONFIG,
        help=(
            "Paths YAML used when --dataset_root is omitted "
            "(default: <project_root>/configs/paths.local.yaml)."
        ),
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        required=True,
        help="Destination CSV path for the inventory.",
    )
    parser.add_argument(
        "--suffixes",
        nargs="+",
        default=list(DEFAULT_SUFFIXES),
        help="File suffixes to include (space- or comma-separated).",
    )
    return parser.parse_args(argv)


def _resolve_dataset_root(
    dataset_root: Path | None,
    paths_config: Path,
) -> Path:
    """Use the CLI dataset root or fall back to the local paths config."""
    if dataset_root is not None:
        return dataset_root

    paths = load_paths_config(paths_config)
    configured_root = paths.get("dataset_root")
    if not configured_root:
        raise ValueError(
            f"No dataset_root is defined in paths config: {paths_config}"
        )
    return Path(configured_root)


def main(argv: Sequence[str] | None = None) -> int:
    """Build and save an inventory from command-line arguments."""
    args = parse_args(argv)
    dataset_root = _resolve_dataset_root(args.dataset_root, args.paths_config)
    print(f"Scanning dataset root: {dataset_root}")
    inventory = build_inventory(dataset_root, args.suffixes)
    output_path = write_csv(inventory, args.out_csv)
    print(f"Saved inventory to: {output_path}")
    _print_summary(inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
