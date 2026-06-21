"""Run one-case forward-pass checks for classification and segmentation paths."""

from __future__ import annotations

import argparse
import gc
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from prostate_iqa.data.segmentation_transforms import (
    get_segmentation_val_transforms,
)
from prostate_iqa.data.transforms import get_val_transforms
from prostate_iqa.models.densenet_quality import build_densenet121
from prostate_iqa.models.unet_segmentation import build_unet3d
from prostate_iqa.utils.io import read_json, write_json
from prostate_iqa.utils.seed import set_global_seed


def _is_present(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "nan", "none"}


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = read_json(path)
    if not isinstance(records, list) or not all(
        isinstance(record, Mapping) for record in records
    ):
        raise ValueError(f"Datalist must contain a JSON list of objects: {path}")
    return [dict(record) for record in records]


def _find_usable_record(
    records: Sequence[Mapping[str, Any]],
    path_keys: Sequence[str],
    target_key: str | None,
) -> dict[str, Any]:
    required = (*path_keys, *((target_key,) if target_key else ()))
    for source in records:
        if not all(_is_present(source.get(key)) for key in required):
            continue
        if not all(Path(str(source[key])).is_file() for key in path_keys):
            continue
        if target_key and target_key.endswith("_mask"):
            if not Path(str(source[target_key])).is_file():
                continue
        return dict(source)
    raise ValueError(
        "No usable case contains existing paths for " + ", ".join(required)
    )


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return requested


def _case_identity(record: Mapping[str, Any]) -> dict[str, str]:
    return {
        "patient_id": str(record.get("patient_id", "")),
        "scan_id": str(record.get("scan_id", "")),
    }


def run_smoke_test(args: argparse.Namespace) -> dict[str, Any]:
    """Load real volumes and run all requested model heads once."""
    set_global_seed(args.seed)
    records = _load_records(args.datalist_json)
    device = _device(args.device)
    report: dict[str, Any] = {
        "datalist_json": str(args.datalist_json.resolve()),
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "roi_size": list(args.roi_size),
    }

    if not args.skip_classification:
        record = _find_usable_record(
            records,
            args.classification_image_keys,
            args.classification_target_key,
        )
        transformed = get_val_transforms(
            args.classification_image_keys, args.roi_size
        )(
            {key: record[key] for key in args.classification_image_keys}
        )
        image = transformed["image"].unsqueeze(0).to(device)
        output_shapes: dict[str, list[int]] = {}
        with torch.inference_mode():
            for classes in (2, 3):
                model = build_densenet121(image.shape[1], classes).to(device).eval()
                logits = model(image)
                output_shapes[f"densenet_{classes}_class"] = list(logits.shape)
                del model, logits
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        report["classification"] = {
            **_case_identity(record),
            "target_key": args.classification_target_key,
            "target": record[args.classification_target_key],
            "image_keys": list(args.classification_image_keys),
            "input_shape": list(image.shape),
            "output_shapes": output_shapes,
        }
        if args.check_backward:
            model = build_densenet121(image.shape[1], 2).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            target = torch.tensor(
                [int(float(record[args.classification_target_key]))],
                dtype=torch.long,
                device=device,
            )
            logits = model(image)
            loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            report["classification"]["backward_loss"] = float(loss.detach().cpu())
            del model, optimizer, target, logits, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del image, transformed
        gc.collect()

    if not args.skip_segmentation:
        record = _find_usable_record(
            records,
            args.segmentation_image_keys,
            args.segmentation_label_key,
        )
        transformed = get_segmentation_val_transforms(
            args.segmentation_image_keys,
            args.segmentation_label_key,
            args.roi_size,
        )(
            {
                **{key: record[key] for key in args.segmentation_image_keys},
                args.segmentation_label_key: record[args.segmentation_label_key],
            }
        )
        image = transformed["image"].unsqueeze(0).to(device)
        label = transformed["label"]
        with torch.inference_mode():
            model = build_unet3d(image.shape[1], 2).to(device).eval()
            logits = model(image)
        report["segmentation"] = {
            **_case_identity(record),
            "label_key": args.segmentation_label_key,
            "image_keys": list(args.segmentation_image_keys),
            "input_shape": list(image.shape),
            "label_shape": list(label.shape),
            "label_values": sorted(int(value) for value in label.unique().tolist()),
            "output_shape": list(logits.shape),
        }
        del model, logits
        if args.check_backward:
            model = build_unet3d(image.shape[1], 2).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            target = label.unsqueeze(0)[:, 0].to(device)
            logits = model(image)
            loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            report["segmentation"]["backward_loss"] = float(loss.detach().cpu())

    if args.out_json is not None:
        write_json(report, args.out_json)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one real case and smoke-test the binary, ternary, and "
            "segmentation 3D model paths without training."
        )
    )
    parser.add_argument("--datalist_json", type=Path, required=True)
    parser.add_argument(
        "--classification_image_keys",
        nargs="+",
        default=("dwi", "t2", "prostate_mask"),
    )
    parser.add_argument("--classification_target_key", default="pirads_ge4")
    parser.add_argument(
        "--segmentation_image_keys", nargs="+", default=("dwi", "t2")
    )
    parser.add_argument("--segmentation_label_key", default="prostate_mask")
    parser.add_argument("--roi_size", nargs=3, type=int, default=(64, 64, 32))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_json", type=Path, default=None)
    parser.add_argument("--skip_classification", action="store_true")
    parser.add_argument("--skip_segmentation", action="store_true")
    parser.add_argument(
        "--check_backward",
        action="store_true",
        help="Also run one AdamW training step for each enabled model path.",
    )
    args = parser.parse_args(argv)
    if any(value <= 0 for value in args.roi_size):
        parser.error("--roi_size values must be positive")
    if args.skip_classification and args.skip_segmentation:
        parser.error("At least one model path must be enabled")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_smoke_test(args)
    print(f"Smoke test passed on {report['device']}.")
    for task in ("classification", "segmentation"):
        if task in report:
            details = report[task]
            print(
                f"  {task}: {details['patient_id']} / {details['scan_id']} "
                f"input={details['input_shape']}"
            )
    if args.out_json is not None:
        print(f"Saved report to: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
