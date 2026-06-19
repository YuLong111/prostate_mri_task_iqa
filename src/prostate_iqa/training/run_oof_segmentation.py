"""Generate patient-level OOF predictions for a 3D segmentation task."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from sklearn.model_selection import KFold

from prostate_iqa.data.segmentation_transforms import (
    get_segmentation_train_transforms,
    get_segmentation_val_transforms,
)
from prostate_iqa.models.unet_segmentation import build_unet3d
from prostate_iqa.training.run_oof_downstream import _patient_key
from prostate_iqa.training.train_binary_task import _load_datalist, _reject_test_input
from prostate_iqa.training.train_segmentation_task import (
    _checkpoint,
    evaluate_segmentation,
    prepare_segmentation_items,
)
from prostate_iqa.utils.io import ensure_dir, write_csv, write_json
from prostate_iqa.utils.seed import set_global_seed


def _patient_rows(
    items: Sequence[dict[str, Any]],
) -> tuple[list[str], dict[str, list[int]]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        key = _patient_key(item.get("patient_id"))
        if not key:
            raise ValueError(f"Segmentation row {index} is missing patient_id.")
        grouped[key].append(index)
    return sorted(grouped), grouped


def _loader(
    items: Sequence[dict[str, Any]],
    transform: Any,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        Dataset(list(items), transform=transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )


def run_oof(args: argparse.Namespace) -> dict[str, Any]:
    """Train all folds and score every training acquisition exactly once."""
    set_global_seed(args.seed)
    set_determinism(seed=args.seed)
    output_dir = ensure_dir(args.out_dir)
    source_path = Path(args.train_json)
    raw = _load_datalist(source_path)
    _reject_test_input(source_path, raw)
    items = prepare_segmentation_items(
        raw, args.image_keys, args.label_key, "OOF segmentation datalist"
    )
    patients, rows_by_patient = _patient_rows(items)
    if args.num_folds < 2 or args.num_folds > len(patients):
        raise ValueError("num_folds must be between 2 and the number of patients.")
    splitter = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    all_metrics: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    seen_holdout: set[str] = set()
    patient_positions = list(range(len(patients)))
    print(
        f"OOF segmentation: {len(items)} rows, {len(patients)} patients, "
        f"{args.num_folds} folds on {device}."
    )
    for fold, (train_positions, holdout_positions) in enumerate(
        splitter.split(patient_positions)
    ):
        fold_seed = args.seed + fold
        set_global_seed(fold_seed)
        train_patients = [patients[int(index)] for index in train_positions]
        holdout_patients = [patients[int(index)] for index in holdout_positions]
        if set(train_patients) & set(holdout_patients):
            raise AssertionError("Patient leakage inside segmentation fold.")
        if seen_holdout & set(holdout_patients):
            raise AssertionError("Patient appears in multiple held-out folds.")
        seen_holdout.update(holdout_patients)
        train_items = [items[i] for patient in train_patients for i in rows_by_patient[patient]]
        holdout_items = [items[i] for patient in holdout_patients for i in rows_by_patient[patient]]
        train_transform = get_segmentation_train_transforms(
            args.image_keys, args.label_key, args.roi_size
        )
        train_transform.set_random_state(seed=fold_seed)
        holdout_transform = get_segmentation_val_transforms(
            args.image_keys, args.label_key, args.roi_size
        )
        train_loader = _loader(
            train_items,
            train_transform,
            args.batch_size,
            args.num_workers,
            True,
            fold_seed,
        )
        holdout_loader = _loader(
            holdout_items,
            holdout_transform,
            args.batch_size,
            args.num_workers,
            False,
            fold_seed,
        )
        model = build_unet3d(len(args.image_keys), 2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses: list[float] = []
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            mean_loss = sum(losses) / max(len(losses), 1)
            history.append({"fold": fold, "epoch": epoch, "train_loss": mean_loss})
            print(
                f"  fold={fold} epoch={epoch:03d}/{args.epochs:03d} "
                f"train_loss={mean_loss:.4f}"
            )
        metrics, frame = evaluate_segmentation(
            model,
            holdout_loader,
            device,
            args.task_name,
            args.label_key,
        )
        frame["fold"] = fold
        all_metrics.append(frame)
        fold_summary = {"fold": fold, **metrics}
        fold_summaries.append(fold_summary)
        torch.save(
            _checkpoint(model, optimizer, args.epochs, metrics, args),
            output_dir / f"fold_{fold}_best.pt",
        )
        write_csv(frame, output_dir / f"fold_{fold}_segmentation_metrics.csv")
        print(f"  held-out fold={fold} mean_dice={float(metrics['mean_dice'] or 0):.4f}")
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if seen_holdout != set(patients):
        raise AssertionError("Not every patient appeared in exactly one held-out fold.")
    oof = pd.concat(all_metrics, ignore_index=True)
    identity = ["patient_id", "scan_id", "acquisition_id"]
    if oof.duplicated(identity).any() or len(oof) != len(items):
        raise AssertionError("OOF segmentation rows are not one-to-one with inputs.")
    aggregate = {
        "task_name": args.task_name,
        "task_type": "segmentation",
        "label_key": args.label_key,
        "num_cases": len(oof),
        "num_patients": len(patients),
        "num_folds": args.num_folds,
        **{
            f"mean_{metric}": float(pd.to_numeric(oof[metric], errors="coerce").mean())
            for metric in ("dice", "iou", "asd", "hd95")
        },
    }
    write_csv(oof, output_dir / "oof_segmentation_metrics.csv")
    write_csv(pd.DataFrame(fold_summaries), output_dir / "fold_metrics.csv")
    write_csv(pd.DataFrame(history), output_dir / "training_history.csv")
    write_json(aggregate, output_dir / "oof_metrics.json")
    print(f"Aggregate OOF mean Dice={aggregate['mean_dice']:.4f}; outputs: {output_dir}")
    return aggregate


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate leakage-safe patient-level OOF segmentation metrics."
    )
    parser.add_argument("--train_json", type=Path, required=True)
    parser.add_argument("--image_keys", nargs="+", required=True)
    parser.add_argument("--label_key", default="prostate_mask")
    parser.add_argument("--task_name", default="prostate_segmentation")
    parser.add_argument("--num_folds", type=_positive_int, default=5)
    parser.add_argument("--roi_size", nargs=3, type=_positive_int, default=(160, 160, 64))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=_positive_int, default=100)
    parser.add_argument("--batch_size", type=_positive_int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive.")
    run_oof(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
