from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from monai.data import CacheDataset, list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    CropForegroundd,
    ResizeWithPadOrCropd,
    ScaleIntensityRangePercentilesd,
    Lambdad,
    ConcatItemsd,
    DeleteItemsd,
)
from monai.networks.nets import DenseNet121


def load_json_list(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    items = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(items, list) or (len(items) > 0 and not isinstance(items[0], dict)):
        raise ValueError(f"Expected a JSON list[dict] at {p}")
    return items


def binarize_mask(x):
    return (x > 0).astype(np.float32)


def make_transforms(
    roi_size: Tuple[int, int, int],
    use_t2: bool,
) -> Compose:
    load_keys = ["dwi", "mask"] + (["t2"] if use_t2 else [])
    spatial_keys = ["dwi", "mask"] + (["t2"] if use_t2 else [])

    xforms = [
        LoadImaged(keys=load_keys, image_only=False),
        EnsureChannelFirstd(keys=load_keys),
        EnsureTyped(keys=load_keys),
        Lambdad(keys="mask", func=binarize_mask),
        CropForegroundd(keys=spatial_keys, source_key="mask", margin=5),
        ResizeWithPadOrCropd(keys=spatial_keys, spatial_size=roi_size),
        ScaleIntensityRangePercentilesd(
            keys="dwi",
            lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True
        ),
    ]

    if use_t2:
        xforms.append(
            ScaleIntensityRangePercentilesd(
                keys="t2",
                lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True
            )
        )

    concat_keys = ["dwi"] + (["t2"] if use_t2 else []) + ["mask"]
    xforms.extend([
        ConcatItemsd(keys=concat_keys, name="image", dim=0),
        Lambdad(keys="label_bin", func=lambda y: torch.tensor(np.int64(y), dtype=torch.long)),
        DeleteItemsd(keys=[k for k in concat_keys if k in ("dwi", "t2", "mask")]),
        EnsureTyped(keys=["image"]),
    ])

    return Compose(xforms)


@torch.no_grad()
def evaluate_with_details(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    model.eval()
    all_probs = []
    all_y = []
    rows: List[Dict[str, Any]] = []

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label_bin"].to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]

        probs_cpu = probs.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy().astype(int)
        preds = (probs_cpu >= threshold).astype(int)

        ids = batch.get("id", [""] * len(y_cpu))
        bvals = batch.get("b_value", [""] * len(y_cpu))

        for i in range(len(y_cpu)):
            rows.append({
                "id": ids[i],
                "b_value": bvals[i],
                "label_bin": int(y_cpu[i]),
                "pred_bin": int(preds[i]),
                "prob_1": float(probs_cpu[i]),
                "correct": int(preds[i] == y_cpu[i]),
            })

        all_probs.append(probs.detach().cpu())
        all_y.append(y.detach().cpu())

    probs = torch.cat(all_probs).numpy()
    y = torch.cat(all_y).numpy().astype(int)
    pred = (probs >= threshold).astype(int)

    acc = float((pred == y).mean())
    tpr = float((pred[y == 1] == 1).mean()) if (y == 1).any() else 0.0
    tnr = float((pred[y == 0] == 0).mean()) if (y == 0).any() else 0.0
    bal_acc = float(0.5 * (tpr + tnr))

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, probs))
    except Exception:
        auc = float("nan")

    return {"acc": acc, "bal_acc": bal_acc, "auc": auc, "tpr": tpr, "tnr": tnr}, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_json", default="data/splits/datalist_test.json")
    ap.add_argument("--ckpt", required=True, help="Path to best.pt or best_finetune.pt")
    ap.add_argument("--out_dir", default="runs/test_eval")
    ap.add_argument("--threshold", type=float, default=0.5, help="If not given, use checkpoint threshold if available, else 0.5")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})

    use_t2 = bool(cfg.get("use_t2", False))
    roi_size = tuple(cfg.get("roi_size", [96, 96, 32]))
    threshold = args.threshold if args.threshold is not None else cfg.get("threshold", 0.5)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_items = load_json_list(args.test_json)
    test_tf = make_transforms(roi_size=roi_size, use_t2=use_t2)

    test_ds = CacheDataset(test_items, transform=test_tf, cache_rate=1.0, num_workers=args.num_workers)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 3 if use_t2 else 2
    model = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=2).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    metrics, rows = evaluate_with_details(model, test_loader, device, threshold=threshold)

    print(
        f"TEST | acc={metrics['acc']:.3f} | "
        f"bal_acc={metrics['bal_acc']:.3f} | "
        f"auc={metrics['auc']:.3f} | "
        f"sensitivity={metrics['tpr']:.3f} | "
        f"specificity={metrics['tnr']:.3f}"
    )

    with (out_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "b_value", "label_bin", "pred_bin", "prob_1", "correct"])
        for r in rows:
            w.writerow([r["id"], r["b_value"], r["label_bin"], r["pred_bin"], f"{r['prob_1']:.6f}", r["correct"]])

    with (out_dir / "test_metrics.txt").open("w", encoding="utf-8") as f:
        f.write(
            f"checkpoint: {args.ckpt}\n"
            f"use_t2: {use_t2}\n"
            f"roi_size: {roi_size}\n"
            f"threshold: {threshold}\n"
            f"acc: {metrics['acc']:.6f}\n"
            f"bal_acc: {metrics['bal_acc']:.6f}\n"
            f"auc: {metrics['auc']:.6f}\n"
            f"sensitivity: {metrics['tpr']:.6f}\n"
            f"specificity: {metrics['tnr']:.6f}\n"
        )

    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()