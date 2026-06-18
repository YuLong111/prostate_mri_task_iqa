from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from monai.data import CacheDataset, list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    CropForegroundd,
    ResizeWithPadOrCropd,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
    Lambdad,
    ConcatItemsd,
    DeleteItemsd,
)

from monai.networks.nets import DenseNet121


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json_list(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    items = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(items, list) or (len(items) > 0 and not isinstance(items[0], dict)):
        raise ValueError(f"Expected a JSON list[dict] at {p}")
    return items


def make_transforms(
    roi_size: Tuple[int, int, int],
    use_t2: bool,
    augment: bool,
) -> Compose:
    load_keys = ["dwi", "mask"] + (["t2"] if use_t2 else [])
    spatial_keys = ["dwi", "mask"] + (["t2"] if use_t2 else [])

    xforms = [
        LoadImaged(keys=load_keys, image_only=False),
        EnsureChannelFirstd(keys=load_keys),
        EnsureTyped(keys=load_keys),
        Lambdad(keys="mask", func=lambda x: (x > 0).astype(np.float32)),
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

    if augment:
        xforms.extend([
            RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=2),
            RandAffined(
                keys=spatial_keys,
                prob=0.3,
                rotate_range=(0.05, 0.05, 0.05),
                translate_range=(5, 5, 3),
                scale_range=(0.05, 0.05, 0.05),
                mode=("bilinear", "nearest") + (("bilinear",) if use_t2 else ()),
                padding_mode="border",
            ),
            RandGaussianNoised(keys="dwi", prob=0.2, mean=0.0, std=0.02),
            RandShiftIntensityd(keys="dwi", prob=0.2, offsets=0.05),
        ])
        if use_t2:
            xforms.extend([
                RandGaussianNoised(keys="t2", prob=0.2, mean=0.0, std=0.02),
                RandShiftIntensityd(keys="t2", prob=0.2, offsets=0.05),
            ])

    concat_keys = ["dwi"] + (["t2"] if use_t2 else []) + ["mask"]
    xforms.extend([
        ConcatItemsd(keys=concat_keys, name="image", dim=0),
        Lambdad(keys="label_bin", func=lambda y: np.int64(y), overwrite=True),
        Lambdad(keys="label_bin", func=lambda y: torch.tensor(y, dtype=torch.long)),
        DeleteItemsd(keys=[k for k in concat_keys if k in ("dwi", "t2", "mask")]),
        EnsureTyped(keys=["image"]),
    ])

    return Compose(xforms)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_y = []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label_bin"].to(device)  
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.detach().cpu())
        all_y.append(y.detach().cpu())

    probs = torch.cat(all_probs).numpy()
    y = torch.cat(all_y).numpy().astype(int)


    pred = (probs >= 0.5).astype(int)
    acc = float((pred == y).mean())
    tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else 0.0
    tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else 0.0
    bal_acc = float(0.5 * (tpr + tnr))


    auc = None
    try:
        from sklearn.metrics import roc_auc_score 
        auc = float(roc_auc_score(y, probs))
    except Exception:
        auc = float("nan")

    return {"acc": acc, "bal_acc": bal_acc, "auc": auc}


def build_sampler(items: List[Dict[str, Any]]) -> WeightedRandomSampler:
    labels = np.array([int(it["label_bin"]) for it in items], dtype=int)
    class_counts = np.bincount(labels, minlength=2).astype(np.float32)
    class_weights = 1.0 / np.clip(class_counts, 1.0, None)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="data/splits/datalist_train.json")
    ap.add_argument("--val_json", default="data/splits/datalist_val.json")
    ap.add_argument("--out_dir", default="runs/exp_quality_bin")
    ap.add_argument("--use_t2", action="store_true", help="Use T2 as an additional input channel.")
    ap.add_argument("--roi_size", nargs=3, type=int, default=[96, 96, 32])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_rate", type=float, default=0.3)
    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_items = load_json_list(args.train_json)
    val_items = load_json_list(args.val_json)

    def dist(name: str, items: List[Dict[str, Any]]) -> None:
        y = [int(it["label_bin"]) for it in items]
        print(f"{name}: n={len(items)} label_dist={{0:{y.count(0)}, 1:{y.count(1)}}}")

    dist("TRAIN", train_items)
    dist("VAL", val_items)

    train_tf = make_transforms(tuple(args.roi_size), use_t2=args.use_t2, augment=True)
    val_tf = make_transforms(tuple(args.roi_size), use_t2=args.use_t2, augment=False)

    train_ds = CacheDataset(train_items, transform=train_tf, cache_rate=args.cache_rate, num_workers=args.num_workers)
    val_ds = CacheDataset(val_items, transform=val_tf, cache_rate=1.0, num_workers=args.num_workers)

    sampler = build_sampler(train_items)  
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 3 if args.use_t2 else 2
    model = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=2).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label_bin"].to(device) 
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            running += float(loss.item()) * x.shape[0]
            n_seen += x.shape[0]

        train_loss = running / max(n_seen, 1)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['acc']:.3f} | "
            f"val_bal_acc={val_metrics['bal_acc']:.3f} | "
            f"val_auc={val_metrics['auc']:.3f}"
        )

        torch.save(
            {"model_state": model.state_dict(), "config": vars(args), "epoch": epoch, "best_auc": best_auc},
            out_dir / "last.pt",
        )
        score = val_metrics["auc"]
        if np.isnan(score):
            score = val_metrics["bal_acc"]

        if score > best_auc:
            best_auc = float(score)
            torch.save(
                {"model_state": model.state_dict(), "config": vars(args), "epoch": epoch, "best_auc": best_auc},
                out_dir / "best.pt",
            )
            print(f"  ✔ Saved new best checkpoint (score={best_auc:.3f})")

    print(f"\nDone. Best score={best_auc:.3f}. Checkpoints in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
