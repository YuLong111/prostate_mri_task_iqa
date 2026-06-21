"""MONAI transforms for generic prostate MRI segmentation tasks."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import torch
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    ConcatItemsd,
    CopyItemsd,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
)

from prostate_iqa.data.transforms import RobustNonzeroPercentileScaled, _is_mask_key


class BinarizeMaskd(MapTransform):
    """Convert one or more loaded masks to binary integer tensors."""

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        output = dict(data)
        for key in self.key_iterator(output):
            output[key] = (torch.as_tensor(output[key]) > 0).to(torch.long)
        return output


def _validate(
    image_keys: Sequence[str],
    label_key: str,
    roi_size: Sequence[int],
) -> tuple[tuple[str, ...], str, tuple[int, int, int]]:
    keys = tuple(str(key).strip() for key in image_keys)
    label = str(label_key).strip()
    size = tuple(int(value) for value in roi_size)
    if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("image_keys must contain unique non-empty names.")
    if not label or label in keys:
        raise ValueError("label_key must be non-empty and cannot also be an image input.")
    if len(size) != 3 or any(value <= 0 for value in size):
        raise ValueError("roi_size must contain three positive integers.")
    return keys, label, size


def _base(
    image_keys: tuple[str, ...],
    label_key: str,
    roi_size: tuple[int, int, int],
) -> list[Any]:
    all_keys = (*image_keys, label_key)
    intensity_keys = tuple(key for key in image_keys if not _is_mask_key(key))
    input_mask_keys = tuple(key for key in image_keys if _is_mask_key(key))
    transforms: list[Any] = [
        LoadImaged(keys=all_keys, image_only=True),
        EnsureChannelFirstd(keys=all_keys),
        EnsureTyped(keys=all_keys, dtype=torch.float32),
    ]
    if intensity_keys:
        transforms.append(RobustNonzeroPercentileScaled(keys=intensity_keys))
    transforms.extend(
        [
            BinarizeMaskd(keys=(*input_mask_keys, label_key)),
            ResizeWithPadOrCropd(
                keys=all_keys, spatial_size=roi_size, mode="constant"
            ),
        ]
    )
    return transforms


def _final(image_keys: tuple[str, ...], label_key: str) -> list[Any]:
    return [
        ConcatItemsd(keys=image_keys, name="image", dim=0),
        CopyItemsd(keys=label_key, names="label"),
        DeleteItemsd(keys=(*image_keys, label_key)),
        EnsureTyped(keys="image", dtype=torch.float32, track_meta=False),
        EnsureTyped(keys="label", dtype=torch.long, track_meta=False),
    ]


def get_segmentation_train_transforms(
    image_keys: Sequence[str],
    label_key: str,
    roi_size: Sequence[int],
) -> Compose:
    """Return synchronized image/mask transforms with light 3D augmentation."""
    images, label, size = _validate(image_keys, label_key, roi_size)
    all_keys = (*images, label)
    intensity_keys = tuple(key for key in images if not _is_mask_key(key))
    mask_keys = tuple(key for key in images if _is_mask_key(key))
    interpolation_modes = tuple(
        "nearest" if _is_mask_key(key) else "bilinear" for key in images
    ) + ("nearest",)
    transforms = _base(images, label, size)
    transforms.extend(
        [
            RandFlipd(keys=all_keys, prob=0.2, spatial_axis=0),
            RandFlipd(keys=all_keys, prob=0.2, spatial_axis=1),
            RandFlipd(keys=all_keys, prob=0.2, spatial_axis=2),
            RandAffined(
                keys=all_keys,
                spatial_size=size,
                prob=0.25,
                rotate_range=(0.05, 0.05, 0.05),
                translate_range=(4.0, 4.0, 2.0),
                scale_range=(0.05, 0.05, 0.05),
                mode=interpolation_modes,
                padding_mode="border",
            ),
        ]
    )
    if intensity_keys:
        transforms.extend(
            [
                RandGaussianNoised(
                    keys=intensity_keys, prob=0.15, mean=0.0, std=0.01
                ),
                RandScaleIntensityd(keys=intensity_keys, prob=0.2, factors=0.1),
                RandShiftIntensityd(keys=intensity_keys, prob=0.2, offsets=0.05),
            ]
        )
    transforms.append(BinarizeMaskd(keys=(*mask_keys, label)))
    transforms.extend(_final(images, label))
    return Compose(transforms)


def get_segmentation_val_transforms(
    image_keys: Sequence[str],
    label_key: str,
    roi_size: Sequence[int],
) -> Compose:
    """Return deterministic transforms for validation or locked-test inference."""
    images, label, size = _validate(image_keys, label_key, roi_size)
    transforms = _base(images, label, size)
    transforms.extend(_final(images, label))
    return Compose(transforms)
