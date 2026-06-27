"""MONAI transforms for prostate MRI classification and IQA models."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    ConcatItemsd,
    CropForegroundd,
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


def _is_mask_key(key: str) -> bool:
    """Infer whether a dictionary key represents a discrete mask."""
    normalized = key.lower()
    return any(token in normalized for token in ("mask", "seg", "label_map"))


def _validate_inputs(
    image_keys: Sequence[str],
    roi_size: Sequence[int],
) -> tuple[tuple[str, ...], tuple[int, int, int]]:
    """Validate transform inputs and return immutable values."""
    keys = tuple(str(key) for key in image_keys)
    if not keys:
        raise ValueError("image_keys must contain at least one key.")
    if any(not key.strip() for key in keys):
        raise ValueError("image_keys cannot contain empty names.")
    if len(set(keys)) != len(keys):
        raise ValueError("image_keys must be unique.")

    size = tuple(int(value) for value in roi_size)
    if len(size) != 3 or any(value <= 0 for value in size):
        raise ValueError("roi_size must contain three positive integers.")
    return keys, size


def _validate_margin(
    margin: Sequence[int] | None,
) -> tuple[int, int, int]:
    """Validate foreground-crop margins."""
    if margin is None:
        return (16, 16, 8)
    parsed = tuple(int(value) for value in margin)
    if len(parsed) != 3 or any(value < 0 for value in parsed):
        raise ValueError("crop_margin must contain three non-negative integers.")
    return parsed


def _scale_tensor_channel(channel: torch.Tensor) -> torch.Tensor:
    """Robustly scale one tensor channel using finite nonzero p1/p99."""
    finite = torch.isfinite(channel)
    values = channel[finite & (channel != 0)]
    if values.numel() < 2:
        values = channel[finite]
    if values.numel() < 2:
        return torch.zeros_like(channel, dtype=torch.float32)

    values = values.float()
    low = torch.quantile(values, 0.01)
    high = torch.quantile(values, 0.99)
    if not torch.isfinite(low) or not torch.isfinite(high) or high <= low:
        return torch.zeros_like(channel, dtype=torch.float32)

    source = channel.float()
    scaled = (torch.clamp(source, low, high) - low) / (high - low)
    scaled = torch.where(finite, scaled, torch.zeros_like(scaled))
    return torch.where(channel == 0, torch.zeros_like(scaled), scaled)


def _scale_numpy_channel(channel: np.ndarray) -> np.ndarray:
    """Robustly scale one NumPy channel using finite nonzero p1/p99."""
    finite = np.isfinite(channel)
    values = channel[finite & (channel != 0)]
    if values.size < 2:
        values = channel[finite]
    if values.size < 2:
        return np.zeros_like(channel, dtype=np.float32)

    low, high = np.percentile(values.astype(np.float32), (1, 99))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(channel, dtype=np.float32)

    source = np.nan_to_num(channel.astype(np.float32), nan=low, posinf=high, neginf=low)
    scaled = (np.clip(source, low, high) - low) / (high - low)
    scaled[channel == 0] = 0.0
    return scaled.astype(np.float32, copy=False)


def _robust_scale(image: Any) -> Any:
    """Scale each channel independently while preserving tensor metadata."""
    if image.ndim < 2:
        raise ValueError(f"Expected channel-first image, received shape {image.shape}.")

    if isinstance(image, torch.Tensor):
        result = image.clone().float()
        for channel_index in range(result.shape[0]):
            result[channel_index] = _scale_tensor_channel(result[channel_index])
        return result

    array = np.asarray(image)
    result = np.empty(array.shape, dtype=np.float32)
    for channel_index in range(array.shape[0]):
        result[channel_index] = _scale_numpy_channel(array[channel_index])
    return result


class RobustNonzeroPercentileScaled(MapTransform):
    """Dictionary transform for channel-wise nonzero percentile scaling."""

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self,
        data: Mapping[Hashable, Any],
    ) -> dict[Hashable, Any]:
        output = dict(data)
        for key in self.key_iterator(output):
            output[key] = _robust_scale(output[key])
        return output


def _base_transforms(
    image_keys: tuple[str, ...],
    roi_size: tuple[int, int, int],
    crop_margin: tuple[int, int, int],
    mask_crop: bool,
) -> list[Any]:
    """Create deterministic loading, normalization, and sizing transforms."""
    intensity_keys = tuple(key for key in image_keys if not _is_mask_key(key))
    transforms: list[Any] = [
        LoadImaged(keys=image_keys, image_only=True),
        EnsureChannelFirstd(keys=image_keys),
        EnsureTyped(keys=image_keys, dtype=torch.float32),
    ]
    if intensity_keys:
        transforms.append(RobustNonzeroPercentileScaled(keys=intensity_keys))
    if mask_crop and "prostate_mask" in image_keys:
        transforms.append(
            CropForegroundd(
                keys=image_keys,
                source_key="prostate_mask",
                margin=crop_margin,
                allow_smaller=True,
            )
        )
    transforms.append(
        ResizeWithPadOrCropd(
            keys=image_keys,
            spatial_size=roi_size,
            mode="constant",
        )
    )
    return transforms


def _final_transforms(image_keys: tuple[str, ...]) -> list[Any]:
    """Concatenate channels and remove redundant modality dictionary entries."""
    return [
        ConcatItemsd(keys=image_keys, name="image", dim=0),
        DeleteItemsd(keys=image_keys),
        EnsureTyped(keys="image", dtype=torch.float32, track_meta=False),
    ]


def get_train_transforms(
    image_keys: Sequence[str],
    roi_size: Sequence[int],
    crop_margin: Sequence[int] | None = None,
    mask_crop: bool = True,
) -> Compose:
    """Return loading, preprocessing, and light training augmentation transforms."""
    keys, size = _validate_inputs(image_keys, roi_size)
    margin = _validate_margin(crop_margin)
    intensity_keys = tuple(key for key in keys if not _is_mask_key(key))
    interpolation_modes = tuple(
        "nearest" if _is_mask_key(key) else "bilinear" for key in keys
    )

    transforms = _base_transforms(keys, size, margin, mask_crop)
    transforms.extend(
        [
            RandFlipd(keys=keys, prob=0.2, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.2, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.2, spatial_axis=2),
            RandAffined(
                keys=keys,
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
                    keys=intensity_keys,
                    prob=0.15,
                    mean=0.0,
                    std=0.01,
                ),
                RandScaleIntensityd(
                    keys=intensity_keys,
                    prob=0.2,
                    factors=0.1,
                ),
                RandShiftIntensityd(
                    keys=intensity_keys,
                    prob=0.2,
                    offsets=0.05,
                ),
            ]
        )
    transforms.extend(_final_transforms(keys))
    return Compose(transforms)


def get_val_transforms(
    image_keys: Sequence[str],
    roi_size: Sequence[int],
    crop_margin: Sequence[int] | None = None,
    mask_crop: bool = True,
) -> Compose:
    """Return deterministic validation and locked-test transforms."""
    keys, size = _validate_inputs(image_keys, roi_size)
    margin = _validate_margin(crop_margin)
    transforms = _base_transforms(keys, size, margin, mask_crop)
    transforms.extend(_final_transforms(keys))
    return Compose(transforms)
