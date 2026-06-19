"""Model builder for generic 3D prostate MRI segmentation tasks."""

from __future__ import annotations

from collections.abc import Sequence

from monai.networks.nets import UNet


def build_unet3d(
    in_channels: int,
    out_channels: int = 2,
    channels: Sequence[int] = (16, 32, 64, 128, 256),
    strides: Sequence[int] = (2, 2, 2, 2),
    num_res_units: int = 2,
) -> UNet:
    """Build a configurable MONAI 3D UNet for binary mask segmentation."""
    channel_values = tuple(int(value) for value in channels)
    stride_values = tuple(int(value) for value in strides)
    if in_channels <= 0 or out_channels <= 1:
        raise ValueError("in_channels must be positive and out_channels at least 2.")
    if len(channel_values) != len(stride_values) + 1:
        raise ValueError("channels must contain exactly one more value than strides.")
    return UNet(
        spatial_dims=3,
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        channels=channel_values,
        strides=stride_values,
        num_res_units=int(num_res_units),
        norm="INSTANCE",
    )
