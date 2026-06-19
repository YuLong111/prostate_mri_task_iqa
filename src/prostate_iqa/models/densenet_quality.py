"""Model builders for 3D DenseNet task and image-quality classifiers."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from monai.networks.nets import DenseNet121


CheckpointType = str | PathLike[str] | Mapping[str, Any]


def _first_conv_key(state_dict: Mapping[str, Any]) -> str:
    """Find MONAI DenseNet's initial convolution in a state dictionary."""
    exact_matches = [
        key
        for key, value in state_dict.items()
        if key.endswith("features.conv0.weight")
        and isinstance(value, torch.Tensor)
        and value.ndim == 5
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError(
            "Checkpoint contains multiple possible DenseNet first-convolution weights: "
            + ", ".join(exact_matches)
        )

    fallback_matches = [
        key
        for key, value in state_dict.items()
        if key.endswith("conv0.weight")
        and isinstance(value, torch.Tensor)
        and value.ndim == 5
    ]
    if len(fallback_matches) == 1:
        return fallback_matches[0]
    raise KeyError("Could not find a 3D DenseNet conv0.weight in the state dictionary.")


def _adapt_conv_tensor(
    weight: torch.Tensor,
    old_in_channels: int,
    new_in_channels: int,
) -> torch.Tensor:
    """Adapt input filters using an intensity-first, mask-last convention."""
    if weight.ndim != 5:
        raise ValueError(
            f"Expected 3D convolution weights with five dimensions, got {weight.shape}."
        )
    if weight.shape[1] != old_in_channels:
        raise ValueError(
            f"old_in_channels={old_in_channels} does not match checkpoint weight "
            f"shape {tuple(weight.shape)}."
        )
    if old_in_channels <= 0 or new_in_channels <= 0:
        raise ValueError("Channel counts must be positive integers.")
    if old_in_channels == new_in_channels:
        return weight.clone()

    if new_in_channels == 1:
        return weight.mean(dim=1, keepdim=True)

    # The project channel order places MRI intensity channels first and the
    # prostate mask last. New intensity filters begin from the mean learned
    # intensity filter; old channels are copied into compatible positions.
    old_intensity_count = old_in_channels - 1 if old_in_channels > 1 else 1
    old_intensity = weight[:, :old_intensity_count]
    intensity_template = old_intensity.mean(dim=1, keepdim=True)
    repeat_shape = [1, new_in_channels, *([1] * (weight.ndim - 2))]
    adapted = intensity_template.repeat(*repeat_shape)

    new_intensity_count = new_in_channels - 1
    copied_intensity_count = min(old_intensity_count, new_intensity_count)
    adapted[:, :copied_intensity_count] = weight[:, :copied_intensity_count]

    if old_in_channels > 1:
        adapted[:, -1] = weight[:, -1]
    return adapted


def adapt_first_conv_weights(
    state_dict: Mapping[str, Any],
    old_in_channels: int,
    new_in_channels: int,
) -> dict[str, Any]:
    """Return a state dict with DenseNet's first convolution channel-adapted.

    The input mapping is not modified. Channel order is assumed to be MRI
    intensity channels followed by a mask. Consequently, adapting a two-channel
    ``[DWI, mask]`` checkpoint to four channels creates
    ``[DWI, ADC, T2, mask]``: DWI and mask weights are copied exactly, while ADC
    and T2 are initialized from the learned DWI/intensity filter template.
    """
    adapted_state = dict(state_dict)
    key = _first_conv_key(adapted_state)
    adapted_state[key] = _adapt_conv_tensor(
        adapted_state[key],
        old_in_channels,
        new_in_channels,
    )
    return adapted_state


def _load_checkpoint_payload(checkpoint: CheckpointType) -> Mapping[str, Any]:
    """Load a checkpoint path safely or accept an in-memory checkpoint mapping."""
    if isinstance(checkpoint, Mapping):
        return checkpoint

    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older supported PyTorch versions.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint must contain a mapping, received: {type(payload)}")
    return payload


def _extract_state_dict(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Extract tensor weights from common checkpoint wrapper formats."""
    candidate: Mapping[str, Any] = payload
    for key in ("state_dict", "model_state_dict", "model", "network", "net"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidate = value
            break

    state_dict = {
        str(key): value
        for key, value in candidate.items()
        if isinstance(value, torch.Tensor)
    }
    if not state_dict:
        raise ValueError("Checkpoint does not contain a tensor state dictionary.")
    return state_dict


def _strip_wrapper_prefixes(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove prefixes commonly added by trainers and distributed wrappers."""
    prefixes = ("module.", "model.", "network.", "net.", "_orig_mod.")
    stripped: dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        key = raw_key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
                    break
        stripped[key] = value
    return stripped


def _load_compatible_weights(
    model: DenseNet121,
    checkpoint: CheckpointType,
    in_channels: int,
) -> None:
    """Load compatible checkpoint tensors, adapting the first convolution."""
    payload = _load_checkpoint_payload(checkpoint)
    state_dict = _strip_wrapper_prefixes(_extract_state_dict(payload))

    try:
        first_conv_key = _first_conv_key(state_dict)
    except KeyError:
        first_conv_key = ""
    if first_conv_key:
        old_in_channels = int(state_dict[first_conv_key].shape[1])
        if old_in_channels != in_channels:
            state_dict = adapt_first_conv_weights(
                state_dict,
                old_in_channels=old_in_channels,
                new_in_channels=in_channels,
            )

    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    incompatible: list[str] = []
    unexpected: list[str] = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
        elif model_state[key].shape != value.shape:
            incompatible.append(
                f"{key}: checkpoint {tuple(value.shape)} != model "
                f"{tuple(model_state[key].shape)}"
            )
        else:
            compatible[key] = value

    if not compatible:
        raise ValueError("No checkpoint tensors are compatible with the DenseNet model.")
    model.load_state_dict(compatible, strict=False)

    if incompatible:
        warnings.warn(
            "Skipped shape-incompatible checkpoint tensors (often an expected "
            "classifier-head change): " + "; ".join(incompatible),
            stacklevel=2,
        )
    if unexpected:
        warnings.warn(
            f"Ignored {len(unexpected)} unexpected checkpoint tensors.",
            stacklevel=2,
        )


def build_densenet121(
    in_channels: int,
    out_channels: int,
    pretrained_ckpt: CheckpointType | None = None,
) -> DenseNet121:
    """Build a 3D MONAI DenseNet121 and optionally load compatible weights.

    ``out_channels`` is intentionally generic: use one or two logits for a
    binary formulation and three logits for ternary/ordinal quality classes.
    """
    if in_channels <= 0:
        raise ValueError("in_channels must be a positive integer.")
    if out_channels <= 0:
        raise ValueError("out_channels must be a positive integer.")

    model = DenseNet121(
        spatial_dims=3,
        in_channels=int(in_channels),
        out_channels=int(out_channels),
    )
    if pretrained_ckpt is not None:
        _load_compatible_weights(model, pretrained_ckpt, int(in_channels))
    return model
