"""Loss functions used by prostate MRI task and quality models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def ordinal_ce_mae_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    lambda_ord: float = 0.5,
) -> torch.Tensor:
    """Combine categorical cross entropy with an ordinal expected-score MAE.

    The softmax probabilities define an expected class score. Penalizing its
    distance from the true ordinal class makes a two-level error (for example,
    reject predicted as accept) cost more than an adjacent-class error.
    """
    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape [batch, classes], "
            f"received {tuple(logits.shape)}."
        )
    if logits.shape[1] < 2:
        raise ValueError("ordinal loss requires at least two ordered classes.")
    if target.ndim != 1 or target.shape[0] != logits.shape[0]:
        raise ValueError(
            "target must have shape [batch] matching logits, "
            f"received {tuple(target.shape)}."
        )
    if not math.isfinite(lambda_ord) or lambda_ord < 0:
        raise ValueError("lambda_ord must be a finite non-negative number.")

    target_long = target.to(device=logits.device, dtype=torch.long)
    if target_long.numel() and (
        int(target_long.min()) < 0 or int(target_long.max()) >= logits.shape[1]
    ):
        raise ValueError(
            f"target values must be between 0 and {logits.shape[1] - 1}."
        )

    cross_entropy = F.cross_entropy(logits, target_long)
    probabilities = torch.softmax(logits, dim=1)
    ordinal_scores = torch.arange(
        logits.shape[1],
        device=logits.device,
        dtype=logits.dtype,
    )
    expected_score = (probabilities * ordinal_scores.unsqueeze(0)).sum(dim=1)
    ordinal_mae = F.l1_loss(expected_score, target_long.to(logits.dtype))
    return cross_entropy + float(lambda_ord) * ordinal_mae
