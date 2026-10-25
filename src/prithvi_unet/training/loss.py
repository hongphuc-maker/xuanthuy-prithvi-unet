"""Weighted cross-entropy (WCE) with ignored pixels.

Ported from xuan-thuy-dmi-segmentation @ 0775770:
    class_weights_from_counts          https://github.com/hongphuc-maker/xuan-thuy-dmi-segmentation/blob/0775770ae87b83b7d2982ed727226128685f3fba/src/xuanthuy_seg/losses/cross_entropy.py#L10-L29
    deterministic_masked_cross_entropy https://github.com/hongphuc-maker/xuan-thuy-dmi-segmentation/blob/0775770ae87b83b7d2982ed727226128685f3fba/src/xuanthuy_seg/losses/cross_entropy.py#L32-L63

Class weights (computed once from training-pixel counts n_c):
    w_c = sqrt(median(n) / n_c), clipped to [0.5, 5], then divided by their mean (mean = 1)
    -> rare classes weigh more, but at most 10x the most common ones.

Loss for one batch (pixels with target 255 are ignored):
    WCE = sum_i w_{y_i} * (-log softmax(z_i)[y_i]) / sum_i w_{y_i}
Written with log_softmax and one-hot instead of F.cross_entropy so that it is
deterministic on CUDA.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

IGNORE_INDEX = 255


def class_weights_from_counts(counts: Sequence[int] | torch.Tensor, minimum: float = 0.5, maximum: float = 5.0) -> torch.Tensor:
    values = torch.as_tensor(counts, dtype=torch.float64)
    if values.ndim != 1 or torch.any(values <= 0):
        raise ValueError("Class counts must be positive")
    weights = torch.sqrt(torch.median(values) / values).clamp(min=minimum, max=maximum)
    return (weights / weights.mean()).to(torch.float32)


def weighted_cross_entropy(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor | None = None,
                           ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """logits [B, C, H, W], target [B, H, W] -> scalar."""
    if logits.ndim != 4 or target.ndim != 3 or logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
        raise ValueError("Expected logits [B, C, H, W] and target [B, H, W] of the same size")
    valid = target != ignore_index
    if not bool(valid.any()):
        raise ValueError("No labelled pixel in this batch")
    safe_target = torch.where(valid, target, torch.zeros_like(target)).long()
    if int(safe_target[valid].max()) >= logits.shape[1]:
        raise ValueError("Target class index out of range")
    one_hot = F.one_hot(safe_target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).to(logits.dtype)
    per_pixel = -(one_hot * F.log_softmax(logits, dim=1)).sum(dim=1)  # -log p(true class)
    pixel_weight = valid.to(logits.dtype)
    if class_weights is not None:
        pixel_weight = pixel_weight * class_weights.to(logits.device, logits.dtype)[safe_target]
    return (per_pixel * pixel_weight).sum() / pixel_weight.sum()
