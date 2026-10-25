"""Sliding-window prediction over a whole scene.

Windows of 224 x 224 with stride 112 (half overlap). For every pixel the softmax
probabilities of all windows that contain it are averaged, then argmax.

Why overlap: a convolution with zero padding sees less context near a window
border, so border predictions are weaker. With stride 112 every pixel away from
the raster edge is also covered by a window in which it lies far from the border.
This plays the role of the "overlap-tile" strategy of the original U-Net paper.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from ..data.scene import stack_crops


def window_starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)  # last window flush with the edge
    return starts


def scene_windows(shape: tuple[int, int], valid: np.ndarray, size: int, stride: int) -> list[dict[str, int]]:
    h, w = shape
    return [
        {"row": r, "col": c, "height": min(size, h - r), "width": min(size, w - c)}
        for r in window_starts(h, size, stride)
        for c in window_starts(w, size, stride)
        if valid[r : r + size, c : c + size].any()
    ]


@torch.inference_mode()
def predict_probabilities(model: torch.nn.Module, scene: Any, windows: list[dict[str, int]], device: torch.device,
                          num_classes: int, domain: np.ndarray | None = None, progress: bool = True
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean softmax [C, H, W], number of windows per pixel [H, W])."""
    was_training = model.training
    model.eval()
    h, w = scene.shape
    prob_sum = np.zeros((num_classes, h, w), np.float32)
    counts = np.zeros((h, w), np.uint16)
    try:
        for win in tqdm(windows, desc="Sliding windows", disable=not progress):
            r, c, hh, ww = win["row"], win["col"], win["height"], win["width"]
            crop = scene.crop(r, c, hh, ww, domain=domain)
            batch = stack_crops([crop], device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(batch["image"], batch["temporal_coords"], batch["location_coords"])
            prob = logits.float().softmax(dim=1)[0, :, :hh, :ww].cpu().numpy()  # drop padding
            mask = crop["valid"]
            prob_sum[:, r : r + hh, c : c + ww] += prob * mask[None]
            counts[r : r + hh, c : c + ww] += mask.astype(np.uint16)
    finally:
        model.train(was_training)
    np.divide(prob_sum, counts[None], out=prob_sum, where=counts[None] > 0)
    return prob_sum, counts
