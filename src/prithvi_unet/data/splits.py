"""Spatial train/validation split: one validation band of columns (or rows) with a guard.

Ported from xuan-thuy-dmi-segmentation @ 0775770, ``build_split_domains`` (band strategies only):
https://github.com/hongphuc-maker/xuan-thuy-dmi-segmentation/blob/0775770ae87b83b7d2982ed727226128685f3fba/src/xuanthuy_seg/data/splits.py#L125-L209
Output is identical (tests/test_sampling.py).

    columns:  [ train | guard 48 | validation 1122..1388 | guard 48 | train ]

* ``validation_metric``: pixels that are scored.
* ``validation_input``: metric band + guards. Validation crops stay inside it.
* ``train_input``: everything else. Training crops stay inside it, so no input
  pixel is shared between training and validation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EXCLUDED, TRAIN, VALIDATION, GUARD = np.uint8(0), np.uint8(1), np.uint8(2), np.uint8(3)


def build_split_domains(shape: tuple[int, int], valid_mask: np.ndarray, split: dict[str, Any]) -> dict[str, np.ndarray]:
    if valid_mask.shape != shape or valid_mask.dtype != bool:
        raise ValueError("valid_mask must be bool and have the raster shape")
    strategy = str(split["strategy"])
    if strategy not in {"vertical_band", "horizontal_band"}:
        raise ValueError(f"Unsupported split strategy: {strategy}")
    p = split["parameters"]
    axis = 1 if strategy == "vertical_band" else 0
    extent = shape[axis]
    start, stop, guard = int(p["validation_start"]), int(p["validation_stop"]), int(p["guard_px_each_side"])
    if not 0 <= start < stop <= extent:
        raise ValueError(f"Invalid validation interval [{start}, {stop})")

    coords = np.arange(extent)
    metric_1d = (coords >= start) & (coords < stop)
    input_1d = (coords >= max(0, start - guard)) & (coords < min(extent, stop + guard))

    def broadcast(line: np.ndarray) -> np.ndarray:
        return np.broadcast_to(line[None, :] if axis == 1 else line[:, None], shape).copy()

    validation_metric = broadcast(metric_1d)
    validation_input = broadcast(input_1d)
    guard_domain = validation_input & ~validation_metric
    train_input = ~validation_input

    split_mask = np.full(shape, EXCLUDED, dtype=np.uint8)
    split_mask[valid_mask & train_input] = TRAIN
    split_mask[valid_mask & guard_domain] = GUARD
    split_mask[valid_mask & validation_metric] = VALIDATION
    return {
        "split_mask": split_mask,
        "train_input": train_input,
        "validation_input": validation_input,
        "validation_metric": validation_metric,
        "guard": guard_domain,
    }
