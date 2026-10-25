"""Validation scoring and map export.

Ported from the completed runs, ``validation`` and ``create_map``:
https://github.com/hongphuc-maker/xuan-thuy-prithvi-segmentation/blob/42290823beeb42e44ec87bb8845e439e10c18735/reference/wce_completed/prithvi_xtseg/inference.py#L61-L123
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch

from .metrics import acceptance_gate, confusion_matrix, per_class_metrics
from .tiling import predict_probabilities, scene_windows


def validate(model: torch.nn.Module, scene: Any, windows: pd.DataFrame, device: torch.device
             ) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score the validation band: sliding windows inside the validation input domain only."""
    k = len(scene.config["class_map"])
    prob, counts = predict_probabilities(model, scene, windows.to_dict("records"), device, k,
                                         domain=scene.domains["validation_input"], progress=False)
    mask = scene.val_mask
    if not np.all(counts[mask] > 0):
        raise ValueError("Some scored pixels were not covered by a window")
    truth = scene.label[mask].astype(np.int64) - 1
    p = prob[:, mask]
    summary, table = per_class_metrics(confusion_matrix(truth, p.argmax(0), k), scene.config["class_map"])
    weights = scene.class_weights.numpy()[truth]
    summary["WCE"] = float((weights * -np.log(np.maximum(p[truth, np.arange(len(truth))], 1e-12))).sum() / weights.sum())
    summary["acceptance_passed"] = acceptance_gate(summary, table, scene.config["acceptance"])
    return summary, table


def predict_map(model: torch.nn.Module, scene: Any, device: torch.device, num_classes: int, stride: int
                ) -> dict[str, np.ndarray]:
    """Whole-scene prediction: classes (1..11, 0 = nodata), max softmax, normalised entropy."""
    windows = scene_windows(scene.shape, scene.image_valid, scene.patch_size, stride)
    prob, counts = predict_probabilities(model, scene, windows, device, num_classes)
    valid = scene.image_valid & (counts > 0)
    classes = np.zeros(scene.shape, np.uint8)
    classes[valid] = prob[:, valid].argmax(0).astype(np.uint8) + 1
    confidence = np.full(scene.shape, np.nan, np.float32)
    confidence[valid] = prob[:, valid].max(0)
    entropy = np.full(scene.shape, np.nan, np.float32)
    pv = prob[:, valid]
    entropy[valid] = -(pv * np.log(np.maximum(pv, 1e-12))).sum(0) / np.log(num_classes)
    return {"classes": classes, "max_softmax": confidence, "entropy": entropy, "validity": valid.astype(np.uint8)}


def write_geotiffs(arrays: dict[str, np.ndarray], profile: dict[str, Any], out_dir: str | Path, prefix: str,
                   tags: dict[str, str]) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, array in arrays.items():
        path = out_dir / f"{prefix}_{name}.tif"
        temporary = path.with_name(path.stem + ".tmp.tif")
        nodata = np.nan if np.issubdtype(array.dtype, np.floating) else 0
        p = profile.copy()
        p.update(count=1, dtype=str(array.dtype), nodata=nodata, compress="deflate")
        with rasterio.open(temporary, "w", **p) as dst:
            dst.write(array, 1)
            dst.update_tags(**tags)
        temporary.replace(path)
        written[name] = str(path)
    return written
