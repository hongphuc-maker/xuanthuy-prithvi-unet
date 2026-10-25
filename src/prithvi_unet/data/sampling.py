"""Which 224 x 224 crops are used for training and validation, and in which batches.

Ported from xuan-thuy-dmi-segmentation @ 0775770, ``xuanthuy_seg/data/preparation.py``:
    _sample_manifest           https://github.com/hongphuc-maker/xuan-thuy-dmi-segmentation/blob/0775770ae87b83b7d2982ed727226128685f3fba/src/xuanthuy_seg/data/preparation.py#L162-L268
    _build_schedule            https://github.com/hongphuc-maker/xuan-thuy-dmi-segmentation/blob/0775770ae87b83b7d2982ed727226128685f3fba/src/xuanthuy_seg/data/preparation.py#L271-L361
    _build_validation_manifest https://github.com/hongphuc-maker/xuan-thuy-dmi-segmentation/blob/0775770ae87b83b7d2982ed727226128685f3fba/src/xuanthuy_seg/data/preparation.py#L118-L159
    window-sum helpers         https://github.com/hongphuc-maker/xuan-thuy-dmi-segmentation/blob/0775770ae87b83b7d2982ed727226128685f3fba/src/xuanthuy_seg/data/preparation.py#L52-L115 The random-number calls are kept in the same order, so the same config
gives the same patches and batches as the completed runs
(``tests/test_sampling.py`` checks this when the original package is installed).

Training patches (``sample_training_patches``)
    A top-left corner is eligible when the whole 224 x 224 window lies in the
    training input domain, at least ``minimum_train_fraction`` of it is labelled
    training pixels, and its centre pixel is a labelled training pixel.
    First ``rare_quota_each`` patches are drawn for each rare class (patches that
    contain at least ``dmi_support_floor_px`` pixels of it), then the rest
    uniformly.

Batches (``build_batch_schedule``)
    Every batch of 16 gets one anchor patch per rare class, and the remaining
    slots are shuffled until every batch contains every class (deterministic
    retries with seed + attempt).

Validation windows (``build_validation_windows``)
    Windows of 224 with stride 112 that cover the validation band; predictions
    are averaged where they overlap.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- window sums
def integral_image(array: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero first row/column: window sums in O(1)."""
    return np.pad(array.astype(np.int64).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def all_window_sums(array: np.ndarray, size: int) -> np.ndarray:
    """Sum of ``array`` in every size x size window, indexed by the top-left corner."""
    s = integral_image(array)
    return s[size:, size:] - s[:-size, size:] - s[size:, :-size] + s[:-size, :-size]


def window_sums_at(integral: np.ndarray, rows: np.ndarray, cols: np.ndarray, size: int) -> np.ndarray:
    return (
        integral[rows + size, cols + size]
        - integral[rows, cols + size]
        - integral[rows + size, cols]
        + integral[rows, cols]
    )


def cover_starts(start: int, stop: int, size: int, stride: int) -> list[int]:
    """Window starts covering [start, stop) with the given stride; the last window ends at ``stop``."""
    if stop - start <= size:
        return [start]
    starts = list(range(start, stop - size + 1, stride))
    if starts[-1] != stop - size:
        starts.append(stop - size)
    return starts


# ----------------------------------------------------------------------------- training patches
def sample_training_patches(
    label: np.ndarray,
    common_valid: np.ndarray,
    split_mask: np.ndarray,
    train_input: np.ndarray,
    split: dict[str, Any],
    class_map: dict[int, str],
) -> pd.DataFrame:
    patch = int(split["patch_policy"]["size"])
    if patch % 2:
        raise ValueError("Patch size must be even (centre convention)")
    half = patch // 2
    minimum_fraction = float(split["patch_policy"].get("minimum_train_fraction", 1.0))
    sampling = split["sampling"]
    n_patches = int(sampling["n_patches"])
    rare_codes = [int(code) for code in sampling.get("rare_codes", [])]
    rare_quota = int(sampling.get("rare_quota_each", 0))
    support_floor = int(sampling.get("dmi_support_floor_px", 1))
    rng = np.random.default_rng(int(sampling["manifest_seed"]))

    # Eligible top-left corners.
    footprint_ok = all_window_sums(train_input, patch) == patch * patch
    train_target = common_valid & (split_mask == 1)
    footprint_ok &= all_window_sums(train_target, patch) >= math.ceil(minimum_fraction * patch * patch)
    top_rows, left_cols = np.nonzero(footprint_ok)
    center_rows, center_cols = top_rows + half, left_cols + half
    center_ok = train_target[center_rows, center_cols]
    top_rows, left_cols = top_rows[center_ok], left_cols[center_ok]
    center_rows, center_cols = center_rows[center_ok], center_cols[center_ok]
    center_classes = label[center_rows, center_cols].astype(np.int16)
    if len(top_rows) < n_patches:
        raise ValueError(f"Only {len(top_rows)} eligible centres for {n_patches} requested patches")

    # Rare-class quotas first.
    rare_support = {
        code: window_sums_at(integral_image(train_target & (label == code)), top_rows, left_cols, patch)
        for code in rare_codes
    }
    selected: list[int] = []
    streams: dict[int, str] = {}
    used = np.zeros(len(top_rows), dtype=bool)
    for code in rare_codes:
        eligible = np.flatnonzero((rare_support[code] >= support_floor) & ~used)
        if len(eligible) < rare_quota:
            raise ValueError(f"Class {code}: {len(eligible)} eligible patches < rare_quota_each={rare_quota}")
        chosen = rng.choice(eligible, size=rare_quota, replace=False)
        used[chosen] = True
        for index in chosen.tolist():
            streams[index] = f"rare_{code}_{class_map[code]}"
        selected.extend(chosen.tolist())

    # Uniform remainder.
    remainder = n_patches - len(selected)
    if remainder < 0:
        raise ValueError("Rare quotas exceed n_patches")
    chosen = rng.choice(np.flatnonzero(~used), size=remainder, replace=False)
    for index in chosen.tolist():
        streams[index] = "uniform_remainder"
    selected.extend(chosen.tolist())
    sel = np.asarray(selected, dtype=np.int64)

    rows, cols = top_rows[sel], left_cols[sel]
    manifest = pd.DataFrame(
        {
            "patch_id": [f"train_cr{r:04d}_cc{c:04d}" for r, c in zip(center_rows[sel], center_cols[sel], strict=True)],
            "center_row": center_rows[sel],
            "center_col": center_cols[sel],
            "row": rows,
            "col": cols,
            "height": patch,
            "width": patch,
            "center_class": center_classes[sel],
            "sampling_stream": [streams[i] for i in selected],
        }
    )
    manifest["train_valid_px"] = window_sums_at(integral_image(train_target), rows, cols, patch)
    manifest["train_valid_fraction"] = manifest["train_valid_px"] / (patch * patch)
    class_columns = []
    for code in sorted(class_map):
        column = f"class_{code}_px"
        class_columns.append(column)
        manifest[column] = window_sums_at(integral_image(train_target & (label == code)), rows, cols, patch)
    manifest["n_classes"] = (manifest[class_columns].to_numpy() > 0).sum(axis=1)
    manifest["class_codes"] = [
        "|".join(str(code) for code, n in zip(sorted(class_map), values, strict=True) if n > 0)
        for values in manifest[class_columns].to_numpy()
    ]
    if manifest.patch_id.duplicated().any():
        raise AssertionError("Duplicate patch centres")
    return manifest


# ----------------------------------------------------------------------------- batches
def build_batch_schedule(
    manifest: pd.DataFrame, split: dict[str, Any], class_map: dict[int, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampling = split["sampling"]
    batch_size = int(sampling["batch_size"])
    floor = int(sampling.get("dmi_support_floor_px", 1))
    rare_codes = [int(code) for code in sampling.get("rare_codes", [])]
    n_patches = len(manifest)
    if n_patches % batch_size:
        raise ValueError("n_patches must be divisible by batch_size")
    n_batches = n_patches // batch_size
    if batch_size <= len(rare_codes):
        raise ValueError("batch_size must exceed the number of rare anchors")

    # One anchor per rare class per batch.
    anchors: dict[int, np.ndarray] = {}
    reserved: set[int] = set()
    base_rng = np.random.default_rng(int(sampling["schedule_seed"]))
    for code in rare_codes:
        candidates = manifest.index[
            manifest["sampling_stream"].str.startswith(f"rare_{code}_") & (manifest[f"class_{code}_px"] >= floor)
        ].to_numpy(np.int64)
        if len(candidates) < n_batches:
            raise ValueError(f"Not enough class-{code} anchors for {n_batches} batches")
        anchors[code] = base_rng.permutation(candidates)[:n_batches]
        if any(int(i) in reserved for i in anchors[code]):
            raise AssertionError("Rare anchor pools overlap")
        reserved.update(int(i) for i in anchors[code])

    # Fill the other slots; retry deterministically until every batch sees every class.
    extras = np.asarray([i for i in manifest.index if int(i) not in reserved], dtype=np.int64)
    free_slots = batch_size - len(rare_codes)
    class_columns = [f"class_{code}_px" for code in sorted(class_map)]
    accepted, accepted_attempt = None, -1
    for attempt in range(512):
        shuffled = np.random.default_rng(int(sampling["schedule_seed"]) + attempt).permutation(extras)
        batches = []
        for b in range(n_batches):
            batch = [int(anchors[code][b]) for code in rare_codes]
            batch.extend(int(v) for v in shuffled[b * free_slots : (b + 1) * free_slots])
            batches.append(batch)
        exposure = np.asarray([manifest.loc[batch, class_columns].to_numpy(np.int64).sum(axis=0) for batch in batches])
        if np.all(exposure >= floor):
            accepted, accepted_attempt = batches, attempt
            break
    if accepted is None:
        raise ValueError("No class-complete batch schedule after 512 deterministic attempts")

    rows, audit = [], []
    for b, batch in enumerate(accepted):
        anchor_of = {int(anchors[code][b]): code for code in rare_codes}
        for slot, index in enumerate(batch):
            rows.append({"pass_index": 0, "batch_index": b, "slot": slot, "manifest_index": index,
                         "patch_id": manifest.loc[index, "patch_id"], "anchor_code": anchor_of.get(index, 0),
                         "schedule_attempt": accepted_attempt})
        exposure = manifest.loc[batch, class_columns].to_numpy(np.int64).sum(axis=0)
        audit.append({"batch_index": b, "n_patches": len(batch), "minimum_class_exposure_px": int(exposure.min()),
                      **{f"class_{code}_exposure_px": int(exposure[k]) for k, code in enumerate(sorted(class_map))}})
    schedule = pd.DataFrame(rows)
    if schedule.manifest_index.nunique() != n_patches:
        raise AssertionError("Schedule must use every patch exactly once")
    return schedule, pd.DataFrame(audit)


def batches_from_schedule(schedule: pd.DataFrame) -> list[np.ndarray]:
    return [g.sort_values("slot").manifest_index.to_numpy(np.int64) for _, g in schedule.groupby("batch_index", sort=True)]


# ----------------------------------------------------------------------------- validation windows
def validation_regions(shape: tuple[int, int], split: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    height, width = shape
    p = split["parameters"]
    guard = int(p["guard_px_each_side"])
    if split["strategy"] == "vertical_band":
        return [(0, height, max(0, int(p["validation_start"]) - guard), min(width, int(p["validation_stop"]) + guard))]
    if split["strategy"] == "horizontal_band":
        return [(max(0, int(p["validation_start"]) - guard), min(height, int(p["validation_stop"]) + guard), 0, width)]
    raise ValueError(f"Unsupported split strategy: {split['strategy']}")


def build_validation_windows(
    shape: tuple[int, int],
    split: dict[str, Any],
    image_valid: np.ndarray,
    metric_mask: np.ndarray,
    validation_input: np.ndarray,
) -> pd.DataFrame:
    size = int(split["patch_policy"]["size"])
    stride = int(split["patch_policy"].get("validation_stride", size // 2))
    coverage = np.zeros(shape, dtype=np.uint16)
    rows = []
    for region_index, (r0, r1, c0, c1) in enumerate(validation_regions(shape, split)):
        for row in cover_starts(r0, r1, size, stride):
            for col in cover_starts(c0, c1, size, stride):
                h, w = min(size, r1 - row), min(size, c1 - col)
                region = np.s_[row : row + h, col : col + w]
                if not validation_input[region].all():
                    raise ValueError("A validation window crosses its input domain")
                valid_px, metric_px = int(image_valid[region].sum()), int(metric_mask[region].sum())
                if valid_px == 0 and metric_px == 0:
                    continue
                coverage[region] += validation_input[region].astype(np.uint16)
                rows.append({"window_id": f"val_g{region_index:03d}_r{row:04d}_c{col:04d}", "region_index": region_index,
                             "row": row, "col": col, "height": h, "width": w, "image_valid_px": valid_px,
                             "image_valid_fraction": valid_px / max(h * w, 1), "validation_metric_px": metric_px})
    if not rows or not np.all(coverage[metric_mask] > 0):
        raise ValueError("Validation windows do not cover every scored pixel")
    return pd.DataFrame(rows)
