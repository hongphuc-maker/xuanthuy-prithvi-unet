"""Patch sampling, batch schedule and split are deterministic, and identical to the
original package when it is installed (xuan-thuy-dmi-segmentation @ 0775770)."""

import numpy as np
import pandas as pd
import pytest

from prithvi_unet.data.sampling import build_batch_schedule, build_validation_windows, sample_training_patches
from prithvi_unet.data.splits import build_split_domains

CLASS_MAP = {i: f"c{i}" for i in range(1, 12)}
SPLIT = {
    "strategy": "vertical_band",
    "parameters": {"validation_start": 450, "validation_stop": 520, "guard_px_each_side": 16},
    "patch_policy": {"size": 224, "minimum_train_fraction": 0.5, "validation_stride": 112, "inference_stride": 112},
    "sampling": {"manifest_seed": 1, "schedule_seed": 2, "n_patches": 8, "batch_size": 4,
                 "rare_codes": [5, 6, 7], "rare_quota_each": 2, "dmi_support_floor_px": 16},
}


def synthetic(shape=(300, 700)):
    cols = np.arange(shape[1])
    label = np.broadcast_to((cols // 9) % 11 + 1, shape).astype(np.uint8).copy()
    label[:, ::31] = 0  # some unlabelled pixels
    valid = np.ones(shape, bool)
    valid[:5] = False
    return label, valid


def run_new(label, valid):
    common = valid & (label > 0)
    d = build_split_domains(label.shape, common, SPLIT)
    m = sample_training_patches(label, common, d["split_mask"], d["train_input"], SPLIT, CLASS_MAP)
    s, a = build_batch_schedule(m, SPLIT, CLASS_MAP)
    v = build_validation_windows(label.shape, SPLIT, valid, common & d["validation_metric"], d["validation_input"])
    return d, m, s, v


def test_split_has_no_shared_input_pixels():
    label, valid = synthetic()
    d = build_split_domains(label.shape, valid & (label > 0), SPLIT)
    assert not (d["train_input"] & d["validation_input"]).any()
    assert d["validation_metric"][:, 450:520].all() and not d["validation_metric"][:, :450].any()


def test_deterministic():
    label, valid = synthetic()
    _, m1, s1, v1 = run_new(label, valid)
    _, m2, s2, v2 = run_new(label, valid)
    pd.testing.assert_frame_equal(m1, m2)
    pd.testing.assert_frame_equal(s1, s2)
    assert set(s1.manifest_index) == set(range(8))


def test_identical_to_original_package():
    prep = pytest.importorskip("xuanthuy_seg.data.preparation")
    splits = pytest.importorskip("xuanthuy_seg.data.splits")
    label, valid = synthetic()
    common = valid & (label > 0)
    d_old = splits.build_split_domains(label.shape, common, SPLIT)
    m_old = prep._sample_manifest(label, common, d_old["split_mask"], d_old["train_input"], SPLIT, CLASS_MAP)
    s_old, _ = prep._build_schedule(m_old, SPLIT, CLASS_MAP)
    v_old = prep._build_validation_manifest(label.shape, SPLIT, valid, common & d_old["validation_metric"],
                                            d_old["validation_input"])
    d, m, s, v = run_new(label, valid)
    for key in d:
        np.testing.assert_array_equal(d[key], d_old[key])
    pd.testing.assert_frame_equal(m, m_old)
    pd.testing.assert_frame_equal(s, s_old)
    pd.testing.assert_frame_equal(v, v_old)
