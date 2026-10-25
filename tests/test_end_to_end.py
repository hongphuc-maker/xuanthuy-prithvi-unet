"""Train one step, validate, select, map and predict on a small synthetic scene (CPU).

Slow (~2 min, full 300M model, random weights): run with ``RUN_SLOW=1 pytest tests/test_end_to_end.py``.
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

pytest.importorskip("terratorch")
pytestmark = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1")

from prithvi_unet.config import load_config  # noqa: E402
from prithvi_unet.inference import predict_new_dates  # noqa: E402
from prithvi_unet.training import Trainer  # noqa: E402
from prithvi_unet.utils import sha256_file  # noqa: E402

from .conftest import CONFIGS  # noqa: E402
from .test_sampling import SPLIT, synthetic  # noqa: E402

PROFILE = dict(driver="GTiff", crs="EPSG:32648", transform=from_origin(650500, 2245550, 10, 10))


def write_image(path, day, shape, seed):
    dn = np.random.default_rng(seed).integers(900, 4000, size=(6, *shape)).astype(np.uint16)
    with rasterio.open(path, "w", height=shape[0], width=shape[1], count=6, dtype="uint16", nodata=0, **PROFILE) as dst:
        dst.write(dn)
        for i, band in enumerate(["B02", "B03", "B04", "B8A", "B11", "B12"], start=1):
            dst.set_band_description(i, band)
        dst.update_tags(acquisition_date=day)


def test_train_map_predict(tmp_path):
    label, _ = synthetic()
    shape = label.shape
    data = tmp_path / "data"
    data.mkdir()
    base = load_config(CONFIGS / "p3_single_date.json")
    images = []
    for k, spec in enumerate(base["images"]):
        write_image(data / spec["path"], spec["date"], shape, k)
        images.append({**spec, "sha256": sha256_file(data / spec["path"])})
    with rasterio.open(data / "label.tif", "w", height=shape[0], width=shape[1], count=1, dtype="uint8", **PROFILE) as dst:
        dst.write(label, 1)

    config = copy.deepcopy(base)
    config.update(images=images, expected_shape=list(shape))
    config["label"] = {**config["label"], "path": "label.tif", "sha256": sha256_file(data / "label.tif")}
    config["split"] = SPLIT
    config["training"].update(effective_batch_size=4, maximum_steps=1, validation_interval_steps=1, save_interval_steps=1)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    trainer = Trainer(config_path, data, tmp_path / "run")
    trainer.load_model(pretrained=False)  # no download in tests
    result = trainer.train(stop_after_steps=1)
    assert result["completed"] and result["step"] == 1
    assert (tmp_path / "run" / "best_any.pt").exists()
    written = trainer.map()
    with rasterio.open(written["classes"]) as src:
        classes = src.read(1)
    assert classes.shape == shape and set(np.unique(classes)) <= set(range(12))

    new_image = tmp_path / "new.tif"
    write_image(new_image, "2026-10-01", shape, 99)
    maps = predict_new_dates(config_path, tmp_path / "run" / "best_any.pt", [new_image], tmp_path / "pred")
    with rasterio.open(maps["classes"]) as src:
        assert src.tags()["map_date"] == "2026-10-01"
