"""Label-free map for new images (another date, or another area) from a trained checkpoint.

Inputs
    config      the experiment config the checkpoint was trained with (model shape, bands,
                radiometry, normalisation, class names). Its image/label lists are not read.
    checkpoint  new format, or a legacy ``best_any.pt`` / ``best_accepted.pt`` of the completed runs
    images      exactly T GeoTIFFs (T = number of active dates in the config), one per date,
                on one common grid, with band descriptions and an ``acquisition_date`` tag
                (scripts/export_sentinel2.py writes both; scripts/set_band_metadata.py adds them).
The images are sorted by date; the last one is the date of the map. Any size and any
projected CRS work; the location embedding is computed from each window's centre.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import rasterio
import torch

from ..config import load_config
from ..data.scene import RasterStack
from ..model import build_model, convert_legacy_state_dict
from ..utils import sha256_file
from .maps import predict_map, write_geotiffs


def load_any_checkpoint(path: str | Path, model: torch.nn.Module) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)  # no extra copy in RAM
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if not any(k.startswith("encoder.vit.") for k in state):
        state = convert_legacy_state_dict(state)  # checkpoint of the completed runs
    model.load_state_dict(state, strict=True)
    return payload if isinstance(payload, dict) else {}


def estimate_memory_gb(height: int, width: int, frames: int, bands: int, classes: int) -> float:
    """Raw DN (uint16) + summed probabilities (float32) + outputs; the model itself adds ~1.3 GB."""
    pixels = height * width
    return (pixels * frames * bands * 2 + pixels * classes * 4 + pixels * 12) / 1e9


def predict_new_dates(config_path: str | Path, checkpoint: str | Path, images: list[str | Path],
                      out_dir: str | Path, stride: int | None = None, final_upsample: str | None = None,
                      dn_offset: float | str | None = None) -> dict[str, str]:
    """Write <date>_classes.tif, _max_softmax.tif, _entropy.tif, _validity.tif to ``out_dir``.

    final_upsample  override "bilinear" (as trained) / "nearest", to compare without retraining
    dn_offset       override the config: a number, or "auto" (per image from its tags or date)
    """
    config = copy.deepcopy(load_config(config_path))
    if final_upsample:
        config["model"]["final_upsample"] = final_upsample
    if dn_offset is not None:
        config["radiometry"]["dn_offset"] = dn_offset
    frames = len(config["active_dates"])

    specs = []
    for path in images:
        with rasterio.open(path) as src:
            day = src.tags().get("acquisition_date")
            if not day:
                raise ValueError(f"{path}: no acquisition_date tag (add it with scripts/set_band_metadata.py)")
            specs.append({"path": str(Path(path).resolve()), "date": day})
            height, width = src.height, src.width
    specs.sort(key=lambda s: s["date"])
    if len(specs) != frames:
        raise ValueError(f"This model takes {frames} date(s); got {len(specs)} image(s)")
    need = estimate_memory_gb(height, width, frames, len(config["bands"]), len(config["class_map"]))
    if need > 6:
        print(f"Warning: about {need:.1f} GB of RAM is needed for a {height} x {width} scene; "
              "clip the area into smaller pieces if the runtime runs out of memory.")

    scene = RasterStack(specs, ".", config["bands"], config["radiometry"], config["normalization"],
                        config["split"]["patch_policy"]["size"], verify_checksums=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, pretrained=False)
    load_any_checkpoint(checkpoint, model)
    model.to(device)
    stride = stride or config["split"]["patch_policy"]["inference_stride"]
    arrays = predict_map(model, scene, device, len(config["class_map"]), stride)
    tags = {"map_date": specs[-1]["date"], "input_dates": ",".join(s["date"] for s in specs),
            "dn_offsets": ",".join(str(i["dn_offset"]) for i in scene.images), "bands": ",".join(config["bands"]),
            "final_upsample": config["model"].get("final_upsample", "bilinear"), "stride": str(stride),
            "checkpoint": Path(checkpoint).name, "checkpoint_sha256": sha256_file(checkpoint),
            "experiment_id": config["experiment_id"], "note": "No date-matched labels: map is not scored"}
    return write_geotiffs(arrays, scene.profile, out_dir, specs[-1]["date"], tags)
