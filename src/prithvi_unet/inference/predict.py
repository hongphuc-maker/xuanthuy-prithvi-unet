"""Label-free map for new dates (e.g. 1 October 2026) from a trained checkpoint.

Inputs: an experiment config (model shape, radiometry), a checkpoint (new format or a
legacy ``best_any.pt`` from the completed runs) and exactly T GeoTIFFs exported by
the Sentinel-2 export notebook (6 bands, ``acquisition_date`` tag, same grid).
The images are sorted by date; the last one is the date of the map.
"""

from __future__ import annotations

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
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if not any(k.startswith("encoder.vit.") for k in state):
        state = convert_legacy_state_dict(state)  # checkpoint of the completed runs
    model.load_state_dict(state, strict=True)
    return payload if isinstance(payload, dict) else {}


def predict_new_dates(config_path: str | Path, checkpoint: str | Path, images: list[str | Path],
                      out_dir: str | Path, stride: int | None = None) -> dict[str, str]:
    config = load_config(config_path)
    frames = len(config["active_dates"])
    specs = []
    for path in images:
        with rasterio.open(path) as src:
            specs.append({"path": str(Path(path).resolve()), "date": src.tags()["acquisition_date"]})
    specs.sort(key=lambda s: s["date"])
    if len(specs) != frames:
        raise ValueError(f"This model takes {frames} date(s); got {len(specs)} image(s)")

    scene = RasterStack(specs, ".", config["radiometry"], config["normalization"],
                        config["split"]["patch_policy"]["size"], verify_checksums=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, pretrained=False)
    load_any_checkpoint(checkpoint, model)
    model.to(device)
    stride = stride or config["split"]["patch_policy"]["inference_stride"]
    arrays = predict_map(model, scene, device, len(config["class_map"]), stride)
    tags = {"map_date": specs[-1]["date"], "input_dates": ",".join(s["date"] for s in specs),
            "checkpoint": Path(checkpoint).name, "checkpoint_sha256": sha256_file(checkpoint),
            "experiment_id": config["experiment_id"], "note": "No date-matched labels: map is not scored"}
    return write_geotiffs(arrays, scene.profile, out_dir, specs[-1]["date"], tags)
