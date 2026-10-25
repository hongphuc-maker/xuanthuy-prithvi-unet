"""Experiment configuration: one JSON file per experiment (``configs/``).

The schema is the one used by the completed P2/P3 runs (``prithvi-xtseg-v1``), so
old configs, checksums and prepared data remain valid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SENTINEL2_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(c: dict[str, Any]) -> None:
    if c.get("schema") != "prithvi-xtseg-v1":
        raise ValueError("Unsupported experiment schema")
    dates = [x["date"] for x in c["images"]]
    if dates != sorted(set(dates)) or not set(c["active_dates"]) <= set(dates):
        raise ValueError("Image dates must be unique and sorted, and contain every active date")
    if c["active_dates"] != sorted(set(c["active_dates"])):
        raise ValueError("active_dates must be sorted and unique")
    if c["target_date"] != c["active_dates"][-1]:
        raise ValueError("The target date must be the last active date (the model reads out the last date)")
    if c["bands"] != SENTINEL2_BANDS:
        raise ValueError(f"Band order must be {SENTINEL2_BANDS} (the order of the pretrained Conv3d channels)")
    if c["training"]["effective_batch_size"] != c["split"]["sampling"]["batch_size"]:
        raise ValueError("training.effective_batch_size and split.sampling.batch_size disagree")
    m = c["model"]
    if m["decoder"] != "UNetDecoder":
        raise ValueError("Only UNetDecoder is implemented")
    if len(m["out_indices"]) != 4 or len(m["decoder_channels"]) != 4:
        raise ValueError("Four encoder readouts and four decoder blocks are required")
    for key in ("mean", "std"):
        if len(c["normalization"][key]) != len(SENTINEL2_BANDS):
            raise ValueError(f"normalization.{key} needs one value per band")
