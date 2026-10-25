"""Experiment configuration: one JSON file per experiment (``configs/``).

The schema is the one used by the completed P2/P3 runs (``prithvi-xtseg-v1``), so
old configs, checksums and prepared data remain valid. Fields added since then are
optional and default to the old behaviour:

    bands                       any subset of the six Prithvi bands, e.g. ["B02","B03","B04","B8A"]
    normalization.mean / std    one value per entry of ``bands``
    radiometry.dn_offset        a number (e.g. -1000) or "auto" (per image, see data/radiometry.py)
    model.final_upsample        "bilinear" (default, as trained) or "nearest"
    training.freeze_encoder     false (default) or true: train only neck, decoder and head
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SENTINEL2_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]  # bands Prithvi was pretrained on (HLS names below)
S2_TO_HLS = {"B02": "BLUE", "B03": "GREEN", "B04": "RED", "B8A": "NIR_NARROW", "B11": "SWIR_1", "B12": "SWIR_2"}


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
    bands = c["bands"]
    if not bands or len(set(bands)) != len(bands) or not set(bands) <= set(SENTINEL2_BANDS):
        raise ValueError(f"bands must be distinct names from {SENTINEL2_BANDS}")
    for key in ("mean", "std"):
        if len(c["normalization"][key]) != len(bands):
            raise ValueError(f"normalization.{key} needs one value per band in 'bands' ({len(bands)})")
    offset = c["radiometry"]["dn_offset"]
    if offset != "auto" and not isinstance(offset, (int, float)):
        raise ValueError("radiometry.dn_offset must be a number or 'auto'")
    if c["training"]["effective_batch_size"] != c["split"]["sampling"]["batch_size"]:
        raise ValueError("training.effective_batch_size and split.sampling.batch_size disagree")
    m = c["model"]
    if m["decoder"] != "UNetDecoder":
        raise ValueError("Only UNetDecoder is implemented")
    if len(m["out_indices"]) != 4 or len(m["decoder_channels"]) != 4:
        raise ValueError("Four encoder readouts and four decoder blocks are required")
    if m.get("final_upsample", "bilinear") not in ("bilinear", "nearest"):
        raise ValueError("model.final_upsample must be 'bilinear' or 'nearest'")
    codes = sorted(int(k) for k in c["class_map"])
    if codes != list(range(1, len(codes) + 1)):
        raise ValueError("class_map keys must be 1..K (label value 0 = no label)")
