"""Sentinel-2 L2A digital numbers -> model input. Fixed formula, no learned weights.

Step 1, reflectance (physical unit, 0..1):
    rho = (DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE
    Since processing baseline 04.00 (25 Jan 2022) ESA stores L2A with
    BOA_ADD_OFFSET = -1000; earlier products have no offset. Planetary Computer
    keeps the DN as delivered.
    - ESA, Level-2A products (offset section):
      https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a-algorithms-products
    - Planetary Computer, baseline change and how to harmonise:
      https://nbviewer.org/github/microsoft/PlanetaryComputerExamples/blob/main/datasets/sentinel-2-l2a/baseline-change.ipynb

Step 2, z-score with Prithvi's pretraining statistics (HLS, per band):
    z = (rho - mean / 10000) / (std / 10000)
    The pretrained Conv3d weights expect this scale.
    - Constants: TerraTorch 1.2.12, PRITHVI_V2_MEAN / PRITHVI_V2_STD
      https://github.com/IBM/terratorch/blob/v1.2.12/terratorch/models/backbones/prithvi_vit.py#L30-L31
    - Units (HLS reflectance x 10000, int16, scale 0.0001): HLS User Guide v2, Tables 6-7
      https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf

Both steps together equal the shorter form used in the completed runs:
    z = (DN - 1000 - mean) / std
"""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np

# Prithvi statistics by Sentinel-2 band name (reflectance x 10000). Same values and order as
# PRITHVI_V2_MEAN / PRITHVI_V2_STD (HLS bands BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2).
PRITHVI_V2_MEAN_BY_BAND = {"B02": 1087.0, "B03": 1342.0, "B04": 1433.0, "B8A": 2734.0, "B11": 1958.0, "B12": 1363.0}
PRITHVI_V2_STD_BY_BAND = {"B02": 2248.0, "B03": 2179.0, "B04": 2178.0, "B8A": 1850.0, "B11": 1242.0, "B12": 1049.0}
PRITHVI_V2_MEAN = tuple(PRITHVI_V2_MEAN_BY_BAND.values())
PRITHVI_V2_STD = tuple(PRITHVI_V2_STD_BY_BAND.values())

OFFSET_INTRODUCED = date(2022, 1, 25)  # processing baseline 04.00


def prithvi_normalization(bands: list[str]) -> dict[str, list[float]]:
    """Mean/std for a subset of bands, e.g. ["B02", "B03", "B04", "B8A"]."""
    return {"mean": [PRITHVI_V2_MEAN_BY_BAND[b] for b in bands], "std": [PRITHVI_V2_STD_BY_BAND[b] for b in bands]}


def resolve_dn_offset(setting: float | str, tags: dict[str, str], acquisition_date: str) -> float:
    """Offset to add to the DN of one image.

    ``setting`` is a number (same offset for every image, e.g. -1000) or "auto":
    1. tag ``boa_add_offset`` written by scripts/export_sentinel2.py, else
    2. tag ``s2_processing_baseline``: >= 04.00 -> -1000, older -> 0, else
    3. the acquisition date: on or after 25 Jan 2022 -> -1000 (warning, check with
       scripts/band_statistics.py: water in B11/B12 must sit near DN 1000 with offset).
    """
    if setting != "auto":
        return float(setting)
    if tags.get("boa_add_offset") not in (None, ""):
        return float(tags["boa_add_offset"])
    baseline = tags.get("s2_processing_baseline")
    if baseline:
        return -1000.0 if float(baseline) >= 4.0 else 0.0
    warnings.warn(f"No offset/baseline tag for {acquisition_date}; offset chosen from the date", stacklevel=2)
    return -1000.0 if date.fromisoformat(acquisition_date) >= OFFSET_INTRODUCED else 0.0


def dn_to_reflectance(dn: np.ndarray, dn_offset: float | np.ndarray = -1000.0, scale: float = 10000.0) -> np.ndarray:
    """Step 1. ``dn`` [..., C, H, W] -> surface reflectance (float32).

    ``dn_offset`` may be an array broadcastable to ``dn`` (one offset per date: shape [T, 1, 1, 1]).
    """
    return (dn.astype(np.float32) + np.asarray(dn_offset, np.float32)) / np.float32(scale)


def standardize(reflectance: np.ndarray, mean_x10000: tuple[float, ...] = PRITHVI_V2_MEAN,
                std_x10000: tuple[float, ...] = PRITHVI_V2_STD) -> np.ndarray:
    """Step 2. Per-band z-score; band axis is -3 ([..., C, H, W])."""
    mean = np.asarray(mean_x10000, np.float32)[:, None, None] / np.float32(10000)
    std = np.asarray(std_x10000, np.float32)[:, None, None] / np.float32(10000)
    return (reflectance - mean) / std


def dn_to_model_input(dn: np.ndarray, radiometry: dict, normalization: dict,
                      dn_offset: float | np.ndarray | None = None) -> np.ndarray:
    """Steps 1 + 2. ``dn_offset`` overrides ``radiometry["dn_offset"]`` (needed when it is "auto")."""
    offset = radiometry["dn_offset"] if dn_offset is None else dn_offset
    rho = dn_to_reflectance(dn, offset, radiometry["scale"])
    return standardize(rho, tuple(normalization["mean"]), tuple(normalization["std"]))
