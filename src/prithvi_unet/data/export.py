"""Helpers for scripts/export_sentinel2.py (pure functions, unit-tested).

The export reproduces the notebook used for the 2026 Xuan Thuy images
(old repo ``notebooks/00_sentinel2_export.ipynb``, cells 2-13):
https://github.com/hongphuc-maker/xuan-thuy-prithvi-segmentation/blob/42290823beeb42e44ec87bb8845e439e10c18735/notebooks/00_sentinel2_export.ipynb
and adds the tags the pipeline reads: band descriptions, acquisition_date,
s2_processing_baseline and boa_add_offset.
"""

from __future__ import annotations

import math


def offset_from_baseline(processing_baseline: str | None) -> float | None:
    """BOA_ADD_OFFSET implied by the L2A processing baseline: -1000 from 04.00, 0 before."""
    if not processing_baseline:
        return None
    return -1000.0 if float(processing_baseline) >= 4.0 else 0.0


def snapped_grid(bounds: tuple[float, float, float, float], resolution: float) -> tuple[float, float, int, int]:
    """Bounds snapped outward to multiples of the resolution -> (left, top, width, height)."""
    minx, miny, maxx, maxy = bounds
    left = math.floor(minx / resolution) * resolution
    bottom = math.floor(miny / resolution) * resolution
    right = math.ceil(maxx / resolution) * resolution
    top = math.ceil(maxy / resolution) * resolution
    return left, top, int(round((right - left) / resolution)), int(round((top - bottom) / resolution))


def pick_one_per_date(candidates: list[dict]) -> list[dict]:
    """Keep the least cloudy item for each date (several tiles/orbits can cover the area the same day)."""
    best: dict[str, dict] = {}
    for c in candidates:
        if c["date"] not in best or c["cloud"] < best[c["date"]]["cloud"]:
            best[c["date"]] = c
    return [best[d] for d in sorted(best)]
