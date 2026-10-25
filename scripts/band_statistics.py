"""Descriptive statistics of the bands: DN, reflectance and normalised values.

    python scripts/band_statistics.py data/*.tif --out band_stats.json
    python scripts/band_statistics.py data/*.tif --offset auto     # per-image offset from tags/date

Checks the radiometry assumption: with the -1000 offset, water pixels in B11/B12
sit near DN 1000 (reflectance ~ 0). Without an offset (data before 25 Jan 2022)
they sit near DN 0. Run this on every new area or period before predicting.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio

from prithvi_unet.config import SENTINEL2_BANDS
from prithvi_unet.data.radiometry import PRITHVI_V2_MEAN_BY_BAND, PRITHVI_V2_STD_BY_BAND, resolve_dn_offset

PERCENTILES = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--offset", default="-1000", help="a number, or 'auto' (per image)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    setting = args.offset if args.offset == "auto" else float(args.offset)

    pooled: dict[str, list[np.ndarray]] = {b: [] for b in SENTINEL2_BANDS}
    per_image = []
    for path in sorted(args.images):
        with rasterio.open(path) as src:
            names, tags = list(src.descriptions), src.tags()
            dn = src.read().astype(np.float64)
        day = tags.get("acquisition_date", "unknown")
        offset = resolve_dn_offset(setting, tags, day) if day != "unknown" else float(setting if setting != "auto" else -1000)
        valid = (dn > 0).all(0)  # nodata = 0
        row = {"file": path, "date": day, "dn_offset": offset, "valid_pixels": int(valid.sum())}
        for b in SENTINEL2_BANDS:
            if b not in names:
                continue
            v = dn[names.index(b)][valid]
            pooled[b].append(v + offset)  # offset-corrected DN = reflectance x 10000
            row[b] = {"p1": float(np.percentile(v, 1)), "median": float(np.median(v)),
                      "below_1000_fraction": float((v < 1000).mean())}
        per_image.append(row)
        print(f"{day}  offset {offset:+.0f}  valid {row['valid_pixels']:,}")

    summary = {}
    print(f"\n{'band':5} {'p1':>6} {'median':>7} {'p99':>6} {'rho_mean':>8} {'z_mean':>7} {'z_std':>6}  (reflectance x 10000)")
    for b, parts in pooled.items():
        if not parts:
            continue
        v = np.concatenate(parts)
        z = (v - PRITHVI_V2_MEAN_BY_BAND[b]) / PRITHVI_V2_STD_BY_BAND[b]
        q = np.percentile(v, PERCENTILES)
        summary[b] = {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
                      "percentiles": dict(zip(map(str, PERCENTILES), q.tolist())),
                      "reflectance_mean": float(v.mean() / 10000), "z_mean": float(z.mean()), "z_std": float(z.std())}
        s = summary[b]
        print(f"{b:5} {q[1]:6.0f} {q[4]:7.0f} {q[7]:6.0f} {s['reflectance_mean']:8.4f} {s['z_mean']:7.2f} {s['z_std']:6.2f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"pooled_offset_corrected": summary, "per_image": per_image}, f, indent=2)


if __name__ == "__main__":
    main()
