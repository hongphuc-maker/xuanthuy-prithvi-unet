"""Descriptive statistics of the 6 bands: DN, reflectance and normalised values.

    python scripts/band_statistics.py data/Sentinel2_XuanThuy_2026_6bands_10m/*.tif --out band_stats.json

Checks the radiometry assumption: with the -1000 offset, water pixels in B11/B12
must sit near DN 1000 (reflectance ~ 0). Without it they would read ~10 %.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio

from prithvi_unet.config import SENTINEL2_BANDS
from prithvi_unet.data.radiometry import PRITHVI_V2_MEAN, PRITHVI_V2_STD

PERCENTILES = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--offset", type=float, default=-1000.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pooled = [[] for _ in SENTINEL2_BANDS]
    per_image = []
    for path in sorted(args.images):
        with rasterio.open(path) as src:
            dn = src.read().astype(np.float64)
            day = src.tags().get("acquisition_date", path)
        valid = (dn > 0).all(0)  # nodata = 0
        row = {"date": day, "valid_pixels": int(valid.sum())}
        for b, name in enumerate(SENTINEL2_BANDS):
            v = dn[b][valid]
            pooled[b].append(v)
            row[name] = {"p1": float(np.percentile(v, 1)), "median": float(np.median(v)),
                         "below_1000_fraction": float((v < 1000).mean())}
        per_image.append(row)

    summary = {}
    print(f"{'band':5} {'p1':>6} {'median':>7} {'p99':>6} {'<1000':>7} {'rho_mean':>8} {'z_mean':>7} {'z_std':>6}")
    for b, name in enumerate(SENTINEL2_BANDS):
        v = np.concatenate(pooled[b])
        rho = (v + args.offset) / 10000.0
        z = (v + args.offset - PRITHVI_V2_MEAN[b]) / PRITHVI_V2_STD[b]
        q = np.percentile(v, PERCENTILES)
        summary[name] = {"n": int(v.size), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
                         "std": float(v.std()), "percentiles": dict(zip(map(str, PERCENTILES), q.tolist())),
                         "below_1000_fraction": float((v < 1000).mean()), "reflectance_mean": float(rho.mean()),
                         "z_mean": float(z.mean()), "z_std": float(z.std())}
        s = summary[name]
        print(f"{name:5} {q[1]:6.0f} {q[4]:7.0f} {q[7]:6.0f} {100 * s['below_1000_fraction']:6.2f}% "
              f"{s['reflectance_mean']:8.4f} {s['z_mean']:7.2f} {s['z_std']:6.2f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"pooled": summary, "per_image": per_image}, f, indent=2)


if __name__ == "__main__":
    main()
