"""Add the metadata the pipeline needs to a GeoTIFF from another source (in place).

    python scripts/set_band_metadata.py image.tif --bands B02 B03 B04 B8A B11 B12 --date 2026-10-01 \
        [--boa-add-offset -1000] [--processing-baseline 05.11]

* ``--bands``: one name per band, in file order (the reader selects bands by these names).
* ``--date``: acquisition date, used for the date embedding and to order multi-date input.
* ``--boa-add-offset``: offset already present in the DN (-1000 for raw L2A from baseline 04.00,
  0 for data before 25 Jan 2022 or data already harmonised, e.g. Earth Engine S2_SR_HARMONIZED).
"""

from __future__ import annotations

import argparse
from datetime import date

import rasterio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--bands", nargs="+", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--boa-add-offset", type=float, default=None)
    parser.add_argument("--processing-baseline", default=None)
    args = parser.parse_args()
    date.fromisoformat(args.date)  # validates YYYY-MM-DD

    with rasterio.open(args.image, "r+") as dst:
        if len(args.bands) != dst.count:
            raise SystemExit(f"{args.image} has {dst.count} bands; got {len(args.bands)} names")
        for i, name in enumerate(args.bands, start=1):
            dst.set_band_description(i, name)
        tags = {"acquisition_date": args.date}
        if args.boa_add_offset is not None:
            tags["boa_add_offset"] = str(args.boa_add_offset)
        if args.processing_baseline:
            tags["s2_processing_baseline"] = args.processing_baseline
        dst.update_tags(**tags)
        if dst.nodata is None:
            dst.nodata = 0
    print(f"updated {args.image}: bands {args.bands}, tags {tags}")


if __name__ == "__main__":
    main()
