"""Export Sentinel-2 L2A images for any area and period from Microsoft Planetary Computer.

    pip install -e ".[export]"
    python scripts/export_sentinel2.py --aoi my_area.geojson --start 2026-10-01 --end 2026-10-31 \
        --out data/my_area --dry-run              # list matching scenes only
    python scripts/export_sentinel2.py --aoi my_area.geojson --start 2026-10-01 --end 2026-10-31 --out data/my_area

One GeoTIFF per date: DN as delivered (uint16, nodata 0), bands named B02..B12, on a grid
snapped to the resolution in the AOI's projected CRS (or --crs). 20 m bands are resampled
bilinearly to 10 m, as for the 2026 Xuan Thuy images. Tags: acquisition_date, item_id,
s2_processing_baseline, boa_add_offset, eo_cloud_cover, aoi_coverage_percent.

Scene selection: scene cloud cover < --max-cloud and footprint covering >= --min-coverage % of
the AOI; one scene per date (least cloudy). Scene cloud cover is not a per-pixel cloud mask.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from prithvi_unet.config import SENTINEL2_BANDS
from prithvi_unet.data.export import offset_from_baseline, pick_one_per_date, snapped_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aoi", required=True, help="vector file of the area (GeoJSON, shapefile, ...)")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-cloud", type=float, default=10.0)
    parser.add_argument("--min-coverage", type=float, default=99.9)
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--crs", default=None, help="target CRS, e.g. EPSG:32648 (default: AOI CRS or its UTM zone)")
    parser.add_argument("--bands", nargs="+", default=SENTINEL2_BANDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import geopandas as gpd
    import planetary_computer
    import rasterio
    from pystac_client import Client
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union

    aoi = gpd.read_file(args.aoi)
    if aoi.crs is None:
        raise SystemExit("The AOI file has no CRS")
    target_crs = args.crs or (aoi.crs if aoi.crs.is_projected else aoi.estimate_utm_crs())
    area = unary_union(aoi.to_crs(target_crs).geometry)
    search_geom = mapping(unary_union(aoi.to_crs("EPSG:4326").geometry))

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    items = list(catalog.search(collections=["sentinel-2-l2a"], intersects=search_geom,
                                datetime=f"{args.start}T00:00:00Z/{args.end}T23:59:59Z").items())
    candidates = []
    for item in items:
        footprint = gpd.GeoSeries([shape(item.geometry)], crs="EPSG:4326").to_crs(target_crs).iloc[0]
        coverage = 100 * footprint.intersection(area).area / area.area
        cloud = item.properties.get("eo:cloud_cover", 999)
        if cloud < args.max_cloud and coverage >= args.min_coverage:
            candidates.append({"item": item, "date": item.datetime.strftime("%Y-%m-%d"), "cloud": cloud,
                               "coverage": coverage, "baseline": item.properties.get("s2:processing_baseline")})
    selected = pick_one_per_date(candidates)
    print(f"{len(items)} scenes found, {len(selected)} selected (cloud < {args.max_cloud} %, coverage >= {args.min_coverage} %)")
    for c in selected:
        print(f"  {c['date']}  {c['item'].id}  cloud {c['cloud']:.2f} %  coverage {c['coverage']:.1f} %  baseline {c['baseline']}")
    if args.dry_run or not selected:
        return

    left, top, width, height = snapped_grid(area.bounds, args.resolution)
    transform = from_origin(left, top, args.resolution, args.resolution)
    inside = geometry_mask([mapping(area)], out_shape=(height, width), transform=transform, invert=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in selected:
        item = c["item"]
        stack = np.zeros((len(args.bands), height, width), np.uint16)
        for b, band in enumerate(args.bands):
            with rasterio.open(item.assets[band].href) as src, WarpedVRT(
                src, crs=target_crs, transform=transform, width=width, height=height,
                resampling=Resampling.bilinear, src_nodata=0, nodata=0,
            ) as vrt:
                array = vrt.read(1)
            array[~inside] = 0
            stack[b] = array
        offset = offset_from_baseline(c["baseline"])
        path = out_dir / f"Sentinel2_{c['date']}_{item.id}_{len(args.bands)}bands_{int(args.resolution)}m.tif"
        profile = {"driver": "GTiff", "height": height, "width": width, "count": len(args.bands), "dtype": "uint16",
                   "crs": target_crs, "transform": transform, "nodata": 0, "compress": "deflate", "tiled": True}
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(stack)
            for i, band in enumerate(args.bands, start=1):
                dst.set_band_description(i, band)
            dst.update_tags(acquisition_date=c["date"], source="Microsoft Planetary Computer sentinel-2-l2a",
                            item_id=item.id, s2_processing_baseline=str(c["baseline"] or ""),
                            boa_add_offset="" if offset is None else str(offset),
                            eo_cloud_cover=f"{c['cloud']:.4f}", aoi_coverage_percent=f"{c['coverage']:.2f}")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
