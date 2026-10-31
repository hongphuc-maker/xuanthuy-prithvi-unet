"""Agreement of class maps with verified reference points (OA and per-class F1).

    python scripts/score_points.py --points data/xuanthuy/verified_points_shapefiles.zip \
        --maps maps/p3/2026-10-01_classes.tif maps/p2/2026-10-01_classes.tif
    python scripts/score_points.py --points data/thaithuy/verified_points_shapefiles.zip --maps tt_map.tif --out scores/

--config only supplies the class names (class_map); any config of the same legend works.
Prints one line per map; with --out also writes <map name>_summary.json, _per_class.csv, _points.csv.
The points are of unconfirmed date: report the result as agreement, and do not use it to choose checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prithvi_unet.config import load_config
from prithvi_unet.inference.points import score_map_on_points
from prithvi_unet.utils import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", required=True, help="folder or ZIP with one point shapefile per class")
    parser.add_argument("--maps", nargs="+", required=True, help="class-map GeoTIFFs (codes 1..K, 0 = no data)")
    parser.add_argument("--config", default="configs/p3_single_date.json", help="source of class_map")
    parser.add_argument("--out", default=None, help="folder for summary/per-class/points files")
    args = parser.parse_args()
    class_map = load_config(args.config)["class_map"]
    for map_path in args.maps:
        summary, table, points = score_map_on_points(map_path, args.points, class_map)
        print(f"{Path(map_path).name}: OA {summary['OA']:.3f} ({summary['correct_points']}/{summary['usable_points']}), "
              f"macro-F1 over {len(summary['represented_classes'])} classes {summary['macro_F1_represented']:.3f}")
        print(table[table.support > 0].round(3).to_string(index=False))
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            stem = Path(map_path).stem
            atomic_json(summary, out / f"{stem}_summary.json")
            table.to_csv(out / f"{stem}_per_class.csv", index=False)
            points.to_csv(out / f"{stem}_points.csv", index=False)
    if args.out:
        print("written to", args.out)


if __name__ == "__main__":
    main()
