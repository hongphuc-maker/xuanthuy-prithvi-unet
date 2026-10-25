"""Map new dates with a trained checkpoint (no labels needed).

P3 (one date):
    python scripts/predict.py --config configs/p3_single_date.json --checkpoint runs/p3/best_any.pt \
        --images data/Sentinel2_2026-10-01_*.tif --out maps/
P2 (six dates, oldest first is not required; images are sorted by their acquisition_date tag):
    python scripts/predict.py --config configs/p2_multitemporal.json --checkpoint runs/p2/best_any.pt \
        --images a.tif b.tif c.tif d.tif e.tif f.tif --out maps/

Images of another area or year: same command. Images from before 25 Jan 2022 have no L2A offset;
use --dn-offset auto (reads the export tags) or --dn-offset 0.
Compare the final resize without retraining: add --upsample nearest and a different --out.
Legacy checkpoints from the old repository (best_any.pt, best_accepted.pt) are converted on the fly.
"""

from __future__ import annotations

import argparse
import json

from prithvi_unet.inference import predict_new_dates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=None, help="sliding-window stride (default: config, 112)")
    parser.add_argument("--upsample", choices=["bilinear", "nearest"], default=None,
                        help="final x4 resize of class scores (default: as trained, bilinear)")
    parser.add_argument("--dn-offset", default=None,
                        help="override radiometry.dn_offset: a number (e.g. -1000, 0) or 'auto' (per image)")
    args = parser.parse_args()
    offset = args.dn_offset if args.dn_offset in (None, "auto") else float(args.dn_offset)
    result = predict_new_dates(args.config, args.checkpoint, args.images, args.out, args.stride, args.upsample, offset)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
