"""Map new dates with a trained checkpoint (no labels needed).

P3 (one date):
    python scripts/predict.py --config configs/p3_single_date.json --checkpoint runs/p3/best_any.pt \
        --images data/Sentinel2_2026-10-01_*.tif --out maps/
P2 (six dates, oldest first is not required; images are sorted by their acquisition_date tag):
    python scripts/predict.py --config configs/p2_multitemporal.json --checkpoint runs/p2/best_any.pt \
        --images a.tif b.tif c.tif d.tif e.tif f.tif --out maps/

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
    args = parser.parse_args()
    print(json.dumps(predict_new_dates(args.config, args.checkpoint, args.images, args.out, args.stride), indent=2))


if __name__ == "__main__":
    main()
