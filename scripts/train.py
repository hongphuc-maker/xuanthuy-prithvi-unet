"""Train P2 or P3 and write the map of the target date.

    python scripts/train.py --config configs/p3_single_date.json --data-root /content/data --run-dir /content/runs/p3
    python scripts/train.py ... --steps 2          # short check on CPU/GPU, resumable
    python scripts/train.py ... --map-only         # map from the selected checkpoint
"""

from __future__ import annotations

import argparse
import json

from prithvi_unet.training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True, help="folder with the GeoTIFFs and label raster named in the config")
    parser.add_argument("--run-dir", required=True, help="new folder per experiment (checkpoints, logs, map)")
    parser.add_argument("--steps", type=int, default=None, help="stop after N optimiser steps (resumable)")
    parser.add_argument("--map-only", action="store_true")
    parser.add_argument("--no-checksums", action="store_true", help="skip SHA-256 checks of the input files")
    parser.add_argument("--prithvi-weights", default=None,
                        help="local Prithvi_EO_V2_300M_TL.pt (default: $PRITHVI_WEIGHTS or Hugging Face download)")
    args = parser.parse_args()

    trainer = Trainer(args.config, args.data_root, args.run_dir, verify_checksums=not args.no_checksums,
                      prithvi_weights=args.prithvi_weights)
    if not args.map_only:
        print(json.dumps(trainer.train(stop_after_steps=args.steps), indent=2))
    if args.map_only or (trainer.run / "TRAINING_COMPLETE.json").exists():
        print(json.dumps(trainer.map(), indent=2))


if __name__ == "__main__":
    main()
