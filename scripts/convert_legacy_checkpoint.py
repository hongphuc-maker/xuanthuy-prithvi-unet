"""Convert a checkpoint of the old repository to the new layout.

    python scripts/convert_legacy_checkpoint.py old_run/best_any.pt converted/p3_best_any.pt \
        --config configs/p3_single_date.json
"""

from __future__ import annotations

import argparse

import torch

from prithvi_unet.config import load_config
from prithvi_unet.model import build_model, convert_legacy_state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    state = convert_legacy_state_dict(payload["model"])
    model = build_model(load_config(args.config), pretrained=False)
    model.load_state_dict(state, strict=True)  # proves every key maps
    torch.save({"format": "prithvi_unet/v1", "model": state, "summary": payload.get("summary"),
                "converted_from": args.source}, args.target)
    print(f"converted {len(state)} tensors -> {args.target}")


if __name__ == "__main__":
    main()
