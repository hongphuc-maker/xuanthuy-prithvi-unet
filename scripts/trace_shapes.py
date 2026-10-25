"""Print every stage of the model with its output shape and learned-parameter count.

    python scripts/trace_shapes.py --config configs/p3_single_date.json
Runs one random input through the model (random weights, no download needed).
"""

from __future__ import annotations

import argparse

import torch

from prithvi_unet.config import load_config
from prithvi_unet.model import build_model, parameter_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p3_single_date.json")
    args = parser.parse_args()
    config = load_config(args.config)
    frames = len(config["active_dates"])
    model = build_model(config, pretrained=False).eval()

    image = torch.randn(1, 6, frames, 224, 224)
    temporal = torch.tensor([[[2026.0, 264.0]] * frames])
    location = torch.tensor([[20.25, 106.55]])
    with torch.inference_mode():
        tokens = model.encoder.embed(image, temporal, location)
        trace = model.forward_trace(image, temporal, location)

    print(f"input image                    {tuple(image.shape)}")
    print(f"tokens after embedding + CLS   {tuple(tokens.shape)}")
    for i, r in zip(model.encoder.out_indices, trace["readouts"]):
        print(f"readout block {i:<2}               {tuple(r.shape)}")
    for i, f in zip(model.encoder.out_indices, trace["feature_maps"]):
        print(f"feature map (block {i:<2})         {tuple(f.shape)}")
    for name, p in zip(["stride 4", "stride 8", "stride 16", "stride 32"], trace["pyramid"]):
        print(f"pyramid {name:<10}             {tuple(p.shape)}")
    print(f"decoder output                 {tuple(trace['decoded'].shape)}")
    print(f"head (logits at 56 x 56)       {tuple(trace['logits_low'].shape)}")
    print(f"logits after bilinear x4       {tuple(trace['logits'].shape)}")
    print()
    total = 0
    for row in parameter_table(model):
        total += row["params"]
        print(f"{row['stage']:<42} {row['params']:>12,}")
    print(f"{'total learned parameters':<42} {total:>12,}")


if __name__ == "__main__":
    main()
