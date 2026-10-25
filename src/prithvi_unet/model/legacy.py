"""Convert checkpoints of the completed runs (TerraTorch EncoderDecoderFactory model)
to the explicit ``PrithviUNet`` layout, so trained P2/P3 weights can be reused.

Old key (xuan-thuy-prithvi-segmentation, reference/wce_completed)  ->  new key
    encoder.blocks.N.inner.X               ->  encoder.vit.blocks.N.X   (gradient-checkpoint wrapper removed)
    encoder.X                              ->  encoder.vit.X
    neck.2.fpn1.{0,1,3}.X                  ->  pyramid.to_stride4.{0,1,3}.X
    neck.2.fpn2.0.X                        ->  pyramid.to_stride8.X
    decoder.decoder.blocks.N.convK.{0,1}.X ->  decoder.blocks.N.convK.{0,1}.X
    head.head.2.X                          ->  head.classifier.X
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

_RULES: list[tuple[str, str]] = [
    (r"^encoder\.blocks\.(\d+)\.inner\.", r"encoder.vit.blocks.\1."),
    (r"^encoder\.", r"encoder.vit."),
    (r"^neck\.2\.fpn1\.", r"pyramid.to_stride4."),
    (r"^neck\.2\.fpn2\.0\.", r"pyramid.to_stride8."),
    (r"^decoder\.decoder\.blocks\.", r"decoder.blocks."),
    (r"^head\.head\.2\.", r"head.classifier."),
]


def convert_legacy_state_dict(old: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new: dict[str, torch.Tensor] = {}
    for key, value in old.items():
        for pattern, replacement in _RULES:
            if re.match(pattern, key):
                new_key = re.sub(pattern, replacement, key, count=1)
                break
        else:
            raise KeyError(f"No conversion rule for legacy key {key!r}")
        if new_key.startswith("encoder.vit.encoder.vit."):
            raise AssertionError(new_key)
        new[new_key] = value
    return new


def load_legacy_checkpoint(path: str | Path, model: torch.nn.Module) -> dict[str, Any]:
    """Load ``best_any.pt`` / ``best_accepted.pt`` / ``latest_resume.pt`` from a completed run."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(convert_legacy_state_dict(state), strict=True)
    return {k: v for k, v in payload.items() if k != "model"} if isinstance(payload, dict) else {}
