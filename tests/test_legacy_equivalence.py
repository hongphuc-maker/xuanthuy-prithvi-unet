"""The explicit PrithviUNet must compute the same function as the completed runs.

Both models are built with random weights; the legacy state dict is converted and
loaded into the new model; both run on the same input in eval mode.
"""

from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("terratorch")

from prithvi_unet.config import load_config  # noqa: E402
from prithvi_unet.model import build_model, convert_legacy_state_dict  # noqa: E402
from tests.legacy.terratorch_model import build_legacy_model, legacy_logits  # noqa: E402

from .conftest import CONFIGS  # noqa: E402


def _inputs(frames: int, batch: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    image = torch.randn(batch, 6, frames, 224, 224, generator=g)
    temporal = torch.tensor([[2026.0, 9.0 + 30 * t] for t in range(frames)]).expand(batch, -1, -1).clone()
    location = torch.tensor([[20.25, 106.55]]).expand(batch, -1).clone()
    return image, temporal, location


@pytest.mark.parametrize("config_name", ["p3_single_date.json", "p2_multitemporal.json"])
def test_same_logits_as_legacy(config_name):
    config = load_config(CONFIGS / config_name)
    torch.manual_seed(1)
    legacy = build_legacy_model(config).eval()
    # Make norm/scale parameters non-trivial so a wrong mapping cannot hide.
    with torch.no_grad():
        for name, p in legacy.named_parameters():
            if name.endswith(".scale") or ".1.weight" in name or "norm" in name:
                p.add_(0.05 * torch.randn_like(p))
    new = build_model(copy.deepcopy(config), pretrained=False).eval()
    new.load_state_dict(convert_legacy_state_dict(legacy.state_dict()), strict=True)

    frames = len(config["active_dates"])
    image, temporal, location = _inputs(frames)
    with torch.inference_mode():
        expected = legacy_logits(legacy, image, temporal, location)
        actual = new(image, temporal, location)
    assert actual.shape == (1, 11, 224, 224)
    max_diff = (actual - expected).abs().max().item()
    assert max_diff < 1e-4, f"max |new - legacy| = {max_diff}"


def test_parameter_counts_match_legacy():
    config = load_config(CONFIGS / "p3_single_date.json")
    legacy = build_legacy_model(config)
    new = build_model(copy.deepcopy(config), pretrained=False)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(new.encoder) == count(legacy.encoder) == 303_886_338
    assert count(new.pyramid) == count(legacy.neck) == 4_720_896
    assert count(new.decoder) == count(legacy.decoder) == 15_597_312
    assert count(new.head) == count(legacy.head) == 715
