"""Band reduction: configs, reading bands by name, pretrained Conv3d channels kept by name."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from prithvi_unet.config import load_config
from prithvi_unet.data.radiometry import prithvi_normalization
from prithvi_unet.data.scene import read_sentinel2

from .conftest import CONFIGS


def test_band_configs_load_with_matching_statistics():
    for name, bands in [("p3_single_date_4bands.json", ["B02", "B03", "B04", "B8A"]),
                        ("p3_single_date_3bands.json", ["B02", "B03", "B04"])]:
        c = load_config(CONFIGS / name)
        assert c["bands"] == bands
        assert c["normalization"]["mean"] == prithvi_normalization(bands)["mean"]
        assert c["images"] == load_config(CONFIGS / "p3_single_date.json")["images"]  # same 6-band files


def test_reader_selects_bands_by_name_and_keeps_six_band_validity(tmp_path):
    path = tmp_path / "img.tif"
    data = np.arange(6 * 4 * 5, dtype=np.uint16).reshape(6, 4, 5) + 1
    data[5, 0, 0] = 0  # nodata only in B12
    with rasterio.open(path, "w", driver="GTiff", height=4, width=5, count=6, dtype="uint16", nodata=0,
                       crs="EPSG:32648", transform=from_origin(0, 40, 10, 10)) as dst:
        dst.write(data)
        for i, b in enumerate(["B02", "B03", "B04", "B8A", "B11", "B12"], start=1):
            dst.set_band_description(i, b)
        dst.update_tags(acquisition_date="2026-09-21")
    dn, valid, _, _ = read_sentinel2(path, ["B04", "B02"], "2026-09-21")
    np.testing.assert_array_equal(dn, data[[2, 0]])
    assert not valid[0, 0] and valid.sum() == 19  # same pixels as a 6-band run


terratorch = pytest.importorskip("terratorch")


@pytest.mark.parametrize("bands", [["B02", "B03", "B04", "B8A"], ["B02", "B04", "B11"]])
def test_pretrained_conv3d_channels_are_selected_by_band_name(bands, tmp_path):
    from prithvi_unet.model.encoder import PrithviEncoder

    torch.manual_seed(0)
    full = PrithviEncoder(num_frames=1)  # stands in for the pretrained 6-band checkpoint
    fake_mae = {"encoder." + k: v for k, v in full.vit.state_dict().items()}
    fake_mae["decoder.decoder_embed.weight"] = torch.zeros(1)  # MAE decoder keys are dropped by the filter
    path = tmp_path / "fake_prithvi.pt"
    torch.save(fake_mae, path)

    subset = PrithviEncoder(num_frames=1, bands=bands)
    subset.load_pretrained(revision="unused", local_path=path)
    six = ["B02", "B03", "B04", "B8A", "B11", "B12"]
    expected = full.vit.patch_embed.proj.weight[:, [six.index(b) for b in bands]]
    assert torch.equal(subset.vit.patch_embed.proj.weight, expected)
    assert torch.equal(subset.vit.blocks[3].attn.qkv.weight, full.vit.blocks[3].attn.qkv.weight)
    x = torch.randn(1, len(bands), 1, 224, 224)
    out = subset(x, torch.tensor([[[2026.0, 264.0]]]), torch.tensor([[20.25, 106.55]]))
    assert [tuple(o.shape) for o in out] == [(1, 197, 1024)] * 4


def test_final_upsample_switch_changes_only_the_last_resize():
    from prithvi_unet.model import build_model

    config = load_config(CONFIGS / "p3_single_date.json")
    torch.manual_seed(0)
    model = build_model(config, pretrained=False).eval()
    x = torch.randn(1, 6, 1, 224, 224)
    t, loc = torch.tensor([[[2026.0, 264.0]]]), torch.tensor([[20.25, 106.55]])
    with torch.inference_mode():
        bilinear = model.forward_trace(x, t, loc)
        model.final_upsample = "nearest"
        nearest = model.forward_trace(x, t, loc)
    assert torch.equal(bilinear["logits_low"], nearest["logits_low"])
    repeated = nearest["logits_low"].repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
    assert torch.equal(nearest["logits"], repeated)
    changed = (bilinear["logits"].argmax(1) != nearest["logits"].argmax(1)).float().mean().item()
    assert 0.0 <= changed < 0.5
