from prithvi_unet.config import SENTINEL2_BANDS, load_config
from prithvi_unet.data.radiometry import PRITHVI_V2_MEAN, PRITHVI_V2_STD

from .conftest import CONFIGS


def test_p2_and_p3_differ_only_in_dates():
    p2 = load_config(CONFIGS / "p2_multitemporal.json")
    p3 = load_config(CONFIGS / "p3_single_date.json")
    assert len(p2["active_dates"]) == 6 and p3["active_dates"] == ["2026-09-21"]
    for key in ("training", "split", "acceptance", "radiometry", "normalization", "bands", "class_map", "images", "label"):
        assert p2[key] == p3[key], key
    without_note = lambda m: {k: v for k, v in m.items() if k != "temporal_readout"}  # noqa: E731
    assert without_note(p2["model"]) == without_note(p3["model"])


def test_radiometry_and_normalisation_constants():
    c = load_config(CONFIGS / "p3_single_date.json")
    assert c["bands"] == SENTINEL2_BANDS
    assert c["radiometry"]["dn_offset"] == -1000.0 and c["radiometry"]["scale"] == 10000.0
    assert tuple(c["normalization"]["mean"]) == PRITHVI_V2_MEAN
    assert tuple(c["normalization"]["std"]) == PRITHVI_V2_STD
    assert c["model"]["out_indices"] == [5, 11, 17, 23]
    assert c["model"]["decoder_channels"] == [512, 256, 128, 64]
