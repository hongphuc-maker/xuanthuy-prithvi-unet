import numpy as np
import pytest

from prithvi_unet.data.export import offset_from_baseline, pick_one_per_date, snapped_grid
from prithvi_unet.data.radiometry import dn_to_reflectance, resolve_dn_offset


def test_fixed_offset_is_used_as_given():
    assert resolve_dn_offset(-1000, {}, "2019-05-01") == -1000.0
    assert resolve_dn_offset(0, {"boa_add_offset": "-1000"}, "2026-05-01") == 0.0


def test_auto_offset_prefers_tags_then_baseline_then_date():
    assert resolve_dn_offset("auto", {"boa_add_offset": "-1000.0"}, "2019-01-01") == -1000.0
    assert resolve_dn_offset("auto", {"s2_processing_baseline": "05.11"}, "2019-01-01") == -1000.0
    assert resolve_dn_offset("auto", {"s2_processing_baseline": "03.01"}, "2021-06-01") == 0.0
    with pytest.warns(UserWarning):
        assert resolve_dn_offset("auto", {}, "2021-06-01") == 0.0
    with pytest.warns(UserWarning):
        assert resolve_dn_offset("auto", {}, "2022-01-25") == -1000.0


def test_one_offset_per_date():
    dn = np.full((2, 6, 1, 1), 1200, np.uint16)  # 2 dates
    rho = dn_to_reflectance(dn, np.array([-1000.0, 0.0], np.float32)[:, None, None, None])
    assert np.allclose(rho[0], 0.02) and np.allclose(rho[1], 0.12)


def test_export_helpers():
    assert offset_from_baseline("05.11") == -1000.0 and offset_from_baseline("02.14") == 0.0
    assert offset_from_baseline(None) is None
    assert snapped_grid((650503.0, 2231807.0, 668707.0, 2245549.0), 10) == (650500.0, 2245550.0, 1821, 1375)
    picked = pick_one_per_date([{"date": "2026-10-01", "cloud": 5}, {"date": "2026-10-01", "cloud": 1},
                                {"date": "2026-09-21", "cloud": 2}])
    assert [(p["date"], p["cloud"]) for p in picked] == [("2026-09-21", 2), ("2026-10-01", 1)]
