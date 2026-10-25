import numpy as np

from prithvi_unet.data.radiometry import PRITHVI_V2_MEAN, PRITHVI_V2_STD, dn_to_model_input, dn_to_reflectance

RADIOMETRY = {"dn_offset": -1000.0, "scale": 10000.0}
NORMALIZATION = {"mean": list(PRITHVI_V2_MEAN), "std": list(PRITHVI_V2_STD)}


def test_two_step_formula_equals_completed_runs():
    dn = np.random.default_rng(0).integers(500, 12000, size=(2, 6, 8, 8)).astype(np.uint16)
    new = dn_to_model_input(dn, RADIOMETRY, NORMALIZATION)
    mean = np.asarray(PRITHVI_V2_MEAN, np.float32)[:, None, None]
    std = np.asarray(PRITHVI_V2_STD, np.float32)[:, None, None]
    old = ((dn.astype(np.float32) - 1000.0) * (10000.0 / 10000.0) - mean) / std  # old data.py crop()
    np.testing.assert_allclose(new, old, rtol=0, atol=2e-6)


def test_water_swir_example():
    dn = np.full((6, 1, 1), 1022, np.uint16)
    rho = dn_to_reflectance(dn)
    assert abs(float(rho[5, 0, 0]) - 0.0022) < 1e-6  # B12 water: reflectance ~ 0
    z = dn_to_model_input(dn, RADIOMETRY, NORMALIZATION)
    assert abs(float(z[5, 0, 0]) - (22 - 1363) / 1049) < 1e-5  # -1.28


def test_dividing_by_10000_only_is_not_the_same_input():
    dn = np.full((6, 1, 1), 2500, np.uint16)
    z = dn_to_model_input(dn, RADIOMETRY, NORMALIZATION)
    divided = dn.astype(np.float32) / 10000.0
    assert np.abs(z - divided).max() > 0.3
