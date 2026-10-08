"""Field statistics, canopy profiles and spanwise spectra against analytic answers."""

import numpy as np
import pytest
from evaluation.turbulence import (
    band_energy_ratio,
    field_rmse,
    field_statistics,
    fluid_mask,
    intrinsic_profile,
    spanwise_spectra,
)

from .test_evaluation_state_rmse import _write_boxes


def test_field_statistics_known_moments():
    """TKE of known variances and u'w' of a known covariance, per cell."""
    rng = np.random.default_rng(0)
    n_time, shape = 20000, (2, 2, 2)
    a, b, c = (rng.standard_normal((n_time, *shape)) for _ in range(3))
    u = 5.0 + 2.0 * a  # var 4
    v = 3.0 * b  # var 9
    w = -0.5 * a + np.sqrt(0.75) * c  # var 1, cov(u, w) = 2 * -0.5 = -1
    stats = field_statistics(u, v, w)

    assert set(stats) == {"u", "v", "w", "tke", "uw"}
    np.testing.assert_allclose(stats["u"], 5.0, atol=0.05)
    np.testing.assert_allclose(stats["tke"], 0.5 * (4 + 9 + 1), rtol=0.03)
    np.testing.assert_allclose(stats["uw"], -1.0, atol=0.05)


def test_field_statistics_exact_on_two_frames():
    """With ddof=1 two frames give the sample (co)variance exactly."""
    u = np.array([1.0, 3.0])[:, None, None, None]
    w = np.array([2.0, -2.0])[:, None, None, None]
    stats = field_statistics(u, np.zeros_like(u), w)
    assert stats["uw"].item() == pytest.approx(-4.0)
    assert stats["tke"].item() == pytest.approx(0.5 * (2.0 + 8.0))


def test_fluid_mask_dilates_the_solid_by_one_cell(tmp_path):
    """A one-cell box removes itself and its six face neighbours."""
    stl = _write_boxes(tmp_path / "box.stl", [((2, 3), (2, 3), (0, 1))])
    centres = np.arange(6) + 0.5
    fluid = fluid_mask(stl, centres[:4], centres, centres)
    assert (~fluid).sum() == 1 + 5  # no neighbour below the ground level
    assert not fluid[0, 2, 2] and not fluid[1, 2, 2] and not fluid[0, 2, 3]
    assert fluid[1, 3, 3] and fluid[2, 2, 2]


def test_intrinsic_profile_averages_fluid_cells_only():
    field = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    fluid = np.ones_like(field, dtype=bool)
    fluid[0, :, 0] = False  # level 0: solid values 0 and 3 excluded
    fluid[1] = False
    profile = intrinsic_profile(field, fluid)
    assert profile[0] == pytest.approx(np.mean([1, 2, 4, 5]))
    assert np.isnan(profile[1])


def test_field_rmse_over_fluid_cells_and_per_level():
    truth = np.zeros((2, 2, 2))
    prediction = np.full_like(truth, 1.0)
    prediction[1] = 3.0
    prediction[1, 0, 0] = 100.0  # masked out
    fluid = np.ones_like(truth, dtype=bool)
    fluid[1, 0, 0] = False
    rmse, profile = field_rmse(prediction, truth, fluid)
    np.testing.assert_allclose(profile, [1.0, 3.0])
    assert rmse == pytest.approx(np.sqrt((4 * 1 + 3 * 9) / 7))


def test_spanwise_spectra_known_modes_and_masked_block():
    """Energy lands at the wavenumber of each group's mode; lines through the
    block, upstream of it and at the outflow are never used."""
    n_time, n_z, n_y, n_x, dy = 3, 6, 16, 10, 2.0
    y = np.arange(n_y) * dy
    length = n_y * dy
    fluid = np.ones((n_z, n_y, n_x), dtype=bool)
    fluid[:2, 4:8, 3:5] = False  # a building at x 3-4, two levels high
    field = np.zeros((n_time, n_z, n_y, n_x))
    # In canopy: mode 2 with amplitude 1 on every line.
    field[:, :2] = np.cos(2 * np.pi * 2 * y / length)[None, None, :, None]
    # Above canopy: mode 5 with amplitude 2.
    field[:, 2:] = 2 * np.cos(2 * np.pi * 5 * y / length)[None, None, :, None]
    # Poison lines that must be excluded: through the block, upstream of the
    # first building, at the outflow and in the top two levels.
    field[:, :2, :, 3] += 1e3 * np.sin(2 * np.pi * 7 * y / length)[None, :]
    field[:, :, :, :3] += 1e3 * np.sin(2 * np.pi * 7 * y / length)[None, None, :, None]
    field[:, :, :, -2:] += 1e3 * np.sin(2 * np.pi * 7 * y / length)[None, None, :, None]
    field[:, -2:] += 1e3 * np.sin(2 * np.pi * 7 * y / length)[None, :, None]

    k, spectra = spanwise_spectra(field, fluid, dy)
    dk = 1.0 / length
    np.testing.assert_allclose(k, np.arange(1, n_y // 2 + 1) * dk)
    for group, mode, amplitude in (("in_canopy", 2, 1.0), ("above_canopy", 5, 2.0)):
        expected = np.zeros(k.size)
        expected[mode - 1] = amplitude**2 / 2 / dk  # variance / dk
        np.testing.assert_allclose(spectra[group], expected, atol=1e-9)


def test_spanwise_spectra_group_without_lines_is_none():
    fluid = np.ones((5, 8, 6), dtype=bool)
    fluid[:2, 3, :] = False  # a wall along x: no fully fluid canopy line
    _, spectra = spanwise_spectra(np.zeros((1, 5, 8, 6)), fluid, 1.0)
    assert spectra["in_canopy"] is None
    assert spectra["above_canopy"] is not None


def test_band_energy_ratio():
    k = np.arange(1, 21) / 80.0  # 80 m period, dy = 2 m: lambda/dy = 40/n
    truth = k ** (-5.0 / 3.0)
    same = band_energy_ratio(k, truth, truth, 2.0)
    assert set(same) == {"large", "mid", "near_cutoff"}
    for value in same.values():
        assert value == pytest.approx(0.0)
    doubled = truth.copy()
    doubled[10:] *= 2.0  # n = 11..20: lambda in [2, 4) dy
    ratio = band_energy_ratio(k, np.stack([truth, doubled]), truth, 2.0)
    np.testing.assert_allclose(ratio["near_cutoff"], [0.0, 10 * np.log10(2.0)])
    np.testing.assert_allclose(ratio["mid"], [0.0, 0.0])
