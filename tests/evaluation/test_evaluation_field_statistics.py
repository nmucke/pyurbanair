"""Field statistics, canopy profiles and spanwise spectra against analytic answers."""

import numpy as np
import pytest
import xarray
from evaluation.turbulence import (
    band_energy_ratio,
    field_statistics,
    fluid_mask,
    intrinsic_profile,
    member_field_reductions,
    spanwise_spectra,
    statistic_rmse,
)

from .test_evaluation_solid_mask import _write_boxes


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


def test_statistic_rmse_over_fluid_cells_and_per_level():
    truth = np.zeros((2, 2, 2))
    prediction = np.full_like(truth, 1.0)
    prediction[1] = 3.0
    prediction[1, 0, 0] = 100.0  # masked out
    fluid = np.ones_like(truth, dtype=bool)
    fluid[1, 0, 0] = False
    rmse, profile = statistic_rmse(prediction, truth, fluid)
    np.testing.assert_allclose(profile, [1.0, 3.0])
    assert rmse == pytest.approx(np.sqrt((4 * 1 + 3 * 9) / 7))


def test_spanwise_spectra_known_modes_and_masked_block():
    """Energy lands at the wavenumber of each group's fluctuating mode; a
    steady mode carries none; lines through the block, upstream of it and at
    the outflow are never used."""
    n_time, n_z, n_y, n_x, dy = 2, 6, 16, 10, 2.0
    y = np.arange(n_y) * dy
    length = n_y * dy
    sign = np.array([1.0, -1.0])[:, None, None, None]  # zero time mean
    fluid = np.ones((n_z, n_y, n_x), dtype=bool)
    fluid[:2, 4:8, 3:5] = False  # a building at x 3-4, two levels high
    field = np.zeros((n_time, n_z, n_y, n_x))
    # In canopy: mode 2 with amplitude 1 on every line.
    field[:, :2] = sign * np.cos(2 * np.pi * 2 * y / length)[None, None, :, None]
    # Above canopy: mode 5 with amplitude 2.
    field[:, 2:] = 2 * sign * np.cos(2 * np.pi * 5 * y / length)[None, None, :, None]
    # A steady mode everywhere: the time mean, so no energy.
    field += 5 * np.cos(2 * np.pi * 3 * y / length)[None, None, :, None]
    # Poison lines that must be excluded: through the block, upstream of the
    # first building, at the outflow and in the top two levels.
    poison = 1e3 * sign * np.sin(2 * np.pi * 7 * y / length)[None, None, :, None]
    field[:, :2, :, 3:4] += poison[:, 0]
    field[:, :, :, :3] += poison
    field[:, :, :, -2:] += poison
    field[:, -2:] += poison

    k, spectra = spanwise_spectra(field, fluid, dy)
    dk = 1.0 / length
    np.testing.assert_allclose(k, np.arange(1, n_y // 2 + 1) * dk)
    for group, mode, amplitude in (("in_canopy", 2, 1.0), ("above_canopy", 5, 2.0)):
        expected = np.zeros(k.size)
        expected[mode - 1] = amplitude**2 / 2 / dk  # variance / dk
        np.testing.assert_allclose(spectra[group], expected, atol=1e-9)


def test_spanwise_spectra_ignore_a_solid_ground_plane():
    """A fully solid level (uDALES' ground plane) does not move the x-range
    upstream: it still starts at the first building."""
    n_y, dy = 8, 1.0
    y = np.arange(n_y) * dy
    fluid = np.ones((5, n_y, 8), dtype=bool)
    fluid[0] = False  # the ground plane
    fluid[1, 2:5, 3] = False  # a building at x 3
    field = np.zeros((2, 5, n_y, 8))
    # Fluctuating energy only upstream of the building.
    field[:, :, :, :3] = (
        np.array([1.0, -1.0])[:, None, None, None]
        * np.cos(2 * np.pi * 2 * y / (n_y * dy))[None, None, :, None]
    )
    _, spectra = spanwise_spectra(field, fluid, dy)
    for group in ("in_canopy", "above_canopy"):
        np.testing.assert_allclose(spectra[group], 0.0, atol=1e-12)


def test_member_spectra_on_udales_grid_skip_the_ground_and_upstream(tmp_path):
    """uDALES' w sits on zm, whose zm = 0 level is under the STL's ground
    plane from x = 0 on: w's spectra still start at the first building, as
    u's and v's do. A member stored in another dim order reads the same."""
    stl = _write_boxes(
        tmp_path / "case.stl",
        [((0, 20), (0, 16), (-0.5, 0)), ((10, 12), (4, 8), (0, 3))],
    )
    xm, ym, zm = (
        np.arange(-10.0, 20.0, 2),
        np.arange(0.0, 16.0, 2),
        np.arange(0.0, 12.0, 2),
    )
    coords = {"xm": xm, "xt": xm + 1, "ym": ym, "yt": ym + 1, "zm": zm, "zt": zm + 1}
    sign = np.array([1.0, -1.0])[:, None, None, None]
    w = np.zeros((2, zm.size, ym.size, xm.size))
    # Fluctuations upstream of the building only (xt < 10).
    w[..., (xm + 1) < 10] = (
        sign * np.cos(2 * np.pi * 2 * (ym + 1) / 16)[None, None, :, None]
    )
    zeros = np.zeros_like(w)
    member = xarray.Dataset(
        {
            "u": (("time", "zt", "yt", "xm"), zeros),
            "v": (("time", "zt", "ym", "xt"), zeros),
            "w": (("time", "zm", "yt", "xt"), w),
        },
        coords={"time": [1.0, 2.0], **coords},
    )
    spectra = member_field_reductions(member, "udales", stl).spectrum
    np.testing.assert_allclose(spectra.sel(component="w"), 0.0, atol=1e-12)

    shuffled = member.assign(w=member.w.transpose("xt", "time", "yt", "zm"))
    np.testing.assert_allclose(
        member_field_reductions(shuffled, "udales", stl).spectrum, spectra
    )


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
