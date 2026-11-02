"""scripts/utils/surrogate_metrics.py on analytic fields with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import load_script

sm = load_script("scripts/utils/surrogate_metrics.py")

SPACING = (1.0, 2.0, 3.0)  # dz != dy != dx


def grid(shape: tuple[int, int, int] = (5, 6, 7)) -> tuple[np.ndarray, ...]:
    """Physical (z, y, x) cell coordinates on SPACING."""
    index = np.meshgrid(*(np.arange(n, dtype=float) for n in shape), indexing="ij")
    return tuple(i * h for i, h in zip(index, SPACING))


def test_interior_cells() -> None:
    fluid = np.ones((5, 5, 5), dtype=bool)
    fluid[2, 2, 3] = False
    interior = sm.interior_cells(fluid)
    expected = np.zeros_like(fluid)
    expected[1:-1, 1:-1, 1:-1] = True
    # The solid cell and its neighbours ((2, 2, 4) is on the edge anyway).
    for cell in ((2, 2, 3), (2, 2, 2), (1, 2, 3), (3, 2, 3), (2, 1, 3), (2, 3, 3)):
        expected[cell] = False
    np.testing.assert_array_equal(interior, expected)


def test_vorticity_and_divergence_are_exact_on_linear_fields() -> None:
    z, y, x = grid()
    interior = sm.interior_cells(np.ones(z.shape, dtype=bool))
    # Solid-body rotation about z, u = -y, v = x: ω = (0, 0, 2), div = 0.
    rotation = np.stack([-y, x, 0 * z])[None]
    omega = sm.vorticity(rotation, SPACING)[0][:, interior]
    np.testing.assert_allclose(omega, [[0.0], [0.0], [2.0]] * np.ones_like(omega))
    np.testing.assert_allclose(sm.divergence_rms(rotation, interior, SPACING), [0.0])
    # Shear u = 3 z, v = 5 x, w = 7 y: ω = (7, 3, 5).
    shear = np.stack([3 * z, 5 * x, 7 * y])[None]
    omega = sm.vorticity(shear, SPACING)[0][:, interior]
    np.testing.assert_allclose(omega.mean(axis=1), [7.0, 3.0, 5.0])
    np.testing.assert_allclose(omega.std(axis=1), 0.0, atol=1e-12)
    # u = x, v = 2 y, w = -z: div = 1 + 2 - 1 = 2.
    linear = np.stack([x, 2 * y, -z])[None]
    np.testing.assert_allclose(sm.divergence_rms(linear, interior, SPACING), [2.0])
    # The vorticity error of the shear against the rotation: (7, 3, 3).
    pooled, per = sm.vorticity_rmse(shear, rotation, interior, SPACING)
    np.testing.assert_allclose(per, [[7.0, 3.0, 3.0]])
    np.testing.assert_allclose(pooled, [np.sqrt((49 + 9 + 9) / 3)])


def test_rmse_and_acc() -> None:
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(4, 3, 50))
    mean = truth.mean(axis=0)
    pooled, per = sm.rmse(truth + np.array([1.0, 2.0, 2.0])[:, None], truth)
    np.testing.assert_allclose(per, [[1.0, 2.0, 2.0]] * 4)
    np.testing.assert_allclose(pooled, np.sqrt(3.0))
    for sign in (1, -1):
        pooled, per = sm.acc(mean + sign * (truth - mean), truth, mean)
        np.testing.assert_allclose(pooled, sign)
        np.testing.assert_allclose(per, sign)


def test_tke() -> None:
    # u = ±1 alternating in time, v = w = 0: TKE 0.5 about the own mean, even
    # with an offset; a steady field has none.
    values = np.zeros((4, 3, 10))
    values[:, 0] = np.array([1.0, -1.0, 1.0, -1.0])[:, None] + 5.0
    np.testing.assert_allclose(sm.tke(values), 0.5)
    np.testing.assert_allclose(sm.tke_field(values[..., None, None]), 0.5)
    np.testing.assert_allclose(sm.tke(np.ones((4, 3, 10))), 0.0)


def test_spanwise_spectrum_recovers_a_sine() -> None:
    ny, dy, amplitude, m = 16, 2.0, 3.0, 3
    y = np.arange(ny)
    lines = (
        amplitude * np.sin(2 * np.pi * m * y / ny)[None, :, None] * np.ones((2, 1, 4))
    )
    k, energy = sm.spanwise_spectrum(lines, dy)
    # Angular wavenumbers 2 pi / wavelength.
    np.testing.assert_allclose(k, 2 * np.pi * np.arange(1, ny // 2 + 1) / (ny * dy))
    expected = np.zeros(ny // 2)
    expected[m - 1] = amplitude**2 / 2 / k[0]  # sum(E) dk = variance
    np.testing.assert_allclose(energy, [expected] * 2, atol=1e-12)


def test_integral_length_of_a_cosine_correlation() -> None:
    # A sine of the domain's wavelength has R(m) = cos(2 pi m / ny); with ny =
    # 10 the first R <= 0 is m = 3. Trapezoid rule: minus R(0) / 2. A spanwise
    # mean per line and frame (here 5 and -2) is removed first.
    ny, dy = 10, 2.0
    lines = np.sin(2 * np.pi * np.arange(ny) / ny + np.arange(3)[:, None])[None]
    lines = np.moveaxis(lines, 1, 2) + np.array([5.0, -2.0])[:, None, None]
    expected = dy * (sum(np.cos(2 * np.pi * m / ny) for m in range(3)) - 0.5)
    np.testing.assert_allclose(sm.integral_length(lines, dy), [expected] * 2)


def test_integral_time_scale() -> None:
    dt = 2.5
    # Trapezoid rule: T = dt (sum_{tau < tau0} R - 1/2).
    alternating = np.array([1.0, -1.0] * 4)  # R(1) = -7/8: T = dt / 2
    pairs = np.array([1.0, 1.0, -1.0, -1.0] * 2)  # R(1) = 1/8, R(2) = -6/8
    scales = sm.integral_time_scale(np.stack([alternating, pairs, 0 * pairs]), dt)
    np.testing.assert_allclose(scales, [dt / 2, dt * (1 / 8 + 1 / 2), np.nan])


def test_reference_scales_of_a_box() -> None:
    fluid = np.ones((4, 3, 5), dtype=bool)
    fluid[:2, :, 1] = False  # 3 columns of 2 cells
    fluid[:1, :, 3] = False  # 3 columns of 1 cell
    dz = SPACING[0]
    series = np.array([1.0, 1.0, -1.0, -1.0] * 2)
    truth = np.zeros((8, 3, *fluid.shape))
    truth[:, 0] = 4.0 + series[:, None, None, None]
    truth[:, 1:] = series[:, None, None, None, None]
    ref = sm.reference_scales(truth, fluid, SPACING, dt=2.0, chunk=7)
    assert ref["k_c"] == 2 and ref["z_c"] == 2.5 * dz
    assert ref["H"] == pytest.approx(1.5 * dz)
    assert ref["u_ref"] == pytest.approx(4.0)
    assert ref["T"] == pytest.approx(2.0 * (1 / 8 + 1 / 2))
    with pytest.raises(ValueError, match="canopy"):
        sm.canopy_level(np.zeros_like(fluid))
    with pytest.raises(ValueError, match="no solid cell"):
        sm.mean_building_height(np.ones_like(fluid), dz)


def test_acc_horizon() -> None:
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])  # start frame, then 4 leads
    threshold = 0.5
    # The time of the first lead below the threshold, not the last one above.
    assert sm.acc_horizon(np.array([0.9, 0.6, 0.4, 0.7]), times, threshold) == (
        1.5,
        False,
    )
    assert sm.acc_horizon(np.array([0.3, 0.9, 0.9, 0.9]), times, threshold) == (
        0.5,
        False,
    )
    # NaN (a diverged rollout) counts as below.
    assert sm.acc_horizon(np.array([0.9, np.nan, np.nan, np.nan]), times, 0.5) == (
        1.0,
        False,
    )
    # Never below: censored at the window's last lead time.
    assert sm.acc_horizon(np.array([0.9, 0.8, 0.7, 0.5]), times, threshold) == (
        2.0,
        True,
    )


def test_bin_spectrum() -> None:
    kh = np.array([1.0, 2.0, 3.0, 4.0])
    spectrum = np.array([[10.0, 20.0, 30.0, 40.0]])
    edges = np.array([0.5, 1.2, 1.5, 3.5, 8.0, 16.0])
    centre = np.sqrt(1.2 * 1.5)  # empty bin: interpolated in log k
    weight = np.log(centre / 1.0) / np.log(2.0)
    np.testing.assert_allclose(
        sm.bin_spectrum(kh, spectrum, edges),
        [[10.0, 10 + 10 * weight, 25.0, 40.0, np.nan]],
    )


def test_density() -> None:
    # 1 of 5 values falls out: the density still integrates to 4/5.
    values = np.array([0.1, 0.2, 0.6, 1.5, 9.0])
    np.testing.assert_allclose(
        sm.density(values, np.array([0.0, 0.5, 2.0])), [2 / 5 / 0.5, 2 / 5 / 1.5]
    )


def test_window_rms() -> None:
    assert sm.window_rms(np.array([3.0, 4.0])) == pytest.approx(np.sqrt(12.5))
    assert np.isnan(sm.window_rms(np.array([1.0, np.nan])))
