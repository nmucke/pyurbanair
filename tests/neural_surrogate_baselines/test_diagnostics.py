"""The comparison diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from neural_surrogate_baselines import diagnostics


def test_persistence_repeats_the_start_frame() -> None:
    truth = np.arange(5.0).reshape(5, 1, 1, 1, 1) * np.ones((5, 2, 2, 3, 3))
    pred = diagnostics.persistence(truth, start=1)
    assert np.array_equal(pred[:2], truth[:2])
    assert np.all(pred[2:] == truth[1])


def test_masked_errors_ignore_solid_cells() -> None:
    truth = np.zeros((2, 3, 2, 2, 2))
    pred = truth.copy()
    fluid = np.ones((2, 2, 2), dtype=bool)
    fluid[0, 0, 0] = False
    pred[:, :, 0, 0, 0] = 100.0  # junk inside a building
    pred[1, 0, 1, 1, 1] = 3.0
    rmse, mae = diagnostics.masked_errors(pred, truth, fluid)
    n = 3 * fluid.sum()
    assert rmse.tolist() == pytest.approx([0.0, np.sqrt(9.0 / n)])
    assert mae.tolist() == pytest.approx([0.0, 3.0 / n])


@pytest.mark.parametrize("ny", [8, 7])  # type: ignore[misc]
def test_lateral_spectrum_sums_to_the_variance(ny: int) -> None:
    rng = np.random.default_rng(0)
    field = rng.standard_normal((4, ny, 5))
    fluid = np.ones((ny, 5), dtype=bool)
    k, energy = diagnostics.lateral_spectrum(field, fluid, dy=2.0)
    assert energy.sum() == pytest.approx((field**2).mean())
    assert k[1] == pytest.approx(2 * np.pi / (ny * 2.0))


def test_lateral_spectrum_finds_a_wave() -> None:
    y = np.arange(16)
    field = np.cos(2 * np.pi * 3 * y / 16)[None, :, None] * np.ones((2, 16, 4))
    _, energy = diagnostics.lateral_spectrum(field, np.ones((16, 4), dtype=bool))
    assert int(np.argmax(energy)) == 3
    assert energy[3] == pytest.approx(0.5)


def test_profiles() -> None:
    rng = np.random.default_rng(0)
    fields = rng.standard_normal((50, 3, 2, 4, 4))
    fields[:, 2] = fields[:, 0]  # w = u: -<u'w'> = -<u'^2>
    fluid = np.ones((2, 4, 4), dtype=bool)
    prof = diagnostics.profiles(fields, fluid)
    prime = diagnostics.fluctuations(fields, fluid)
    np.testing.assert_allclose(prof["mean_u"], fields[:, 0].mean(axis=(0, 2, 3)))
    np.testing.assert_allclose(prof["uw"], -(prime[:, 0] ** 2).mean(axis=(0, 2, 3)))
    np.testing.assert_allclose(
        prof["tke"], 0.5 * (prime**2).sum(axis=1).mean(axis=(0, 2, 3))
    )


def test_spatial_std_detects_smoothing() -> None:
    rng = np.random.default_rng(0)
    truth = rng.standard_normal((3, 3, 2, 4, 4))
    fluid = np.ones((2, 4, 4), dtype=bool)
    ratio = diagnostics.spatial_std(0.5 * truth, fluid) / diagnostics.spatial_std(
        truth, fluid
    )
    np.testing.assert_allclose(ratio, 0.5)
