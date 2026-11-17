"""The online adaptive observation error (data_assimilation.adaptive_observation_error).

A linear-Gaussian toy, cycled like the parameter-only smoother over windows:
the true observation error is k times the configured one (k per height group),
and the adaptive option must recover k, bring the Desroziers ratio back to ~1
and the posterior back to calibrated within a few windows. With observation
statistics every statistic row is a group of its own.
"""

import logging
import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from data_assimilation.adaptive_observation_error import (
    AdaptiveObservationError,
    observation_heights,
    observation_rows,
)
from data_assimilation.filtering.analysis import stochastic_enkf_update

HEIGHTS = np.repeat([2.0, 8.0, 20.0], 12)  # three groups of 12 observations
STD = 0.1  # configured std
TRUE_FACTOR = np.repeat([2.0, 4.0, 6.0], 12)  # true std / configured std
N_PARAMS, N_E, N_STEPS, N_WINDOWS = 3, 64, 4, 6


def _cycle(
    adaptive: AdaptiveObservationError | None,
    seed: int,
    num_windows: int = N_WINDOWS,
    heights: np.ndarray = HEIGHTS,
    true_factor: np.ndarray = TRUE_FACTOR,
    rows: np.ndarray | None = None,
) -> dict:
    """One toy run: per window the ratio of the Desroziers std to the std used,
    and the posterior z-scores of the true parameters."""
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    H = 0.5 * rng.standard_normal((heights.size, N_PARAMS))
    truth = rng.standard_normal(N_PARAMS)
    members = rng.standard_normal((N_PARAMS, N_E))
    configured = np.full(heights.size, STD**2)
    ratio, z = [], []
    for _ in range(num_windows):
        y = H @ truth + STD * true_factor * rng.standard_normal(heights.size)
        used = (
            configured
            if adaptive is None
            else adaptive.variance(configured, heights, rows)
        )
        forecast = H @ members
        for _ in range(N_STEPS):
            key, sub = jax.random.split(key)
            members = np.asarray(
                stochastic_enkf_update(
                    jnp.asarray(members),
                    jnp.asarray(H @ members),
                    jnp.asarray(y),
                    jnp.asarray(used),
                    sub,
                    alpha=float(N_STEPS),
                )
            )
        analysis = H @ members
        if adaptive is not None:
            adaptive.record(y, forecast, analysis, configured, heights, rows)
        cross = (y - forecast.mean(1)) * (y - analysis.mean(1))
        ratio.append(np.sqrt(np.mean(cross / used)))
        z.append((members.mean(1) - truth) / members.std(1, ddof=1))
    return {"ratio": np.array(ratio), "z": np.array(z)}


def _trials(initial_inflation: float | None) -> dict:
    """12 toy runs; `None` keeps the configured error throughout."""
    runs = [
        _cycle(
            (
                None
                if initial_inflation is None
                else AdaptiveObservationError(initial_inflation=initial_inflation)
            ),
            seed,
        )
        for seed in range(12)
    ]
    return {key: np.stack([r[key] for r in runs]) for key in runs[0]}


@pytest.fixture(scope="module")  # type: ignore[misc]
def toy() -> dict:
    return {
        "fixed": _trials(None),
        # Window 0 with the configured (too small) error, or conservatively
        # inflated beyond the true error.
        "adaptive": _trials(1.0),
        "conservative": _trials(6.0),
    }


def test_adaptive_error_recovers_a_desroziers_ratio_of_one(toy: dict) -> None:
    # The configured error is 2-6x too small: the ratio says so, every window.
    assert toy["fixed"]["ratio"][:, 3:].mean() > 3.0
    # Adapted from window 1 on, so by windows 3-5 it is ~1 whatever window 0 used.
    for arm in ("adaptive", "conservative"):
        assert 0.85 < toy[arm]["ratio"][:, 3:].mean() < 1.15, arm


def test_adaptive_error_calibrates_the_posterior(toy: dict) -> None:
    def rms(z: np.ndarray) -> float:
        return float(np.sqrt(np.mean(z[:, 3:] ** 2)))

    # Fixed R: overconfident (true parameters many posterior stds away).
    assert rms(toy["fixed"]["z"]) > 2.0
    # A window 0 with the too-small error has already collapsed the ensemble,
    # which a parameter-only smoother never re-inflates: the right R later
    # comes too late. A conservative window 0 keeps the spread for it.
    assert rms(toy["adaptive"]["z"]) > 2.0
    assert 0.6 < rms(toy["conservative"]["z"]) < 1.5


@pytest.mark.parametrize("initial_inflation", [1.0, 6.0])  # type: ignore[misc]
def test_inflation_recovers_the_true_factors(initial_inflation: float) -> None:
    # Window 0's ratios (assimilated with a wrong R) set window 1 only, so a
    # far-off initial inflation does not bias the later estimate.
    final = []
    for seed in range(6):
        adaptive = AdaptiveObservationError(initial_inflation=initial_inflation)
        _cycle(adaptive, seed, num_windows=8)
        final.append(adaptive.inflation(adaptive.history[-1]["groups"]))
    np.testing.assert_allclose(np.mean(final, axis=0), [2.0, 4.0, 6.0], rtol=0.15)


def test_inflation_pools_and_bounds() -> None:
    heights = np.array([2.0, 2.0, 20.0, 20.0])
    configured = np.ones(4)
    adaptive = AdaptiveObservationError(initial_inflation=2.0, max_inflation=5.0)
    # Window 0: the initial inflation.
    np.testing.assert_allclose(adaptive.variance(configured, heights), 4.0)
    # d_a d_f / R = 9 at z=2 (x3) and 100 at z=20 (x10, capped at 5).
    obs = np.zeros(4)
    forecast = np.array([[3.0], [3.0], [10.0], [10.0]])
    adaptive.record(obs, forecast, forecast, configured, heights)
    np.testing.assert_allclose(
        adaptive.variance(configured, heights), [9.0, 9.0, 25.0, 25.0]
    )
    assert adaptive.history[0]["window_estimate"] == pytest.approx([3.0, 10.0])
    assert adaptive.history[1]["groups"] == ["z=2", "z=20"]

    # From window 2 on, window 0 is left out: windows 1 and 2 pool to (1+4)/2.
    small = np.array([[1.0], [1.0], [1.0], [1.0]])
    adaptive.record(obs, small, small, configured, heights)
    np.testing.assert_allclose(adaptive.variance(configured, heights)[0], 1.0)
    adaptive.record(obs, 2 * small, 2 * small, configured, heights)
    np.testing.assert_allclose(adaptive.variance(configured, heights)[0], 2.5)

    # memory=1 keeps only the last window.
    a = AdaptiveObservationError(memory=1)
    for pred in (forecast, small, 2 * small):
        a.record(obs, pred, pred, configured, heights)
    assert a.variance(configured, heights)[0] == pytest.approx(4.0)

    # Floors at the configured error: a negative sample cannot deflate it.
    a = AdaptiveObservationError()
    a.record(obs, forecast, -forecast, configured, heights)
    np.testing.assert_allclose(a.variance(configured, heights), 1.0)


def test_non_finite_predictions_are_dropped(caplog: pytest.LogCaptureFixture) -> None:
    heights = np.array([2.0, 2.0, 20.0, 20.0])
    configured = np.ones(4)
    adaptive = AdaptiveObservationError(initial_inflation=2.0)
    adaptive.variance(configured, heights)
    # One NaN at z=2, only NaNs at z=20.
    forecast = np.array([[3.0], [np.nan], [np.nan], [np.inf]])
    with caplog.at_level(logging.WARNING):
        adaptive.record(np.zeros(4), forecast, forecast, configured, heights)
    assert "dropped 3 of 4" in caplog.text
    assert adaptive.history[0]["window_estimate"] == [3.0, None]
    # z=2 from its finite ratio; z=20 keeps its previous inflation.
    np.testing.assert_allclose(
        adaptive.variance(configured, heights), [9.0, 9.0, 4.0, 4.0]
    )
    assert np.all(np.isfinite(adaptive.history[-1]["inflation"]))


def test_a_band_without_sensors_keeps_its_inflation() -> None:
    adaptive = AdaptiveObservationError(
        height_bands=[{"min_z": 0.0, "max_z": 10.0}, {"min_z": 10.0, "max_z": 40.0}],
        initial_inflation=2.0,
    )
    heights, configured = np.array([2.0, 8.0]), np.ones(2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no RuntimeWarning from an empty mean
        adaptive.variance(configured, heights)
        forecast = np.array([[3.0], [3.0]])
        adaptive.record(np.zeros(2), forecast, forecast, configured, heights)
        np.testing.assert_allclose(adaptive.variance(configured, heights), 9.0)
    assert adaptive.history[0]["std_used"] == [2.0, None]
    assert adaptive.history[0]["window_estimate"] == [3.0, None]
    assert adaptive.history[1]["inflation"] == [3.0, 2.0]


def test_height_bands_group_and_must_cover_every_sensor() -> None:
    adaptive = AdaptiveObservationError(
        height_bands=[{"min_z": 0.0, "max_z": 10.0}, {"min_z": 10.0, "max_z": 40.0}]
    )
    ids, names = adaptive.groups(np.array([2.0, 8.0, 20.0]))
    assert ids.tolist() == [0, 0, 1] and names == ["z=[0,10)", "z=[10,40)"]
    with pytest.raises(ValueError, match="no adaptive height band"):
        adaptive.groups(np.array([50.0]))
    with pytest.raises(ValueError, match="must not overlap"):
        AdaptiveObservationError(
            height_bands=[{"min_z": 10.0, "max_z": 40.0}, {"min_z": 0.0, "max_z": 12.0}]
        )
    with pytest.raises(ValueError, match="initial_inflation"):
        AdaptiveObservationError(initial_inflation=0.5)


def _statistics_layout(num_sensors: int, sensor_heights: np.ndarray) -> SimpleNamespace:
    """A resolved statistics error laid out by hand (sensor innermost): two
    components; bins (3 rows), tke (1 row), spectral (2 bands) of
    ``2 * num_sensors`` entries per row, and a ``stress`` row of one entry per
    sensor (one component pair), fewer than ``2 * num_sensors``."""
    rows = {"bins": 3, "tke": 1, "spectral": 2, "stress": 1}
    per_row = {"bins": 2, "tke": 2, "spectral": 2, "stress": 1}
    slices, start = {}, 0
    for block, count in rows.items():
        size = count * per_row[block] * num_sensors
        slices[block] = slice(start, start + size)
        start += size
    return SimpleNamespace(
        covariance_diag=np.full(start, STD**2), block_slices=slices, block_rows=rows
    )


def test_statistic_rows_are_grouped_apart() -> None:
    operator = SimpleNamespace(obs_z=np.array([2.0, 2.0, 20.0]))
    resolved = _statistics_layout(3, operator.obs_z)
    heights = observation_heights(resolved, operator)
    rows = observation_rows(resolved)
    assert rows is not None and heights.size == rows.size == 39
    # Entry j is sensor j % 3 in every block, the stress row included.
    np.testing.assert_array_equal(heights, np.tile([2.0, 2.0, 20.0], 13))
    assert (
        rows.tolist()
        == ([""] * 18 + ["tke"] * 6 + ["spectral[0]"] * 6 + ["spectral[1]"] * 6)
        + ["stress"] * 3
    )
    ids, names = AdaptiveObservationError().groups(heights, rows)
    # The bins keep the height groups and names of a run without statistics.
    assert names[:2] == AdaptiveObservationError().groups(heights)[1]
    assert names == [
        "z=2",
        "z=20",
        "tke z=2",
        "tke z=20",
        "spectral[0] z=2",
        "spectral[0] z=20",
        "spectral[1] z=2",
        "spectral[1] z=20",
        "stress z=2",
        "stress z=20",
    ]
    np.testing.assert_array_equal(ids[:18], np.tile([0, 0, 1], 6))
    np.testing.assert_array_equal(ids[18:24], np.tile([2, 2, 3], 2))
    np.testing.assert_array_equal(ids[24:30], [4, 4, 5] * 2)
    np.testing.assert_array_equal(ids[30:36], [6, 6, 7] * 2)
    np.testing.assert_array_equal(ids[-3:], [8, 8, 9])
    # Height bands split the statistic rows the same way.
    banded = AdaptiveObservationError(
        height_bands=[{"min_z": 0.0, "max_z": 10.0}, {"min_z": 10.0, "max_z": 40.0}]
    )
    assert banded.groups(heights, rows)[1][-2:] == [
        "stress z=[0,10)",
        "stress z=[10,40)",
    ]
    # Without statistics there are no rows: the plain height groups.
    plain = SimpleNamespace(sensor_indices=(0, 1, 2), times=np.zeros(2))
    assert observation_rows(plain) is None
    np.testing.assert_array_equal(
        observation_heights(plain, operator), np.tile([2.0, 2.0, 20.0], 2)
    )


def test_inflation_recovers_the_factors_per_statistic_row() -> None:
    """The bins are consistent; tke, the two spectral bands and the stress row
    carry an error 3, 1.5, 4 and 2 times the configured one, at both
    heights. Each group must recover its own factor."""
    operator = SimpleNamespace(obs_z=np.repeat([2.0, 20.0], 6))
    resolved = _statistics_layout(12, operator.obs_z)
    heights = observation_heights(resolved, operator)
    rows = observation_rows(resolved)
    assert rows is not None
    factors = {"": 1.0, "tke": 3.0, "spectral[0]": 1.5, "spectral[1]": 4.0}
    factors["stress"] = 2.0
    true_factor = np.array([factors[r] for r in rows])
    final = []
    for seed in range(6):
        adaptive = AdaptiveObservationError(initial_inflation=1.0)
        _cycle(adaptive, seed, 8, heights, true_factor, rows)
        names = adaptive.history[-1]["groups"]
        final.append(adaptive.inflation(names))
    estimate = dict(zip(names, np.mean(final, axis=0)))
    for name, value in estimate.items():
        row = name.rpartition(" ")[0]
        if row:
            assert value == pytest.approx(factors[row], rel=0.15), name
        else:  # the bins: consistent, floored at 1
            assert 1.0 <= value < 1.15, name
