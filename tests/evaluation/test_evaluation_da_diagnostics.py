"""The DA diagnostics ``compute_metrics.py`` writes: the spread--skill ratio of
:func:`vector_sensor_metrics`' spread, and the innovation χ² and Desroziers
estimate of :func:`observation_fit`, each ≈ its calibrated value on synthetic
data where the answer is known."""

import numpy as np
import xarray
from evaluation.scores import observation_fit, spread_skill, vector_sensor_metrics


def test_spread_skill_is_one_for_a_calibrated_ensemble() -> None:
    # Truth and members drawn from the same distribution: the truth is just
    # another member, so the vector spread and the ensemble-mean error agree
    # once spread_skill corrects for the finite ensemble.
    rng = np.random.default_rng(0)
    n_members, n_time, n_sensors = 20, 200, 10
    coords = {"component": ["u", "v", "w"], "time": np.arange(n_time, dtype=float)}
    truth = xarray.DataArray(
        rng.standard_normal((3, n_time, n_sensors)),
        dims=("component", "time", "sensor"),
        coords=coords,
    )
    members = xarray.DataArray(
        rng.standard_normal((3, n_members, n_time, n_sensors)),
        dims=("component", "ensemble", "time", "sensor"),
        coords=coords,
    )

    vector = vector_sensor_metrics(truth, members)

    assert vector["spread"].shape == (n_time,)
    assert abs(spread_skill(vector["spread"], vector["rmse"], n_members) - 1) < 0.05


def test_spread_is_nan_for_a_single_member() -> None:
    coords = {"component": ["u", "v", "w"], "time": [0.0, 1.0]}
    truth = xarray.DataArray(
        np.zeros((3, 2, 4)), dims=("component", "time", "sensor"), coords=coords
    )
    member = truth.expand_dims(ensemble=1, axis=1)

    assert np.isnan(vector_sensor_metrics(truth, member)["spread"]).all()


def test_acc_scores_the_pattern_of_the_fluctuations() -> None:
    """ACC is 1 for the truth's fluctuations at any amplitude, -1 for their
    opposite and about 0 for unrelated ones."""
    rng = np.random.default_rng(0)
    coords = {"component": ["u", "v", "w"], "time": np.arange(50, dtype=float)}
    truth = xarray.DataArray(
        5.0 + rng.standard_normal((3, 50, 40)),
        dims=("component", "time", "sensor"),
        coords=coords,
    )
    mean = truth.mean("time")

    def acc(members: xarray.DataArray) -> np.ndarray:
        return vector_sensor_metrics(truth, members.expand_dims(ensemble=1, axis=1))[
            "acc"
        ]

    np.testing.assert_allclose(acc(mean + 0.3 * (truth - mean)), 1.0)
    np.testing.assert_allclose(acc(mean - (truth - mean)), -1.0)
    unrelated = truth.copy(data=5.0 + rng.standard_normal(truth.shape))
    assert abs(acc(unrelated).mean()) < 0.05


def test_member_acc_is_below_the_ensemble_means() -> None:
    """Members = the truth's fluctuations plus their own noise: averaging the
    members removes the noise, so the ensemble mean correlates better."""
    rng = np.random.default_rng(0)
    coords = {"component": ["u", "v", "w"], "time": np.arange(50, dtype=float)}
    truth = xarray.DataArray(
        rng.standard_normal((3, 50, 40)),
        dims=("component", "time", "sensor"),
        coords=coords,
    )
    members = truth.expand_dims(ensemble=20, axis=1) + rng.standard_normal(
        (3, 20, 50, 40)
    )
    vector = vector_sensor_metrics(truth, members)
    # Signal and noise of equal variance: corr ~ 1/sqrt(2) per member, ~ 1 for
    # the mean of 20 (noise variance / 20).
    np.testing.assert_allclose(vector["member_acc"].mean(), 1 / np.sqrt(2), atol=0.02)
    np.testing.assert_allclose(vector["acc"].mean(), 1 / np.sqrt(1.05), atol=0.02)
    # One member: the two are the same.
    one = vector_sensor_metrics(truth, members.isel(ensemble=[0]))
    np.testing.assert_allclose(one["acc"], one["member_acc"])


def test_observation_fit_recovers_chi2_and_the_observation_error() -> None:
    # A calibrated forecast (truth and members from the same distribution), the
    # observations with a known error std, and the analysis mean the optimal
    # Kalman gain gives: χ² ≈ 1 and Desroziers returns that std.
    rng = np.random.default_rng(0)
    n_obs, n_members, sigma_b, sigma_o = 20000, 50, 2.0, 0.5
    truth = sigma_b * rng.standard_normal(n_obs)
    obs = truth + sigma_o * rng.standard_normal(n_obs)
    forecast = sigma_b * rng.standard_normal((n_obs, n_members))
    # The forecast mean's error variance includes its own sampling error.
    background = sigma_b**2 * (1 + 1 / n_members)
    gain = background / (background + sigma_o**2)
    analysis = forecast + gain * (obs - forecast.mean(axis=1))[:, None]

    fit = observation_fit(obs, np.full(n_obs, sigma_o), forecast, analysis)

    assert abs(fit["innovation_chi2_diag"] - 1) < 0.05
    assert abs(fit["obs_std_estimated"] - sigma_o) < 0.05 * sigma_o
    assert fit["forecast_rmse"] > fit["analysis_rmse"]
    assert fit["rmse_ratio"] == fit["forecast_rmse"] / fit["analysis_rmse"]
