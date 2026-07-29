"""Correctness tests for finite-ensemble data-assimilation metrics."""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
import xarray

from pyurbanair.plotting import _crps_ensemble, compute_parameter_metrics
from pyurbanair.utils.da_metrics import (
    ensemble_uniqueness,
    per_knot_crps,
    summary_scalars,
)
from scripts.esmda._esmda_common import (
    _energy_score,
    parameter_metric_summary,
    read_yaml,
    write_yaml,
)
from scripts.figspec.metrics import spread_skill


def _normal_crps(y: float, mean: float = 0.0, std: float = 1.0) -> float:
    """Analytic CRPS of a Gaussian forecast at a deterministic observation."""
    z = (y - mean) / std
    phi = math.exp(-(z**2) / 2.0) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return std * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def test_fair_crps_matches_analytic_gaussian() -> None:
    rng = np.random.default_rng(42)
    members = rng.normal(size=(10_000, 1))
    truth = np.array([0.7])

    score = per_knot_crps(members, truth)[0]

    assert score == pytest.approx(_normal_crps(float(truth[0])), abs=1e-2)


def test_fair_crps_removes_small_ensemble_upward_bias() -> None:
    rng = np.random.default_rng(7)
    n_members = 8
    n_trials = 20_000
    members = rng.normal(size=(n_members, n_trials))
    truth = rng.normal(size=n_trials)

    fair = float(per_knot_crps(members, truth).mean())
    diffs = np.abs(members[:, None, :] - members[None, :, :])
    biased = float(
        (
            np.mean(np.abs(members - truth[None, :]), axis=0)
            - 0.5 * diffs.mean(axis=(0, 1))
        ).mean()
    )
    population_score = 1.0 / math.sqrt(math.pi)

    assert fair == pytest.approx(population_score, abs=1e-2)
    assert biased > population_score + 0.04


def test_all_fair_score_sites_exclude_the_zero_diagonal() -> None:
    members = np.array([[0.0], [2.0]])
    truth = np.array([0.0])
    # term1 = 1; fair pairwise half-term = (0 + 2 + 2 + 0) / (2*2*1) = 1.
    assert per_knot_crps(members, truth)[0] == pytest.approx(0.0)
    assert _crps_ensemble(members, truth)[0] == pytest.approx(0.0)

    vector_members = members.T[:, :, None, None]  # (component, ensemble, time, sensor)
    vector_truth = truth[:, None, None]  # (component, time, sensor)
    assert _energy_score(vector_members, vector_truth)[0] == pytest.approx(0.0)


def test_single_member_scores_reduce_to_absolute_error() -> None:
    members = np.array([[2.5]])
    truth = np.array([1.0])
    assert per_knot_crps(members, truth)[0] == pytest.approx(1.5)
    assert _crps_ensemble(members, truth)[0] == pytest.approx(1.5)

    vector_members = members.T[:, :, None, None]
    vector_truth = truth[:, None, None]
    assert _energy_score(vector_members, vector_truth)[0] == pytest.approx(1.5)


def test_summary_spread_is_root_mean_variance() -> None:
    members = np.array([[0.0, 0.0], [0.0, 2.0]])
    truth = np.zeros(2)
    expected = math.sqrt(np.mean(np.std(members, axis=0, ddof=1) ** 2))

    assert summary_scalars(members, truth)["time_avg_spread"] == pytest.approx(expected)


def test_spread_skill_fortin_factor_calibrates_exchangeable_ensemble() -> None:
    rng = np.random.default_rng(123)
    n_members = 10
    members = rng.normal(size=(n_members, 100_000))
    truth = rng.normal(size=members.shape[1])
    spread = members.std(axis=0, ddof=1)
    error = members.mean(axis=0) - truth

    corrected = spread_skill(spread, error, n_members)
    uncorrected = float(np.sqrt(np.mean(spread**2)) / np.sqrt(np.mean(error**2)))

    assert corrected == pytest.approx(1.0, abs=1e-2)
    assert uncorrected == pytest.approx(
        math.sqrt(n_members / (n_members + 1)), abs=1e-2
    )


def test_ensemble_uniqueness_detects_exact_clone() -> None:
    members = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ]
    )

    health = ensemble_uniqueness(members)

    assert health["n_members"] == 4
    assert health["n_unique"] == 3
    assert health["min_pairwise"] == 0.0
    median = health["median_pairwise"]
    assert median is not None and median > 0.0


def test_parameter_metrics_include_crps_skill_against_prior() -> None:
    coords = {"ensemble": np.arange(3), "time": [0.0, 1.0]}
    posterior = xarray.Dataset(
        {
            "inflow_angle": (
                ("ensemble", "time"),
                [[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5]],
            )
        },
        coords=coords,
    )
    prior = xarray.Dataset(
        {
            "inflow_angle": (
                ("ensemble", "time"),
                [[-2.0, -2.0], [2.0, 2.0], [3.0, 3.0]],
            )
        },
        coords=coords,
    )
    truth = xarray.Dataset(
        {"inflow_angle": (("time",), [0.0, 0.0])},
        coords={"time": coords["time"]},
    )

    metrics = compute_parameter_metrics(posterior, truth, prior)
    summary = parameter_metric_summary(posterior, truth, prior)

    assert "prior_crps" in metrics["inflow_angle"]
    assert summary["inflow_angle"]["prior_crps_mean"] > 0.0
    assert summary["inflow_angle"]["crps_reduction_vs_prior"] > 0.0


def test_skip_viz_summary_has_version_and_ensemble_health(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = tmp_path / "run"
    windows_dir = run_dir / "windows"
    windows_dir.mkdir(parents=True)
    write_yaml({"run": {"skip_viz": True}}, run_dir / "config.yaml")
    write_yaml({"configuration": {"ensemble_size": 4}}, run_dir / "run_info.yaml")

    coords = {"ensemble": np.arange(4), "time": [0.0, 1.0]}
    posterior = xarray.Dataset(
        {
            "inflow_angle": (
                ("ensemble", "time"),
                [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
            )
        },
        coords=coords,
    )
    prior = xarray.Dataset(
        {
            "inflow_angle": (
                ("ensemble", "time"),
                [[-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0]],
            )
        },
        coords=coords,
    )
    truth = xarray.Dataset(
        {"inflow_angle": (("time",), [0.0, 0.0])},
        coords={"time": coords["time"]},
    )
    posterior.to_netcdf(run_dir / "posterior_params.nc")
    posterior.to_netcdf(windows_dir / "window_0_posterior_params.nc")
    prior.to_netcdf(run_dir / "prior_params.nc")
    truth.to_netcdf(run_dir / "true_params.nc")

    compute_metrics(run_dir)

    summary = read_yaml(run_dir / "run_summary.yaml")
    assert summary["metrics_version"] == 2
    assert summary["ensemble_health"] == {
        "n_members": 4,
        "n_unique": 3,
        "n_unique_per_window": [3],
        "min_over_median_pairwise": 0.0,
    }
    assert (
        summary["parameter_metrics"]["inflow_angle"]["crps_reduction_vs_prior"]
        is not None
    )


def test_sweep_comparison_warns_on_mixed_metric_versions(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.figure_creation.compare_sweep_results import load_runs

    for name, version in (
        ("pylbm_nx10_ny10_nz4_ens10_steps2", None),
        ("pylbm_nx10_ny10_nz4_ens20_steps2", 2),
    ):
        run_dir = tmp_path / name
        run_dir.mkdir()
        summary: dict[str, object] = {"configuration": {"assimilation_model": "pylbm"}}
        if version is not None:
            summary["metrics_version"] = version
        write_yaml(summary, run_dir / "run_summary.yaml")

    with pytest.warns(UserWarning, match="mismatched metrics versions"):
        runs = load_runs(tmp_path, models=None)

    assert set(runs["metrics_version"]) == {1, 2}


def test_sweep_metrics_omit_legacy_sensor_scores_without_truth_access(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.figure_creation.compute_sweep_metrics import process_run

    run_dir = tmp_path / "legacy_run"
    out_dir = tmp_path / "sweep_metrics"
    run_dir.mkdir()
    write_yaml(
        {
            "configuration": {"assimilation_model": "pylbm"},
            "sensor_metrics": {
                "assimilation": {
                    "vel_magnitude_crps": {"mean": 123.0},
                }
            },
        },
        run_dir / "run_summary.yaml",
    )
    write_yaml(
        {
            "obs": {
                "mode": "points",
                "x_points": [0.0],
                "y_points": [0.0],
                "z_points": [0.0],
            }
        },
        run_dir / "config.yaml",
    )

    coords = {"ensemble": [0, 1]}
    posterior = xarray.Dataset(
        {"inflow_angle": (("ensemble",), [0.0, 1.0])}, coords=coords
    )
    prior = xarray.Dataset(
        {"inflow_angle": (("ensemble",), [-1.0, 2.0])}, coords=coords
    )
    truth = xarray.Dataset({"inflow_angle": 0.5})
    posterior.to_netcdf(run_dir / "posterior_params.nc")
    prior.to_netcdf(run_dir / "prior_params.nc")
    truth.to_netcdf(run_dir / "true_params.nc")

    status = process_run(run_dir, out_dir)
    metrics = read_yaml(out_dir / "metrics.yaml")

    assert metrics["metrics_version"] == 2
    assert "parameter_metrics" in metrics
    assert "sensor_metrics" not in metrics
    assert status["note"] == (
        "no truth_access.yaml -> sensor metrics omitted (re-run ESMDA)"
    )
