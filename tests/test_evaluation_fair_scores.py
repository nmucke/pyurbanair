"""WP1.1: the fair finite-ensemble estimators and the run-summary guards.

Every score here is used to *certify* an ESMDA posterior, so an estimator whose
optimum is a collapsed ensemble makes the certification circular. The biased
``M**2`` pairwise form has exactly that defect and its error is ~O(1/M) -- small
enough to look like noise, large enough to reorder a sweep at M=50. Nothing in
the pipeline notices the difference, so it is pinned here:

  * the fair CRPS converges to the analytic Gaussian CRPS, the biased one does
    not (``..._matches_analytic_gaussian`` / ``..._upward_bias``);
  * every pairwise site excludes the zero diagonal (``..._zero_diagonal``);
  * the spread reductions are roots of mean *variances*, with the Fortin
    finite-ensemble factor, so a calibrated ensemble scores 1 and not
    ``sqrt(M/(M+1))``;
  * ``run_summary.yaml`` carries ``metrics_version`` and ``ensemble_health``,
    and the comparison scripts refuse to silently mix estimator generations.
"""

from __future__ import annotations

import math
import pathlib
import warnings

import numpy as np
import pytest
import xarray
from evaluation.scores import (
    _energy_score,
    compute_parameter_metrics,
    crps_ensemble,
    ensemble_uniqueness,
    parameter_metric_summary,
    spread_skill,
    summary_scalars,
)


def _analytic_normal_crps(y: float, mean: float = 0.0, std: float = 1.0) -> float:
    """CRPS of a N(mean, std) forecast at the deterministic observation ``y``."""
    z = (y - mean) / std
    pdf = math.exp(-(z**2) / 2.0) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return std * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


def _biased_crps(ens: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """The pre-WP1.1 estimator: pairwise mean over all ``M**2`` pairs."""
    diffs = np.abs(ens[:, None, :] - ens[None, :, :])
    return np.mean(np.abs(ens - truth[None, :]), axis=0) - 0.5 * diffs.mean(axis=(0, 1))


# ---------------------------------------------------------------------------
# Fair pairwise estimators
# ---------------------------------------------------------------------------


def test_fair_crps_matches_the_pairwise_definition() -> None:
    # crps_ensemble sums sorted samples instead of building the (M, M, K)
    # pairwise tensor. That identity is not obvious by inspection, so it is
    # checked against the literal definition on ragged, unsorted data.
    rng = np.random.default_rng(11)
    ens = rng.standard_normal((7, 4))
    truth = rng.standard_normal(4)

    diffs = np.abs(ens[:, None, :] - ens[None, :, :])
    n = ens.shape[0]
    expected = np.mean(np.abs(ens - truth[None, :]), axis=0) - 0.5 * diffs.sum(
        axis=(0, 1)
    ) / (n * (n - 1))

    np.testing.assert_allclose(crps_ensemble(ens, truth), expected, rtol=1e-12)


def test_fair_crps_matches_analytic_gaussian() -> None:
    # A large sample from a known distribution: the fair estimator is unbiased
    # for the population CRPS, so it must land on the closed form.
    rng = np.random.default_rng(42)
    ens = rng.normal(size=(10_000, 1))
    truth = np.array([0.7])

    assert crps_ensemble(ens, truth)[0] == pytest.approx(
        _analytic_normal_crps(0.7), abs=1e-2
    )


def test_fair_crps_removes_the_small_ensemble_upward_bias() -> None:
    # The discriminating case: at M=8 the biased form overstates the CRPS by
    # ~1/(2M) of the mean pairwise distance. Averaged over many independent
    # trials the fair form converges to the population value 1/sqrt(pi) (the
    # CRPS of a standard normal forecast against a standard normal draw) while
    # the biased one sits clearly above it.
    rng = np.random.default_rng(7)
    n_members, n_trials = 8, 20_000
    ens = rng.normal(size=(n_members, n_trials))
    truth = rng.normal(size=n_trials)
    population = 1.0 / math.sqrt(math.pi)

    fair = float(crps_ensemble(ens, truth).mean())
    biased = float(_biased_crps(ens, truth).mean())

    assert fair == pytest.approx(population, abs=1e-2)
    assert biased > population + 0.04


def test_every_pairwise_site_excludes_the_zero_diagonal() -> None:
    # Two members straddling the truth at equal distance: term1 = 1 and the fair
    # pairwise half-term = (2 + 2) / (2 * 2 * 1) = 1, so the score is exactly 0.
    # The biased form keeps the two zero diagonal entries and returns 0.5.
    ens = np.array([[0.0], [2.0]])
    truth = np.array([0.0])
    assert crps_ensemble(ens, truth)[0] == pytest.approx(0.0)
    assert _biased_crps(ens, truth)[0] == pytest.approx(0.5)

    # Same numbers through the multivariate site, shaped
    # (component, ensemble, time, sensor) / (component, time, sensor).
    vector_ens = ens.T[:, :, None, None]
    vector_truth = truth[:, None, None]
    assert _energy_score(vector_ens, vector_truth)[0] == pytest.approx(0.0)


def test_single_member_scores_degenerate_to_the_absolute_error() -> None:
    # M(M-1) is 0 at M=1: the pairwise term must be dropped, not divided by zero.
    ens = np.array([[2.5]])
    truth = np.array([1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a 0/0 would surface as a RuntimeWarning
        assert crps_ensemble(ens, truth)[0] == pytest.approx(1.5)
        assert _energy_score(ens.T[:, :, None, None], truth[:, None, None])[
            0
        ] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Spread reductions
# ---------------------------------------------------------------------------


def test_summary_spread_is_the_root_of_the_mean_variance() -> None:
    # One zero-spread knot and one with spread: the mean of the stds understates
    # the ensemble variance (Jensen) and is not comparable with time_avg_error,
    # which is already an RMS.
    ens = np.array([[0.0, 0.0], [0.0, 2.0]])
    truth = np.zeros(2)
    stds = np.std(ens, axis=0, ddof=1)

    spread = summary_scalars(ens, truth)["time_avg_spread"]

    assert spread == pytest.approx(math.sqrt(float(np.mean(stds**2))))
    assert spread > float(np.mean(stds))


def test_spread_skill_fortin_factor_calibrates_an_exchangeable_ensemble() -> None:
    # Truth and members drawn from the same distribution = perfect calibration by
    # construction, so a correct spread-skill ratio is 1. Without the factor the
    # same data scores sqrt(M/(M+1)) < 1 and reads as over-confident.
    rng = np.random.default_rng(123)
    n_members = 10
    ens = rng.normal(size=(n_members, 100_000))
    truth = rng.normal(size=ens.shape[1])
    spread = ens.std(axis=0, ddof=1)
    error = ens.mean(axis=0) - truth

    corrected = spread_skill(spread, error, n_members)
    uncorrected = float(np.sqrt(np.mean(spread**2)) / np.sqrt(np.mean(error**2)))

    assert corrected == pytest.approx(1.0, abs=1e-2)
    assert uncorrected == pytest.approx(
        math.sqrt(n_members / (n_members + 1)), abs=1e-2
    )


def test_spread_skill_rejects_a_nonsensical_ensemble_size() -> None:
    with pytest.raises(ValueError, match="n_members"):
        spread_skill(np.ones(3), np.ones(3), 0)


# ---------------------------------------------------------------------------
# Duplicate-member guard
# ---------------------------------------------------------------------------


def test_ensemble_uniqueness_counts_an_exact_clone() -> None:
    # Rows 1 and 2 are the bit-identical copy a resampling policy leaves behind.
    members = np.array([[0.0, 1.0], [2.0, 3.0], [2.0, 3.0], [4.0, 5.0]])

    health = ensemble_uniqueness(members)

    assert health["n_members"] == 4
    assert health["n_unique"] == 3
    assert health["min_pairwise"] == 0.0
    assert health["median_pairwise"] > 0.0


def test_ensemble_uniqueness_keeps_near_duplicates_distinct() -> None:
    # Exact matching only. Two very close but distinct members are two members;
    # the distance ratio is what flags them.
    members = np.array([[0.0], [1e-9], [5.0]])

    health = ensemble_uniqueness(members)

    assert health["n_unique"] == 3
    assert health["min_pairwise"] == pytest.approx(1e-9)


def test_ensemble_uniqueness_has_no_distances_below_two_members() -> None:
    health = ensemble_uniqueness(np.array([[1.0, 2.0]]))

    assert health == {
        "n_members": 1,
        "n_unique": 1,
        "min_pairwise": None,
        "median_pairwise": None,
    }


def test_ensemble_uniqueness_rejects_a_non_matrix() -> None:
    with pytest.raises(ValueError, match="n_members, n_features"):
        ensemble_uniqueness(np.zeros((2, 3, 4)))


# ---------------------------------------------------------------------------
# CRPSS vs the prior
# ---------------------------------------------------------------------------


def _param_dataset(members: list[list[float]]) -> xarray.Dataset:
    return xarray.Dataset(
        {"inflow_angle": (("ensemble", "time"), members)},
        coords={"ensemble": np.arange(len(members)), "time": [0.0, 1.0]},
    )


def test_parameter_summary_reports_crps_skill_against_the_prior() -> None:
    # A posterior tight around the truth against a wide, biased prior: both the
    # RMSE and the CRPS must improve.
    posterior = _param_dataset([[0.1, 0.1], [0.6, 0.6], [-0.4, -0.4]])
    prior = _param_dataset([[-2.0, -2.0], [2.0, 2.0], [3.0, 3.0]])
    truth = xarray.Dataset(
        {"inflow_angle": (("time",), [0.0, 0.0])}, coords={"time": [0.0, 1.0]}
    )

    metrics = compute_parameter_metrics(posterior, truth, prior)
    summary = parameter_metric_summary(posterior, truth, prior)["inflow_angle"]

    assert "prior_crps" in metrics["inflow_angle"]
    assert summary["prior_crps_mean"] > 0.0
    assert 0.0 < summary["crps_reduction_vs_prior"] <= 1.0
    # The pre-existing RMSE keys are untouched (additive-only invariant).
    assert summary["rmse_reduction_vs_prior"] > 0.0


def test_crps_skill_is_negative_when_the_posterior_over_contracts() -> None:
    # The failure the CRPSS exists to catch: the posterior collapses onto a
    # value that is *wrong*, so it beats a wide prior on nothing.
    posterior = _param_dataset([[3.0, 3.0], [3.01, 3.01], [2.99, 2.99], [3.0, 3.0]])
    prior = _param_dataset([[-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0]])
    truth = xarray.Dataset(
        {"inflow_angle": (("time",), [0.0, 0.0])}, coords={"time": [0.0, 1.0]}
    )

    summary = parameter_metric_summary(posterior, truth, prior)["inflow_angle"]

    assert summary["crps_reduction_vs_prior"] < 0.0


def test_parameter_summary_omits_prior_keys_without_a_prior() -> None:
    # Invariant 3: the block degrades to the posterior-only entries rather than
    # emitting nulls or raising.
    posterior = _param_dataset([[0.0, 0.0], [1.0, 1.0]])
    truth = xarray.Dataset(
        {"inflow_angle": (("time",), [0.0, 0.0])}, coords={"time": [0.0, 1.0]}
    )

    summary = parameter_metric_summary(posterior, truth, None)["inflow_angle"]

    assert set(summary) == {"rmse", "crps"}


# ---------------------------------------------------------------------------
# run_summary.yaml wiring and the comparison-script guards
# ---------------------------------------------------------------------------


def _write_run_dir(run_dir: pathlib.Path) -> None:
    """A minimal skip_viz ESMDA run dir: config, run_info and parameter files."""
    from scripts.esmda._esmda_common import write_yaml

    windows = run_dir / "windows"
    windows.mkdir(parents=True)
    write_yaml({"run": {"skip_viz": True}}, run_dir / "config.yaml")
    write_yaml({"configuration": {"ensemble_size": 4}}, run_dir / "run_info.yaml")

    # Members 1 and 2 are identical -> 3 unique, in the assembled file and in
    # the single window it was concatenated from.
    posterior = _param_dataset([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    prior = _param_dataset([[-4.0, -4.0], [-3.0, -3.0], [3.0, 3.0], [4.0, 4.0]])
    truth = xarray.Dataset(
        {"inflow_angle": (("time",), [0.0, 0.0])}, coords={"time": [0.0, 1.0]}
    )
    posterior.to_netcdf(run_dir / "posterior_params.nc")
    posterior.to_netcdf(windows / "window_0_posterior_params.nc")
    prior.to_netcdf(run_dir / "prior_params.nc")
    truth.to_netcdf(run_dir / "true_params.nc")


def test_run_summary_carries_the_version_marker_and_ensemble_health(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = tmp_path / "run"
    _write_run_dir(run_dir)

    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert summary["metrics_version"] == 2
    assert summary["ensemble_health"] == {
        "n_members": 4,
        "n_unique": 3,
        "n_unique_per_window": [3],
        "min_over_median_pairwise": 0.0,
    }
    assert summary["parameter_metrics"]["inflow_angle"]["crps_reduction_vs_prior"] > 0.0


def test_sweep_comparison_warns_on_mixed_metric_versions(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.esmda._esmda_common import write_yaml
    from scripts.figure_creation.compare_sweep_results import load_runs

    for name, version in (
        ("pylbm_nx10_ny10_nz4_ens10_steps2", None),  # pre-WP1.1: no marker
        ("pylbm_nx10_ny10_nz4_ens20_steps2", 2),
    ):
        run_dir = tmp_path / name
        run_dir.mkdir()
        summary: dict = {"configuration": {"assimilation_model": "pylbm"}}
        if version is not None:
            summary["metrics_version"] = version
        write_yaml(summary, run_dir / "run_summary.yaml")

    with pytest.warns(UserWarning, match="mismatched metrics versions"):
        runs = load_runs(tmp_path, models=None)

    assert set(runs["metrics_version"]) == {1, 2}


def test_state_run_comparison_warns_on_mixed_metric_versions(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.esmda._esmda_common import write_yaml
    from scripts.figure_creation.compare_state_runs import collect_runs

    for name, version in (("run_a_ic", None), ("run_b_ic", 2)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        write_yaml({"esmda": {"final_time_smoothing": False}}, run_dir / "config.yaml")
        summary: dict = {"parameter_metrics": {}}
        if version is not None:
            summary["metrics_version"] = version
        write_yaml(summary, run_dir / "run_summary.yaml")

    with pytest.warns(UserWarning, match="mismatched metrics versions"):
        rows = collect_runs(tmp_path, mode_filter="both")

    assert {r["metrics_version"] for r in rows} == {1, 2}


def test_comparisons_are_silent_when_every_run_shares_a_version(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.esmda._esmda_common import write_yaml
    from scripts.figure_creation.compare_sweep_results import load_runs

    for name in (
        "pylbm_nx10_ny10_nz4_ens10_steps2",
        "pylbm_nx10_ny10_nz4_ens20_steps2",
    ):
        run_dir = tmp_path / name
        run_dir.mkdir()
        write_yaml({"metrics_version": 2}, run_dir / "run_summary.yaml")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        load_runs(tmp_path, models=None)
