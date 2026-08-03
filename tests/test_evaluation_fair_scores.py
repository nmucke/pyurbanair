"""WP1.1: the fair finite-ensemble estimators and the run-summary guards.

Every score here is used to *certify* an ESMDA posterior, so an estimator whose
optimum is a collapsed ensemble makes the certification circular. The biased
``M**2`` pairwise form has exactly that defect and its error is ~O(1/M) -- small
enough to look like noise, large enough to reorder a sweep at M=50. Nothing in
the pipeline notices the difference, so it is pinned here:

  * the fair CRPS converges to the analytic Gaussian CRPS at an ensemble size
    the pipeline actually uses, where the biased one is outside the band
    (``..._matches_analytic_gaussian``), and its finite-ensemble bias is gone
    (``..._upward_bias``);
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
    # The fair estimator is unbiased for the population CRPS at *any* ensemble
    # size, so averaging many independent M=50 ensembles (the size the sweeps
    # actually run) must land on the closed form.
    #
    # M matters here. At M=10**4 the biased form's error is E|X-X'|/(2M) ~ 6e-5
    # and it would pass any tolerance loose enough for the Monte-Carlo noise --
    # such a test cannot tell the two estimators apart. At M=50 that same bias
    # is 0.0113, comfortably outside the band asserted below.
    rng = np.random.default_rng(42)
    n_members, n_trials = 50, 20_000
    ens = rng.normal(size=(n_members, n_trials))
    truth = np.full(n_trials, 0.7)
    analytic = _analytic_normal_crps(0.7)

    fair = float(crps_ensemble(ens, truth).mean())
    # E|X - X'| = 2 sigma / sqrt(pi) for independent standard normals; the M**2
    # form keeps the M zero diagonal entries, shrinking its pairwise term by
    # exactly 1/M and inflating the score by that much.
    biased = fair + (2.0 / math.sqrt(math.pi)) / (2 * n_members)

    assert fair == pytest.approx(analytic, abs=5e-3)
    assert abs(biased - analytic) > 1e-2


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


def test_fair_energy_score_matches_its_definition_on_a_realistic_shape() -> None:
    # The M=2 case above pins the diagonal; this pins the estimator itself on a
    # full (component, ensemble, time, sensor) block against a literal
    # transcription of the formula.
    rng = np.random.default_rng(19)
    members = rng.standard_normal((3, 6, 4, 5))  # C, E, T, S
    truth = rng.standard_normal((3, 4, 5))
    n_ens = members.shape[1]

    expected = []
    for t in range(members.shape[2]):
        m, v = members[:, :, t, :], truth[:, t, :]
        term1 = np.sqrt(np.sum((m - v[:, None, :]) ** 2, axis=0)).mean(axis=0)
        pair = np.sqrt(np.sum((m[:, :, None, :] - m[:, None, :, :]) ** 2, axis=0))
        term2 = 0.5 * pair.sum(axis=(0, 1)) / (n_ens * (n_ens - 1))
        expected.append(float((term1 - term2).mean()))

    np.testing.assert_allclose(_energy_score(members, truth), expected, rtol=1e-12)


def test_fair_crps_is_accurate_in_float32_far_from_the_origin() -> None:
    # The sorted-sample form sums terms whose weights cancel to zero, so without
    # centering it loses ~4 orders of magnitude on data offset from zero -- and
    # the parameters scored here include an inflow angle near 270 degrees, held
    # as float32 on disk. Pins the dtype contract's "~1e-7" accuracy claim.
    rng = np.random.default_rng(5)
    ens32 = (rng.normal(scale=0.1, size=(50, 200)) + 270.0).astype(np.float32)
    truth32 = (rng.normal(scale=0.1, size=200) + 270.0).astype(np.float32)

    scored = crps_ensemble(ens32, truth32)
    exact = crps_ensemble(ens32.astype(float), truth32.astype(float))

    assert scored.dtype == np.float32
    assert float(np.max(np.abs(scored.astype(float) - exact))) < 1e-6


def test_fair_crps_does_not_wrap_on_integer_ensembles() -> None:
    # Signed weights and a signed centering, so nothing may be accumulated in
    # the samples' own dtype: an unsigned ensemble would wrap silently.
    truth = np.array([3])
    expected = crps_ensemble(np.array([[1.0], [5.0], [9.0]]), truth.astype(float))[0]

    for dtype in (np.uint8, np.int8, np.int32):
        ens = np.array([[1], [5], [9]], dtype=dtype)
        assert crps_ensemble(ens, truth.astype(dtype))[0] == pytest.approx(
            expected, rel=1e-6
        )


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

    assert health["n_members"] == 1
    assert health["n_unique"] == 1
    assert health["min_pairwise"] is None
    assert health["median_pairwise"] is None


def test_ensemble_uniqueness_matches_diverged_nan_members() -> None:
    # The case the guard exists for: a solver blew up and the resampling policy
    # cloned the failed member. Value comparison calls two all-NaN rows distinct
    # (NaN != NaN) and would report a healthy ensemble; the counts are bitwise.
    members = np.array([[np.nan, 1.0], [np.nan, 1.0], [0.0, 1.0]])

    health = ensemble_uniqueness(members)

    assert health["n_members"] == 3
    assert health["n_unique"] == 2
    # Every pair involves a NaN row here, so there is no finite distance to
    # report -- but that must be None, not NaN leaking into the YAML.
    assert health["min_pairwise"] is None
    assert health["median_pairwise"] is None


def test_ensemble_uniqueness_ignores_nan_pairs_when_finite_ones_exist() -> None:
    members = np.array([[np.nan], [0.0], [0.0], [4.0]])

    health = ensemble_uniqueness(members)

    assert health["n_unique"] == 3
    assert health["min_pairwise"] == 0.0
    assert health["median_pairwise"] == pytest.approx(4.0)


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


def test_crps_skill_is_null_and_logged_at_the_smoke_shape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The 2-member CI smoke shape. The fair CRPS is *identically* zero whenever
    # the truth is bracketed (term1 and the pairwise term coincide), so the
    # skill score divides by zero. The master plan calls for null + a log line,
    # not a special case -- and WP1.2 builds on this key, so the null must be
    # explained somewhere.
    posterior = _param_dataset([[-1.0, -1.0], [1.0, 1.0]])
    prior = _param_dataset([[-3.0, -3.0], [3.0, 3.0]])
    truth = xarray.Dataset(
        {"inflow_angle": (("time",), [0.0, 0.0])}, coords={"time": [0.0, 1.0]}
    )

    with caplog.at_level("WARNING", logger="evaluation.scores"):
        summary = parameter_metric_summary(posterior, truth, prior)["inflow_angle"]

    assert summary["crps"]["mean"] == 0.0
    assert summary["prior_crps_mean"] == 0.0
    assert summary["crps_reduction_vs_prior"] is None
    assert "skill score is undefined" in caplog.text
    # The RMSE reduction is unaffected -- it is not a pairwise score.
    assert summary["rmse_reduction_vs_prior"] > 0.0


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
    health = summary["ensemble_health"]
    # Subset, not equality: invariant 1 permits new keys here.
    assert health["n_members"] == 4
    assert health["n_unique"] == 3
    assert health["n_unique_per_window"] == [3]
    assert health["min_over_median_pairwise"] == 0.0
    assert summary["parameter_metrics"]["inflow_angle"]["crps_reduction_vs_prior"] > 0.0


def test_ensemble_health_survives_a_run_dir_with_no_windows(
    tmp_path: pathlib.Path,
) -> None:
    # Invariant 3: an old or partial run dir degrades, it does not abort the
    # metric stage.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = tmp_path / "run"
    _write_run_dir(run_dir)
    (run_dir / "windows" / "window_0_posterior_params.nc").unlink()

    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert summary["ensemble_health"]["n_unique_per_window"] == []
    assert summary["ensemble_health"]["n_unique"] == 3


def test_ensemble_health_skips_an_unreadable_window(tmp_path: pathlib.Path) -> None:
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = tmp_path / "run"
    _write_run_dir(run_dir)
    # A window truncated by a killed job must cost its own count, not the run's
    # metrics.
    (run_dir / "windows" / "window_1_posterior_params.nc").write_bytes(b"not netcdf")

    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert summary["ensemble_health"]["n_unique_per_window"] == [3]


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


def test_sweep_metrics_omit_sensor_scores_it_cannot_recompute(
    tmp_path: pathlib.Path,
) -> None:
    # Without truth_access.yaml the sensor scores cannot be recomputed. Copying
    # the run's own (possibly biased) ones forward would make a single
    # metrics_version describe a file whose parameter block is fair and whose
    # sensor block is not, defeating the mixing guard; the block is dropped.
    from scripts.esmda._esmda_common import read_yaml, write_yaml
    from scripts.figure_creation.compute_sweep_metrics import process_run

    run_dir = tmp_path / "legacy_run"
    out_dir = tmp_path / "sweep_metrics" / "legacy_run"
    run_dir.mkdir()
    write_yaml(
        {
            "configuration": {"assimilation_model": "pylbm"},
            "sensor_metrics": {"assimilation": {"vel_magnitude_crps": {"mean": 123.0}}},
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
    xarray.Dataset(
        {"inflow_angle": (("ensemble",), [0.0, 1.0])}, coords=coords
    ).to_netcdf(run_dir / "posterior_params.nc")
    xarray.Dataset(
        {"inflow_angle": (("ensemble",), [-1.0, 2.0])}, coords=coords
    ).to_netcdf(run_dir / "prior_params.nc")
    xarray.Dataset({"inflow_angle": 0.5}).to_netcdf(run_dir / "true_params.nc")

    status = process_run(run_dir, out_dir)
    metrics = read_yaml(out_dir / "metrics.yaml")

    assert metrics["metrics_version"] == 2
    assert "parameter_metrics" in metrics
    assert "sensor_metrics" not in metrics
    assert "sensor metrics omitted" in status["note"]


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
