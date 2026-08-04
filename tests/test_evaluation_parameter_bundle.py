"""WP1.2: the parameter calibration bundle (metrics doc §3).

The accuracy scores already in ``run_summary.yaml`` answer "how close is the
posterior?". They cannot answer the question ESMDA actually fails at: **is the
posterior honest about its own uncertainty?** A filter that halves its error
while quartering its spread has got *worse*, and every RMSE/CRPS number in the
summary goes the right way while it happens.

That is what is pinned here:

  * the z-score is ``(θ* − θ̄ᵃ)/σᵃ`` with a ``ddof=1`` spread and that sign --
    both easy to write backwards, both silently plausible if you do
    (``..._exact_and_signed_toward_the_truth``, ``..._mirror_of_the_z_score``);
  * pooled over knots it is ~N(0, 1) for a calibrated posterior
    (``..._look_standard_normal_...``), which is the reading the docs promise;
  * the canonical failure -- tight *and* wrong -- shows up as a small
    contraction ratio with a large ``|z|`` (``..._over_confident_...``);
  * degenerate scales (a pinned parameter's zero prior spread, a collapsed
    posterior, a one-member ensemble) produce ``null``, never ``inf``: an
    ``inf`` poisons every reduction it enters and reaches ``run_summary.yaml``
    as ``.inf`` (``..._pinned_parameter_...``, ``..._collapsed_...``);
  * the block is additive -- every pre-WP1.2 key keeps its value
    (``..._additive_over_the_existing_keys``).
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
import xarray
from evaluation.scores import (
    compute_parameter_bundles,
    compute_parameter_metrics,
    parameter_bundle,
    parameter_metric_summary,
    z_score_stats,
)

# ---------------------------------------------------------------------------
# The estimator itself
# ---------------------------------------------------------------------------


def test_z_score_is_exact_and_signed_toward_the_truth() -> None:
    # Two members at -1 and +1: mean 0, ddof=1 spread sqrt(2) (a ddof=0 spread
    # would be 1.0 and the z-score exactly 1.0, so this pins the ddof).
    bundle = parameter_bundle(np.array([[-1.0], [1.0]]), None, np.array([1.0]))

    assert bundle["posterior_std"] == pytest.approx(math.sqrt(2.0))
    assert bundle["z_score"] == pytest.approx(1.0 / math.sqrt(2.0))


def test_z_score_is_negative_when_the_posterior_mean_overshoots_the_truth() -> None:
    bundle = parameter_bundle(np.array([[1.0], [3.0]]), None, np.array([0.0]))

    assert bundle["z_score"][0] < 0.0


def test_normalized_error_is_the_mirror_of_the_z_score_scaled_by_the_prior() -> None:
    # The two share a numerator up to sign and differ in which spread they
    # divide by; writing either with the other's convention is the easy bug.
    post = np.array([[1.0, 1.0], [3.0, 3.0]])  # mean 2, sd sqrt(2)
    prior = np.array([[-4.0, -4.0], [4.0, 4.0]])  # mean 0, sd sqrt(32)
    truth = np.array([0.0, 0.0])

    bundle = parameter_bundle(post, prior, truth)

    assert bundle["prior_std"] == pytest.approx(math.sqrt(32.0))
    # Posterior mean is *above* the truth: the error is positive, the z is not.
    assert bundle["normalized_error"] == pytest.approx(2.0 / math.sqrt(32.0))
    assert bundle["z_score"] == pytest.approx(-2.0 / math.sqrt(2.0))


def test_contraction_ratio_is_the_posterior_spread_over_the_prior_spread() -> None:
    post = np.array([[-1.0], [1.0]])  # sd sqrt(2)
    prior = np.array([[-4.0], [4.0]])  # sd sqrt(32) = 4 sqrt(2)
    bundle = parameter_bundle(post, prior, np.array([0.0]))

    assert bundle["contraction_ratio"] == pytest.approx(0.25)


def test_pooled_z_scores_look_standard_normal_for_a_calibrated_posterior() -> None:
    """The linear-Gaussian toy: a *correctly* calibrated posterior.

    The ensemble and the truth are independent draws from the same Gaussian --
    the definition of a calibrated posterior -- so the z-scores must be ~N(0, 1).
    At this ensemble size the exact sampling distribution
    (``sqrt(1 + 1/M) · t_(M-1)``) has std 1.008, indistinguishable from 1 here;
    the point is the shape, which a variance-for-std or missing-sqrt slip
    destroys.
    """
    rng = np.random.default_rng(20260804)
    n_members, n_replicates = 200, 2000

    draws = rng.standard_normal((n_members + 1, n_replicates))
    # One knot per replicate: members are rows 0..M-1, the truth is the extra,
    # independent row.
    bundle = parameter_bundle(draws[:n_members], None, draws[n_members])
    stats = z_score_stats(bundle["z_score"])

    assert stats is not None
    assert stats["n"] == n_replicates
    assert abs(stats["mean"]) < 0.05
    assert stats["std"] == pytest.approx(1.0, abs=0.05)
    # ~4.6 % of a standard normal lies outside +-2.
    assert stats["frac_abs_gt_2"] == pytest.approx(0.0455, abs=0.015)


def test_over_confident_posterior_shows_small_contraction_and_a_large_z() -> None:
    # The canonical ESMDA failure: the posterior contracted hard onto a value
    # that is wrong. Accuracy alone cannot distinguish this from success --
    # jointly, the two numbers can.
    post = np.array([[2.99, 2.99], [3.0, 3.0], [3.01, 3.01]])
    prior = np.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
    truth = np.array([0.0, 0.0])

    bundle = parameter_bundle(post, prior, truth)

    assert np.all(bundle["contraction_ratio"] < 0.02)
    assert np.all(np.abs(bundle["z_score"]) > 3.0)


# ---------------------------------------------------------------------------
# Degenerate scales: null, never inf
# ---------------------------------------------------------------------------


def test_a_pinned_parameter_yields_null_rather_than_infinite_ratios() -> None:
    # A pinned parameter has zero prior spread by construction, so both
    # prior-relative quantities are undefined -- not infinite.
    post = np.array([[0.5], [1.5]])
    prior = np.array([[1.0], [1.0]])
    bundle = parameter_bundle(post, prior, np.array([0.0]))

    assert np.isnan(bundle["normalized_error"]).all()
    assert np.isnan(bundle["contraction_ratio"]).all()
    assert not np.isinf(bundle["normalized_error"]).any()
    # The posterior half is still perfectly well defined.
    assert np.isfinite(bundle["z_score"]).all()


def test_a_pinned_parameter_reaches_the_summary_as_null() -> None:
    # The same degeneracy through the reduction: null, not a missing key and
    # not `.inf`. Pinned parameters are a supported ESMDA configuration.
    posterior = _param_dataset(inflow_angle=[[-0.5, -0.5], [1.5, 1.5]])
    prior = _param_dataset(inflow_angle=[[1.0, 1.0], [1.0, 1.0]])
    truth = _truth_dataset(inflow_angle=[0.0, 0.0])

    summary = parameter_metric_summary(posterior, truth, prior)["inflow_angle"]

    assert summary["normalized_error"] is None
    assert summary["contraction_ratio"] is None
    assert summary["z_score"]["n"] == 2


def test_a_collapsed_posterior_yields_a_null_z_score() -> None:
    post = np.array([[2.0], [2.0], [2.0]])
    bundle = parameter_bundle(post, None, np.array([0.0]))

    assert np.isnan(bundle["z_score"]).all()
    assert not np.isinf(bundle["z_score"]).any()


def test_a_single_member_ensemble_has_no_defined_spread() -> None:
    bundle = parameter_bundle(np.array([[1.0, 2.0]]), None, np.array([0.0, 0.0]))

    assert np.isnan(bundle["posterior_std"]).all()
    assert np.isnan(bundle["z_score"]).all()


def test_parameter_bundle_without_a_prior_reports_only_the_posterior_half() -> None:
    bundle = parameter_bundle(np.array([[-1.0], [1.0]]), None, np.array([0.0]))

    assert set(bundle) == {"z_score", "posterior_std"}


@pytest.mark.parametrize(  # type: ignore[misc]
    "post,prior,truth,match",
    [
        (np.zeros(4), None, np.zeros(4), "n_members, n_knots"),
        (np.zeros((3, 4)), None, np.zeros(5), "match post"),
        (np.zeros((3, 4)), np.zeros((3, 5)), np.zeros(4), "match post"),
    ],
)
def test_parameter_bundle_rejects_mismatched_shapes(
    post: np.ndarray,
    prior: np.ndarray | None,
    truth: np.ndarray,
    match: str,
) -> None:
    # Broadcasting a differently-sampled prior or truth would invent a
    # knot-by-knot correspondence that does not exist.
    with pytest.raises(ValueError, match=match):
        parameter_bundle(post, prior, truth)


# ---------------------------------------------------------------------------
# The z-score reduction
# ---------------------------------------------------------------------------


def test_z_score_stats_is_none_when_nothing_is_finite() -> None:
    assert z_score_stats(np.array([np.nan, np.nan])) is None
    assert z_score_stats(np.array([])) is None


def test_z_score_stats_ignores_the_degenerate_knots() -> None:
    stats = z_score_stats(np.array([1.0, np.nan, -1.0, 3.0]))

    assert stats is not None
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(1.0)
    assert stats["max_abs"] == pytest.approx(3.0)
    assert stats["frac_abs_gt_2"] == pytest.approx(1.0 / 3.0)


def test_z_score_stats_has_no_std_for_a_single_knot() -> None:
    # Undefined, not zero: a single value has no ddof=1 spread.
    stats = z_score_stats(np.array([2.0]))

    assert stats is not None
    assert stats["std"] is None
    assert stats["mean"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Wiring into run_summary.yaml
# ---------------------------------------------------------------------------


def _param_dataset(**members: list[list[float]]) -> xarray.Dataset:
    n_members = len(next(iter(members.values())))
    return xarray.Dataset(
        {name: (("ensemble", "time"), rows) for name, rows in members.items()},
        coords={"ensemble": np.arange(n_members), "time": [0.0, 1.0]},
    )


def _truth_dataset(**values: list[float]) -> xarray.Dataset:
    return xarray.Dataset(
        {name: (("time",), series) for name, series in values.items()},
        coords={"time": [0.0, 1.0]},
    )


def _toy_run() -> tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]:
    """Two parameters, a contracted posterior around a slightly offset truth."""
    posterior = _param_dataset(
        inflow_angle=[[-0.5, -0.5], [0.5, 0.5], [1.5, 1.5]],
        velocity_magnitude=[[4.2, 4.2], [5.2, 5.2], [6.2, 6.2]],
    )
    # Deliberately asymmetric about the truth: a prior ensemble spaced evenly
    # around it scores a fair CRPS of *exactly* zero, which would make the
    # CRPSS null here for a reason that has nothing to do with this WP.
    prior = _param_dataset(
        inflow_angle=[[-6.0, -6.0], [-1.0, -1.0], [7.0, 7.0]],
        velocity_magnitude=[[1.0, 1.0], [4.0, 4.0], [10.0, 10.0]],
    )
    truth = _truth_dataset(inflow_angle=[0.0, 0.0], velocity_magnitude=[5.0, 5.0])
    return posterior, prior, truth


def test_parameter_summary_carries_the_calibration_block() -> None:
    posterior, prior, truth = _toy_run()

    summary = parameter_metric_summary(posterior, truth, prior)

    for name in ("inflow_angle", "velocity_magnitude"):
        entry = summary[name]
        assert entry["z_score"]["n"] == 2  # two knots
        assert entry["normalized_error"] is not None
        assert 0.0 < entry["contraction_ratio"]["mean"] < 1.0
    # Pooled across *both* parameters and every knot -- legitimate only because
    # a z-score is dimensionless, which is half the reason it is reported.
    assert summary["pooled"]["z_score"]["n"] == 4


def test_parameter_summary_is_additive_over_the_existing_keys() -> None:
    # Invariant 1: WP1.2 may only add keys. The accuracy half must be
    # bit-identical to what compute_parameter_metrics produces on its own.
    posterior, prior, truth = _toy_run()

    summary = parameter_metric_summary(posterior, truth, prior)
    metrics = compute_parameter_metrics(posterior, truth, prior)

    entry = summary["inflow_angle"]
    assert set(entry) >= {
        "rmse",
        "crps",
        "prior_rmse_mean",
        "rmse_reduction_vs_prior",
        "prior_crps_mean",
        "crps_reduction_vs_prior",
    }
    assert entry["rmse"]["mean"] == pytest.approx(
        float(np.mean(metrics["inflow_angle"]["rmse"]))
    )
    assert entry["crps"]["mean"] == pytest.approx(
        float(np.mean(metrics["inflow_angle"]["crps"]))
    )


def test_summary_reports_null_and_logs_when_the_posterior_has_collapsed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    posterior = _param_dataset(inflow_angle=[[3.0, 3.0], [3.0, 3.0]])
    prior = _param_dataset(inflow_angle=[[-2.0, -2.0], [2.0, 2.0]])
    truth = _truth_dataset(inflow_angle=[0.0, 0.0])

    with caplog.at_level("WARNING"):
        summary = parameter_metric_summary(posterior, truth, prior)

    assert summary["inflow_angle"]["z_score"] is None
    assert summary["pooled"]["z_score"] is None
    assert "calibration of this parameter is unknown" in caplog.text
    # The contraction ratio still reports the collapse it caused.
    assert summary["inflow_angle"]["contraction_ratio"]["mean"] == 0.0


def test_a_differently_sampled_prior_drops_only_the_prior_relative_entries() -> None:
    # The pre-existing guard (a prior on a different knot grid is not
    # comparable knot-by-knot) now governs the bundle too.
    posterior, _prior, truth = _toy_run()
    prior = xarray.Dataset(
        {"inflow_angle": (("ensemble", "time"), [[-6.0], [0.0], [6.0]])},
        coords={"ensemble": np.arange(3), "time": [0.0]},
    )

    bundles = compute_parameter_bundles(posterior, truth, prior)
    summary = parameter_metric_summary(posterior, truth, prior)

    assert set(bundles["inflow_angle"]) == {"z_score", "posterior_std"}
    assert "contraction_ratio" not in summary["inflow_angle"]
    assert "rmse_reduction_vs_prior" not in summary["inflow_angle"]
    assert summary["inflow_angle"]["z_score"]["n"] == 2


def test_run_summary_serializes_the_calibration_block(tmp_path: pathlib.Path) -> None:
    # End to end through the real metric stage: the values must survive
    # yaml.safe_dump as plain scalars (an inf or a numpy float would not).
    from scripts.esmda._esmda_common import read_yaml, write_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = tmp_path / "run"
    (run_dir / "windows").mkdir(parents=True)
    write_yaml({"run": {"skip_viz": True}}, run_dir / "config.yaml")
    write_yaml({"configuration": {"ensemble_size": 3}}, run_dir / "run_info.yaml")
    posterior, prior, truth = _toy_run()
    posterior.to_netcdf(run_dir / "posterior_params.nc")
    prior.to_netcdf(run_dir / "prior_params.nc")
    truth.to_netcdf(run_dir / "true_params.nc")

    compute_metrics(run_dir)
    written = read_yaml(run_dir / "run_summary.yaml")["parameter_metrics"]

    entry = written["inflow_angle"]
    assert isinstance(entry["z_score"]["mean"], float)
    assert math.isfinite(entry["z_score"]["mean"])
    assert math.isfinite(entry["contraction_ratio"]["mean"])
    assert math.isfinite(written["pooled"]["z_score"]["max_abs"])
    # ``.inf`` / ``.nan`` are how yaml.safe_dump spells a non-finite float; on a
    # well-conditioned run neither may appear (an undefined entry is ``null``).
    text = (run_dir / "run_summary.yaml").read_text()
    assert ".inf" not in text and ".nan" not in text
