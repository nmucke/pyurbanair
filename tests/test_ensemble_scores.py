"""Closed-form and property tests for the shared ensemble scores.

Every assertion here is either an analytic value, a brute-force reference
implementation, or a statistical property of a synthetic calibrated ensemble
with a fixed seed -- no snapshots, so the tests stay meaningful if the
implementation is rewritten. ``M = 2`` (the smoke-shape ensemble size used
throughout the test suite) is exercised explicitly for every function.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyurbanair.plotting import _crps_ensemble
from pyurbanair.utils.da_metrics import per_knot_crps
from pyurbanair.utils.ensemble_scores import (
    coverage,
    coverage_indicator,
    coverage_nominal_alpha,
    crpss,
    fair_crps,
    fair_energy_score,
    max_abs_zscore_reference,
    max_nominal_alpha,
    pit_rank,
    rank_histogram,
    rank_histogram_weights,
    spread_skill_ratio,
    wasserstein_over_floor,
    wasserstein_over_sigma,
    wasserstein_self_floor,
    zscore,
    zscore_exceedance,
    zscore_nominal_exceedance,
)
from scripts.esmda._esmda_common import _energy_score
from scripts.figspec.metrics import spread_skill


def _brute_force_crps(members: np.ndarray, truth: float) -> float:
    """Fair CRPS by the definition: O(N**2) double loop, no identities."""
    n = len(members)
    term1 = float(sum(abs(float(x) - truth) for x in members)) / n
    if n < 2:
        return term1
    pairwise = float(sum(abs(float(a) - float(b)) for a in members for b in members))
    return term1 - 0.5 * pairwise / (n * (n - 1))


def _brute_force_energy(members: np.ndarray, truth: np.ndarray) -> float:
    """Fair energy score by the definition, one vector ensemble at a time."""
    n = len(members)
    term1 = float(np.mean([np.linalg.norm(m - truth) for m in members]))
    if n < 2:
        return term1
    pairwise = sum(float(np.linalg.norm(a - b)) for a in members for b in members)
    return term1 - 0.5 * pairwise / (n * (n - 1))


# ---------------------------------------------------------------------------
# fair_crps
# ---------------------------------------------------------------------------


def test_fair_crps_single_member_is_absolute_error() -> None:
    members = np.array([[2.5, -1.0]])
    truth = np.array([1.0, 1.0])

    assert fair_crps(members, truth) == pytest.approx([1.5, 2.0])


def test_fair_crps_matches_hand_computed_value() -> None:
    # members {0, 1, 2}, truth 3:
    #   term1 = (3 + 2 + 1)/3 = 2
    #   sum_{i,j} |x_i - x_j| = 2*(1 + 2 + 1) = 8 -> term2 = 0.5*8/(3*2) = 2/3
    members = np.array([[0.0], [1.0], [2.0]])
    truth = np.array([3.0])

    assert fair_crps(members, truth)[0] == pytest.approx(2.0 - 2.0 / 3.0)


def test_fair_crps_matches_brute_force_on_random_input() -> None:
    rng = np.random.default_rng(7)
    members = rng.normal(size=(9, 13))
    truth = rng.normal(size=13)

    scores = fair_crps(members, truth)
    expected = [_brute_force_crps(members[:, k], truth[k]) for k in range(13)]

    assert scores == pytest.approx(expected)


def test_fair_crps_preserves_multidimensional_batch_axes() -> None:
    rng = np.random.default_rng(11)
    members = rng.normal(size=(6, 4, 3))
    truth = rng.normal(size=(4, 3))

    scores = fair_crps(members, truth)

    assert scores.shape == (4, 3)
    for i in range(4):
        for j in range(3):
            assert scores[i, j] == pytest.approx(
                _brute_force_crps(members[:, i, j], truth[i, j])
            )


def test_fair_crps_agrees_with_the_call_sites_it_replaced() -> None:
    rng = np.random.default_rng(3)
    members = rng.normal(size=(8, 17))
    truth = rng.normal(size=17)

    shared = fair_crps(members, truth)

    # Both legacy entry points are now wrappers; this pins them bit-for-bit.
    assert per_knot_crps(members, truth) == pytest.approx(shared, rel=0, abs=0)
    assert _crps_ensemble(members, truth) == pytest.approx(shared, rel=0, abs=0)


# ---------------------------------------------------------------------------
# fair_energy_score
# ---------------------------------------------------------------------------


def test_energy_score_reduces_to_crps_in_one_dimension() -> None:
    """The defining property of the energy score."""
    rng = np.random.default_rng(19)
    members = rng.normal(size=(7, 5))
    truth = rng.normal(size=5)

    energy = fair_energy_score(members[..., None], truth[..., None])

    assert energy == pytest.approx(fair_crps(members, truth))


def test_energy_score_matches_brute_force_on_random_vectors() -> None:
    rng = np.random.default_rng(23)
    members = rng.normal(size=(6, 4, 3, 2))  # (M, time, sensor, component)
    truth = rng.normal(size=(4, 3, 2))

    scores = fair_energy_score(members, truth)

    assert scores.shape == (4, 3)
    for t in range(4):
        for s in range(3):
            assert scores[t, s] == pytest.approx(
                _brute_force_energy(members[:, t, s, :], truth[t, s, :])
            )


def test_energy_score_single_member_is_euclidean_error() -> None:
    members = np.array([[[3.0, 4.0]]])  # (M=1, batch=1, C=2)
    truth = np.array([[0.0, 0.0]])

    assert fair_energy_score(members, truth)[0] == pytest.approx(5.0)


def test_esmda_energy_score_adapter_matches_the_shared_score() -> None:
    """``_esmda_common._energy_score`` is now an axis adapter, nothing more."""
    rng = np.random.default_rng(29)
    members = rng.normal(size=(3, 5, 4, 6))  # (component, ensemble, time, sensor)
    truth = rng.normal(size=(3, 4, 6))  # (component, time, sensor)

    per_time = _energy_score(members, truth)
    shared = fair_energy_score(
        np.moveaxis(members, 0, -1), np.moveaxis(truth, 0, -1)
    ).mean(axis=-1)

    assert per_time.shape == (4,)
    assert per_time == pytest.approx(shared, rel=0, abs=0)


# ---------------------------------------------------------------------------
# crpss
# ---------------------------------------------------------------------------


def test_crpss_is_one_minus_the_score_ratio() -> None:
    assert crpss(0.25, 1.0) == pytest.approx(0.75)
    assert crpss(2.0, 1.0) == pytest.approx(-1.0)


def test_crpss_returns_none_for_degenerate_references() -> None:
    nan = float("nan")
    for post, prior in [(0.5, 0.0), (0.5, -1.0), (0.5, nan), (nan, 1.0)]:
        assert crpss(post, prior) is None, f"post={post}, prior={prior}"


# ---------------------------------------------------------------------------
# zscore
# ---------------------------------------------------------------------------


def test_zscore_matches_the_definition() -> None:
    members = np.array([[0.0], [2.0], [4.0]])
    truth = np.array([4.0])
    expected = (4.0 - 2.0) / float(np.std([0.0, 2.0, 4.0], ddof=1))

    assert zscore(members, truth)[0] == pytest.approx(expected)


def test_zscore_is_nan_without_raising_for_degenerate_spread() -> None:
    single = np.array([[1.0, 2.0]])
    flat = np.zeros((4, 2))
    truth = np.array([1.0, 2.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a RuntimeWarning would fail the test
        single_z = zscore(single, truth)
        flat_z = zscore(flat, np.array([1.0, 0.0]))

    assert np.all(np.isnan(single_z))
    assert np.all(np.isnan(flat_z))  # zero spread, including a nonzero error


# ---------------------------------------------------------------------------
# z-score calibration reference
# ---------------------------------------------------------------------------


def _calibrated_zscores(
    n_members: int, shape: tuple[int, ...], seed: int
) -> np.ndarray:
    """z-scores of a *perfectly* calibrated ensemble: truth is one more draw."""
    rng = np.random.default_rng(seed)
    ens = rng.normal(size=(n_members,) + shape)
    truth = rng.normal(size=shape)
    return zscore(ens, truth)


@pytest.mark.parametrize(  # type: ignore[misc]
    "n_members, threshold",
    [(4, 2.0), (8, 3.0), (32, 2.0), (32, 3.0), (50, 3.0)],
)
def test_zscore_nominal_exceedance_matches_a_calibrated_ensemble(
    n_members: int, threshold: float
) -> None:
    """The stated null is the empirical one, at every M that matters."""
    z = _calibrated_zscores(n_members, (200_000,), seed=n_members)
    observed = float(np.mean(np.abs(z) > threshold))
    nominal = zscore_nominal_exceedance(n_members, threshold)

    # Columns are independent ensembles here, so binomial sigma applies.
    sigma = np.sqrt(nominal * (1.0 - nominal) / z.size)
    assert observed == pytest.approx(nominal, abs=5.0 * sigma)


def test_zscore_null_is_a_scaled_t_not_a_plain_t_nor_a_normal() -> None:
    """Both corrections are needed; neither is a rounding-level detail.

    The ``ddof=1`` denominator gives the ``t``; ``truth - mean_m`` having
    variance ``sigma**2 (1 + 1/M)`` gives the ``sqrt((M + 1)/M)`` scale.
    Dropping either one biases the reference LOW, i.e. toward calling a
    calibrated ensemble overconfident.
    """
    from scipy import stats

    z = _calibrated_zscores(32, (200_000,), seed=901)
    observed = float(np.mean(np.abs(z) > 3.0))
    nominal = zscore_nominal_exceedance(32, 3.0)
    plain_t = float(2.0 * stats.t.sf(3.0, df=31))
    normal = float(2.0 * stats.norm.sf(3.0))

    # measured: observed 0.0058, nominal 0.0059, plain t 0.0053, normal 0.0027.
    assert observed == pytest.approx(nominal, rel=0.1)
    assert plain_t < 0.93 * nominal  # ~11% low, ~7 sigma at this sample size
    assert normal < 0.5 * nominal  # the normal reference is off by >2x
    # ... and the gap widens as M shrinks: at M = 4 the normal reference is
    # nearly 4x too tight even at |z| > 2.
    assert zscore_nominal_exceedance(4, 2.0) > 3.5 * float(2.0 * stats.norm.sf(2.0))


def test_zscore_nominal_exceedance_converges_to_the_normal_limit() -> None:
    """Large M is the one regime where the textbook reference is right."""
    from scipy import stats

    assert zscore_nominal_exceedance(100_000, 3.0) == pytest.approx(
        float(2.0 * stats.norm.sf(3.0)), rel=1e-3
    )


def test_zscore_exceedance_reports_observed_beside_nominal() -> None:
    """The emitted mapping is self-describing: four parallel lists + metadata."""
    z = _calibrated_zscores(32, (5_000,), seed=17)

    block = zscore_exceedance(z, 32)

    assert block["thresholds"] == [2.0, 3.0]
    assert block["n_samples"] == 5_000
    assert block["df"] == 31
    assert block["null_scale"] == pytest.approx(np.sqrt(33.0 / 32.0))
    assert block["counts"] == [
        int(np.count_nonzero(np.abs(z) > 2.0)),
        int(np.count_nonzero(np.abs(z) > 3.0)),
    ]
    assert block["observed"] == pytest.approx(
        [c / 5_000 for c in block["counts"]], rel=0, abs=0
    )
    assert block["nominal"] == pytest.approx(
        [zscore_nominal_exceedance(32, t) for t in (2.0, 3.0)], rel=0, abs=0
    )
    # Every list is threshold-aligned and the same length -- a consumer zips them.
    keys = ("thresholds", "counts", "observed", "nominal", "nominal_normal")
    lengths = {len(block[key]) for key in keys}
    assert lengths == {2}


def test_zscore_exceedance_drops_non_finite_scores() -> None:
    z = np.array([0.5, np.nan, 4.0, np.inf, -3.5])

    block = zscore_exceedance(z, 8)

    assert block["n_samples"] == 3  # 0.5, 4.0, -3.5
    assert block["counts"] == [2, 2]

    empty = zscore_exceedance(np.full(4, np.nan), 8)
    assert empty["n_samples"] == 0
    assert empty["counts"] == [0, 0]
    assert all(np.isnan(v) for v in empty["observed"])
    assert all(np.isfinite(v) for v in empty["nominal"])


def test_exceedance_fraction_replaces_a_max_abs_cut_that_flags_sample_size() -> None:
    """The regression this whole reference exists for.

    ``max |z| > 3`` is a fixed cut on an ORDER STATISTIC, so its
    false-positive rate on a perfectly calibrated ensemble grows with the
    number of pooled knots -- at the routine 2-parameter/21-knot/3-window
    shape (315 pooled knots, M = 32) it fires on ~80% of calibrated
    ensembles. The exceedance fraction has a sample-size-independent
    expectation, so more knots make it SHARPER rather than louder.
    """
    n_members, n_trials = 32, 400
    rng = np.random.default_rng(4242)
    ens = rng.normal(size=(n_members, n_trials, 315))
    truth = rng.normal(size=(n_trials, 315))
    z = zscore(ens, truth)  # (n_trials, 315)

    nominal = zscore_nominal_exceedance(n_members, 3.0)

    flag_rate_315 = float(np.mean(np.max(np.abs(z), axis=1) > 3.0))
    flag_rate_21 = float(np.mean(np.max(np.abs(z[:, :21]), axis=1) > 3.0))
    # measured: 0.12 at 21 knots, 0.82 at 315 -- same ensemble, same calibration.
    assert flag_rate_315 > 0.6
    assert flag_rate_315 > 3.0 * flag_rate_21

    # The replacement, on the very same draws: pooled over all trials the
    # observed fraction lands on the nominal level, and the PER-TRIAL spread
    # about it shrinks with the sample size instead of growing.
    pooled = zscore_exceedance(z, n_members, thresholds=(3.0,))
    sigma = np.sqrt(nominal * (1.0 - nominal) / z.size)
    assert pooled["observed"][0] == pytest.approx(nominal, abs=5.0 * sigma)

    per_trial_315 = np.mean(np.abs(z) > 3.0, axis=1)
    per_trial_21 = np.mean(np.abs(z[:, :21]) > 3.0, axis=1)
    assert float(np.std(per_trial_315)) < float(np.std(per_trial_21))
    assert float(np.mean(per_trial_315)) == pytest.approx(nominal, abs=5.0 * sigma)


def test_max_abs_zscore_reference_is_the_quantile_it_claims_to_be() -> None:
    """Closed form checked against the empirical distribution of ``max |z|``."""
    n_members, n_knots = 32, 105
    z = _calibrated_zscores(n_members, (4_000, n_knots), seed=55)
    observed_max = np.max(np.abs(z), axis=1)

    for quantile in (0.5, 0.9):
        predicted = max_abs_zscore_reference(n_members, n_knots, quantile=quantile)
        empirical = float(np.quantile(observed_max, quantile))
        assert predicted == pytest.approx(empirical, rel=0.03)


def test_max_abs_zscore_reference_explains_the_fixed_cut_failure() -> None:
    """The median calibrated ``max |z|`` walks up to the fixed threshold."""
    medians = [max_abs_zscore_reference(32, n) for n in (21, 63, 105, 315)]

    assert medians == sorted(medians)  # strictly grows with the pooled count
    # 2.27 (21 knots) -> 2.96 (105) -> 3.39 (315): the fixed cut at 3.0 sits
    # above the median, on it, then BELOW it -- which is the 12% / 45% / 82%
    # false-positive progression of that cut, independently derived.
    assert 2.2 < medians[0] < 2.4
    assert 2.9 < medians[2] < 3.05  # 105 knots: the cut is a coin flip
    assert medians[-1] > 3.0  # 315 knots: the cut fires more often than not
    # A defensible one-sided ceiling at that shape is well above 3.
    assert max_abs_zscore_reference(32, 315, quantile=0.95) > 4.0


@pytest.mark.parametrize(  # type: ignore[misc]
    "kwargs",
    [
        {"n_members": 1, "n_samples": 10},
        {"n_members": 32, "n_samples": 0},
        {"n_members": 32, "n_samples": 10, "quantile": 0.0},
        {"n_members": 32, "n_samples": 10, "quantile": 1.0},
    ],
)
def test_max_abs_zscore_reference_rejects_nonsense(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        max_abs_zscore_reference(**kwargs)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])  # type: ignore[misc]
def test_zscore_nominal_exceedance_rejects_nonsense_thresholds(bad: float) -> None:
    with pytest.raises(ValueError):
        zscore_nominal_exceedance(32, bad)


def test_zscore_exceedance_rejects_a_single_member_ensemble() -> None:
    # M = 1 has no ddof=1 spread at all, so there is no null to reference.
    with pytest.raises(ValueError):
        zscore_exceedance(np.zeros(5), 1)
    with pytest.raises(ValueError):
        zscore_exceedance(np.zeros(5), 32, thresholds=())


# ---------------------------------------------------------------------------
# pit_rank / rank_histogram
# ---------------------------------------------------------------------------


def test_pit_rank_counts_members_below_the_truth() -> None:
    members = np.array([[0.0], [1.0], [2.0], [3.0]])

    assert pit_rank(members, np.array([-1.0]))[0] == 0
    assert pit_rank(members, np.array([1.5]))[0] == 2
    assert pit_rank(members, np.array([9.0]))[0] == 4


def test_pit_rank_is_uniform_for_a_calibrated_ensemble() -> None:
    rng = np.random.default_rng(101)
    n_members = 9
    n_samples = 20_000
    members = rng.normal(size=(n_members, n_samples))
    truth = rng.normal(size=n_samples)

    ranks = pit_rank(members, truth, rng=0)
    counts = np.bincount(ranks, minlength=n_members + 1)
    expected = n_samples / (n_members + 1)

    assert ranks.min() == 0 and ranks.max() == n_members
    # Multinomial std is ~sqrt(n p (1-p)) ~ 42 here; 5 sigma never flakes.
    assert np.max(np.abs(counts - expected)) < 5.0 * np.sqrt(
        expected * (1.0 - 1.0 / (n_members + 1))
    )


def test_pit_rank_tie_breaking_is_seeded_and_reproducible() -> None:
    members = np.zeros((5, 200))  # every member ties with the truth
    truth = np.zeros(200)

    first = pit_rank(members, truth, rng=0)
    again = pit_rank(members, truth, rng=0)
    other = pit_rank(members, truth, rng=1)
    default_a = pit_rank(members, truth)
    default_b = pit_rank(members, truth)

    assert np.array_equal(first, again)
    assert np.array_equal(default_a, default_b)  # default is deterministic
    assert not np.array_equal(first, other)  # but seeds actually matter
    assert first.min() >= 0 and first.max() <= 5


def test_rank_histogram_is_flat_for_a_calibrated_ensemble() -> None:
    rng = np.random.default_rng(5)
    n_members = 19  # 20 ranks -> 10 bins exactly
    members = rng.normal(size=(n_members, 20_000))
    truth = rng.normal(size=20_000)

    counts = rank_histogram(pit_rank(members, truth, rng=0), n_members, n_bins=10)

    assert counts.shape == (10,)
    assert counts.sum() == 20_000
    assert np.max(np.abs(counts - 2_000)) < 5.0 * np.sqrt(2_000 * 0.9)
    # The M above is the special case where the bins are equal; the general
    # reference is the weights, which here are all 2.
    assert rank_histogram_weights(n_members, n_bins=10).tolist() == [2] * 10


def test_rank_histogram_weights_expose_the_uneven_binning() -> None:
    """The production ensemble size does NOT divide evenly into 10 bins."""
    weights = rank_histogram_weights(32, n_bins=10)

    # 33 rank values over 10 bins: a fixed three-bin comb, +21% / -9% about a
    # flat 3.3. A consumer plotting counts against a flat line would read a
    # perfectly calibrated ensemble as structured.
    assert weights.tolist() == [4, 3, 3, 4, 3, 3, 4, 3, 3, 3]
    assert weights.sum() == 33
    # And below n_bins members some bins are unreachable, not merely uneven.
    assert rank_histogram_weights(8, n_bins=10).tolist() == [1] * 9 + [0]


def test_rank_histogram_of_a_calibrated_m32_ensemble_matches_the_weights() -> None:
    """At M = 32 calibration means matching the weights, not a flat line."""
    rng = np.random.default_rng(32)
    n_members = 32
    n_samples = 60_000
    members = rng.normal(size=(n_members, n_samples))
    truth = rng.normal(size=n_samples)

    counts = rank_histogram(pit_rank(members, truth, rng=0), n_members, n_bins=10)
    weights = rank_histogram_weights(n_members, n_bins=10)
    expected = n_samples * weights / (n_members + 1)

    assert counts.sum() == n_samples
    # Multinomial std per bin is <= sqrt(n p) ~ 85; 5 sigma never flakes.
    assert np.max(np.abs(counts - expected)) < 5.0 * np.sqrt(expected.max())
    # The same data is NOT flat against n_samples / n_bins -- the artifact this
    # reference exists to prevent is real and far outside sampling noise.
    # (measured: the tall bins sit ~1270 counts above flat, ~15 sigma)
    assert np.max(np.abs(counts - n_samples / 10)) > 10.0 * np.sqrt(expected.max())


@pytest.mark.parametrize("bad", [0, -1])  # type: ignore[misc]
def test_rank_histogram_weights_reject_nonsense_shapes(bad: int) -> None:
    with pytest.raises(ValueError):
        rank_histogram_weights(bad, n_bins=10)
    with pytest.raises(ValueError):
        rank_histogram_weights(10, n_bins=bad)


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def test_coverage_band_edges_are_member_order_statistics() -> None:
    # M = 19, alpha = 0.9 -> ceil(0.05*20) = 1 and ceil(0.95*20) = 19, i.e.
    # the band is exactly [min, max] and excludes nothing else.
    members = np.arange(19, dtype=float)[:, None]

    assert coverage_indicator(members, np.array([0.0]))[0]
    assert coverage_indicator(members, np.array([18.0]))[0]
    assert not coverage_indicator(members, np.array([18.5]))[0]
    # np.quantile interpolation would have put the upper edge at 17.1 and
    # (wrongly) called a truth of 17.5 uncovered.
    assert coverage_indicator(members, np.array([17.5]))[0]


def test_coverage_of_a_calibrated_ensemble_matches_alpha() -> None:
    rng = np.random.default_rng(2024)
    n_members = 19  # 0.05*(M+1) and 0.95*(M+1) are integers -> exactly nominal
    members = rng.normal(size=(n_members, 40_000))
    truth = rng.normal(size=40_000)

    for alpha in (0.5, 0.9):
        observed = coverage(members, truth, alpha=alpha)
        # Sampling std at n=40k is ~0.0025; 0.02 is ~8 sigma.
        assert observed == pytest.approx(alpha, abs=0.02)


@pytest.mark.parametrize(  # type: ignore[misc]
    "n_members, alpha, expected",
    [
        (32, 0.5, 16 / 33),  # band [x_(9),  x_(25)] -> 0.4848, not 0.5
        (32, 0.9, 30 / 33),  # band [x_(2),  x_(32)] -> 0.9091, not 0.9
        (8, 0.5, 4 / 9),  # band [x_(3),  x_(7)]  -> 0.4444
        (4, 0.5, 2 / 5),  # band [x_(2),  x_(4)]  -> 0.4000
        (19, 0.9, 0.9),  # 0.05*(M+1) integral -> exactly the request
    ],
)
def test_coverage_nominal_alpha_matches_the_realized_band(
    n_members: int, alpha: float, expected: float
) -> None:
    assert coverage_nominal_alpha(n_members, alpha) == pytest.approx(expected)


@pytest.mark.parametrize("alpha", [0.5, 0.9])  # type: ignore[misc]
def test_calibrated_coverage_tracks_the_nominal_level_not_the_request(
    alpha: float,
) -> None:
    """A consumer comparing against ``alpha`` reads discretization as error.

    At M = 32 the alpha_50 band's actual target is 0.4848; the 1.5-point
    shortfall is ~13 sampling sigma here, i.e. it would be reported as a real
    miscalibration by anything checking ``coverage ~ alpha``.
    """
    rng = np.random.default_rng(606)
    n_members = 32
    members = rng.normal(size=(n_members, 200_000))
    truth = rng.normal(size=200_000)

    observed = coverage(members, truth, alpha=alpha)
    nominal = coverage_nominal_alpha(n_members, alpha)
    sigma = np.sqrt(nominal * (1.0 - nominal) / truth.size)

    assert observed == pytest.approx(nominal, abs=5.0 * sigma)
    assert abs(observed - alpha) > 10.0 * sigma  # and NOT the requested level


@pytest.mark.parametrize("n_members", [2, 4, 8, 19, 32, 50])  # type: ignore[misc]
def test_coverage_nominal_alpha_is_a_gap_multiple_and_close_to_the_request(
    n_members: int,
) -> None:
    """Property: the attainable levels are the multiples of 1/(M + 1)."""
    for alpha in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
        nominal = coverage_nominal_alpha(n_members, alpha)
        gaps = nominal * (n_members + 1)
        assert gaps == pytest.approx(round(gaps))  # an integer number of gaps
        assert 0.0 <= nominal <= max_nominal_alpha(n_members)
        if alpha <= max_nominal_alpha(n_members):
            # Rounding to the grid, never further than one gap width.
            assert abs(nominal - alpha) <= 1.0 / (n_members + 1)


def test_coverage_nominal_alpha_collapses_to_zero_below_one_gap() -> None:
    """An alpha finer than the member spacing has no band at all.

    ``alpha < 1/(M + 1)`` rounds both edges onto the same order statistic, so
    the band is one member wide and its nominal coverage is 0 -- a real
    property of the (deliberately un-interpolated) edges, not a bug. Reported
    honestly so a consumer sees "unresolvable at this M" instead of dividing
    an empirical 0.0 by a requested 0.1. Both production levels are far above
    the collapse for every supported M.
    """
    rng = np.random.default_rng(808)
    members = rng.normal(size=(2, 20_000))
    truth = rng.normal(size=20_000)

    assert coverage_nominal_alpha(2, 0.1) == 0.0
    assert coverage(members, truth, alpha=0.1) == pytest.approx(0.0, abs=1e-9)
    # Unreachable for the alphas the metrics stage actually asks for.
    for n_members in (2, 3, 4, 8, 32):
        for alpha in (0.5, 0.9):
            assert coverage_nominal_alpha(n_members, alpha) > 0.0


@pytest.mark.parametrize("n_members", [2, 4, 8, 19, 32, 50])  # type: ignore[misc]
def test_max_nominal_alpha_is_the_supremum_of_the_realized_levels(
    n_members: int,
) -> None:
    """The two coverage references must never disagree about the ceiling."""
    ceiling = max_nominal_alpha(n_members)
    realized = [
        coverage_nominal_alpha(n_members, a) for a in np.linspace(0.01, 0.99, 99)
    ]

    assert max(realized) == pytest.approx(ceiling)
    assert ceiling == pytest.approx((n_members - 1) / (n_members + 1))


def test_coverage_nominal_alpha_rejects_an_out_of_range_alpha() -> None:
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            coverage_nominal_alpha(32, bad)
    with pytest.raises(ValueError):
        max_nominal_alpha(0)


def test_coverage_clamps_instead_of_failing_for_tiny_ensembles() -> None:
    rng = np.random.default_rng(13)
    members = rng.normal(size=(2, 5_000))
    truth = rng.normal(size=5_000)

    # Widest band a 2-member ensemble has is [x_(1), x_(2)] -> 1/3 coverage.
    assert coverage(members, truth, alpha=0.9) == pytest.approx(1.0 / 3.0, abs=0.03)


# ---------------------------------------------------------------------------
# spread_skill_ratio
# ---------------------------------------------------------------------------


def test_spread_skill_ratio_calibrates_an_exchangeable_ensemble() -> None:
    rng = np.random.default_rng(123)
    n_members = 10
    members = rng.normal(size=(n_members, 100_000))
    truth = rng.normal(size=members.shape[1])
    variances = members.var(axis=0, ddof=1)
    sq_errors = (members.mean(axis=0) - truth) ** 2

    assert spread_skill_ratio(variances, sq_errors, n_members) == pytest.approx(
        1.0, abs=1e-2
    )


@pytest.mark.parametrize("n_members", [2, 5, 50])  # type: ignore[misc]
def test_figspec_spread_skill_squares_both_of_its_series(n_members: int) -> None:
    """``figspec.spread_skill`` now delegates here, so what it owns is the squaring.

    Comparing the two functions is tautological once one calls the other; the
    real contract is that the adapter's std/RMSE vocabulary reaches the same
    closed form. The series are deliberately non-constant, so squaring only one
    of them -- or neither -- gives a different answer:
    ``mean([1, 3]**2) = 5`` but ``mean([1, 3])**2 = 4``.
    """
    spread = np.array([1.0, 3.0])  # standard deviations
    error = np.array([2.0, 4.0])  # errors
    expected = np.sqrt((n_members + 1) / n_members) * np.sqrt(5.0 / 10.0)

    assert spread_skill(spread, error, n_members) == pytest.approx(expected)
    # ... and the un-squared reading, which the adapter must not produce.
    assert spread_skill(spread, error, n_members) != pytest.approx(
        np.sqrt((n_members + 1) / n_members) * np.sqrt(2.0 / 3.0)
    )


def test_spread_skill_ratio_treats_an_infinite_spread_as_masked() -> None:
    """``inf`` spread is undefined, not "infinitely over-dispersed".

    The all-non-finite short-circuit keys on ``isfinite``, so it catches ``inf``
    as well as ``nan``. This is the shared contract now that figspec delegates
    here -- its phase-0 body returned ``+inf`` for this input.
    """
    assert np.isnan(spread_skill_ratio(np.full(4, np.inf), np.ones(4), 8))
    # A single inf among finite entries does not hit that short-circuit -- it
    # rides the nanmean through to a non-finite ratio. Either way the caller
    # sees "not a number it can plot"; what must never happen is a finite,
    # plausible-looking ratio computed from a poisoned mean.
    mixed = np.array([1.0, np.inf, 2.0, np.nan])
    assert not np.isfinite(spread_skill_ratio(mixed, np.ones(4), 8))


def test_spread_skill_ratio_is_nan_for_a_zero_error_norm() -> None:
    assert np.isnan(spread_skill_ratio(np.ones(4), np.zeros(4), 5))
    assert np.isnan(spread_skill_ratio(np.ones(4), np.full(4, np.nan), 5))


# ---------------------------------------------------------------------------
# Wasserstein family
# ---------------------------------------------------------------------------


def _ar1(n_samples: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) with unit marginal variance: a series with a correlation time."""
    noise = rng.normal(scale=np.sqrt(1.0 - phi**2), size=n_samples)
    series = np.empty(n_samples)
    series[0] = rng.normal()
    for i in range(1, n_samples):
        series[i] = phi * series[i - 1] + noise[i]
    return series


def test_wasserstein_over_sigma_is_the_scipy_distance_in_truth_sigmas() -> None:
    """The definition, against a direct scipy call on differently sized sets."""
    from scipy import stats

    rng = np.random.default_rng(41)
    samples = rng.normal(loc=0.4, size=57)
    truth = rng.normal(scale=1.7, size=33)  # deliberately a different length

    observed = wasserstein_over_sigma(samples, truth)

    assert observed == pytest.approx(
        float(stats.wasserstein_distance(samples, truth)) / float(np.std(truth, ddof=1))
    )
    # The normalization is the TRUTH spread, not the sample or pooled spread:
    # the truth is the one thing shared by every member being scored, so it is
    # the only divisor that leaves member scores comparable with each other.
    assert observed != pytest.approx(
        float(stats.wasserstein_distance(samples, truth))
        / float(np.std(samples, ddof=1))
    )


def test_wasserstein_over_sigma_is_zero_for_the_same_sample_set() -> None:
    rng = np.random.default_rng(42)
    truth = rng.normal(size=40)

    assert wasserstein_over_sigma(truth, truth) == pytest.approx(0.0, abs=1e-12)
    # It compares distributions, not time series, so a shuffled copy is the
    # same input -- which is exactly why it can score an ensemble window
    # against a truth window sampled at a different cadence.
    assert wasserstein_over_sigma(rng.permutation(truth), truth) == pytest.approx(
        0.0, abs=1e-12
    )


def test_wasserstein_self_floor_is_small_beside_a_real_distribution_shift() -> None:
    """Zero is the wrong reference; the floor is small enough to be a useful one.

    Two independent draws from the *same* distribution sit a finite distance
    apart at finite ``n``, so comparing a raw distance against 0 calls every
    model a failure at these window lengths. The floor has to sit well below a
    genuine mismatch to be worth dividing by, and it does.
    """
    rng = np.random.default_rng(43)
    floors, perfect, shifted = [], [], []
    for _ in range(200):
        truth = rng.normal(size=36)
        floors.append(wasserstein_self_floor(truth))
        perfect.append(wasserstein_over_sigma(rng.normal(size=36), truth))
        shifted.append(wasserstein_over_sigma(rng.normal(loc=1.0, size=36), truth))

    # measured medians: floor 0.36, perfect model 0.27, 1-sigma shift 0.99.
    assert np.median(floors) < 0.5 * np.median(shifted)
    assert np.median(perfect) < np.median(floors)
    # And the floor is emphatically not negligible at a 36-frame window: a raw
    # distance of 0.3 there is indistinguishable from a perfect model.
    assert np.median(floors) > 0.25


def test_wasserstein_over_floor_separates_a_perfect_model_from_a_biased_one() -> None:
    """The headline ratio sits near 1 for a perfect model and far above for a bad one."""
    rng = np.random.default_rng(45)
    perfect, shifted = [], []
    for _ in range(200):
        truth = rng.normal(size=36)
        floor = wasserstein_self_floor(truth)
        perfect.append(
            wasserstein_over_floor(
                wasserstein_over_sigma(rng.normal(size=36), truth), floor
            )
        )
        shifted.append(
            wasserstein_over_floor(
                wasserstein_over_sigma(rng.normal(loc=1.0, size=36), truth), floor
            )
        )

    # measured medians: 0.70 perfect, 2.61 for a 1-sigma mean shift. The
    # perfect median sits below 1 rather than on it -- the floor's halves have
    # n/2 samples against the compared sets' n, the documented conservatism.
    assert 0.4 < float(np.median(perfect)) < 1.2
    assert float(np.median(shifted)) > 2.0


def test_wasserstein_self_floor_splits_contiguously_so_it_sees_the_correlation() -> (
    None
):
    """The one design choice in the floor, and the naive alternative it rejects.

    A random split interleaves the halves through the same slow excursions, so
    both inherit the series' full range and agree far better than two genuinely
    independent stretches of the flow would. It reports the white-noise floor
    whatever the series does -- on a correlated probe that understates the
    floor severalfold, and a perfectly good model then reads as several times
    worse than the floor.
    """
    rng = np.random.default_rng(44)
    contiguous, shuffled, white = [], [], []
    for _ in range(100):
        series = _ar1(200, 0.9, rng)
        contiguous.append(wasserstein_self_floor(series))
        shuffled.append(wasserstein_self_floor(rng.permutation(series)))
        white.append(wasserstein_self_floor(_ar1(200, 0.0, rng)))

    # measured medians: 0.44 contiguous, 0.17 shuffled -- a 2.7x understatement.
    assert float(np.median(contiguous)) > 2.0 * float(np.median(shuffled))
    # ... and the shuffled value is blind to phi: it is the white-noise floor.
    assert float(np.median(shuffled)) == pytest.approx(
        float(np.median(white)), rel=0.25
    )


def test_wasserstein_self_floor_drops_the_middle_sample_of_an_odd_count() -> None:
    """Equal halves, so both carry the same sample-size dependence.

    Keeping the odd sample would make the floor depend on which half it landed
    in, which is a coin flip the caller cannot see. Here the dropped sample is
    a huge outlier, so the two conventions are orders of magnitude apart.
    """
    from scipy import stats

    series = np.array([0.0, 1.0, 100.0, 4.0, 5.0])
    expected = float(stats.wasserstein_distance([0.0, 1.0], [4.0, 5.0])) / float(
        np.std(series, ddof=1)
    )

    assert wasserstein_self_floor(series) == pytest.approx(expected)
    # The outlier still sets the normalizing sigma -- only the distance drops it.
    assert wasserstein_self_floor(series) < 0.2


def test_wasserstein_family_returns_nan_rather_than_raising_when_undefined() -> None:
    """Degenerate sample sets are a supported input, not an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a RuntimeWarning would fail the test
        # Fewer than two finite samples on either side: no distribution.
        assert np.isnan(wasserstein_over_sigma(np.array([1.0]), np.arange(5.0)))
        assert np.isnan(wasserstein_over_sigma(np.arange(5.0), np.array([1.0])))
        assert np.isnan(wasserstein_over_sigma(np.full(4, np.nan), np.arange(5.0)))
        # Zero truth spread: the normalization does not exist.
        assert np.isnan(wasserstein_over_sigma(np.arange(5.0), np.full(5, 2.0)))
        assert np.isnan(wasserstein_self_floor(np.full(8, 2.0)))
        # The floor needs FOUR samples, not two -- a half of one sample is not
        # a distribution, so the three-frame smoke window is undefined here.
        assert np.isnan(wasserstein_self_floor(np.arange(3.0)))
        assert np.isfinite(wasserstein_self_floor(np.arange(4.0)))
        # Non-finite entries are dropped before the split, not after.
        assert wasserstein_self_floor(
            np.array([0.0, 1.0, np.nan, 4.0, 5.0])
        ) == pytest.approx(wasserstein_self_floor(np.array([0.0, 1.0, 4.0, 5.0])))


def test_wasserstein_is_nan_when_the_truth_spread_is_only_float_rounding() -> None:
    """``std > 0`` is the wrong guard; ``std`` above float64 noise is the right one.

    A physically constant sensor is not bitwise constant once it has been
    through ``sqrt(sum_c u_c**2)``: the magnitude of a constant vector comes
    back with a spread of order 1e-16. Dividing by that turns a zero distance
    into ~1e15, which is *finite* -- so it survives the metrics layer's
    finite-or-null filter and is written to the summary as a headline number.
    """
    truth = 1.0 + np.arange(20, dtype=float) * 1e-16
    assert 0.0 < float(np.std(truth, ddof=1)) < 1e-15  # not exactly constant

    assert np.isnan(wasserstein_over_sigma(truth, truth))
    assert np.isnan(wasserstein_self_floor(truth))
    # The guard is relative, so a genuinely tiny but real signal survives it --
    # an absolute epsilon would silently discard a low-speed sensor.
    assert np.isfinite(wasserstein_self_floor(1e-9 * np.arange(20, dtype=float)))


def test_wasserstein_over_floor_clamps_a_zero_floor_instead_of_dividing_by_it() -> None:
    """A zero floor is reachable, and neither ``inf`` nor ``nan`` is the answer.

    An exactly periodic probe signal whose period divides the record in half
    has bitwise-identical halves and a perfectly good spread. ``write_yaml``
    cannot emit the ``inf`` that a bare division gives, and a ``nan`` would
    silently drop the sensor from the median over sensors; a large finite
    number reads as "the perfect-model distance is unresolvable here".
    """
    periodic = np.array([1.0, 2.0] * 4)
    assert wasserstein_self_floor(periodic) == 0.0
    assert float(np.std(periodic, ddof=1)) > 0.0  # the spread is fine; the floor is not

    assert wasserstein_over_floor(0.6, 0.0) == pytest.approx(0.6e12)
    assert wasserstein_over_floor(0.6, 0.3) == pytest.approx(2.0)
    assert wasserstein_over_floor(0.0, 0.3) == 0.0


def test_wasserstein_over_floor_is_nan_on_nan_but_raises_on_a_negative_distance() -> (
    None
):
    # nan is how both producers report "undefined", so it must pass through.
    assert np.isnan(wasserstein_over_floor(float("nan"), 0.3))
    assert np.isnan(wasserstein_over_floor(0.3, float("nan")))
    assert np.isnan(wasserstein_over_floor(float("inf"), 0.3))
    # A negative distance is a caller bug, not a degenerate sample.
    for w1, floor in [(-0.1, 0.3), (0.3, -0.1)]:
        with pytest.raises(ValueError):
            wasserstein_over_floor(w1, floor)


def test_pooling_an_ensemble_can_only_lower_the_wasserstein_distance() -> None:
    """W1 is convex in its first argument, so the two pooling conventions differ.

    ``W1(pooled, truth) <= mean_m W1(member_m, truth)``, always -- the naive
    reading (that pooling spreads the samples out and so inflates the distance)
    is backwards. The gap is the ensemble spread, so both numbers are worth
    reporting: members straddling the truth cancel when pooled, members sharing
    a bias do not.
    """
    rng = np.random.default_rng(46)
    truth = rng.normal(size=400)
    straddling = [rng.normal(loc=mu, size=400) for mu in (-1.0, 1.0)]
    biased = [rng.normal(loc=1.0, size=400) for _ in range(2)]

    ratios = []
    for members in (straddling, biased):
        pooled = wasserstein_over_sigma(np.concatenate(members), truth)
        per_member = float(np.mean([wasserstein_over_sigma(m, truth) for m in members]))
        assert pooled <= per_member + 1e-12
        ratios.append(pooled / per_member)

    # measured: 0.51 straddling, 1.00 sharing a bias.
    assert ratios[0] < 0.7
    assert ratios[1] > 0.95


# ---------------------------------------------------------------------------
# Degenerate shapes: the two-member smoke ensemble must work everywhere
# ---------------------------------------------------------------------------


def test_two_member_smoke_ensemble_works_for_every_score() -> None:
    rng = np.random.default_rng(31)
    members = rng.normal(size=(2, 6))
    truth = rng.normal(size=6)

    crps = fair_crps(members, truth)
    energy = fair_energy_score(members[..., None], truth[..., None])
    z = zscore(members, truth)
    ranks = pit_rank(members, truth, rng=0)
    hist = rank_histogram(ranks, 2, n_bins=3)

    assert np.all(np.isfinite(crps)) and np.all(crps >= 0.0)
    assert energy == pytest.approx(crps)
    assert np.all(np.isfinite(z))  # two members still admit a ddof=1 spread
    assert set(np.unique(ranks)) <= {0, 1, 2}
    assert hist.sum() == 6
    assert np.isfinite(coverage(members, truth, alpha=0.9))
    assert np.isfinite(
        spread_skill_ratio(
            members.var(axis=0, ddof=1), (members.mean(0) - truth) ** 2, 2
        )
    )

    # The references degrade honestly rather than raising: at M = 2 the only
    # band is [x_(1), x_(2)] (1/3 of the three gaps) whatever alpha asks for,
    # and the z-score null is a 1-df t, whose tails are enormous.
    assert coverage_nominal_alpha(2, 0.9) == pytest.approx(1.0 / 3.0)
    assert coverage_nominal_alpha(2, 0.5) == pytest.approx(1.0 / 3.0)
    assert max_nominal_alpha(2) == pytest.approx(1.0 / 3.0)
    block = zscore_exceedance(z, 2)
    assert block["df"] == 1
    assert block["nominal"][1] > 0.15
    assert np.isfinite(max_abs_zscore_reference(2, 6))
