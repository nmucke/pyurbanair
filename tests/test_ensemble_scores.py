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
    crpss,
    fair_crps,
    fair_energy_score,
    pit_rank,
    rank_histogram,
    rank_histogram_weights,
    spread_skill_ratio,
    zscore,
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


def test_spread_skill_ratio_agrees_with_figspec_spread_skill() -> None:
    rng = np.random.default_rng(77)
    for n_members in (2, 5, 50):
        spread = np.abs(rng.normal(size=200)) + 0.1
        error = rng.normal(size=200)

        assert spread_skill_ratio(spread**2, error**2, n_members) == pytest.approx(
            spread_skill(spread, error, n_members), rel=0, abs=0
        )


def test_spread_skill_ratio_is_nan_for_a_zero_error_norm() -> None:
    assert np.isnan(spread_skill_ratio(np.ones(4), np.zeros(4), 5))
    assert np.isnan(spread_skill_ratio(np.ones(4), np.full(4, np.nan), 5))


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
