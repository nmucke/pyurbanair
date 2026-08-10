"""The dtype contract of ``evaluation.scores.crps_ensemble``, and the band
indicator's inclusion rule.

WP0.2 merged two CRPS implementations that looked identical but were not:
``da_metrics.per_knot_crps`` scored in whatever dtype it was handed, while
``plotting._crps_ensemble`` upcast to float64 first. Parameter artifacts are
float32 on disk, and the pairwise term differs between the two at the ~1e-7
level, so the merge had to preserve *both* behaviours: the function itself no
longer casts, and the one caller that relied on the upcast
(:func:`compute_parameter_metrics`) does it explicitly.

Nothing else pins that split. Without these tests, "tidying up" by moving the
cast back inside the function — or deleting it at the call site — silently
shifts every emitted CRPS number, and no other test in the suite notices.

``per_knot_in_band`` is here for the same reason: it is the coverage number in
the parameter diagnostics (``scripts/_common.py`` writes it to the per-knot
table and plots it), it is one boolean expression, and its *closed* interval is
the whole of its content — a truth sitting exactly on a band edge is the only
case that tells a closed rule from an open one, and it is the case a collapsed
or pinned ensemble produces constantly.
"""

import numpy as np
import xarray
from evaluation.scores import compute_parameter_metrics, crps_ensemble, per_knot_in_band


def _param_datasets(
    dtype: str,
) -> tuple[xarray.Dataset, xarray.Dataset]:
    """A tiny (ensemble, time) posterior and its matching truth, in ``dtype``."""
    rng = np.random.default_rng(0)
    time = np.array([0.0, 1.0, 2.0, 3.0])
    members = rng.standard_normal((6, time.size)).astype(dtype)
    truth = rng.standard_normal(time.size).astype(dtype)

    posterior = xarray.Dataset(
        {"inflow_angle": (("ensemble", "time"), members)},
        coords={"time": time, "ensemble": np.arange(members.shape[0])},
    )
    true = xarray.Dataset(
        {"inflow_angle": (("time",), truth)},
        coords={"time": time},
    )
    return posterior, true


def test_crps_ensemble_does_not_upcast() -> None:
    # The merged function inherited per_knot_crps's behaviour: dtype is the
    # caller's business. scripts/_common.py relies on this -- it passes raw
    # float32 parameter history and its metrics.csv digits depend on it.
    ens = np.random.default_rng(1).standard_normal((8, 5)).astype(np.float32)
    truth = np.random.default_rng(2).standard_normal(5).astype(np.float32)

    out = crps_ensemble(ens, truth)

    assert out.dtype == np.float32, (
        "crps_ensemble must not cast; float64 here would silently shift "
        "metrics.csv for every float32 parameter run"
    )


def test_crps_ensemble_float32_and_float64_actually_differ() -> None:
    # Guards the premise of the whole split: if the two accumulation dtypes ever
    # agree bitwise, the explicit cast at the caller is pointless and this file
    # can go.
    #
    # Both calls must see the SAME numbers -- only the dtype they are summed in
    # may vary. Rounding the truth to float32 for one call and not the other
    # would make the assertion pass on the data difference alone, which is
    # exactly the situation compute_parameter_metrics is NOT in: it upcasts
    # members that are already float32-valued.
    ens = np.random.default_rng(3).standard_normal((16, 12)).astype(np.float32)
    truth = np.random.default_rng(4).standard_normal(12).astype(np.float32)

    in_float32 = crps_ensemble(ens, truth)
    in_float64 = crps_ensemble(ens.astype(float), truth.astype(float))

    assert not np.array_equal(in_float32, in_float64)


def test_compute_parameter_metrics_scores_crps_in_float64() -> None:
    # The compensating half: this caller upcast implicitly before the merge
    # (via _crps_ensemble), so it must upcast explicitly after it.
    posterior, true = _param_datasets("float32")

    metrics = compute_parameter_metrics(posterior, true)
    crps = metrics["inflow_angle"]["crps"]

    # Deliberately no dtype assertion: the truth comes back from np.interp as
    # float64, so the first CRPS term is float64 with or without the cast and
    # only the pairwise term moves. The array comparison below is the only
    # thing that discriminates -- do not "simplify" it away.
    members = np.asarray(posterior["inflow_angle"].transpose("ensemble", "time").values)
    truth = np.asarray(true["inflow_angle"].values, dtype=float)
    expected = crps_ensemble(np.asarray(members, dtype=float), truth)

    np.testing.assert_array_equal(crps, expected)


# ---------------------------------------------------------------------------
# per_knot_in_band -- the coverage indicator
# ---------------------------------------------------------------------------


def test_per_knot_in_band_flags_the_central_alpha_interval() -> None:
    # 21 members 0..20 put np.quantile's linear rule on exact members: the 5 %
    # level is index 0.05*20 = 1 and the 95 % level is index 19, so the default
    # alpha = 0.9 band is exactly [1, 19] and every knot below is an integer.
    ens = np.repeat(np.arange(21.0)[:, None], 4, axis=1)
    truth = np.array([10.0, 0.0, 20.0, 19.0])

    assert per_knot_in_band(ens, truth).tolist() == [True, False, False, True]


def test_per_knot_in_band_includes_a_truth_exactly_on_either_edge() -> None:
    # The band is CLOSED. A truth landing on the edge is not a curiosity: with
    # a small or contracted ensemble the quantiles sit on member values, and a
    # truth that a well-calibrated filter has pulled onto one is precisely the
    # outcome the coverage number is meant to reward. An open interval scores
    # those knots as misses and reports a systematically under-covered
    # ensemble -- which reads as a filter that is over-confident.
    ens = np.repeat(np.arange(21.0)[:, None], 2, axis=1)
    on_edges = np.array([1.0, 19.0])  # exactly the 5 % and 95 % quantiles

    assert np.quantile(ens[:, 0], 0.05) == 1.0  # the fixture is on the edge
    assert np.quantile(ens[:, 0], 0.95) == 19.0
    assert per_knot_in_band(ens, on_edges).tolist() == [True, True]


def test_per_knot_in_band_covers_a_collapsed_ensemble_at_its_own_value() -> None:
    # The degenerate edge case, and the common one: a pinned or fully collapsed
    # parameter has lo == hi == the members' shared value. Under a closed rule
    # a truth at that value is covered and one anywhere else is not; under an
    # open rule the knot can never be covered at all, so a pinned parameter
    # that is exactly right reports 0 % coverage.
    ens = np.full((5, 3), 2.5)
    truth = np.array([2.5, 2.5 + 1e-9, 2.5 - 1e-9])

    assert per_knot_in_band(ens, truth).tolist() == [True, False, False]


def test_per_knot_in_band_widens_with_alpha() -> None:
    # ``alpha`` is the interval width, not a threshold: the same knot moves from
    # outside a narrow band to inside a wide one.
    ens = np.repeat(np.arange(21.0)[:, None], 1, axis=1)
    truth = np.array([2.0])

    assert per_knot_in_band(ens, truth, alpha=0.5).tolist() == [False]  # [5, 15]
    assert per_knot_in_band(ens, truth, alpha=0.9).tolist() == [True]  # [1, 19]
