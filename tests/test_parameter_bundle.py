"""Closed-form and property tests for the WP1.1 parameter calibration bundle.

Everything here runs on synthetic ``xarray`` Datasets shaped like the real
artifacts (``posterior_params.nc`` / ``prior_params.nc`` / ``true_params.nc``),
with fixed seeds. Assertions are analytic values, constructed-covariance
answers or statistical properties of a deliberately calibrated ensemble -- no
snapshots -- so they survive a reimplementation.

Two shapes get explicit coverage because they are the ones that bite:

* ``M = 2``, the smoke shape the suite runs everywhere, where a ddof=1 spread,
  an order-statistic band and a 10-bin PIT are all artifacts of the ensemble
  size and must come out ``null`` rather than raising or fabricating;
* a truth sampled on a coarser knot grid than the posterior (19 vs 21 on the
  reference run), where the ``np.interp`` alignment is load-bearing.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import xarray
from omegaconf import DictConfig, OmegaConf

from pyurbanair.utils.ensemble_scores import (
    coverage_nominal_alpha,
    max_nominal_alpha,
    zscore_nominal_exceedance,
)
from scripts.esmda._esmda_common import (
    JOINT_CORR_MAX_K,
    MIN_MEMBERS_CALIBRATION,
    OVERCONFIDENT_MULTIPLIER,
    PIT_BINS,
    _knot_correlation_config,
    joint_parameter_directions,
    parameter_bundle_summary,
    parameter_vector_labels,
)
from scripts.esmda.compute_esmda_metrics import (
    _flatten_parameter_members,
    _window_block_flat,
)

SECONDS_PER_KNOT = 30.0
CORRELATION_LENGTH = 200.0


# ---------------------------------------------------------------------------
# Synthetic run artifacts
# ---------------------------------------------------------------------------


def _cfg(
    correlation_length: float | None = CORRELATION_LENGTH,
    seconds_per_knot: float | None = SECONDS_PER_KNOT,
) -> DictConfig:
    """A saved-config stand-in carrying the GP knot spacing, as runs do."""
    block: dict[str, object] = {}
    if correlation_length is not None:
        block["correlation_length"] = correlation_length
    if seconds_per_knot is not None:
        block["seconds_per_knot"] = seconds_per_knot
    return OmegaConf.create({"prior_params": block})


def _dynamic_dataset(
    members: np.ndarray, name: str = "inflow_angle", n_knots: int | None = None
) -> xarray.Dataset:
    """``(n_members, n_knots)`` members as a time-coordinate-carrying Dataset."""
    members = np.asarray(members, dtype=float)
    n_knots = members.shape[1] if n_knots is None else n_knots
    return xarray.Dataset(
        {name: (("ensemble", "time"), members)},
        coords={
            "ensemble": np.arange(members.shape[0]),
            "time": np.arange(n_knots) * SECONDS_PER_KNOT,
        },
    )


def _truth_dataset(
    values: np.ndarray, name: str = "inflow_angle", times: np.ndarray | None = None
) -> xarray.Dataset:
    values = np.asarray(values, dtype=float)
    times = np.arange(values.size) * SECONDS_PER_KNOT if times is None else times
    return xarray.Dataset(
        {name: (("time", "ensemble"), values[:, None])},
        coords={"time": np.asarray(times, dtype=float), "ensemble": [0]},
    )


def _bundle(
    posterior: xarray.Dataset,
    truth: xarray.Dataset,
    prior: xarray.Dataset | None = None,
    cfg: DictConfig | None = None,
    num_windows: int | None = 3,
    base: dict | None = None,
) -> dict:
    """Run the bundle the way ``compute_esmda_metrics`` does."""
    # `_esmda_common` is an untyped legacy module, so the result arrives as
    # `Any`; bind it to a typed name rather than returning `Any` from a typed
    # helper (mypy `no-any-return`).
    summary: dict = parameter_bundle_summary(
        {} if base is None else base,
        posterior,
        truth,
        prior,
        posterior_flat=_flatten_parameter_members(posterior),
        prior_flat=None if prior is None else _flatten_parameter_members(prior),
        # Sliced exactly as `compute_metrics` does, so the unit tests drive the
        # same code path as the pipeline rather than a shortcut past it.
        final_posterior_window_flat=_window_block_flat(posterior, num_windows, -1),
        initial_prior_window_flat=(
            None if prior is None else _window_block_flat(prior, num_windows, 0)
        ),
        cfg=_cfg() if cfg is None else cfg,
        num_windows=num_windows,
    )
    return summary


def _calibrated(
    n_members: int,
    n_knots: int,
    seed: int = 0,
    spread: float = 1.0,
    truth_spread: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """An ensemble whose members and truth are draws from the same distribution.

    Truth drawn from the ensemble's own distribution is exactly the exchangeable
    case: PIT ranks are uniform and coverage matches its nominal level.
    ``truth_spread`` breaks that on purpose (overconfidence when it exceeds
    ``spread``).
    """
    rng = np.random.default_rng(seed)
    centre = rng.normal(0.0, 5.0, size=n_knots)
    members = centre[None, :] + rng.normal(0.0, spread, size=(n_members, n_knots))
    truth = centre + rng.normal(
        0.0, spread if truth_spread is None else truth_spread, size=n_knots
    )
    return members, truth


# ---------------------------------------------------------------------------
# Calibration: a calibrated ensemble scores as calibrated
# ---------------------------------------------------------------------------


def test_calibrated_ensemble_has_nominal_coverage_and_flat_pit() -> None:
    """Coverage ~ alpha and a near-uniform PIT for an exchangeable ensemble."""
    members, truth = _calibrated(n_members=39, n_knots=4000, seed=1)
    summary = _bundle(_dynamic_dataset(members), _truth_dataset(truth))
    entry = summary["inflow_angle"]

    # 4000 knots: the sampling error on a coverage fraction is ~0.008, so 0.03
    # is loose enough to never flake and tight enough to catch a wrong band.
    assert entry["coverage"]["alpha_50"] == pytest.approx(0.5, abs=0.03)
    assert entry["coverage"]["alpha_90"] == pytest.approx(0.9, abs=0.03)
    # M = 39 -> 40 rank gaps, exactly 4 per bin, so the histogram is unbiased.
    counts = np.asarray(entry["pit_counts"])
    assert counts.sum() == 4000
    assert np.max(np.abs(counts - 400.0)) < 100.0

    # A calibrated ensemble's z-scores are standard normal.
    assert entry["zscore"]["mean"] == pytest.approx(0.0, abs=0.1)
    assert entry["zscore"]["std"] == pytest.approx(1.0, abs=0.1)


def test_overconfident_ensemble_is_flagged() -> None:
    """A truth drawn far wider than the ensemble trips the flag and empties the band."""
    members, truth = _calibrated(
        n_members=39, n_knots=500, seed=2, spread=1.0, truth_spread=6.0
    )
    entry = _bundle(_dynamic_dataset(members), _truth_dataset(truth))["inflow_angle"]

    assert entry["zscore"]["max_abs"] > 3.0
    assert entry["zscore"]["overconfident"] is True
    assert entry["zscore"]["std"] > 3.0
    assert entry["coverage"]["alpha_90"] < 0.5
    # Under-dispersion piles the truth into the outermost rank bins.
    counts = np.asarray(entry["pit_counts"])
    assert counts[0] + counts[-1] > 0.5 * counts.sum()


@pytest.mark.parametrize(  # type: ignore[misc]
    ("offset_in_sigma", "expected_flag"), [(1.0, False), (4.0, True)]
)
def test_zscore_reads_off_a_constructed_offset(
    offset_in_sigma: float, expected_flag: bool
) -> None:
    """Truth placed a known number of spreads from the mean reports that number.

    Constructed rather than sampled: the flag's threshold is exercised from both
    sides with no dependence on which draw a seed happens to produce.
    """
    rng = np.random.default_rng(3)
    members = rng.normal(0.0, 2.0, size=(20, 8))
    truth = members.mean(axis=0) + offset_in_sigma * members.std(axis=0, ddof=1)

    entry = _bundle(_dynamic_dataset(members), _truth_dataset(truth))["inflow_angle"]

    assert entry["zscore"]["mean"] == pytest.approx(offset_in_sigma)
    assert entry["zscore"]["max_abs"] == pytest.approx(offset_in_sigma)
    assert entry["zscore"]["std"] == pytest.approx(0.0, abs=1e-9)
    assert entry["zscore"]["overconfident"] is expected_flag


def test_overconfident_flag_does_not_fire_on_a_bigger_calibrated_ensemble() -> None:
    """The finding-1 regression: the flag must measure calibration, not sample size.

    A fixed cut on ``max |z|`` flags a *perfectly calibrated* M = 32 ensemble
    ~12% of the time at 21 pooled knots and ~85% at 315, because the maximum of
    ``n`` draws grows with ``n``. The routine 2-parameter / 21-knot / 3-window
    shape lands in the worst part of that curve, so the old flag was very nearly
    a constant `true` on real runs.

    Here the SAME calibrated process is scored at four pooling depths. A
    sample-size-driven flag lights up as the knot count grows; an exceedance
    *fraction* has a sample-size-independent expectation, so it must not.
    """
    for n_knots in (21, 63, 105, 315):
        members, truth = _calibrated(n_members=32, n_knots=n_knots, seed=n_knots)
        entry = _bundle(_dynamic_dataset(members), _truth_dataset(truth))[
            "inflow_angle"
        ]
        assert entry["zscore"]["overconfident"] is False, n_knots
        # `max_abs` itself is kept (it is informative) but must arrive with the
        # reference that makes it readable -- which grows with the knot count.
        assert entry["zscore"]["max_abs_calibrated_median"] > 0.0

    # And the reference does grow, which is exactly why a fixed cut cannot work.
    references = [
        _bundle(
            _dynamic_dataset(_calibrated(32, n, seed=n)[0]),
            _truth_dataset(_calibrated(32, n, seed=n)[1]),
        )["inflow_angle"]["zscore"]["max_abs_calibrated_median"]
        for n in (21, 315)
    ]
    assert references[0] < references[1]


def test_overconfident_flag_still_fires_on_real_underdispersion() -> None:
    """Sample-size-robust must not mean blind: a too-narrow ensemble still trips."""
    members, truth = _calibrated(
        n_members=32, n_knots=105, seed=203, spread=1.0, truth_spread=2.0
    )
    entry = _bundle(_dynamic_dataset(members), _truth_dataset(truth))["inflow_angle"]

    assert entry["zscore"]["overconfident"] is True
    # The verdict is reproducible from the emitted keys alone.
    exceedance = entry["zscore"]["exceedance"]
    assert (
        exceedance["observed"][0] > OVERCONFIDENT_MULTIPLIER * exceedance["nominal"][0]
    )


def test_zscore_exceedance_carries_the_scaled_t_reference_not_a_normal_one() -> None:
    """The null is ``sqrt((M+1)/M) * t(M-1)``; a normal table is ~2x too tight.

    WP1.4 plots `observed` against `nominal` from this block, so the reference
    has to travel with the numbers -- and it must be the finite-ensemble one:
    both the mean and the spread come from the same M members.
    """
    members, truth = _calibrated(n_members=32, n_knots=200, seed=204)
    zscore_block = _bundle(_dynamic_dataset(members), _truth_dataset(truth))[
        "inflow_angle"
    ]["zscore"]
    exceedance = zscore_block["exceedance"]

    assert exceedance["df"] == 31
    assert exceedance["null_scale"] == pytest.approx(np.sqrt(33.0 / 32.0))
    assert exceedance["n_samples"] == 200
    for threshold, nominal in zip(exceedance["thresholds"], exceedance["nominal"]):
        assert nominal == pytest.approx(zscore_nominal_exceedance(32, threshold))
    # At M = 32 the 3-sigma tail is 0.59%, 2.2x the normal table's 0.27%.
    assert exceedance["nominal"][1] > 2.0 * exceedance["nominal_normal"][1]
    # The rule string names the keys it is computed from, so the boolean is
    # reproducible from the summary without reading this module.
    assert "exceedance.observed[0]" in zscore_block["overconfident_rule"]
    assert "exceedance.nominal[0]" in zscore_block["overconfident_rule"]


def test_coverage_reports_the_realizable_level_next_to_the_empirical_one() -> None:
    """Finding 2: comparing an empirical fraction against the REQUESTED alpha lies.

    At M = 32 an ``alpha = 0.5`` request is the band ``[x_(9), x_(25)]``, whose
    nominal level is 16/33 = 0.4848. A calibrated ensemble scores that, not 0.5,
    so a consumer holding the number up against 0.5 sees ~13 sampling sigma of
    pure discretization.
    """
    members, truth = _calibrated(n_members=32, n_knots=4000, seed=205)
    block = _bundle(_dynamic_dataset(members), _truth_dataset(truth))["inflow_angle"][
        "coverage"
    ]

    assert block["nominal_alpha_50"] == pytest.approx(16.0 / 33.0)
    assert block["nominal_alpha_90"] == pytest.approx(30.0 / 33.0)
    assert block["nominal_alpha_50"] != pytest.approx(0.5)
    # Both come from `ensemble_scores`, so the metrics layer holds no second
    # copy of the band-edge convention that could drift from `_band_indices`.
    assert block["nominal_alpha_50"] == coverage_nominal_alpha(32, 0.5)
    assert block["max_nominal_alpha"] == max_nominal_alpha(32)
    # The empirical fractions track the realizable levels, not the requested
    # ones (4000 knots: sampling error ~0.008).
    assert block["alpha_50"] == pytest.approx(block["nominal_alpha_50"], abs=0.03)
    assert block["alpha_90"] == pytest.approx(block["nominal_alpha_90"], abs=0.03)


def test_sampling_caveat_sits_on_the_parameter_entry() -> None:
    """`zscore` and `coverage` pool over the same correlated knots `pit` does.

    The effective-sample-size caveat used to live only inside `pit`, which reads
    as though PIT alone needed it. It qualifies all three, so it is hoisted --
    and mirrored under `pit` from the same computation, so the two cannot drift.
    """
    members, truth = _calibrated(n_members=10, n_knots=20, seed=206)
    entry = _bundle(_dynamic_dataset(members), _truth_dataset(truth))["inflow_angle"]

    assert entry["sampling"] == {
        "n_samples": entry["pit"]["n_samples"],
        "n_knots_effective": entry["pit"]["n_knots_effective"],
        "pooling": entry["pit"]["pooling"],
    }
    # ceil(20 * 30 / 200) = 3 independent knots behind 20 pooled ones.
    assert entry["sampling"]["n_knots_effective"] == 3
    assert entry["sampling"]["n_samples"] == 20


# ---------------------------------------------------------------------------
# Contraction ratio
# ---------------------------------------------------------------------------


def test_contraction_ratio_recovers_a_known_spread_ratio() -> None:
    """A posterior scaled by a known factor reports exactly that factor."""
    rng = np.random.default_rng(4)
    prior_members = rng.normal(0.0, 2.0, size=(16, 12))
    factor = 0.25
    posterior_members = prior_members * factor

    summary = _bundle(
        _dynamic_dataset(posterior_members),
        _truth_dataset(np.zeros(12)),
        _dynamic_dataset(prior_members),
    )
    contraction = summary["inflow_angle"]["contraction_ratio"]
    assert contraction["mean"] == pytest.approx(factor)
    assert contraction["min"] == pytest.approx(factor)


def _chained_prior_posterior(
    n_members: int, n_windows: int, knots_per_window: int, ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """The prior structure a real multi-window run actually writes.

    `run_esmda.py` seeds window `w`'s prior from window `w - 1`'s posterior
    (GP-extrapolated when dynamic, assigned bitwise when static), so in the
    concatenated `prior_params.nc` **only block 0 is a genuine prior** and the
    spread ratchets down:

        prior spread[w]     = initial * ratio**w
        posterior spread[w] = initial * ratio**(w + 1)

    Each block gets independent anomalies normalized to an exact sample spread,
    so the per-knot ratios are closed-form while the blocks stay full rank.
    """
    rng = np.random.default_rng(seed)

    def _unit(shape: tuple[int, int]) -> np.ndarray:
        raw = rng.normal(size=shape)
        raw = raw - raw.mean(axis=0, keepdims=True)
        return raw / raw.std(axis=0, ddof=1, keepdims=True)

    prior_blocks, posterior_blocks = [], []
    spread = 1.0
    for _ in range(n_windows):
        prior_blocks.append(_unit((n_members, knots_per_window)) * spread)
        spread *= ratio
        posterior_blocks.append(_unit((n_members, knots_per_window)) * spread)
    return (
        np.concatenate(prior_blocks, axis=1),
        np.concatenate(posterior_blocks, axis=1),
    )


def test_contraction_separates_per_window_from_cumulative_contraction() -> None:
    """The headline finding: the elementwise ratio measures only the LAST update.

    On a prior with the real window chain the two answers differ by the window
    count, and the elementwise one -- the number the schema called
    `contraction_ratio` -- is the smaller claim. A run that cut spread by 78%
    reports 40% if only that number is read.
    """
    n_windows, knots_per_window, ratio = 3, 7, 0.6
    prior, posterior = _chained_prior_posterior(
        n_members=32,
        n_windows=n_windows,
        knots_per_window=knots_per_window,
        ratio=ratio,
        seed=200,
    )
    n_knots = n_windows * knots_per_window

    contraction = _bundle(
        _dynamic_dataset(posterior),
        _truth_dataset(np.zeros(n_knots)),
        _dynamic_dataset(prior),
        num_windows=n_windows,
    )["inflow_angle"]["contraction_ratio"]

    # Per window the update contracts by exactly `ratio`, everywhere.
    assert contraction["vs_window_prior"]["mean"] == pytest.approx(ratio)
    assert contraction["vs_window_prior"]["min"] == pytest.approx(ratio)

    # Against the only genuine prior in the file, window w sits at ratio**(w+1).
    expected = [ratio ** (w + 1) for w in range(n_windows)]
    assert contraction["vs_initial_prior"]["mean"] == pytest.approx(np.mean(expected))
    assert contraction["vs_initial_prior"]["min"] == pytest.approx(ratio**n_windows)
    assert contraction["vs_initial_prior"]["reason"] is None

    # The regression a uniformly-wider prior cannot catch: the two disagree.
    assert (
        contraction["vs_initial_prior"]["mean"] < contraction["vs_window_prior"]["mean"]
    )
    # ... and the legacy top-level keys still alias the per-window block.
    assert contraction["mean"] == contraction["vs_window_prior"]["mean"]
    assert contraction["min"] == contraction["vs_window_prior"]["min"]


def test_cumulative_contraction_handles_a_static_parameter_window_axis() -> None:
    """A static parameter's x-axis IS the window index, so block 0 is column 0.

    Same slice expression as the dynamic case (knots_per_window comes out 1);
    this pins that no separate branch is needed and none was added.
    """
    n_windows, ratio = 4, 0.5
    prior, posterior = _chained_prior_posterior(
        n_members=16, n_windows=n_windows, knots_per_window=1, ratio=ratio, seed=201
    )
    posterior_ds = xarray.Dataset(
        {"sgs_constant": (("ensemble", "time"), 0.15 + posterior)},
        coords={"ensemble": np.arange(16)},  # no `time` coord: the static branch
    )
    prior_ds = xarray.Dataset(
        {"sgs_constant": (("ensemble", "time"), 0.15 + prior)},
        coords={"ensemble": np.arange(16)},
    )
    truth = xarray.Dataset(
        {"sgs_constant": (("ensemble",), np.array([0.15]))}, coords={"ensemble": [0]}
    )

    contraction = _bundle(posterior_ds, truth, prior_ds, num_windows=n_windows)[
        "sgs_constant"
    ]["contraction_ratio"]

    assert contraction["vs_window_prior"]["mean"] == pytest.approx(ratio)
    assert contraction["vs_initial_prior"]["min"] == pytest.approx(ratio**n_windows)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("num_windows", "n_knots", "expected_in_reason"),
    [(None, 12, "num_windows is unknown"), (5, 12, "do not divide")],
)
def test_cumulative_contraction_degrades_to_null_with_a_reason(
    num_windows: int | None, n_knots: int, expected_in_reason: str
) -> None:
    """An unsplittable artifact emits null + a reason, never a silent mis-slice."""
    rng = np.random.default_rng(202)
    prior_members = rng.normal(0.0, 2.0, size=(10, n_knots))

    contraction = _bundle(
        _dynamic_dataset(prior_members * 0.5),
        _truth_dataset(np.zeros(n_knots)),
        _dynamic_dataset(prior_members),
        num_windows=num_windows,
    )["inflow_angle"]["contraction_ratio"]

    # The per-window number is unaffected -- it needs no window structure.
    assert contraction["vs_window_prior"]["mean"] == pytest.approx(0.5)
    assert contraction["vs_initial_prior"]["mean"] is None
    assert contraction["vs_initial_prior"]["min"] is None
    assert expected_in_reason in contraction["vs_initial_prior"]["reason"]


def test_contraction_ratio_is_null_without_a_prior() -> None:
    members, truth = _calibrated(n_members=8, n_knots=10, seed=5)
    summary = _bundle(_dynamic_dataset(members), _truth_dataset(truth), prior=None)

    assert summary["inflow_angle"]["contraction_ratio"] is None
    # ... and the joint block degrades with a reason rather than raising.
    assert summary["joint"]["generalized_eigenvalues"] is None
    assert "prior" in summary["joint"]["reason"]


def test_contraction_ratio_skips_knots_with_zero_prior_spread() -> None:
    """A collapsed prior knot is dropped, not turned into an infinity."""
    rng = np.random.default_rng(6)
    prior_members = rng.normal(0.0, 1.0, size=(10, 4))
    prior_members[:, 2] = 3.0  # zero spread at one knot
    posterior_members = prior_members * 0.5

    contraction = _bundle(
        _dynamic_dataset(posterior_members),
        _truth_dataset(np.zeros(4)),
        _dynamic_dataset(prior_members),
    )["inflow_angle"]["contraction_ratio"]

    assert contraction["mean"] == pytest.approx(0.5)
    assert contraction["min"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Truth alignment
# ---------------------------------------------------------------------------


def test_truth_on_a_coarser_knot_grid_is_interpolated() -> None:
    """Truth with fewer knots than the posterior (19 vs 21) still scores."""
    n_knots = 21
    members = np.tile(np.linspace(0.0, 20.0, n_knots), (12, 1))
    members += np.random.default_rng(7).normal(0.0, 0.1, size=members.shape)
    truth_times = np.linspace(0.0, (n_knots - 1) * SECONDS_PER_KNOT, 19)
    truth_values = truth_times / SECONDS_PER_KNOT  # the same ramp, resampled

    entry = _bundle(
        _dynamic_dataset(members),
        _truth_dataset(truth_values, times=truth_times),
    )["inflow_angle"]

    assert entry["pit"]["n_samples"] == n_knots
    # The truth is the members' own ramp, so every z-score stays small.
    assert entry["zscore"]["max_abs"] < 4.0
    assert entry["zscore"]["overconfident"] is False


# ---------------------------------------------------------------------------
# PIT branches: the discriminator is per parameter, not per run
# ---------------------------------------------------------------------------


def test_both_pit_branches_fire_in_one_dataset() -> None:
    """A time-coordinate parameter and a window-axis one, same Dataset."""
    rng = np.random.default_rng(8)
    n_knots, n_windows, n_members = 20, 3, 12
    dynamic = rng.normal(0.0, 1.0, size=(n_members, n_knots))
    static = rng.normal(0.0, 1.0, size=(n_members, n_windows))

    posterior = xarray.Dataset(
        {
            "inflow_angle": (("ensemble", "time"), dynamic),
            # No `time` coordinate: this is how `_concat_windows` stacks a
            # parameter that has no time axis of its own.
            "vertical_inflow_exponent": (("ensemble", "window"), static),
        },
        coords={
            "ensemble": np.arange(n_members),
            "time": np.arange(n_knots) * SECONDS_PER_KNOT,
        },
    )
    truth = xarray.Dataset(
        {
            "inflow_angle": (("time", "ensemble"), rng.normal(size=(n_knots, 1))),
            "vertical_inflow_exponent": (("ensemble",), np.array([0.25])),
        },
        coords={
            "time": np.arange(n_knots) * SECONDS_PER_KNOT,
            "ensemble": [0],
        },
    )

    summary = _bundle(posterior, truth, num_windows=n_windows)

    dynamic_pit = summary["inflow_angle"]["pit"]
    static_pit = summary["vertical_inflow_exponent"]["pit"]
    assert dynamic_pit["pooling"] == "knots_correlated"
    assert static_pit["pooling"] == "windows_correlated"
    # ceil(20 * 30 / 200) = 3 independent knots, below the 20 raw ones.
    assert dynamic_pit["n_knots_effective"] == 3
    assert dynamic_pit["n_samples"] == n_knots
    # The static parameter pools over windows, so its ceiling is the window count.
    assert static_pit["n_knots_effective"] == n_windows
    assert static_pit["n_samples"] == n_windows


def test_time_dimension_without_a_time_coordinate_pools_over_windows() -> None:
    """A purely static run: `time` is a stacking dim, so the x-axis is the window."""
    rng = np.random.default_rng(9)
    n_windows, n_members = 4, 10
    posterior = xarray.Dataset(
        {
            "sgs_constant": (
                ("ensemble", "time"),
                rng.normal(size=(n_members, n_windows)),
            )
        },
        coords={"ensemble": np.arange(n_members)},  # deliberately no `time` coord
    )
    truth = xarray.Dataset(
        {"sgs_constant": (("ensemble",), np.array([0.1]))}, coords={"ensemble": [0]}
    )

    pit = _bundle(posterior, truth, num_windows=n_windows)["sgs_constant"]["pit"]
    assert pit["pooling"] == "windows_correlated"
    assert pit["n_knots_effective"] == n_windows


def test_static_parameter_broadcast_onto_a_time_axis_counts_windows() -> None:
    """Piecewise-constant knots cannot be more independent than their steps."""
    rng = np.random.default_rng(10)
    n_windows, knots_per_window, n_members = 3, 7, 12
    levels = rng.normal(0.0, 1.0, size=(n_members, n_windows))
    members = np.repeat(levels, knots_per_window, axis=1)

    pit = _bundle(
        _dynamic_dataset(members),
        _truth_dataset(np.zeros(n_windows * knots_per_window)),
        num_windows=n_windows,
    )["inflow_angle"]["pit"]

    assert pit["pooling"] == "knots_correlated"
    assert pit["n_knots_effective"] == n_windows


def test_segment_counting_tolerance_scales_with_the_parameter_spread() -> None:
    """A parameter's offset must not set the "is this knot a step?" threshold.

    `inflow_angle` sits near 270 with an ensemble spread of ~1 degree, so a
    magnitude-relative tolerance (`np.isclose`'s rtol=1e-5) would call any knot
    step below 2.7e-3 degrees constant -- i.e. would clamp a genuinely
    time-varying parameter's effective sample size on the strength of where its
    origin happens to be.
    """
    rng = np.random.default_rng(101)
    n_members, n_knots = 12, 20
    walk = np.cumsum(rng.normal(0.0, 5e-4, size=(n_members, n_knots)), axis=1)
    members = 270.0 + walk  # steps ~5e-4 << np.isclose's 2.7e-3 at this offset

    pit = _bundle(
        _dynamic_dataset(members),
        _truth_dataset(np.full(n_knots, 270.0)),
        num_windows=3,
    )["inflow_angle"]["pit"]

    # Every knot differs, so the segment clamp must not bind; what remains is
    # the GP formula, ceil(20 * 30 / 200) = 3.
    assert pit["n_knots_effective"] == 3

    # The broadcast-static case still resolves to its window count: those knots
    # repeat bitwise-identical values, which no tolerance can confuse.
    levels = 270.0 + rng.normal(0.0, 1.0, size=(n_members, 4))
    broadcast = _bundle(
        _dynamic_dataset(np.repeat(levels, 5, axis=1)),
        _truth_dataset(np.full(20, 270.0)),
        num_windows=4,
    )["inflow_angle"]["pit"]
    assert broadcast["n_knots_effective"] == 3  # min(segments=4, GP bound=3)


def test_missing_knot_correlation_config_emits_null() -> None:
    """No `correlation_length` in the saved config -> null, never a guess."""
    members, truth = _calibrated(n_members=10, n_knots=15, seed=11)
    summary = _bundle(
        _dynamic_dataset(members),
        _truth_dataset(truth),
        cfg=_cfg(correlation_length=None),
    )

    pit = summary["inflow_angle"]["pit"]
    assert pit["n_knots_effective"] is None
    assert pit["pooling"] == "knots_correlated"
    assert pit["n_samples"] == 15


def test_knot_correlation_config_resolves_interpolations() -> None:
    """`seconds_per_knot: ${time.seconds_per_knot}` is how runs actually save it."""
    cfg = OmegaConf.create(
        {
            "time": {"seconds_per_knot": 30.0},
            "prior_params": {
                "correlation_length": 200.0,
                "seconds_per_knot": "${time.seconds_per_knot}",
            },
        }
    )
    assert _knot_correlation_config(cfg) == (200.0, 30.0)
    assert _knot_correlation_config(OmegaConf.create({})) == (None, None)
    assert _knot_correlation_config(None) == (None, None)


# ---------------------------------------------------------------------------
# Joint directions
# ---------------------------------------------------------------------------


def _members_with_covariance(cov: np.ndarray, n_members: int, seed: int) -> np.ndarray:
    """Members whose SAMPLE covariance is exactly ``cov`` (no sampling error)."""
    rng = np.random.default_rng(seed)
    k = cov.shape[0]
    raw = rng.normal(size=(n_members, k))
    raw = raw - raw.mean(axis=0, keepdims=True)
    # Whiten the draw, then colour it: the sample covariance is then exact, so
    # the eigenvalue assertions below are closed-form rather than statistical.
    sample = np.cov(raw, rowvar=False)
    whitening = np.linalg.inv(np.linalg.cholesky(sample))
    colouring = np.linalg.cholesky(cov)
    return raw @ whitening.T @ colouring.T


def test_generalized_eigenvalues_recover_constructed_ratios() -> None:
    """Diagonal covariances -> the eigenvalues are exactly the variance ratios."""
    prior_cov = np.diag([4.0, 1.0, 9.0])
    ratios = np.array([0.01, 0.4, 2.0])
    posterior_cov = np.diag(np.diag(prior_cov) * ratios)

    prior = _members_with_covariance(prior_cov, n_members=20, seed=12)
    posterior = _members_with_covariance(posterior_cov, n_members=20, seed=12)

    joint = joint_parameter_directions(posterior, prior, ["a", "b", "c"])

    assert joint["n_sample_directions"] == 3
    assert joint["rank_deficient"] is False
    assert np.allclose(sorted(joint["generalized_eigenvalues"]), sorted(ratios))
    # Two of the three ratios are below one half.
    assert joint["n_constrained_directions"] == 2
    assert joint["most_constrained"]["eigenvalue"] == pytest.approx(0.01)
    assert joint["least_constrained"]["eigenvalue"] == pytest.approx(2.0)
    # The most-constrained direction is the first coordinate, which was
    # contracted 100-fold; loadings are reported in parameter space.
    assert joint["most_constrained"]["loadings"][0]["parameter"] == "a"
    assert abs(joint["most_constrained"]["loadings"][0]["loading"]) == pytest.approx(
        1.0, abs=1e-6
    )
    assert joint["least_constrained"]["loadings"][0]["parameter"] == "c"


def test_constrained_direction_count_uses_a_strict_threshold() -> None:
    """A direction whose spread halved exactly is not counted as constrained."""
    prior_cov = np.diag([1.0, 1.0])
    posterior_cov = np.diag([0.5, 0.499])
    prior = _members_with_covariance(prior_cov, n_members=10, seed=25)
    posterior = _members_with_covariance(posterior_cov, n_members=10, seed=25)

    joint = joint_parameter_directions(posterior, prior)
    assert joint["n_constrained_directions"] == 1


def test_joint_eigenvalues_are_invariant_to_parameter_rescaling() -> None:
    """Mixed units (degrees vs m/s) must not change the spectrum.

    Generalized eigenvalues are invariant under any invertible rescaling of the
    parameter vector, which is why the joint block can mix units at all. The
    invariance is exact for the raw pencil and holds to rounding once the
    trace-scaled eps ridge is added -- the ridge is a fixed fraction of the mean
    variance, so only an absurd anisotropy (many orders of magnitude beyond the
    ~20x between degrees and m/s) makes it visible.
    """
    prior_cov = np.array([[4.0, 1.0], [1.0, 2.0]])
    posterior_cov = np.array([[1.0, 0.2], [0.2, 1.5]])
    prior = _members_with_covariance(prior_cov, n_members=12, seed=13)
    posterior = _members_with_covariance(posterior_cov, n_members=12, seed=13)
    scale = np.array([100.0, 0.01])

    base = joint_parameter_directions(posterior, prior)
    scaled = joint_parameter_directions(posterior * scale, prior * scale)

    assert np.allclose(
        base["generalized_eigenvalues"], scaled["generalized_eigenvalues"]
    )


def test_joint_handles_a_rank_deficient_ensemble() -> None:
    """M - 1 < K (the routine production shape) truncates instead of exploding."""
    rng = np.random.default_rng(14)
    n_members, k = 8, 20
    prior = rng.normal(0.0, 1.0, size=(n_members, k))
    posterior = prior * 0.3

    joint = joint_parameter_directions(posterior, prior)

    assert joint["rank_deficient"] is True
    assert joint["n_sample_directions"] == n_members - 1
    eigenvalues = np.asarray(joint["generalized_eigenvalues"], dtype=float)
    assert eigenvalues.size == n_members - 1
    assert np.all(eigenvalues > 0.0)
    # Every retained direction was contracted by the same factor.
    assert np.allclose(eigenvalues, 0.3**2)
    assert joint["n_constrained_directions"] == n_members - 1


def test_variance_retained_is_one_when_the_posterior_lives_in_the_prior_span() -> None:
    """One ESMDA update: posterior members are combinations of prior members."""
    rng = np.random.default_rng(140)
    n_members, k = 32, 42
    prior = rng.normal(size=(n_members, k))
    # An ensemble-space transform is exactly what a single ESMDA update applies.
    transform = np.eye(n_members) + 0.3 * rng.normal(size=(n_members, n_members)) / (
        np.sqrt(n_members)
    )

    joint = joint_parameter_directions(transform @ prior, prior)

    assert joint["rank_deficient"] is True  # M - 1 = 31 < K = 42
    assert joint["posterior_variance_retained"] == pytest.approx(1.0, abs=0.01)


def test_variance_retained_exposes_a_posterior_direction_the_prior_lacks() -> None:
    """The multi-window concatenation is NOT one update, so the projection can lose."""
    rng = np.random.default_rng(141)
    n_members, n_windows, knots_per_window = 32, 3, 14
    k = n_windows * knots_per_window
    prior = rng.normal(size=(n_members, k))

    # Each window block carries its own M x M transform, exactly as
    # `posterior_params.nc` (a concatenation of per-window updates) does.
    posterior = np.empty_like(prior)
    for w in range(n_windows):
        block = slice(w * knots_per_window, (w + 1) * knots_per_window)
        transform = np.eye(n_members) + 0.3 * rng.normal(
            size=(n_members, n_members)
        ) / np.sqrt(n_members)
        posterior[:, block] = transform @ prior[:, block]

    multi_window = joint_parameter_directions(posterior, prior)
    # Measured ~0.985 on this construction: a small but real loss, invisible
    # from `rank_deficient` alone.
    assert 0.9 < multi_window["posterior_variance_retained"] < 1.0

    # A genuinely new direction (inflation, a re-sampled window) is dropped
    # wholesale, and only this key says so.
    new_direction = rng.normal(size=(n_members, 1)) @ rng.normal(size=(1, k))
    grown = joint_parameter_directions(prior + 2.0 * new_direction, prior)
    assert grown["posterior_variance_retained"] < 0.9
    assert grown["rank_deficient"] is True  # unchanged -- it cannot see this


def test_variance_retained_is_null_on_every_degraded_path() -> None:
    rng = np.random.default_rng(142)
    joint = joint_parameter_directions(rng.normal(size=(2, 5)), rng.normal(size=(2, 5)))

    assert joint["posterior_variance_retained"] is None


def test_joint_correlation_matrices_are_capped() -> None:
    """Small K keeps the matrices; large K swaps them for the off-diagonal summary."""
    rng = np.random.default_rng(15)
    small = joint_parameter_directions(
        rng.normal(size=(12, JOINT_CORR_MAX_K)),
        rng.normal(size=(12, JOINT_CORR_MAX_K)),
    )
    assert np.shape(small["posterior_corr"]) == (JOINT_CORR_MAX_K, JOINT_CORR_MAX_K)
    assert np.allclose(np.diag(np.asarray(small["posterior_corr"])), 1.0)

    big = joint_parameter_directions(
        rng.normal(size=(12, JOINT_CORR_MAX_K + 1)),
        rng.normal(size=(12, JOINT_CORR_MAX_K + 1)),
    )
    assert big["posterior_corr"] is None
    assert big["prior_corr"] is None
    assert "corr_matrices_omitted" in big
    assert 0.0 <= big["corr_summary"]["posterior_offdiag_abs_mean"] <= 1.0


def test_joint_names_its_prior_reference_and_answers_both_questions() -> None:
    """`n_constrained_directions` is per-window; the cumulative one says so.

    The concatenated prior is a per-window reference (block `w` is posterior
    `w - 1`), so the parent pencil cannot answer "what did the run constrain".
    The cumulative block scores the final posterior window against window 0's
    prior, and both are labelled rather than one standing in for the other.
    """
    n_windows, knots_per_window, ratio = 3, 6, 0.6
    prior, posterior = _chained_prior_posterior(
        n_members=32,
        n_windows=n_windows,
        knots_per_window=knots_per_window,
        ratio=ratio,
        seed=207,
    )
    joint = _bundle(
        _dynamic_dataset(posterior),
        _truth_dataset(np.zeros(n_windows * knots_per_window)),
        _dynamic_dataset(prior),
        num_windows=n_windows,
    )["joint"]

    assert joint["prior_reference"] == "per_window_prior"
    cumulative = joint["vs_initial_prior"]
    assert cumulative["reason"] is None
    # One window wide, against the parent's K = n_windows * knots_per_window.
    assert cumulative["n_parameters"] == knots_per_window
    assert joint["n_parameters"] == n_windows * knots_per_window
    assert cumulative["rank_deficient"] is False
    # Three windows of contraction sit far below one window's.
    assert cumulative["eigenvalue_quantiles"]["median"] < ratio**4
    assert joint["eigenvalue_quantiles"]["median"] > ratio**4


def test_joint_cumulative_block_is_null_when_the_blocks_cannot_be_sliced() -> None:
    """Same degenerate-shape rule as everywhere else: null plus a reason."""
    members, truth = _calibrated(n_members=10, n_knots=13, seed=208)
    # 13 knots do not divide into 3 windows, so `_window_block_flat` declines.
    joint = _bundle(
        _dynamic_dataset(members),
        _truth_dataset(truth),
        _dynamic_dataset(members * 4.0),
        num_windows=3,
    )["joint"]

    cumulative = joint["vs_initial_prior"]
    assert cumulative["n_constrained_directions"] is None
    assert cumulative["eigenvalue_quantiles"] is None
    assert "could not be sliced" in cumulative["reason"]
    # The per-window block is unaffected -- it needs no window structure.
    assert joint["n_constrained_directions"] is not None


def test_window_block_flat_slices_the_requested_window() -> None:
    """The block helper reuses the one flattener on a `time`-sliced Dataset."""
    n_members, n_windows, knots_per_window = 6, 3, 4
    n_knots = n_windows * knots_per_window
    values = np.arange(n_members * n_knots, dtype=float).reshape(n_members, n_knots)
    params = _dynamic_dataset(values)

    first = _window_block_flat(params, n_windows, 0)
    last = _window_block_flat(params, n_windows, -1)

    assert np.array_equal(first, values[:, :knots_per_window])
    assert np.array_equal(last, values[:, -knots_per_window:])
    # Unsplittable or unknown shapes decline rather than mis-slicing.
    assert _window_block_flat(params, None, 0) is None
    assert _window_block_flat(params, 5, 0) is None


def test_joint_is_null_for_non_finite_members() -> None:
    rng = np.random.default_rng(16)
    posterior = rng.normal(size=(10, 5))
    posterior[3, 2] = np.nan

    joint = joint_parameter_directions(posterior, rng.normal(size=(10, 5)))
    assert joint["generalized_eigenvalues"] is None
    assert joint["n_constrained_directions"] is None
    assert "non-finite" in joint["reason"]


def test_joint_is_null_when_the_prior_shape_disagrees() -> None:
    rng = np.random.default_rng(17)
    joint = joint_parameter_directions(
        rng.normal(size=(10, 5)), rng.normal(size=(10, 4))
    )
    assert joint["generalized_eigenvalues"] is None
    assert "do not match" in joint["reason"]


# ---------------------------------------------------------------------------
# The smoke shape and other degenerate inputs
# ---------------------------------------------------------------------------


def test_two_member_smoke_shape_emits_nulls_without_raising() -> None:
    """M = 2: every key is present, every value is null, nothing raises."""
    rng = np.random.default_rng(18)
    posterior = _dynamic_dataset(rng.normal(size=(2, 6)))
    prior = _dynamic_dataset(rng.normal(size=(2, 6)))
    truth = _truth_dataset(rng.normal(size=6))

    summary = _bundle(posterior, truth, prior)
    entry = summary["inflow_angle"]

    assert MIN_MEMBERS_CALIBRATION > 2
    for key in ("zscore", "pit_counts", "pit", "coverage", "contraction_ratio"):
        assert key in entry, key
        assert entry[key] is None, key

    joint = summary["joint"]
    assert joint["n_members"] == 2
    assert joint["generalized_eigenvalues"] is None
    assert joint["n_constrained_directions"] is None
    assert joint["posterior_corr"] is None
    assert joint["reason"]


def test_single_member_ensemble_emits_nulls() -> None:
    posterior = _dynamic_dataset(np.zeros((1, 5)))
    summary = _bundle(
        posterior, _truth_dataset(np.zeros(5)), _dynamic_dataset(np.ones((1, 5)))
    )
    assert summary["inflow_angle"]["zscore"] is None
    assert summary["joint"]["generalized_eigenvalues"] is None


def test_single_knot_has_no_zscore_spread() -> None:
    """One knot gives no spread estimate over knots -- null, not zero."""
    rng = np.random.default_rng(19)
    entry = _bundle(
        _dynamic_dataset(rng.normal(size=(10, 1))),
        _truth_dataset(np.zeros(1)),
    )["inflow_angle"]

    assert entry["zscore"]["mean"] is not None
    assert entry["zscore"]["std"] is None
    assert entry["zscore"]["max_abs"] is not None


# ---------------------------------------------------------------------------
# Schema contracts
# ---------------------------------------------------------------------------


def test_bundle_is_additive_and_does_not_mutate_the_base() -> None:
    """Existing `parameter_metrics` keys survive untouched (figure scripts read them)."""
    members, truth = _calibrated(n_members=10, n_knots=12, seed=20)
    base = {"inflow_angle": {"rmse": {"mean": 1.0}, "crps": {"mean": 2.0}}}
    original = {"inflow_angle": {"rmse": {"mean": 1.0}, "crps": {"mean": 2.0}}}

    summary = _bundle(
        _dynamic_dataset(members),
        _truth_dataset(truth),
        _dynamic_dataset(members * 2.0),
        base=base,
    )

    assert base == original  # the caller's mapping is not mutated
    assert summary["inflow_angle"]["rmse"] == {"mean": 1.0}
    assert summary["inflow_angle"]["crps"] == {"mean": 2.0}
    assert set(summary) == {"inflow_angle", "joint"}


def test_summary_is_yaml_round_trippable(tmp_path: pathlib.Path) -> None:
    """No numpy scalars, no bare nan/inf: `write_yaml` output reloads as floats."""
    from scripts.esmda._esmda_common import read_yaml, write_yaml

    members, truth = _calibrated(n_members=9, n_knots=14, seed=21)
    summary = _bundle(
        _dynamic_dataset(members),
        _truth_dataset(truth),
        _dynamic_dataset(members * 3.0),
    )

    path = tmp_path / "run_summary.yaml"
    write_yaml({"parameter_metrics": summary}, path)
    reloaded = read_yaml(path)["parameter_metrics"]

    assert reloaded["inflow_angle"]["coverage"]["alpha_90"] == pytest.approx(
        summary["inflow_angle"]["coverage"]["alpha_90"]
    )
    assert reloaded["joint"]["n_constrained_directions"] == (
        summary["joint"]["n_constrained_directions"]
    )

    def _no_bare_non_finite(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                _no_bare_non_finite(value)
        elif isinstance(node, list):
            for value in node:
                _no_bare_non_finite(value)
        elif isinstance(node, float):
            assert np.isfinite(node)

    _no_bare_non_finite(reloaded)


def test_pit_counts_have_the_documented_shape() -> None:
    members, truth = _calibrated(n_members=15, n_knots=40, seed=22)
    entry = _bundle(_dynamic_dataset(members), _truth_dataset(truth))["inflow_angle"]

    counts = entry["pit_counts"]
    assert len(counts) == PIT_BINS
    assert all(isinstance(c, int) for c in counts)
    assert sum(counts) == entry["pit"]["n_samples"] == 40


def test_pit_metadata_carries_the_per_bin_reference() -> None:
    """The counts are meaningless without the uneven bin widths that produced them."""
    members, truth = _calibrated(n_members=32, n_knots=40, seed=25)
    pit = _bundle(_dynamic_dataset(members), _truth_dataset(truth))["inflow_angle"][
        "pit"
    ]

    # M = 32: 33 rank values over 10 bins is a +21% / -9% comb, so a consumer
    # normalizing by a flat len(ranks)/n_bins would call this calibrated
    # ensemble structured. `ranks_per_bin` is that reference.
    assert pit["ranks_per_bin"] == [4, 3, 3, 4, 3, 3, 4, 3, 3, 3]
    assert sum(pit["ranks_per_bin"]) == 32 + 1
    assert len(pit["ranks_per_bin"]) == pit["n_bins"] == PIT_BINS


def test_coverage_reports_the_attainable_nominal_ceiling() -> None:
    """A small ensemble cannot offer a 90% band; the reported ceiling says so.

    ``max_nominal_alpha`` bounds the *nominal* level of the widest available
    band, not the realized fraction -- which may land above it by chance, so
    only the monotonicity of the two bands is asserted here.
    """
    members, truth = _calibrated(n_members=4, n_knots=30, seed=23)
    coverage_block = _bundle(_dynamic_dataset(members), _truth_dataset(truth))[
        "inflow_angle"
    ]["coverage"]

    assert coverage_block["max_nominal_alpha"] == pytest.approx(3.0 / 5.0)
    assert coverage_block["alpha_90"] >= coverage_block["alpha_50"]


def test_parameter_vector_labels_match_the_flattener() -> None:
    """One label per column of `_flatten_parameter_members`, in the same order."""
    rng = np.random.default_rng(24)
    n_members = 6
    params = xarray.Dataset(
        {
            "velocity_magnitude": (
                ("ensemble", "time"),
                rng.normal(size=(n_members, 5)),
            ),
            "inflow_angle": (("ensemble", "time"), rng.normal(size=(n_members, 5))),
            "vertical_inflow_exponent": (("ensemble",), rng.normal(size=n_members)),
        },
        coords={"ensemble": np.arange(n_members), "time": np.arange(5.0)},
    )

    flat = _flatten_parameter_members(params)
    labels = parameter_vector_labels(params)

    assert len(labels) == flat.shape[1]
    # Sorted variable order, then row-major within each variable.
    assert labels[0] == "inflow_angle[0]"
    assert labels[4] == "inflow_angle[4]"
    assert labels[5] == "velocity_magnitude[0]"
    assert labels[-1] == "vertical_inflow_exponent"
    # Columns and labels agree element-for-element.
    for column, label in enumerate(labels):
        name = label.split("[")[0]
        index = int(label.split("[")[1][:-1]) if "[" in label else 0
        expected = np.asarray(params[name].transpose("ensemble", ...).values)
        expected = expected.reshape(n_members, -1)[:, index]
        assert np.allclose(flat[:, column], expected), label
