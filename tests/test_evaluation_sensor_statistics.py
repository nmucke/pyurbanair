"""WP1.3: window statistics as the verification object (metrics doc §4.2).

``sensor_metrics`` scores the *instantaneous* sensor series, and for turbulence
that is mostly a phase measurement: two members with identical parameters
decorrelate within an eddy turnover, so the pointwise error is dominated by
something no parameter estimate can control. What the parameters act on is the
statistics, so WP1.3 scores those instead. What is pinned here:

  * the windowing is by the **time coordinate**, not by frame count, because
    the truth and the assimilation routinely write at different cadences and
    the same reduction has to serve both (``..._bins_by_time_...``);
  * the block bootstrap is the sampling floor those statistics are read
    against. It lands on ``std/sqrt(n)`` when there is no correlation to
    preserve and rises well above it when there is -- the whole point, since
    the iid formula understates a turbulent series by 2-3x
    (``..._matches_the_iid_formula_...``, ``..._exceeds_the_iid_formula_...``);
  * the identifiability guard is the ratio of those two spreads. A statistic
    whose across-member spread does not clear its own sampling noise is not
    identifiable at this window length however good its CRPS looks
    (``..._flags_a_statistic_...``);
  * ranks are uniform for a calibrated ensemble and pile at an end for a biased
    one -- the shape check the CRPS and the z-score cannot make
    (``..._uniform_...``, ``..._pile_up_...``);
  * ``ensemble_sensor_series`` reads **one member at a time**. It used to
    ``.load()`` whole multi-GB window files, the last such site in the
    post-processing stack, and the rewrite has to be numerically inert
    (``..._streams_member_at_a_time``, ``..._is_inert``).
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
import xarray
from evaluation.scores import ensemble_rank, window_statistics_summary
from evaluation.sensors import window_masks, window_sampling_std, window_statistics
from evaluation.turbulence import block_bootstrap_std

SIM_TIME = 10.0
NUM_WINDOWS = 3


def _series(
    rng: np.random.Generator,
    n_time: int,
    n_ensemble: int | None = None,
    n_sensors: int = 2,
    offset: float = 0.0,
) -> xarray.DataArray:
    """A ``(component, [ensemble,] time, sensor)`` series on a global time axis."""
    time = np.linspace(0.0, NUM_WINDOWS * SIM_TIME, n_time, endpoint=False)
    shape: tuple[int, ...] = (3, n_time, n_sensors)
    dims: tuple[str, ...] = ("component", "time", "sensor")
    if n_ensemble is not None:
        shape = (3, n_ensemble, n_time, n_sensors)
        dims = ("component", "ensemble", "time", "sensor")
    return xarray.DataArray(
        rng.normal(size=shape) + 2.0 + offset,
        dims=dims,
        coords={
            "component": ["u", "v", "w"],
            "time": time,
            "sensor": np.arange(n_sensors),
        },
    )


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def test_window_masks_bin_by_time_not_by_frame_count() -> None:
    # A boundary frame opens its window; anything past the last boundary (float
    # drift on the final frame) falls in the last window rather than off the end.
    times = np.array([0.0, 9.9, 10.0, 19.9, 20.0, 30.0])
    masks = window_masks(times, SIM_TIME, NUM_WINDOWS)

    assert [list(np.flatnonzero(m)) for m in masks] == [[0, 1], [2, 3], [4, 5]]


def test_window_masks_survive_float_drift_on_a_boundary() -> None:
    # The extraction rebases window w to start at exactly ``w*sim_time``, but
    # ``(w*sim_time)/sim_time`` is not exactly ``w`` in IEEE double for most
    # sim_time. At 10.76 (200 frames at pylbm's default 0.0538 s cadence)
    # windows 7 and 14 come out a ULP short, and a bare ``floor`` scores their
    # opening frame into the *previous* window -- silently, and not necessarily
    # for the same frame on the truth axis as on the ensemble axis.
    sim_time = 10.76
    starts = np.array([w * sim_time for w in range(16)])
    masks = window_masks(starts, sim_time, 16)

    assert [int(np.flatnonzero(m)[0]) for m in masks] == list(range(16))


def test_window_masks_reject_a_run_with_no_windows() -> None:
    with pytest.raises(ValueError, match="num_windows"):
        window_masks(np.array([0.0]), SIM_TIME, 0)


def test_window_statistics_bin_by_time_so_truth_and_ensemble_cadences_may_differ() -> (
    None
):
    # The truth writes 30 frames per window, the assimilation 9. Binning by frame
    # count would put the two on different horizons and silently compare window 0
    # of one against most of window 1 of the other.
    rng = np.random.default_rng(0)
    truth = _series(rng, 90)
    members = _series(rng, 27, n_ensemble=4)

    truth_stats = window_statistics(truth, SIM_TIME, NUM_WINDOWS)
    member_stats = window_statistics(members, SIM_TIME, NUM_WINDOWS)

    assert truth_stats["mean"].sizes["window"] == NUM_WINDOWS
    assert member_stats["mean"].sizes["window"] == NUM_WINDOWS
    expected = truth.sel(component="u").isel(sensor=0).values[30:60].mean()
    assert truth_stats["mean"].sel(quantity="u", window=1).isel(
        sensor=0
    ).item() == pytest.approx(expected)


def test_window_statistics_variance_is_ddof_one() -> None:
    # ddof=1: the window variance estimates the flow's variance from a finite
    # window, not the window's own second moment. Truth and members are reduced
    # identically, so the choice cannot bias the comparison -- but it has to be
    # the same on both sides, which a hand-rolled reduction gets wrong.
    rng = np.random.default_rng(1)
    truth = _series(rng, 60)

    stats = window_statistics(truth, SIM_TIME, NUM_WINDOWS)
    window0 = truth.sel(component="v").isel(sensor=0).values[:20]

    assert stats["variance"].sel(quantity="v", window=0).isel(
        sensor=0
    ).item() == pytest.approx(window0.var(ddof=1))


def test_window_statistics_carry_the_magnitude_beside_the_components() -> None:
    rng = np.random.default_rng(2)
    truth = _series(rng, 60)

    stats = window_statistics(truth, SIM_TIME, NUM_WINDOWS)

    assert list(stats["mean"]["quantity"].values) == ["u", "v", "w", "magnitude"]
    magnitude = np.sqrt((truth**2).sum("component"))
    assert stats["mean"].sel(quantity="magnitude", window=0).isel(
        sensor=0
    ).item() == pytest.approx(magnitude.isel(sensor=0).values[:20].mean())


def test_window_statistics_null_a_window_with_no_frames(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Invariant 3: a truncated run costs the windows it lost, not the block.
    rng = np.random.default_rng(3)
    truth = _series(rng, 40).isel(time=slice(0, 20))  # only window 0 is covered

    with caplog.at_level("WARNING"):
        stats = window_statistics(truth, SIM_TIME, NUM_WINDOWS)

    assert np.isfinite(stats["mean"].sel(window=0).values).all()
    assert np.isnan(stats["mean"].sel(window=2).values).all()
    assert "No sensor frames fall in window 2" in caplog.text


# ---------------------------------------------------------------------------
# The block bootstrap: the sampling floor
# ---------------------------------------------------------------------------


def _ar1(rng: np.random.Generator, n: int, phi: float, n_series: int = 1) -> np.ndarray:
    """AR(1) series of unit marginal variance."""
    noise = rng.normal(scale=np.sqrt(1.0 - phi**2), size=(n_series, n))
    out = np.empty((n_series, n))
    out[:, 0] = rng.normal(size=n_series)
    for t in range(1, n):
        out[:, t] = phi * out[:, t - 1] + noise[:, t]
    return out


def test_block_bootstrap_matches_the_iid_formula_on_independent_data() -> None:
    # Blocking costs nothing when there is no correlation to preserve, which is
    # what makes it safe to apply unconditionally.
    rng = np.random.default_rng(4)
    series = rng.normal(size=(8, 400))

    std = block_bootstrap_std(series, n_blocks=20, n_resamples=400)

    assert std.shape == (8,)
    assert np.median(std) == pytest.approx(1.0 / np.sqrt(400), rel=0.25)


def test_block_bootstrap_exceeds_the_iid_formula_on_correlated_data() -> None:
    # The reason this exists: a correlated series has far fewer independent
    # samples than frames, so std/sqrt(n) understates its sampling spread and
    # would make every statistic look identifiable.
    rng = np.random.default_rng(5)
    series = _ar1(rng, 400, phi=0.9, n_series=8)

    blocked = np.median(block_bootstrap_std(series, n_blocks=20, n_resamples=400))
    iid = np.median(series.std(axis=1, ddof=1) / np.sqrt(400))

    assert blocked > 2.0 * iid


def test_block_bootstrap_is_vectorized_over_leading_dims_not_approximated() -> None:
    # Every row shares one time axis and therefore one index matrix, so the fast
    # path is the same estimator -- not a batched approximation of it.
    rng = np.random.default_rng(6)
    series = rng.normal(size=(3, 5, 200))

    batched = block_bootstrap_std(series, n_blocks=10, n_resamples=64)
    per_row = np.array(
        [
            [
                float(block_bootstrap_std(series[i, j], n_blocks=10, n_resamples=64))
                for j in range(5)
            ]
            for i in range(3)
        ]
    )

    assert batched.shape == (3, 5)
    assert np.allclose(batched, per_row)


def test_block_bootstrap_chunking_moves_no_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The replicate axis is split to bound the temporary. Every shape the tests
    # use fits in one chunk, so without this the "changes no value" claim is
    # never exercised -- and the chunk loop only runs at production scale.
    from evaluation import turbulence

    rng = np.random.default_rng(23)
    series = rng.normal(size=(6, 120))
    unchunked = block_bootstrap_std(series, n_blocks=10, n_resamples=64)

    monkeypatch.setattr(turbulence, "_BOOTSTRAP_CHUNK_BYTES", 1)
    chunked = block_bootstrap_std(series, n_blocks=10, n_resamples=64)

    assert np.array_equal(unchunked, chunked)


def test_block_bootstrap_reproduces_without_a_seed() -> None:
    # Re-running the metric stage on the same run dir must reproduce
    # run_summary.yaml; the default must therefore not be OS entropy.
    rng = np.random.default_rng(7)
    series = rng.normal(size=(4, 100))

    assert np.array_equal(block_bootstrap_std(series), block_bootstrap_std(series))
    assert not np.array_equal(
        block_bootstrap_std(series), block_bootstrap_std(series, rng=1)
    )


def test_block_bootstrap_is_null_on_windows_too_short_to_block() -> None:
    # Routine, not exotic: the default 20 blocks needs 21 samples and the CI
    # smoke shape has four frames per window, so callers must handle the null.
    rng = np.random.default_rng(8)

    assert np.isnan(block_bootstrap_std(rng.normal(size=(2, 3)))).all()  # n < 4
    assert np.isnan(block_bootstrap_std(rng.normal(size=(2, 10)))).all()  # L < 2


def test_block_bootstrap_nulls_only_the_rows_that_are_not_finite() -> None:
    # A diverged member must not blank the floor for the members beside it.
    rng = np.random.default_rng(9)
    series = rng.normal(size=(3, 100))
    series[1, 5] = np.nan

    std = block_bootstrap_std(series, n_blocks=10, n_resamples=64)

    assert np.isnan(std[1])
    assert np.isfinite(std[[0, 2]]).all()


def test_block_bootstrap_of_a_single_block_is_a_measured_zero() -> None:
    # One block spans the series, so there is one legal start and every replicate
    # is the original: the spread really is zero. Exactly zero, not 1e-17 of
    # float rounding, which would survive the ``> 0`` filter downstream and turn
    # an identifiability ratio into ~1e17 instead of a clean null.
    rng = np.random.default_rng(10)

    std = block_bootstrap_std(rng.normal(size=(2, 60)), n_blocks=1, n_resamples=16)

    assert np.array_equal(std, np.zeros(2))


def test_block_bootstrap_rejects_a_statistic_that_ignores_its_axis() -> None:
    rng = np.random.default_rng(11)

    with pytest.raises(ValueError, match="reduce only the axis"):
        block_bootstrap_std(
            rng.normal(size=(2, 100)), statistic=lambda x, axis: np.mean(x)
        )


def test_window_sampling_std_matches_the_statistics_it_floors() -> None:
    # The floor has to be shaped exactly like the statistic it divides into, or
    # the identifiability ratio silently broadcasts across sensors.
    rng = np.random.default_rng(12)
    members = _series(rng, 120, n_ensemble=3)

    stats = window_statistics(members, SIM_TIME, NUM_WINDOWS)
    floor = window_sampling_std(
        members, SIM_TIME, NUM_WINDOWS, n_blocks=5, n_resamples=32
    )

    for name in ("mean", "variance"):
        assert floor[name].dims == stats[name].dims
        assert floor[name].shape == stats[name].shape
        assert np.isfinite(floor[name].values).all()


# ---------------------------------------------------------------------------
# Ranks
# ---------------------------------------------------------------------------


def test_ensemble_rank_counts_the_members_below_the_truth() -> None:
    members = np.array([[0.0], [1.0], [2.0], [3.0]])

    assert ensemble_rank(members, np.array([-1.0])) == 0
    assert ensemble_rank(members, np.array([1.5])) == 2
    assert ensemble_rank(members, np.array([9.0])) == 4


def test_ensemble_ranks_are_uniform_for_a_calibrated_ensemble() -> None:
    # The third calibration view: ranks use only the ordering, so they catch a
    # shape failure the CRPS and the z-score cannot see.
    rng = np.random.default_rng(13)
    n_members, n_knots = 8, 4000
    ranks = ensemble_rank(
        rng.normal(size=(n_members, n_knots)), rng.normal(size=n_knots), rng=rng
    )

    counts = np.bincount(ranks.astype(int), minlength=n_members + 1)
    expected = n_knots / (n_members + 1)
    assert np.abs(counts - expected).max() < 4.0 * np.sqrt(expected)


def test_ensemble_ranks_pile_up_when_the_ensemble_is_biased() -> None:
    rng = np.random.default_rng(14)
    n_knots = 2000
    ranks = ensemble_rank(
        rng.normal(size=(8, n_knots)) - 3.0, rng.normal(size=n_knots), rng=rng
    )

    assert (ranks == 8).mean() > 0.9


def test_ensemble_rank_ties_are_drawn_not_piled_at_one_end() -> None:
    # A fully collapsed ensemble matching the truth is all ties. A deterministic
    # tie-break would put every one of them at rank 0 (or M) and read as a
    # catastrophic bias that is not there.
    members = np.zeros((4, 2000))
    ranks = ensemble_rank(members, np.zeros(2000))

    assert set(np.unique(ranks)) == {0.0, 1.0, 2.0, 3.0, 4.0}
    assert ranks.mean() == pytest.approx(2.0, abs=0.15)


def test_ensemble_rank_is_null_where_a_member_is_not_finite() -> None:
    members = np.array([[0.0, 0.0], [1.0, np.nan]])

    ranks = ensemble_rank(members, np.array([0.5, 0.5]))

    assert ranks[0] == 1
    assert np.isnan(ranks[1])


# ---------------------------------------------------------------------------
# The summary block
# ---------------------------------------------------------------------------


def _summary(
    rng: np.random.Generator,
    n_members: int = 8,
    n_time: int = 120,
    with_prior: bool = True,
    prior_offset: float = 1.0,
) -> dict:
    truth = _series(rng, n_time)
    posterior = _series(rng, n_time, n_ensemble=n_members)
    prior = (
        _series(rng, n_time, n_ensemble=n_members, offset=prior_offset)
        if with_prior
        else None
    )
    boot = {"n_blocks": 5, "n_resamples": 32}
    # ``dict(...)``: ``evaluation.scores`` is mypy-waived and returns ``Any``.
    return dict(
        window_statistics_summary(
            window_statistics(truth, SIM_TIME, NUM_WINDOWS),
            window_statistics(posterior, SIM_TIME, NUM_WINDOWS),
            prior_stats=(
                window_statistics(prior, SIM_TIME, NUM_WINDOWS)
                if prior is not None
                else None
            ),
            posterior_sampling_std=window_sampling_std(
                posterior, SIM_TIME, NUM_WINDOWS, **boot
            ),
            prior_sampling_std=(
                window_sampling_std(prior, SIM_TIME, NUM_WINDOWS, **boot)
                if prior is not None
                else None
            ),
        )
    )


def test_summary_scores_every_statistic_and_quantity_separately() -> None:
    # Separately identifiable: a parameter that fixes the mean wind while
    # leaving the resolved variance halved is exactly what this block names.
    summary = _summary(np.random.default_rng(15))

    assert set(summary["posterior"]) == {
        f"{stat}_{q}"
        for stat in ("mean", "variance")
        for q in ("u", "v", "w", "magnitude")
    }
    entry = summary["posterior"]["mean_magnitude"]
    assert set(entry) >= {"crps", "z_score", "ranks", "identifiability"}
    assert entry["crps"]["mean"] > 0
    assert len(entry["ranks"]) == NUM_WINDOWS * 2  # windows x sensors


def test_summary_scores_the_prior_and_the_skill_against_it() -> None:
    # A prior displaced from the truth must score worse, so the skill is positive.
    summary = _summary(np.random.default_rng(16), prior_offset=3.0)

    assert set(summary["prior"]) == set(summary["posterior"])
    entry = summary["posterior"]["mean_u"]
    assert entry["prior_crps_mean"] > entry["crps"]["mean"]
    assert entry["crps_reduction_vs_prior"] > 0.5


def test_summary_no_ops_on_the_prior_when_it_was_not_saved() -> None:
    # run.save_prior_state is off by default, so absence is the common case.
    summary = _summary(np.random.default_rng(17), with_prior=False)

    assert "prior" not in summary
    assert "crps_reduction_vs_prior" not in summary["posterior"]["mean_u"]


def test_summary_identifiability_flags_a_statistic_the_window_cannot_resolve(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Members that differ only by their own turbulent noise: the across-member
    # spread is the sampling spread, so the ratio sits near 1 and the statistic
    # carries no parameter information however good its CRPS reads.
    with caplog.at_level("WARNING"):
        summary = _summary(np.random.default_rng(18), n_members=6)

    assert summary["posterior"]["mean_u"]["identifiability"]["mean"] < 3.0
    assert "not identifiable at this window length" in caplog.text


def test_summary_identifiability_rises_when_the_members_genuinely_differ() -> None:
    # The direction check the noise-only case cannot make: with members offset
    # well beyond their own sampling spread the ratio must go UP. Inverted
    # (floor/spread) it would go down, and the "< 3" test below still passes.
    rng = np.random.default_rng(24)
    truth = _series(rng, 120)
    members = xarray.concat(
        [_series(rng, 120) + offset for offset in np.linspace(-3.0, 3.0, 6)],
        dim="ensemble",
    ).transpose("component", "ensemble", "time", "sensor")

    summary = window_statistics_summary(
        window_statistics(truth, SIM_TIME, NUM_WINDOWS),
        window_statistics(members, SIM_TIME, NUM_WINDOWS),
        posterior_sampling_std=window_sampling_std(
            members, SIM_TIME, NUM_WINDOWS, n_blocks=5, n_resamples=32
        ),
    )

    assert summary["posterior"]["mean_u"]["identifiability"]["mean"] > 3.0


def test_summary_is_omitted_when_the_members_axis_is_absent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An ensemble-mean-only artifact has nothing to score probabilistically.
    # Invariant 3: the block goes away, the metric stage does not.
    rng = np.random.default_rng(25)
    truth = _series(rng, 60)

    with caplog.at_level("WARNING"):
        summary = window_statistics_summary(
            window_statistics(truth, SIM_TIME, NUM_WINDOWS),
            window_statistics(truth, SIM_TIME, NUM_WINDOWS),
            label="assimilation",
        )

    assert summary == {}
    assert "no ensemble dimension" in caplog.text


def test_summary_identifiability_is_absent_when_no_floor_was_measured() -> None:
    # Short windows give no bootstrap, and a ratio over an unmeasured floor is
    # not "infinitely identifiable" -- it is unknown.
    rng = np.random.default_rng(19)
    truth = _series(rng, 90)
    posterior = _series(rng, 90, n_ensemble=4)

    summary = window_statistics_summary(
        window_statistics(truth, SIM_TIME, NUM_WINDOWS),
        window_statistics(posterior, SIM_TIME, NUM_WINDOWS),
        posterior_sampling_std=window_sampling_std(
            posterior, SIM_TIME, NUM_WINDOWS, n_blocks=200
        ),
    )

    assert "identifiability" not in summary["posterior"]["mean_u"]


def test_summary_degrades_on_the_smoke_shape_rather_than_inventing_numbers() -> None:
    # M=2: z is Cauchy, so no moment of it converges and the whole z_score entry
    # is null by policy (WP1.2). The CRPS and the ranks stay well defined.
    summary = _summary(np.random.default_rng(20), n_members=2, n_time=12)

    entry = summary["posterior"]["mean_magnitude"]
    assert summary["n_members"] == 2
    assert entry["z_score"] is None
    assert entry["crps"]["mean"] >= 0
    assert all(0 <= r <= 2 for r in entry["ranks"])


def test_summary_survives_a_window_whose_frames_are_missing() -> None:
    # Invariant 3 end to end: a truncated run nulls the windows it lost and
    # scores the ones it has, silently (no numpy all-NaN-slice warnings) and
    # without taking the block down.
    rng = np.random.default_rng(22)
    truth = _series(rng, 90).isel(time=slice(0, 60))  # window 2 never ran
    posterior = _series(rng, 90, n_ensemble=4).isel(time=slice(0, 60))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        summary = window_statistics_summary(
            window_statistics(truth, SIM_TIME, NUM_WINDOWS),
            window_statistics(posterior, SIM_TIME, NUM_WINDOWS),
        )

    crps = summary["posterior"]["mean_u"]["crps"]
    assert crps["final"] is None  # the missing last window
    assert crps["mean"] > 0  # the two that ran


def test_summary_is_yaml_safe(tmp_path: pathlib.Path) -> None:
    # The block is written straight to run_summary.yaml through the pipeline's
    # own writer; a numpy scalar surviving into it would raise there (safe_dump
    # has no representer for one), so this goes through the real round trip
    # rather than asserting about types.
    from scripts.esmda._esmda_common import read_yaml, write_yaml

    summary = _summary(np.random.default_rng(21))
    path = tmp_path / "run_summary.yaml"
    write_yaml({"sensor_statistics": {"assimilation": summary}}, path)

    assert "!!python" not in path.read_text()
    assert read_yaml(path)["sensor_statistics"]["assimilation"]["n_members"] == 8


# ---------------------------------------------------------------------------
# The streaming extraction this block is built on
# ---------------------------------------------------------------------------


def _window_state_file(
    tmp_path: pathlib.Path,
    window: int,
    n_ensemble: int = 3,
    n_time: int = 4,
    n_cells: int = 5,
) -> pathlib.Path:
    """A ``(ensemble, time, z, y, x)`` window state file on a pylbm-shaped grid."""
    rng = np.random.default_rng(100 + window)
    axis = np.linspace(0.0, 20.0, n_cells)
    shape = (n_ensemble, n_time, n_cells, n_cells, n_cells)
    ds = xarray.Dataset(
        {
            name: (("ensemble", "time", "z", "y", "x"), rng.normal(size=shape))
            for name in ("u", "v", "w")
        },
        coords={
            "ensemble": np.arange(n_ensemble),
            "time": np.arange(n_time, dtype=float),
            "z": axis,
            "y": axis,
            "x": axis,
        },
    )
    path = tmp_path / f"window_{window}_posterior_state.nc"
    ds.to_netcdf(path)
    return path


_SENSOR_SETS = {
    "assimilation": (
        np.array([3.0, 11.0]),
        np.array([4.0, 12.0]),
        np.array([5.0, 6.0]),
    ),
    "validation": (np.array([7.0]), np.array([8.0]), np.array([9.0])),
}


def test_ensemble_sensor_series_streams_member_at_a_time(
    tmp_path: pathlib.Path,
) -> None:
    # Master-plan invariant 2: full-ensemble window state files are gigabytes and
    # must never be read whole. This was the last site that did.
    from scripts.esmda import _esmda_common

    paths = [_window_state_file(tmp_path, w) for w in range(2)]
    seen: list[int | None] = []
    original = _esmda_common._sensor_component_timeseries

    def _spy(state: xarray.Dataset, *args: object, **kwargs: object) -> object:
        seen.append(state.sizes.get("ensemble"))
        return original(state, *args, **kwargs)

    _esmda_common._sensor_component_timeseries = _spy
    try:
        _esmda_common.ensemble_sensor_series(paths, _SENSOR_SETS, "pylbm", 4.0)
    finally:
        _esmda_common._sensor_component_timeseries = original

    # 2 windows x 3 members x 2 sensor sets, never more than one member at a time.
    assert seen == [1] * 12


def test_ensemble_sensor_series_preserves_the_member_axis(
    tmp_path: pathlib.Path,
) -> None:
    # The per-member slices are concatenated back; a lost or reordered
    # ``ensemble`` coord would silently re-label every member's scores.
    from scripts.esmda._esmda_common import ensemble_sensor_series

    paths = [_window_state_file(tmp_path, w) for w in range(2)]
    series = ensemble_sensor_series(paths, _SENSOR_SETS, "pylbm", 4.0)["assimilation"]

    assert series.sizes["ensemble"] == 3
    assert list(series["ensemble"].values) == [0, 1, 2]


def test_ensemble_sensor_series_is_inert_under_the_streaming_rewrite(
    tmp_path: pathlib.Path,
) -> None:
    # The rewrite is a memory change and nothing else: the same numbers, the same
    # dims, the same global time axis, in the same member order.
    from scripts.esmda._esmda_common import (
        _sensor_component_timeseries,
        ensemble_sensor_series,
    )

    sim_time = 4.0
    paths = [_window_state_file(tmp_path, w) for w in range(2)]
    streamed = ensemble_sensor_series(paths, _SENSOR_SETS, "pylbm", sim_time)

    for name, (ox, oy, oz) in _SENSOR_SETS.items():
        # The pre-WP1.3 implementation: open each window whole, interpolate once.
        pieces = []
        for w, path in enumerate(paths):
            whole = xarray.open_dataset(path).load()
            t = np.asarray(whole["time"].values, dtype=float)
            vel = _sensor_component_timeseries(whole, ox, oy, oz, "pylbm")
            pieces.append(vel.assign_coords(time=(t - t[0]) + w * sim_time))
            whole.close()
        expected = xarray.concat(pieces, dim="time", join="override")

        assert streamed[name].dims == expected.dims
        assert np.array_equal(streamed[name].values, expected.values)
        assert np.array_equal(streamed[name]["time"].values, expected["time"].values)


# ---------------------------------------------------------------------------
# run_summary.yaml wiring
# ---------------------------------------------------------------------------

_RUN_WINDOWS = 2
_RUN_SIM_TIME = 4.0
_RUN_FRAMES = 6  # per window


def _state_dataset(
    n_ensemble: int | None, n_time: int, seed: int, n_cells: int = 5
) -> xarray.Dataset:
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 20.0, n_cells)
    dims: tuple[str, ...] = ("time", "z", "y", "x")
    shape: tuple[int, ...] = (n_time, n_cells, n_cells, n_cells)
    coords = {
        "time": np.arange(n_time, dtype=float) * (_RUN_SIM_TIME / n_time),
        "z": axis,
        "y": axis,
        "x": axis,
    }
    if n_ensemble is not None:
        dims = ("ensemble",) + dims
        shape = (n_ensemble,) + shape
        coords["ensemble"] = np.arange(n_ensemble)
    return xarray.Dataset(
        {n: (dims, rng.normal(size=shape) + 2.0) for n in ("u", "v", "w")},
        coords=coords,
    )


def _full_run_dir(
    tmp_path: pathlib.Path, n_members: int = 4, save_prior_state: bool = False
) -> pathlib.Path:
    """An ESMDA run dir complete enough for the non-``skip_viz`` metric path."""
    from scripts.esmda._esmda_common import write_yaml

    run_dir = tmp_path / "run"
    (run_dir / "windows").mkdir(parents=True)

    write_yaml(
        {
            "run": {"skip_viz": False},
            "obs": {
                "mode": "points",
                "x_points": [4.0, 12.0],
                "y_points": [5.0, 13.0],
                "z_points": [6.0, 7.0],
                "validation_x_points": [8.0],
                "validation_y_points": [9.0],
                "validation_z_points": [10.0],
            },
        },
        run_dir / "config.yaml",
    )
    write_yaml(
        {"configuration": {"ensemble_size": n_members}}, run_dir / "run_info.yaml"
    )

    # Truth: one global time axis across both windows.
    truth = _state_dataset(None, _RUN_WINDOWS * _RUN_FRAMES, seed=1)
    truth = truth.assign_coords(
        time=np.linspace(
            0.0,
            _RUN_WINDOWS * _RUN_SIM_TIME,
            _RUN_WINDOWS * _RUN_FRAMES,
            endpoint=False,
        )
    )
    truth.to_netcdf(run_dir / "truth_state.nc")
    write_yaml(
        {
            "true_state_path": str(run_dir / "truth_state.nc"),
            "n_total": _RUN_WINDOWS * _RUN_FRAMES,
            "x_offset": 0.0,
            "start_idx": 0,
            "t_offset": 0.0,
            "num_windows": _RUN_WINDOWS,
            "n_per_window": _RUN_FRAMES,
            "sim_time": _RUN_SIM_TIME,
            "truth_solver_name": "pylbm",
            "assim_solver_name": "pylbm",
        },
        run_dir / "truth_access.yaml",
    )

    for w in range(_RUN_WINDOWS):
        _state_dataset(n_members, _RUN_FRAMES, seed=10 + w).to_netcdf(
            run_dir / "windows" / f"window_{w}_posterior_state.nc"
        )
        if save_prior_state:
            _state_dataset(n_members, _RUN_FRAMES, seed=50 + w).to_netcdf(
                run_dir / "windows" / f"window_{w}_prior_state.nc"
            )
    _state_dataset(None, _RUN_WINDOWS * _RUN_FRAMES, seed=20).to_netcdf(
        run_dir / "posterior_state_mean.nc"
    )

    members = np.linspace(-1.0, 1.0, n_members)
    for name, values in (
        ("posterior_params", members),
        ("prior_params", members * 4.0),
    ):
        xarray.Dataset(
            {"inflow_angle": (("ensemble", "time"), np.repeat(values[:, None], 2, 1))},
            coords={"ensemble": np.arange(n_members), "time": [0.0, 1.0]},
        ).to_netcdf(run_dir / f"{name}.nc")
    xarray.Dataset(
        {"inflow_angle": (("time",), [0.0, 0.0])}, coords={"time": [0.0, 1.0]}
    ).to_netcdf(run_dir / "true_params.nc")
    return run_dir


def test_run_summary_carries_sensor_statistics_for_every_sensor_set(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = _full_run_dir(tmp_path)
    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    block = summary["sensor_statistics"]
    assert set(block) == {"assimilation", "validation"}
    assimilation = block["assimilation"]
    assert assimilation["n_members"] == 4
    assert assimilation["n_windows"] == _RUN_WINDOWS
    assert assimilation["num_sensors"] == 2
    assert block["validation"]["num_sensors"] == 1
    # Held out from the update but scored all the same -- the point of §4.2.
    assert block["validation"]["posterior"]["mean_magnitude"]["crps"]["mean"] > 0

    # Additive: WP1.1/WP1.2 keys are untouched.
    assert summary["metrics_version"] == 2
    assert "parameter_metrics" in summary and "sensor_metrics" in summary


def test_run_summary_sensor_statistics_no_op_without_saved_prior_states(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = _full_run_dir(tmp_path, save_prior_state=False)
    with caplog.at_level("INFO"):
        compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert "prior" not in summary["sensor_statistics"]["assimilation"]
    assert "No prior sensor statistics" in caplog.text


def test_run_summary_survives_an_unreadable_prior_state_file(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A job killed mid-``to_netcdf`` leaves a prior state file that exists but
    # cannot be opened. Invariant 3: that costs the prior half of one block, not
    # run_summary.yaml -- which would take the parameter, state and sensor
    # metrics down with it.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = _full_run_dir(tmp_path, save_prior_state=True)
    (run_dir / "windows" / "window_1_prior_state.nc").write_bytes(b"not a netcdf")

    with caplog.at_level("WARNING"):
        compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert "Cannot read the prior sensor series" in caplog.text
    assert "prior" not in summary["sensor_statistics"]["assimilation"]
    # Everything the stage had already computed survives.
    assert summary["parameter_metrics"] and summary["sensor_metrics"]
    assert summary["sensor_statistics"]["assimilation"]["posterior"]


def test_run_summary_sensor_statistics_score_the_prior_when_it_was_saved(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = _full_run_dir(tmp_path, save_prior_state=True)
    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    posterior = summary["sensor_statistics"]["assimilation"]["posterior"]
    assert set(summary["sensor_statistics"]["assimilation"]["prior"]) == set(posterior)
    assert "crps_reduction_vs_prior" in posterior["mean_u"]
