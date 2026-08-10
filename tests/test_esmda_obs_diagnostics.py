"""Phase 2: the observation-space artifacts (WP2.1) and ``O_N`` (WP2.2).

The normalized data mismatch is the one ESMDA diagnostic that distinguishes
under-fitting from *fitting the observation noise* — the two failures no RMSE
separates — so what it is worth depends entirely on the ``1/2`` target being the
real one. That is pinned here against an exact linear-Gaussian posterior
(``test_data_mismatch_is_half_on_a_linear_gaussian_posterior``), not against a
recorded number, because a recorded number would survive the formula drifting.

The rest guards the plumbing the target rides on:

  * the flat observation vector's ``(interval, state, sensor)`` decoding matches
    what the observation operator actually concatenates — a silent transposition
    there would mislabel every saved array while leaving ``O_N`` itself correct;
  * the smoother records one predicted-observation block per iteration plus the
    posterior forecast, and **nothing** when the flag is off;
  * every consumer no-ops on a run dir that predates WP2.1 (invariant 3).
"""

from __future__ import annotations

import pathlib
from typing import Callable

import numpy as np
import pytest
import xarray
from evaluation.figures import _d3_box_layout, plot_data_mismatch_decay
from evaluation.scores import (
    DATA_MISMATCH_TARGET,
    data_mismatch,
    data_mismatch_summary,
    data_mismatch_target_band,
)
from matplotlib.axes import Axes
from omegaconf import DictConfig

from scripts.esmda._esmda_common import obs_diagnostics_bundle
from scripts.esmda.run_esmda import _obs_index_coords, _save_obs_diagnostics

# ---------------------------------------------------------------------------
# O_N against the chi-squared target
# ---------------------------------------------------------------------------


def _linear_gaussian_scores(
    rng: np.random.Generator, n_obs: int, n_params: int, n_members: int
) -> tuple[float, float]:
    """``(mean O_N over posterior samples, O_N of the posterior mean)``.

    One realization of ``d = G θ_true + ε`` with a ``N(0, I)`` prior on ``θ``,
    conditioned in closed form — no ESMDA, no approximation, so the only thing
    under test is the estimator.
    """
    sigma = 0.3
    G = rng.normal(size=(n_obs, n_params))
    theta_true = rng.normal(size=n_params)
    obs = G @ theta_true + sigma * rng.normal(size=n_obs)

    covariance = np.linalg.inv(np.eye(n_params) + G.T @ G / sigma**2)
    mean = covariance @ (G.T @ obs / sigma**2)
    theta = mean[:, None] + np.linalg.cholesky(covariance) @ rng.normal(
        size=(n_params, n_members)
    )

    std = np.full(n_obs, sigma)
    samples = data_mismatch(obs, G @ theta, std)
    assert samples.shape == (n_members,)
    return float(samples.mean()), float(data_mismatch(obs, (G @ mean)[:, None], std)[0])


def test_data_mismatch_is_half_on_a_linear_gaussian_posterior() -> None:
    """Posterior samples of an exact linear-Gaussian problem score ``O_N ≈ 1/2``.

    The whole diagnostic is "compare against ½", so the target is verified
    rather than assumed.

    Averaged over independent realizations of ``d``, not over members of one:
    every member is scored against the *same* ``d``, so ``E[O_N | d]`` itself
    fluctuates with ``1/√(2N_d) ≈ 0.04`` and the across-member mean concentrates
    on that, not on ½. A tolerance derived from the member count would be a seed
    lottery — only ~27 % of seeds pass at ``±0.01``. With ``R`` realizations the
    standard error is ``1/√(2·N_d·R)``, and the tolerance below is 3× it.
    """
    rng = np.random.default_rng(7)
    n_obs, n_params, realizations = 300, 5, 12

    means = [
        _linear_gaussian_scores(rng, n_obs, n_params, n_members=400)[0]
        for _ in range(realizations)
    ]
    tolerance = 3.0 / np.sqrt(2.0 * n_obs * realizations)
    assert abs(float(np.mean(means)) - DATA_MISMATCH_TARGET) < tolerance


def test_data_mismatch_of_the_posterior_mean_is_below_the_target() -> None:
    """The posterior *mean* scores ``(N_d − P)/2N_d``, not ½.

    The distinction the formula has to preserve: an implementation that scored
    the ensemble mean instead of the members would land here, and at the
    well-determined shape of the test above (``P/N_d = 1/60``) the two values
    differ by less than the noise. So this uses a deliberately
    poorly-determined shape — ``P = N_d/2``, where the target is ½ and the
    posterior mean is ¼ — and the assertion is that the two are *separated*.
    """
    rng = np.random.default_rng(11)
    n_obs, n_params, realizations = 40, 20, 24

    pairs = [
        _linear_gaussian_scores(rng, n_obs, n_params, n_members=400)
        for _ in range(realizations)
    ]
    sample_mean = float(np.mean([p[0] for p in pairs]))
    posterior_mean = float(np.mean([p[1] for p in pairs]))

    expected = (n_obs - n_params) / (2.0 * n_obs)
    tolerance = 3.0 / np.sqrt(2.0 * n_obs * realizations)
    assert sample_mean == pytest.approx(DATA_MISMATCH_TARGET, abs=tolerance)
    assert posterior_mean == pytest.approx(expected, abs=tolerance)
    # And the two are genuinely distinguishable, which is the point. The true
    # gap is P/2N_d = 1/4 here, comfortably clear of the sampling noise.
    assert posterior_mean < sample_mean - 2.0 * tolerance


def test_data_mismatch_scales_with_the_standardized_residual() -> None:
    """A residual of exactly ``k·σ`` at every observation scores ``k²/2``."""
    obs = np.zeros(16)
    sigma = np.full(16, 0.25)
    for k in (0.0, 1.0, 2.0, 3.0):
        pred = np.full((16, 3), k * 0.25)
        assert data_mismatch(obs, pred, sigma) == pytest.approx(np.full(3, k**2 / 2.0))


def test_data_mismatch_target_band_is_three_sigma() -> None:
    assert data_mismatch_target_band(200) == pytest.approx(3.0 / np.sqrt(400.0))
    # No observations -> no band, rather than an invented tolerance.
    assert data_mismatch_target_band(0) is None
    assert data_mismatch_target_band(-4) is None


def test_data_mismatch_drops_unusable_observations() -> None:
    """A non-finite obs or a non-positive sigma leaves the average, not the run."""
    obs = np.array([0.0, 0.0, np.nan, 0.0])
    sigma = np.array([1.0, 0.0, 1.0, 1.0])  # second is unusable
    pred = np.zeros((4, 2)) + 1.0
    # Only observations 0 and 3 are usable: mean(1**2, 1**2)/2 = 0.5.
    assert data_mismatch(obs, pred, sigma) == pytest.approx(np.full(2, 0.5))

    # Nothing usable at all -> nan per member, never a flattering 0.
    all_bad = data_mismatch(np.full(3, np.nan), np.zeros((3, 2)), np.ones(3))
    assert np.all(np.isnan(all_bad))


def test_data_mismatch_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        data_mismatch(np.zeros(4), np.zeros(4), np.ones(4))
    with pytest.raises(ValueError, match="observations but obs has"):
        data_mismatch(np.zeros(4), np.zeros((5, 2)), np.ones(4))


# ---------------------------------------------------------------------------
# The run_summary block
# ---------------------------------------------------------------------------


def test_data_mismatch_summary_flags_underfit_and_overfit() -> None:
    n_obs = 200
    band = data_mismatch_target_band(n_obs)

    underfit = data_mismatch_summary([np.full(20, 8.0), np.full(20, 4.0)], n_obs)
    assert underfit["underfit_final"] is True
    assert underfit["overfit_final"] is False
    assert underfit["per_step_median"] == pytest.approx([8.0, 4.0])
    assert underfit["target"] == DATA_MISMATCH_TARGET
    assert underfit["target_band"] == pytest.approx(band)
    assert underfit["caveat"] == "no_representativeness_error"

    # Fitting the noise: well below the band, which no RMSE would flag.
    overfit = data_mismatch_summary([np.full(20, 4.0), np.full(20, 0.01)], n_obs)
    assert overfit["overfit_final"] is True
    assert overfit["underfit_final"] is False

    # On target: neither flag.
    healthy_values = 0.5 + 0.02 * np.linspace(-1.0, 1.0, 40)
    healthy = data_mismatch_summary([np.full(40, 4.0), healthy_values], n_obs)
    assert healthy["underfit_final"] is False
    assert healthy["overfit_final"] is False


def test_data_mismatch_summary_flags_collapse_only_when_off_target() -> None:
    """A vanishing across-member IQR is the pathology only away from the target."""
    n_obs = 200
    window = [np.stack([np.full(20, 4.0), np.full(20, 4.0)])]
    collapsed = data_mismatch_summary(
        [np.full(20, 4.0), np.full(20, 4.0)], n_obs, per_window=window
    )
    assert collapsed["collapsed"] is True
    assert collapsed["per_step_iqr"][-1] == pytest.approx(0.0)

    # Identical members that agree ON the target are converged, not collapsed.
    on_target = [np.stack([np.full(20, 4.0), np.full(20, 0.5)])]
    converged = data_mismatch_summary(
        [np.full(20, 4.0), np.full(20, 0.5)], n_obs, per_window=on_target
    )
    assert converged["collapsed"] is False


def test_data_mismatch_summary_judges_collapse_per_window_not_pooled() -> None:
    """The verdict is ACROSS MEMBERS, so a rollout is judged window by window.

    Pooling the windows measures the drift of ``O_N`` from one window to the
    next — entirely healthy — and reads it as spread. A run where every window
    has collapsed on a different value would then report a large pooled IQR and
    ``collapsed: False``, which is backwards.
    """
    n_obs = 200
    # Each window's members sit on top of each other, at different levels.
    per_window = [
        np.stack([np.full(20, 20.0), np.full(20, level)]) for level in (8.0, 2.0)
    ]
    pooled = [np.full(40, 20.0), np.concatenate([np.full(20, 8.0), np.full(20, 2.0)])]

    assert data_mismatch_summary(pooled, n_obs, per_window=per_window)["collapsed"] is (
        True
    )
    # The pooled view alone cannot see it -- which is why `per_window` exists.
    assert data_mismatch_summary(pooled, n_obs)["collapsed"] is None

    # One healthy window is enough to withhold the verdict.
    mixed = [per_window[0], np.stack([np.full(20, 20.0), np.linspace(0.2, 4.0, 20)])]
    assert data_mismatch_summary(pooled, n_obs, per_window=mixed)["collapsed"] is False


def test_data_mismatch_summary_will_not_judge_collapse_on_a_smoke_ensemble() -> None:
    """The 2-member smoke shape has no meaningful IQR, so ``collapsed`` is None.

    Without the guard this is the *default* smoke outcome: two near-identical
    members give a tiny IQR, and a smoke run's median is nowhere near the
    target, so every CI run would publish ``collapsed: true``. Judging per
    window is what keeps the guard effective on a rollout — pooling would sail
    a 2-member run past a pooled count threshold once it had 4 windows.
    """
    smoke_window = [np.array([[9.0, 9.001], [8.0, 8.001]])]
    smoke = data_mismatch_summary(
        [np.array([9.0, 9.001]), np.array([8.0, 8.001])], 200, per_window=smoke_window
    )
    assert smoke["collapsed"] is None
    # The rest of the block is still reported -- only the IQR verdict abstains.
    assert smoke["underfit_final"] is True
    assert smoke["per_step_median"] == pytest.approx([9.0005, 8.0005])

    # Four windows of two members each: still None, not a verdict off 8 values.
    assert (
        data_mismatch_summary(
            [np.full(8, 9.0), np.full(8, 8.0)], 200, per_window=smoke_window * 4
        )["collapsed"]
        is None
    )


def test_data_mismatch_summary_reports_which_step_the_flags_describe() -> None:
    """A failed posterior forecast falls back, and says which step it used."""
    n_obs = 200
    full = data_mismatch_summary([np.full(20, 4.0), np.full(20, 0.5)], n_obs)
    assert full["final_step_index"] == 1

    # The last iteration produced nothing: the flags describe iteration 0, and
    # `final_step_index` is the only way a reader could tell.
    fell_back = data_mismatch_summary([np.full(20, 4.0), np.full(20, np.nan)], n_obs)
    assert fell_back["final_step_index"] == 0
    assert fell_back["underfit_final"] is True
    assert fell_back["per_step_median"][1] is None


def test_data_mismatch_summary_degrades_without_values() -> None:
    assert data_mismatch_summary([], 200) is None
    assert data_mismatch_summary(None, 200) is None
    assert data_mismatch_summary([np.full(4, np.nan)], 200) is None

    # No observations -> the flags are None (unjudgeable), not False (checked).
    # 20 values, so `collapsed` abstains for want of a BAND, not for want of
    # values (which the smoke-shape test above covers separately).
    no_band = data_mismatch_summary([np.full(20, 3.0)], 0)
    assert no_band["target_band"] is None
    assert no_band["underfit_final"] is None
    assert no_band["overfit_final"] is None
    assert no_band["collapsed"] is None


# ---------------------------------------------------------------------------
# The saved arrays: layout and round-trip
# ---------------------------------------------------------------------------


def test_obs_index_coords_matches_the_operator_flattening() -> None:
    """The decoded (interval, state, sensor) labels match a real operator's order.

    ``_obs_index_coords`` hard-codes the nesting the observation operator builds
    by concatenation. Rather than restate the formula, this drives the actual
    operator with a field whose value at a point *encodes* which sensor it is,
    and checks the decoded labels select the right entries.
    """
    from data_assimilation.observation_operator import (
        ObservationOperator,
        TemporalObservationOperator,
    )

    n_sensors, states = 3, ["u", "v"]
    base = ObservationOperator(
        obs_ids_x=list(range(n_sensors)),
        obs_ids_y=[0] * n_sensors,
        obs_ids_z=[0] * n_sensors,
        obs_states=states,
        solver_name="pylbm",
    )
    operator = TemporalObservationOperator(
        base, mode="intervals", interval_seconds=1.0, aggregation_mode="mean"
    )

    # u = 100*interval + sensor, v = u + 10, constant within an interval, so
    # every flat entry is self-identifying.
    times = np.array([0.0, 1.0, 2.0])
    x = np.arange(n_sensors, dtype=float)
    u = times[:, None] * 100.0 + x[None, :]
    state = xarray.Dataset(
        {
            "u": (("time", "z", "y", "x"), u[:, None, None, :]),
            "v": (("time", "z", "y", "x"), (u + 10.0)[:, None, None, :]),
        },
        coords={"time": times, "z": [0.0], "y": [0.0], "x": x},
    )

    flat = np.asarray(operator(state)).ravel()
    coords = _obs_index_coords(operator, flat.size)
    assert set(coords) == {"obs_sensor", "obs_state", "obs_interval"}

    sensor = np.asarray(coords["obs_sensor"][1])
    which = np.asarray(coords["obs_state"][1])
    interval = np.asarray(coords["obs_interval"][1])
    expected = interval * 100.0 + sensor + np.where(which == "v", 10.0, 0.0)
    assert flat == pytest.approx(expected)


def test_obs_index_coords_degrades_on_an_indivisible_length() -> None:
    """An observation count that is not whole blocks costs metadata, not the file."""

    class _Odd:
        num_sensors = 3
        obs_states = ["u", "v"]

    assert _obs_index_coords(_Odd(), 7) == {}
    # And an operator exposing nothing at all.
    assert _obs_index_coords(object(), 12) == {}


def test_saved_obs_diagnostics_round_trip(tmp_path: pathlib.Path) -> None:
    """What ``run_esmda`` writes is what ``obs_diagnostics_bundle`` scores."""
    windows_dir = tmp_path / "windows"
    windows_dir.mkdir()

    n_obs, n_members, n_steps, n_windows = 12, 5, 3, 2
    rng = np.random.default_rng(3)
    sigma = np.full(n_obs, 0.5)
    expected = []
    for window in range(n_windows):
        obs = rng.normal(size=n_obs)
        history = [rng.normal(size=(n_obs, n_members)) for _ in range(n_steps)]
        params = xarray.Dataset(
            {"a": (("esmda_step", "ensemble"), rng.normal(size=(n_steps, n_members)))}
        )
        _save_obs_diagnostics(
            windows_dir, window, obs, obs, sigma, history, params, object()
        )
        expected.append(
            np.stack([data_mismatch(obs, block, sigma) for block in history])
        )

    for window in range(n_windows):
        for suffix in ("obs", "pred_obs", "params_steps"):
            assert (windows_dir / f"window_{window}_{suffix}.nc").exists()

    bundle = obs_diagnostics_bundle(tmp_path)
    assert bundle is not None
    assert bundle["num_windows"] == n_windows
    assert bundle["num_observations"] == n_obs
    assert len(bundle["per_step"]) == n_steps
    for step in range(n_steps):
        # Members pooled across windows, in window order.
        assert bundle["per_step"][step] == pytest.approx(
            np.concatenate([e[step] for e in expected])
        )
    for window in range(n_windows):
        assert bundle["per_window"][window] == pytest.approx(expected[window])


def test_obs_diagnostics_bundle_absent_on_a_pre_wp21_run_dir(
    tmp_path: pathlib.Path,
) -> None:
    """Invariant 3: an old run dir logs and skips rather than raising."""
    (tmp_path / "windows").mkdir()
    assert obs_diagnostics_bundle(tmp_path) is None
    # And a run dir with no ``windows/`` at all.
    assert obs_diagnostics_bundle(tmp_path / "nowhere") is None


def test_obs_diagnostics_bundle_skips_a_window_missing_its_pred_obs(
    tmp_path: pathlib.Path,
) -> None:
    """A half-written window costs itself, not the whole diagnostic."""
    windows_dir = tmp_path / "windows"
    windows_dir.mkdir()
    rng = np.random.default_rng(11)
    sigma = np.full(6, 0.5)
    params = xarray.Dataset(
        {"a": (("esmda_step", "ensemble"), rng.normal(size=(2, 4)))}
    )
    _save_obs_diagnostics(
        windows_dir,
        0,
        rng.normal(size=6),
        rng.normal(size=6),
        sigma,
        [rng.normal(size=(6, 4)), rng.normal(size=(6, 4))],
        params,
        object(),
    )
    # Window 1's obs file exists but its pred_obs never landed.
    _save_obs_diagnostics(
        windows_dir,
        1,
        rng.normal(size=6),
        rng.normal(size=6),
        sigma,
        [],
        params,
        object(),
    )
    assert not (windows_dir / "window_1_pred_obs.nc").exists()

    bundle = obs_diagnostics_bundle(tmp_path)
    assert bundle is not None
    assert bundle["num_windows"] == 1


# ---------------------------------------------------------------------------
# Figure D3
# ---------------------------------------------------------------------------


def _decay_windows(
    rng: np.random.Generator, levels: tuple[float, ...], n_windows: int = 2
) -> list[np.ndarray]:
    """``n_windows`` arrays of ``(len(levels), 8)`` O_N values around ``levels``."""
    return [
        np.stack([np.full(8, lv) + 0.01 * lv * rng.normal(size=8) for lv in levels])
        for _ in range(n_windows)
    ]


def test_plot_data_mismatch_decay_renders(tmp_path: pathlib.Path) -> None:
    rng = np.random.default_rng(5)
    written = plot_data_mismatch_decay(
        _decay_windows(rng, (40.0, 4.0, 0.5)),
        tmp_path / "d3.png",
        num_observations=200,
    )
    assert written is not None and written.exists()


def test_plot_data_mismatch_decay_never_draws_an_inverted_band(
    tmp_path: pathlib.Path,
) -> None:
    """An entirely off-target run must not shade the band upside down.

    The regime the ``C_D`` caveat says to expect: every member above
    ``1/2 + band``, on a log axis. Flooring the band at the smallest plotted
    value without clamping made ``axhspan(ymin > ymax)`` shade everything from
    0.65 up to the lowest box under the band's own legend label — telling the
    reader the posterior landed inside tolerance while ``run_summary.yaml``
    reported ``underfit_final: true`` off the same numbers.
    """
    from unittest.mock import patch

    rng = np.random.default_rng(5)

    def spans_for(levels: tuple[float, ...], name: str) -> list[tuple[float, float]]:
        recorded: list[tuple[float, float]] = []
        with patch.object(Axes, "axhspan", autospec=True) as spy:
            spy.side_effect = lambda ax, ymin, ymax, **kw: recorded.append((ymin, ymax))
            written = plot_data_mismatch_decay(
                _decay_windows(rng, levels, n_windows=1),
                tmp_path / f"{name}.png",
                num_observations=200,
            )
        assert written is not None and written.exists()
        assert all(lo < hi for lo, hi in recorded), f"inverted band, {name}: {recorded}"
        return recorded

    band = data_mismatch_target_band(200)
    true_span = (DATA_MISMATCH_TARGET - band, DATA_MISMATCH_TARGET + band)

    # An entirely off-target run on a log axis: the band is drawn at its TRUE
    # position and the axis grows to include it, so the reader sees how far
    # above it the boxes sit. The old behaviour clipped the lower edge to the
    # data, which is what inverted the span.
    off_target = spans_for((500.0, 80.0, 10.0), "off_target")
    assert off_target == [pytest.approx(true_span)]

    # Healthy, and off-target without spanning decades (linear axis): same band.
    assert spans_for((40.0, 4.0, 0.5), "healthy") == [pytest.approx(true_span)]
    assert spans_for((3.0, 2.5, 2.0), "linear_off") == [pytest.approx(true_span)]


def test_plot_data_mismatch_decay_band_floor_only_matters_at_tiny_observation_counts(
    tmp_path: pathlib.Path,
) -> None:
    """``1/2 − band <= 0`` only when ``N_d <= 18``; that is the sole floored case.

    Everywhere else the true band has a positive lower edge and needs no floor
    at all — which is what removed the inversion rather than papering over it.
    """
    from unittest.mock import patch

    # 3/sqrt(2*18) is exactly 1/2, so 18 is the last count with a non-positive
    # lower edge and 19 the first with a drawable one.
    assert data_mismatch_target_band(19) < DATA_MISMATCH_TARGET
    assert data_mismatch_target_band(18) == pytest.approx(DATA_MISMATCH_TARGET)
    assert data_mismatch_target_band(17) > DATA_MISMATCH_TARGET

    rng = np.random.default_rng(2)

    def spans(
        levels: tuple[float, ...], n_obs: int, name: str
    ) -> list[tuple[float, float]]:
        recorded: list[tuple[float, float]] = []
        with patch.object(Axes, "axhspan", autospec=True) as spy:
            spy.side_effect = lambda ax, ymin, ymax, **kw: recorded.append((ymin, ymax))
            plot_data_mismatch_decay(
                _decay_windows(rng, levels, n_windows=1),
                tmp_path / f"{name}.png",
                num_observations=n_obs,
            )
        assert all(lo < hi for lo, hi in recorded), f"inverted band, {name}"
        return recorded

    # N_d = 8: band 0.75, so the true lower edge is negative. On a LINEAR axis
    # that just means "no lower bound", and the span starts at zero.
    linear = spans((3.0, 2.0, 1.0), 8, "tiny_nd_linear")
    assert linear == [(0.0, pytest.approx(DATA_MISMATCH_TARGET + 0.75))]

    # On a log axis zero is off the scale, so it is floored at the lowest box
    # when that box is inside the band. (Levels chosen to clear the 1.5-decade
    # log threshold: log10(60/0.2) = 2.5.)
    floored = spans((60.0, 2.0, 0.2), 8, "tiny_nd_log")
    assert len(floored) == 1 and floored[0][0] > 0.0

    # ...and dropped when nothing plotted is anywhere near it, since a band with
    # no drawable lower edge and no data inside is a decoration, not a reference.
    assert spans((500.0, 80.0, 10.0), 8, "tiny_nd_far") == []


def test_plot_data_mismatch_decay_boxes_stay_on_their_own_iteration(
    tmp_path: pathlib.Path,
) -> None:
    """Per-window clusters must not bleed into the neighbouring iteration's slot.

    With the boxes centred at ``i ± 0.4`` and 0.32 wide, window 1's step-0 box
    spanned [0.24, 0.56] while window 0's step-1 box spanned [0.44, 0.76] — they
    overlapped, and iteration 1's tick fell between two boxes belonging to
    different iterations.
    """
    for n_windows in (1, 2, 3, 5):
        positions, width = _d3_box_layout(4, n_windows)
        assert positions.shape == (n_windows, 4)
        # Full extent of one iteration's cluster, outer box edges included.
        half_extent = float(np.abs(positions[:, 0]).max()) + width / 2.0
        assert half_extent < 0.5, f"{n_windows} windows: cluster spills its slot"

    written = plot_data_mismatch_decay(
        _decay_windows(np.random.default_rng(1), (40.0, 4.0, 0.5), n_windows=3),
        tmp_path / "clusters.png",
        num_observations=200,
    )
    assert written is not None and written.exists()


def test_plot_data_mismatch_decay_no_ops_without_inputs(
    tmp_path: pathlib.Path,
) -> None:
    """Invariant 3, at the figure: absent or all-nan inputs draw nothing."""
    assert plot_data_mismatch_decay(None, tmp_path / "a.png") is None
    assert plot_data_mismatch_decay([], tmp_path / "b.png") is None
    assert (
        plot_data_mismatch_decay(
            [np.full((3, 4), np.nan)], tmp_path / "c.png", num_observations=10
        )
        is None
    )
    assert not (tmp_path / "c.png").exists()


def test_plot_data_mismatch_decay_survives_a_zero_score(
    tmp_path: pathlib.Path,
) -> None:
    """An O_N of exactly 0 (a member reproducing the obs) has no log axis."""
    windows = [np.array([[10.0, 12.0, 11.0], [0.0, 0.4, 0.5]])]
    written = plot_data_mismatch_decay(
        windows, tmp_path / "zero.png", num_observations=200
    )
    assert written is not None and written.exists()


# ---------------------------------------------------------------------------
# Consumer wiring: the metric and figure stages actually read the bundle
# ---------------------------------------------------------------------------


def _write_window_diagnostics(
    run_dir: pathlib.Path, n_windows: int, *, n_obs: int = 24, n_members: int = 10
) -> None:
    """Phase-2 files for ``n_windows`` windows, converging on the target."""
    rng = np.random.default_rng(4)
    windows_dir = run_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    sigma = np.full(n_obs, 0.5)
    params = xarray.Dataset(
        {"inflow_angle": (("esmda_step", "ensemble"), rng.normal(size=(3, n_members)))}
    )
    for window in range(n_windows):
        obs = rng.normal(size=n_obs)
        # Residuals shrinking from ~4 sigma to ~1 sigma: prior -> posterior.
        history = [
            obs[:, None] + scale * sigma[:, None] * rng.normal(size=(n_obs, n_members))
            for scale in (4.0, 2.0, 1.0)
        ]
        _save_obs_diagnostics(
            windows_dir, window, obs, obs, sigma, history, params, object()
        )


def test_metric_stage_emits_esmda_diagnostics_only_with_the_files(
    tmp_path: pathlib.Path,
) -> None:
    """The block lands in ``run_summary.yaml`` — and is absent, not null, without it.

    The wiring neither the unit tests nor the ``skip_viz`` e2e exercises: a typo
    in a bundle key at this call site would ship silently. Also pins the
    deliberate asymmetry that the block is computed **under** ``run.skip_viz``,
    since it reads only the KB-scale observation files, never the truth.
    """
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics
    from tests.test_evaluation_spectra import _minimal_run_dir

    run_dir = tmp_path / "run"
    _minimal_run_dir(run_dir)

    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")
    assert "esmda_diagnostics" not in summary
    assert summary["parameter_metrics"], "the rest of the summary must be intact"

    _write_window_diagnostics(run_dir, n_windows=2)
    compute_metrics(run_dir)
    block = read_yaml(run_dir / "run_summary.yaml")["esmda_diagnostics"][
        "data_mismatch"
    ]
    assert len(block["per_step_median"]) == 3
    # Residuals shrink 4σ -> 1σ, so O_N descends towards the ½ target.
    assert block["per_step_median"][0] > block["per_step_median"][-1]
    assert block["final_step_index"] == 2
    assert block["num_observations"] == 24
    assert block["caveat"] == "no_representativeness_error"


def test_metric_stage_ignores_windows_left_by_an_earlier_run(
    tmp_path: pathlib.Path,
) -> None:
    """``paths.results_dir`` is fixed and ``windows/`` is never cleared.

    A rerun with fewer windows leaves the earlier run's files in place. Every
    other block in ``run_summary.yaml`` is bounded by ``truth_access.yaml``'s
    ``num_windows``; this one must be too, or one summary reports two windows'
    sensors beside four windows' data mismatch.
    """
    from scripts.esmda._esmda_common import obs_diagnostics_bundle, write_yaml
    from tests.test_evaluation_spectra import _minimal_run_dir

    run_dir = tmp_path / "run"
    _minimal_run_dir(run_dir)
    _write_window_diagnostics(run_dir, n_windows=4)

    # No truth_access.yaml yet: nothing to bound against, so all four are used.
    unbounded = obs_diagnostics_bundle(run_dir)
    assert unbounded is not None and unbounded["num_windows"] == 4

    write_yaml({"num_windows": 2}, run_dir / "truth_access.yaml")
    bundle = obs_diagnostics_bundle(run_dir)
    assert bundle is not None
    assert bundle["num_windows"] == 2
    assert bundle["window_indices"] == [0, 1]


def _disable_obs_diagnostics(run_dir: pathlib.Path) -> None:
    """Record ``esmda.save_obs_diagnostics=false`` in the run's ``run_info.yaml``."""
    from scripts.esmda._esmda_common import read_yaml, write_yaml

    info = read_yaml(run_dir / "run_info.yaml")
    info.setdefault("configuration", {})["save_obs_diagnostics"] = False
    write_yaml(info, run_dir / "run_info.yaml")


def test_metric_stage_ignores_windows_left_when_diagnostics_are_disabled(
    tmp_path: pathlib.Path,
) -> None:
    """A flag-off rerun must not republish the *previous* run's diagnostic.

    ``conf/run_esmda.yaml`` offers ``esmda.save_obs_diagnostics=false`` to
    reproduce the pre-phase-2 artifact set exactly. But ``paths.results_dir`` is
    fixed and ``windows/`` is never cleared, so the earlier run's files survive
    with a window count that still matches -- the ``num_windows`` bound catches
    only a *shrinking* run. Deciding on file existence alone would then put the
    earlier assimilation's ``O_N`` beside this run's parameter and state metrics.
    """
    from scripts.esmda._esmda_common import obs_diagnostics_bundle, read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics
    from tests.test_evaluation_spectra import _minimal_run_dir

    run_dir = tmp_path / "run"
    _minimal_run_dir(run_dir)
    _write_window_diagnostics(run_dir, n_windows=2)

    # The flag-on run these files are left over from: the bundle is scored.
    assert obs_diagnostics_bundle(run_dir) is not None

    _disable_obs_diagnostics(run_dir)
    assert obs_diagnostics_bundle(run_dir) is None
    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")
    assert "esmda_diagnostics" not in summary
    assert summary["parameter_metrics"], "the rest of the summary must be intact"


def test_figure_stage_draws_no_d3_when_diagnostics_are_disabled(
    tmp_path: pathlib.Path,
) -> None:
    """The other consumer of the same stale files -- D3 must skip itself too."""
    from scripts.esmda.make_esmda_figures import make_figures
    from tests.test_evaluation_figures import _figure_run_dir

    run_dir = _figure_run_dir(tmp_path)
    _write_window_diagnostics(run_dir, n_windows=2)
    _disable_obs_diagnostics(run_dir)

    make_figures(run_dir)
    assert not (run_dir / "data_mismatch_decay.png").exists()
    # A skipped figure must not cost the rest of the set.
    assert (run_dir / "sensor_fans.png").is_file()


def test_figure_stage_draws_d3_only_with_the_files(tmp_path: pathlib.Path) -> None:
    """The figure wiring, on the WP1.5 suite's own complete run dir.

    ``_figure_run_dir`` is imported rather than re-fabricated, for the reason
    the WP3 suite gives: it is ``run_esmda.py``'s output layout, and a second
    copy here would be the thing that drifts.
    """
    from scripts.esmda.make_esmda_figures import make_figures
    from tests.test_evaluation_figures import _figure_run_dir

    run_dir = _figure_run_dir(tmp_path)
    make_figures(run_dir)
    assert not (run_dir / "data_mismatch_decay.png").exists()

    _write_window_diagnostics(run_dir, n_windows=2)
    make_figures(run_dir)
    assert (run_dir / "data_mismatch_decay.png").is_file()
    # A new figure must not cost an old one.
    assert (run_dir / "sensor_fans.png").is_file()


# ---------------------------------------------------------------------------
# End to end: the smoother records what the runner persists
# ---------------------------------------------------------------------------


# The cheapest ESMDA mode in the suite: static parameters, one window, one MDA
# step, the global (unlocalized) update the 2-member smoke ensemble needs.
# Restated rather than imported from ``tests.test_run_esmda._overrides``: that
# module carries pre-existing untyped-def errors, and importing it would drag
# them into the mypy hook for every commit touching this file.
_E2E_OVERRIDES = [
    "model@truth_model=pylbm",
    "model@assim_model=pylbm",
    "esmda/smoother=static",
    "params@prior_params=static",
    "params@truth_params=static_truth",
    "esmda.localization=null",
    "esmda.num_steps=1",
    "esmda.num_assimilation_windows=1",
    "run.skip_viz=true",
    "run.truth_dir=null",
    # The smoke domain is [0,20]^2, so the case's full-domain sensors would all
    # fall outside it; these sit in its open N-S lanes.
    "obs.x_points=[2.5,2.5,18.0,18.0]",
    "obs.y_points=[5.0,15.0,5.0,15.0]",
    "obs.z_points=[3.0,3.0,3.0,3.0]",
    "obs.interval_seconds=3.0",
    "truth_model.forward_model.cuda=false",
    "assim_model.forward_model.cuda=false",
]


def _run_e2e(compose_test_cfg: Callable[..., DictConfig], flag: bool) -> DictConfig:
    """One smoke assimilation with ``esmda.save_obs_diagnostics`` set to ``flag``."""
    from scripts.esmda.run_esmda import run

    cfg = compose_test_cfg(
        [*_E2E_OVERRIDES, f"esmda.save_obs_diagnostics={str(flag).lower()}"],
        config_name="run_esmda",
    )
    run(cfg)
    return cfg


def _run_dir_listing(cfg: DictConfig) -> set[str]:
    """Every artifact path a run produced, relative to its output dir."""
    root = pathlib.Path(cfg.paths.results_dir)
    return {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and "hydra" not in p.parts
    }


def test_run_esmda_persists_every_iteration_when_the_flag_is_on(
    compose_test_cfg: Callable[..., DictConfig],
) -> None:
    """``num_steps + 1`` iterations persisted, with the observations beside them.

    What is under test is the plumbing, not the assimilation, hence the cheapest
    mode. The ``esmda_step`` length is the assertion that matters: it is
    ``num_steps + 1`` only if the posterior forecast's predicted observations
    were recorded after the loop, which is the one entry no ``_one_step``
    produces.
    """
    cfg = _run_e2e(compose_test_cfg, True)
    windows_dir = pathlib.Path(cfg.paths.results_dir) / "windows"

    with xarray.open_dataset(windows_dir / "window_0_pred_obs.nc") as pred:
        assert pred["pred_obs"].dims == ("esmda_step", "obs_index", "ensemble")
        assert pred.sizes["esmda_step"] == int(cfg.esmda.num_steps) + 1
        assert pred.sizes["ensemble"] == int(cfg.ensemble.ensemble_size)
        n_obs = pred.sizes["obs_index"]

    with xarray.open_dataset(windows_dir / "window_0_obs.nc") as obs_ds:
        assert obs_ds.sizes["obs_index"] == n_obs
        # `obs` must survive the round trip as a DATA VARIABLE: a variable whose
        # name equals its dimension is silently promoted to a coordinate, which
        # is why the dimension is `obs_index`.
        assert "obs" in obs_ds.data_vars
        # The noise is what separates the two: obs = obs_clean + sqrt(C_D)·z.
        assert not np.allclose(obs_ds["obs"].values, obs_ds["obs_clean"].values)
        assert obs_ds["obs_error_std"].values == pytest.approx(
            float(cfg.esmda.obs_error_std)
        )

    assert (windows_dir / "window_0_params_steps.nc").exists()


def test_run_esmda_flag_off_reproduces_the_pre_phase2_artifact_set(
    compose_test_cfg: Callable[..., DictConfig],
) -> None:
    """The plan's acceptance criterion: a directory-listing comparison.

    Not a name-pattern check — that would pass a change which quietly *dropped*
    an existing artifact as well as one that added nothing. The flag-off listing
    must equal the flag-on listing minus exactly the three new files.
    """
    on_listing = _run_dir_listing(_run_e2e(compose_test_cfg, True))
    off_listing = _run_dir_listing(_run_e2e(compose_test_cfg, False))

    added = {
        "windows/window_0_obs.nc",
        "windows/window_0_pred_obs.nc",
        "windows/window_0_params_steps.nc",
    }
    assert on_listing - off_listing == added
    assert off_listing - on_listing == set()
