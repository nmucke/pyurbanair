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
from evaluation.figures import plot_data_mismatch_decay
from evaluation.scores import (
    DATA_MISMATCH_TARGET,
    data_mismatch,
    data_mismatch_summary,
    data_mismatch_target_band,
)
from omegaconf import DictConfig

from scripts.esmda._esmda_common import obs_diagnostics_bundle
from scripts.esmda.run_esmda import _obs_index_coords, _save_obs_diagnostics

# ---------------------------------------------------------------------------
# O_N against the chi-squared target
# ---------------------------------------------------------------------------


def test_data_mismatch_is_half_on_a_linear_gaussian_posterior() -> None:
    """Posterior samples of an exact linear-Gaussian problem score ``O_N ≈ 1/2``.

    The whole diagnostic is "compare against 1/2", so the target is verified
    rather than assumed: build ``d = G θ_true + ε`` with a Gaussian prior on
    ``θ``, sample the *exact* posterior in closed form, and check the mean of
    ``O_N`` over those samples. ``E[(d − Gθ_m)ᵀ C_D⁻¹ (d − Gθ_m)] = N_d`` for
    posterior samples, hence ``O_N → 1/2`` — and note this is the *sample* value,
    not the posterior mean's, which sits at ``(N_d − P)/2N_d`` and is the number
    a "fit the mean" implementation would produce instead.
    """
    rng = np.random.default_rng(7)
    n_obs, n_params, n_members = 300, 5, 4000
    sigma = 0.3

    G = rng.normal(size=(n_obs, n_params))
    theta_true = rng.normal(size=n_params)
    obs = G @ theta_true + sigma * rng.normal(size=n_obs)

    precision = np.eye(n_params) + G.T @ G / sigma**2
    covariance = np.linalg.inv(precision)
    mean = covariance @ (G.T @ obs / sigma**2)
    theta = mean[:, None] + np.linalg.cholesky(covariance) @ rng.normal(
        size=(n_params, n_members)
    )

    scores = data_mismatch(obs, G @ theta, np.full(n_obs, sigma))
    assert scores.shape == (n_members,)
    # 3 sigma of the mean over the members: Var[O_N] = 1/(2 N_d) per member.
    tolerance = 3.0 / np.sqrt(2.0 * n_obs * n_members)
    assert abs(float(scores.mean()) - DATA_MISMATCH_TARGET) < max(tolerance, 0.01)

    # The posterior MEAN sits measurably below the target -- the distinction the
    # formula has to preserve.
    mean_score = float(
        data_mismatch(obs, (G @ mean)[:, None], np.full(n_obs, sigma))[0]
    )
    assert mean_score == pytest.approx((n_obs - n_params) / (2.0 * n_obs), abs=0.05)


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
    collapsed = data_mismatch_summary([np.full(20, 4.0), np.full(20, 4.0)], n_obs)
    assert collapsed["collapsed"] is True
    assert collapsed["per_step_iqr"][-1] == pytest.approx(0.0)

    # Identical members that agree ON the target are converged, not collapsed.
    converged = data_mismatch_summary([np.full(20, 4.0), np.full(20, 0.5)], n_obs)
    assert converged["collapsed"] is False


def test_data_mismatch_summary_degrades_without_values() -> None:
    assert data_mismatch_summary([], 200) is None
    assert data_mismatch_summary(None, 200) is None
    assert data_mismatch_summary([np.full(4, np.nan)], 200) is None

    # No observations -> the flags are None (unjudgeable), not False (checked).
    no_band = data_mismatch_summary([np.full(4, 3.0)], 0)
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


def test_plot_data_mismatch_decay_renders(tmp_path: pathlib.Path) -> None:
    rng = np.random.default_rng(5)
    bundle = {
        "per_window": [
            np.stack(
                [
                    np.full(8, level) + 0.1 * rng.normal(size=8)
                    for level in (40.0, 4.0, 0.5)
                ]
            )
            for _ in range(2)
        ],
        "num_observations": 200,
    }
    written = plot_data_mismatch_decay(bundle, tmp_path / "d3.png")
    assert written is not None and written.exists()


def test_plot_data_mismatch_decay_no_ops_without_inputs(
    tmp_path: pathlib.Path,
) -> None:
    """Invariant 3, at the figure: absent or all-nan inputs draw nothing."""
    assert plot_data_mismatch_decay(None, tmp_path / "a.png") is None
    assert plot_data_mismatch_decay({}, tmp_path / "b.png") is None
    assert (
        plot_data_mismatch_decay(
            {"per_window": [np.full((3, 4), np.nan)], "num_observations": 10},
            tmp_path / "c.png",
        )
        is None
    )
    assert not (tmp_path / "c.png").exists()


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


@pytest.mark.parametrize("save_obs_diagnostics", [True, False])  # type: ignore[misc]
def test_run_esmda_obs_diagnostics_flag(
    save_obs_diagnostics: bool, compose_test_cfg: Callable[..., DictConfig]
) -> None:
    """Flag on -> ``num_steps + 1`` iterations persisted; flag off -> no files.

    What is under test is the plumbing, not the assimilation, hence the cheapest
    mode. The ``esmda_step`` length is the assertion that matters: it is
    ``num_steps + 1`` only if the posterior forecast's predicted observations
    were recorded after the loop, which is the one entry no ``_one_step``
    produces.
    """
    from scripts.esmda.run_esmda import run

    overrides = [
        *_E2E_OVERRIDES,
        f"esmda.save_obs_diagnostics={str(save_obs_diagnostics).lower()}",
    ]
    cfg = compose_test_cfg(overrides, config_name="run_esmda")
    run(cfg)

    windows_dir = pathlib.Path(cfg.paths.results_dir) / "windows"
    produced = sorted(p.name for p in windows_dir.glob("window_0_*"))

    if not save_obs_diagnostics:
        assert not [
            name for name in produced if "obs" in name or "params_steps" in name
        ]
        return

    with xarray.open_dataset(windows_dir / "window_0_pred_obs.nc") as pred:
        assert pred["pred_obs"].dims == ("esmda_step", "obs", "ensemble")
        assert pred.sizes["esmda_step"] == int(cfg.esmda.num_steps) + 1
        assert pred.sizes["ensemble"] == int(cfg.ensemble.ensemble_size)
        n_obs = pred.sizes["obs"]

    with xarray.open_dataset(windows_dir / "window_0_obs.nc") as obs_ds:
        assert obs_ds.sizes["obs"] == n_obs
        # The noise is what separates the two: obs = obs_clean + sqrt(C_D)·z.
        assert not np.allclose(obs_ds["obs"].values, obs_ds["obs_clean"].values)
        assert obs_ds["obs_error_std"].values == pytest.approx(
            float(cfg.esmda.obs_error_std)
        )

    assert (windows_dir / "window_0_params_steps.nc").exists()
