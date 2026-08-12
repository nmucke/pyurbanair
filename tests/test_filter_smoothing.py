"""Unit tests for filter smoothing (data_assimilation.filter_smoothing).

Covers the Phase 1 deliverables of docs/plans/filter_smoothing_windowed_esmda.md
§6: the pre-analysis forecast-observation recording hook, the per-iteration
state reset, the final consistency pass, temporal credit assignment (with and
without temporal localization), the temporal taper itself, the alpha schedule
and common random numbers, and the per-cycle segment restriction (plan §9: the
knot grid is independent of the cycle grid, and each cycle forecasts with the
trajectory restricted to its segment). Everything runs on toy in-memory forward
models — no CFD solver.

The algorithm's semantics are all about *which* ensemble each inner pass is
given and *when* an observation is recorded, so the stubs below record every
``(state, params)`` pair (the forward model) and every predicted-observation
array (the observation operator) in call order. A test then reads that call
log rather than reverse-engineering the result.
"""

import pathlib
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray
from data_assimilation.augmentation import ParamAugmentation
from data_assimilation.filter_smoothing import (
    FilterSmoothingESMDA,
    TemporalLocalization,
)
from data_assimilation.filtering import EnsembleKalmanFilter
from data_assimilation.localization.base import resolve_row_inflation, taper_inflation

# ---------------------------------------------------------------------------
# Toy forward models and observation operators
# ---------------------------------------------------------------------------


def _segment_mean(params: xarray.Dataset, name: str) -> jnp.ndarray:
    """The stub backends' reading of one segment's parameter schedule.

    A cycle's forecast is handed the trajectory RESTRICTED to that cycle's
    segment: a ``(time, ensemble)`` schedule on a segment-local clock that a
    real backend interpolates over the run. These toys have no internal clock,
    so they collapse the schedule to its **mean over the segment** — the
    simplest reduction that (a) actually consumes the whole segment rather than
    one knot of it and (b) keeps the map ``Theta -> D`` LINEAR, which is what
    lets the linear-Gaussian tests below state an exact posterior.

    Scalar (no-``time``) parameters pass straight through, so the same stub also
    serves the plain filter, whose ``_params_for_cycle`` is the identity.
    """
    variable = params[name]
    if "time" not in variable.dims:
        values = jnp.asarray(variable.values)
        assert values.ndim == 1, f"{name} arrived with shape {values.shape}."
        return values
    assert float(variable.coords["time"].values[0]) == 0.0, (
        "the segment schedule must arrive on a SEGMENT-LOCAL clock starting at "
        f"0, got {variable.coords['time'].values}"
    )
    segment = jnp.asarray(variable.transpose("time", ...).values)
    return jnp.mean(segment, axis=0)


class _TrajectoryToyModel:
    """``x_{k+1} = decay * x_k + param_effect * mean(theta over segment k)``.

    Records every ``(state, params)`` pair it is called with, in call order, so
    a test can see which initial condition and which piece of the trajectory
    each inner pass received. The forecast carries a two-frame ``time``
    dimension so the filter's end-of-segment selection (``isel(time=-1)``) is
    exercised.

    The parameter enters through :func:`_segment_mean` — see there for why the
    segment is averaged rather than sampled at one knot.

    ``param_effect=0.0`` makes the forecast blind to the parameters: the map
    ``Theta -> D`` is then constant, which is what isolates the inner PRNG in
    the common-random-numbers test.
    """

    save_on_disk = False
    results_dir: Optional[pathlib.Path] = None

    def __init__(
        self,
        param_effect: float = 1.0,
        decay: float = 1.0,
        param_name: str = "theta",
    ) -> None:
        self.param_effect = param_effect
        self.decay = decay
        self.param_name = param_name
        self.states: list[xarray.Dataset] = []
        self.params: list[Optional[xarray.Dataset]] = []

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> xarray.Dataset:
        assert state is not None
        self.states.append(state.copy(deep=True))
        self.params.append(None if params is None else params.copy(deep=True))

        x = jnp.asarray(state["u"].values)  # (N_e, nx)
        x_new = self.decay * x
        if params is not None and self.param_effect:
            theta = _segment_mean(params, self.param_name)
            x_new = x_new + self.param_effect * theta[:, None]
        frames = jnp.stack([x, x_new], axis=1)  # (N_e, 2, nx)
        return xarray.Dataset(
            {"u": (("ensemble", "time", "x"), frames)},
            coords={
                "ensemble": np.arange(x.shape[0]),
                "time": np.array([0.0, 1.0]),
                "x": np.asarray(state.coords["x"].values),
            },
        )

    def apply_failure_substitutions_to_params(
        self, params: Optional[xarray.Dataset]
    ) -> Optional[xarray.Dataset]:
        return params

    def apply_failure_substitutions_to_state(
        self, state: Optional[xarray.Dataset]
    ) -> Optional[xarray.Dataset]:
        return state


class _DelayedParamModel(_TrajectoryToyModel):
    """``u_{k+1} = s_k``, ``s_{k+1} = m_k``: a segment is seen one cycle late.

    ``m_k`` is segment ``k``'s mean parameter (:func:`_segment_mean`). Only
    ``u`` is observed, so cycle ``k``'s forecast observation carries
    ``m_{k-1}`` and says nothing about ``m_k``. From a zero-spread
    initial condition, cycle 0's predicted observations are identical across
    members, so its inner analysis has *exactly* zero gain and cannot erase the
    hidden ``s`` in which ``theta_0`` is stored. The last observation of the
    window is therefore the only one informative about the first knot — the
    linear-Gaussian toy for temporal credit assignment.
    """

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> xarray.Dataset:
        assert state is not None and params is not None
        self.states.append(state.copy(deep=True))
        self.params.append(params.copy(deep=True))

        u = jnp.asarray(state["u"].values)  # (N_e, 1)
        s = jnp.asarray(state["s"].values)  # (N_e, 1)
        theta = _segment_mean(params, self.param_name)
        u_new = s
        s_new = jnp.broadcast_to(theta[:, None], s.shape)
        return xarray.Dataset(
            {
                "u": (("ensemble", "time", "x"), jnp.stack([u, u_new], axis=1)),
                "s": (("ensemble", "time", "x"), jnp.stack([s, s_new], axis=1)),
            },
            coords={
                "ensemble": np.arange(u.shape[0]),
                "time": np.array([0.0, 1.0]),
                "x": np.asarray(state.coords["x"].values),
            },
        )


class _ToyObsOp:
    """Linear observation of the forecast's final frame: ``y = H u_T``."""

    def __init__(self, H: np.ndarray) -> None:
        self.H = jnp.asarray(H)  # (N_d, nx)

    def __call__(self, state: xarray.Dataset) -> jnp.ndarray:
        u = jnp.asarray(state["u"].isel(time=-1).values)  # (N_e, nx)
        return u @ self.H.T  # (N_e, N_d)


class _RecordingObsOp(_ToyObsOp):
    """Toy operator that keeps every predicted-observation array it produced.

    One call per cycle, so the log is ``(iteration, cycle)`` in row-major
    order — the per-iteration blocks of ``D`` as the inner pass produced them.
    """

    def __init__(self, H: np.ndarray) -> None:
        super().__init__(H)
        self.calls: list[np.ndarray] = []

    def __call__(self, state: xarray.Dataset) -> jnp.ndarray:
        predicted = super().__call__(state)
        self.calls.append(np.asarray(predicted))
        return predicted


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def _scalar_state(values: np.ndarray) -> xarray.Dataset:
    """One-cell state ensemble: ``u`` of shape ``(N_e, 1)``."""
    return xarray.Dataset(
        {"u": (("ensemble", "x"), jnp.asarray(values)[:, None])},
        coords={"ensemble": np.arange(values.shape[0]), "x": np.array([0.0])},
    )


def _delayed_state(n_e: int) -> xarray.Dataset:
    """Zero-spread ``(u, s)`` initial condition for :class:`_DelayedParamModel`."""
    zeros = jnp.zeros((n_e, 1))
    return xarray.Dataset(
        {"u": (("ensemble", "x"), zeros), "s": (("ensemble", "x"), zeros)},
        coords={"ensemble": np.arange(n_e), "x": np.array([0.0])},
    )


def _trajectory_params(
    knots: np.ndarray,
    static: Optional[np.ndarray] = None,
    name: str = "theta",
    spacing: float = 1.0,
) -> xarray.Dataset:
    """A ``(time, ensemble)`` trajectory prior on a uniform grid from ``t = 0``.

    ``knots`` has shape ``(n_knots, N_e)``. The ``time`` coordinate holds the
    knots' times in SECONDS — the grid the estimator places each cycle's segment
    on, and what the temporal localization's row coordinates are built from. The
    default ``spacing`` equals the toys' 1 s cycle (:data:`_CYCLE_LENGTH`), so
    knot index and cycle index coincide; pass a different one for a knot grid
    coarser or finer than the cycle.
    """
    n_knots, n_e = knots.shape
    data_vars: dict[str, Any] = {name: (("time", "ensemble"), jnp.asarray(knots))}
    if static is not None:
        data_vars["gamma"] = (("ensemble",), jnp.asarray(static))
    return xarray.Dataset(
        data_vars,
        coords={
            "time": np.arange(n_knots, dtype=float) * spacing,
            "ensemble": np.arange(n_e),
        },
    )


#: One cycle's forecast segment, in seconds. The toy models have no clock of
#: their own (and so no ``simulation_time`` for the estimator to read), so every
#: smoother below is told the cycle length explicitly; 1 s keeps the cycle axis
#: and the default knot grid numerically identical.
_CYCLE_LENGTH = 1.0


def _smoothing_kwargs(
    forward_model: Any,
    observation_operator: Optional[Any] = None,
    variance: float = 0.01,
    cycle_length: float = _CYCLE_LENGTH,
) -> dict:
    return {
        "observation_operator": (
            _ToyObsOp(np.array([[1.0]]))
            if observation_operator is None
            else observation_operator
        ),
        "forward_model": forward_model,
        "C_D": jnp.array([variance]),
        "cycle_length": cycle_length,
    }


# ---------------------------------------------------------------------------
# (1) Forecast observations are recorded BEFORE the analysis
# ---------------------------------------------------------------------------


def test_collect_pred_obs_records_the_forecast_not_the_analysis() -> None:
    """``pred_obs_history`` holds ``H(x_f)``, recorded before the update.

    The paper's Eq. (7) stores ``d_k`` *before* ``y_k`` is assimilated; this
    hook is the only source of ``D``, so recording the analyzed state instead
    would silently turn the outer ESMDA update into a self-referential one.
    The observation here is far from the forecast, so the analysis visibly
    shifts the state and the two candidates cannot be confused.
    """
    n_e = 32
    x0 = np.asarray(jax.random.normal(jax.random.PRNGKey(0), (n_e,)))
    state = _scalar_state(x0)
    theta = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (n_e,)))
    model = _TrajectoryToyModel()
    params = xarray.Dataset(
        {"theta": (("ensemble",), jnp.asarray(theta))},
        coords={"ensemble": np.arange(n_e)},
    )

    enkf = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(np.array([[1.0]])),
        forward_model=model,
        C_D=jnp.array([0.01]),
        mode="state",
        rng_key=jax.random.PRNGKey(2),
    )
    # Off by default: nothing is recorded and nothing changes.
    assert not getattr(enkf, "pred_obs_history", None)

    enkf.collect_pred_obs = True
    enkf.run(state=state, params=params, observations=jnp.array([[5.0], [5.0]]))

    history = enkf.pred_obs_history
    assert len(history) == 2
    # Cycle 0's record is H of the FORECAST from x_0: (N_d, N_e).
    np.testing.assert_allclose(
        np.asarray(history[0]), (x0 + theta)[None, :], rtol=1e-5, atol=1e-6
    )
    # ... and not H of the analyzed state, which is the next cycle's warm start.
    analyzed = np.asarray(model.states[1]["u"].values)[:, 0]
    assert not np.allclose(np.asarray(history[0])[0], analyzed, atol=1e-3)

    # A fresh run rebinds the history rather than appending to the old one.
    enkf.run(state=state, params=params, observations=jnp.array([[5.0]]))
    assert len(enkf.pred_obs_history) == 1


def test_filter_smoothing_enables_the_recording_hook() -> None:
    """The outer loop's ``D`` comes from that hook, so it must switch it on."""
    smoother = FilterSmoothingESMDA(
        num_steps=1, **_smoothing_kwargs(_TrajectoryToyModel())
    )
    assert smoother._inner.collect_pred_obs is True


# ---------------------------------------------------------------------------
# (2) Reset semantics: every iteration starts from the same x_0
# ---------------------------------------------------------------------------


def test_every_iteration_restarts_from_the_same_initial_state() -> None:
    """Each inner pass is handed the identical ``x_0`` (§4.4, §9).

    The analyzed state trajectories of a previous iteration must never leak
    into the next one — otherwise the ESMDA iterations would keep conditioning
    a state ensemble that has already seen the window's observations.
    """
    n_e, num_cycles, num_steps = 16, 2, 2
    x0 = np.asarray(jax.random.normal(jax.random.PRNGKey(3), (n_e,)))
    state = _scalar_state(x0)
    knots = np.asarray(jax.random.normal(jax.random.PRNGKey(4), (num_cycles + 1, n_e)))
    model = _TrajectoryToyModel()

    FilterSmoothingESMDA(
        num_steps=num_steps,
        rng_key=jax.random.PRNGKey(5),
        **_smoothing_kwargs(model),
    ).run(
        state=state,
        params=_trajectory_params(knots),
        observations=jnp.array([[2.0], [2.0]]),
    )

    # num_steps inner passes plus the final consistency pass, each of num_cycles
    # forecasts.
    assert len(model.states) == (num_steps + 1) * num_cycles
    for iteration in range(num_steps + 1):
        first_cycle = model.states[iteration * num_cycles]
        np.testing.assert_array_equal(
            np.asarray(first_cycle["u"].values)[:, 0], x0.astype(np.float32)
        )
    # The second cycle of an iteration does NOT see x_0 — otherwise the check
    # above would hold trivially for a filter that never advances its state.
    assert not np.allclose(np.asarray(model.states[1]["u"].values)[:, 0], x0)


# ---------------------------------------------------------------------------
# (3) The final consistency pass runs with the returned trajectory
# ---------------------------------------------------------------------------


def test_final_pass_uses_the_returned_trajectory() -> None:
    """The returned state comes from an extra pass driven by ``Theta^{N_a}``.

    Eq. (16): the filtered state must be consistent with the trajectory the
    method reports, not with the last iteration's (pre-update) one.
    """
    n_e, num_cycles, num_steps = 24, 2, 2
    x0 = np.asarray(jax.random.normal(jax.random.PRNGKey(6), (n_e,)))
    knots = np.asarray(jax.random.normal(jax.random.PRNGKey(7), (num_cycles + 1, n_e)))
    prior = _trajectory_params(knots)
    model = _TrajectoryToyModel()

    result = FilterSmoothingESMDA(
        num_steps=num_steps,
        rng_key=jax.random.PRNGKey(8),
        **_smoothing_kwargs(model),
    ).run(
        state=_scalar_state(x0),
        params=prior,
        observations=jnp.array([[3.0], [3.0]]),
        return_history=True,
    )

    assert result.params is not None and result.state is not None
    assert len(result.iteration_diagnostics) == num_steps
    assert result.params_history is not None

    # The trajectory really moved, so "the final pass saw the posterior" is not
    # vacuously the same statement as "it saw the prior".
    assert not np.allclose(np.asarray(result.params["theta"].values), knots, atol=1e-4)

    # The final pass's segments are the POSTERIOR trajectory restricted cycle by
    # cycle: with one knot per cycle, segment k runs from knot k to knot k+1.
    final_pass_segments = model.params[-num_cycles:]
    for cycle, seen in enumerate(final_pass_segments):
        assert seen is not None
        np.testing.assert_allclose(
            np.asarray(seen["theta"].values),
            np.asarray(result.params["theta"].isel(time=[cycle, cycle + 1]).values),
            rtol=1e-5,
            atol=1e-6,
        )
    assert result.final_pass is not None
    assert result.final_pass.state is not None
    np.testing.assert_allclose(
        np.asarray(result.state["u"].values),
        np.asarray(result.final_pass.state["u"].values),
        rtol=1e-6,
        atol=1e-6,
    )


def test_static_parameters_ride_along_and_stay_out_of_the_update() -> None:
    """No-``time`` variables are forecast unchanged and not estimated (§9).

    Phase 1 carries e.g. ``vertical_inflow_exponent`` through every forecast so
    the backend sees a complete parameter set, but excludes it from the outer
    trajectory update (including it later is a mask away).
    """
    n_e, num_cycles = 20, 2
    x0 = np.asarray(jax.random.normal(jax.random.PRNGKey(9), (n_e,)))
    knots = np.asarray(jax.random.normal(jax.random.PRNGKey(10), (num_cycles + 1, n_e)))
    gamma = np.asarray(0.25 + 0.05 * jax.random.normal(jax.random.PRNGKey(11), (n_e,)))
    model = _TrajectoryToyModel()

    result = FilterSmoothingESMDA(
        num_steps=2,
        rng_key=jax.random.PRNGKey(12),
        **_smoothing_kwargs(model),
    ).run(
        state=_scalar_state(x0),
        params=_trajectory_params(knots, static=gamma),
        observations=jnp.array([[3.0], [3.0]]),
    )

    for seen in model.params:
        assert seen is not None
        np.testing.assert_allclose(
            np.asarray(seen["gamma"].values), gamma, rtol=1e-6, atol=1e-6
        )
    assert result.params is not None
    np.testing.assert_allclose(
        np.asarray(result.params["gamma"].values), gamma, rtol=1e-6, atol=1e-6
    )
    assert not np.allclose(np.asarray(result.params["theta"].values), knots, atol=1e-4)


# ---------------------------------------------------------------------------
# (4) Temporal credit assignment
# ---------------------------------------------------------------------------


def _credit_assignment_run(
    temporal_localization: Optional[TemporalLocalization],
) -> tuple[np.ndarray, np.ndarray]:
    """One two-cycle delayed-effect window; returns (prior knots, posterior theta_0).

    ``y_1 = 3`` is the only informative observation, and it is informative
    about ``theta_0`` *only through the dynamics* (the hidden ``s`` carries
    segment 0's mean into the next segment). ``y_0`` equals the forecast
    exactly, so the first cycle contributes no innovation.
    """
    n_e, num_cycles = 400, 2
    knots = np.asarray(jax.random.normal(jax.random.PRNGKey(13), (num_cycles + 1, n_e)))
    prior = _trajectory_params(knots)

    result = FilterSmoothingESMDA(
        num_steps=1,
        temporal_localization=temporal_localization,
        rng_key=jax.random.PRNGKey(14),
        **_smoothing_kwargs(_DelayedParamModel()),
    ).run(
        state=_delayed_state(n_e),
        params=prior,
        observations=jnp.array([[0.0], [3.0]]),
    )

    assert result.params is not None
    return knots, np.asarray(result.params["theta"].isel(time=0).values)


def test_last_observation_updates_the_first_knot() -> None:
    """The defining feature (§4.3): credit flows back through the window.

    The toy is scalar-Gaussian and the map ``Theta -> D`` is linear — segment 0
    is the ramp from knot 0 to knot 1, so ``D_1 = (theta_0 + theta_1) / 2`` —
    which makes one full-weight ESMDA iteration the exact Kalman update. It is
    written out here from the prior ensemble's OWN sample covariances rather
    than from the population values (``Cov = 1/2``, ``Var = 1/2``, gain
    ``0.5/0.51``): the update uses the sample gain, and at a few hundred
    members that differs by a few percent — a discrepancy which would otherwise
    have to be hidden under a tolerance wide enough to also hide a real bug.

    A method that only ever assimilated the *current* cycle's observation into
    the *current* knot would leave this mean at the prior's 0.
    """
    knots, posterior = _credit_assignment_run(temporal_localization=None)
    prior = knots[0]

    assert abs(float(prior.mean())) < 0.2  # the prior knows nothing about y_1

    segment_mean = 0.5 * (knots[0] + knots[1])
    covariance = np.cov(np.stack([prior, segment_mean]), ddof=1)
    gain = covariance[0, 1] / (covariance[1, 1] + 0.01)
    expected = float(prior.mean()) + gain * (3.0 - float(segment_mean.mean()))
    assert float(posterior.mean()) == pytest.approx(expected, abs=0.05)
    # ... and that update really is most of the way to y_1, not a nudge.
    assert float(posterior.mean()) == pytest.approx(3.0, abs=0.3)


def test_tight_temporal_localization_blocks_the_credit() -> None:
    """A radius shorter than the window severs the same link, exactly.

    Knot 0 sits at ``t = 0`` and the informative observation batch at the end
    of the second segment (``t = 2``), so a 0.5 s radius excludes it: the
    localized update returns those rows untouched (the paper's identity
    transition). ``block_grouping=False`` keeps this a per-knot statement —
    with grouping on, knot 0 would share knot 1's observation selection.
    """
    knots, posterior = _credit_assignment_run(
        temporal_localization=TemporalLocalization(
            temporal_radius=0.5,
            tapering_beta=0.5,
            max_inflation=2.0,
            block_grouping=False,
        )
    )

    np.testing.assert_allclose(posterior, knots[0], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# (5) The temporal taper itself
# ---------------------------------------------------------------------------


def test_temporal_localization_matches_the_shared_taper() -> None:
    """Eq. (17) is ``taper_inflation`` on ``|t_row - t_obs|``.

    Pinned against the shared formula rather than hard-coded numbers, plus the
    three qualitative anchors: full weight at zero lag, a taper strictly
    between 1 and ``E_max`` inside the radius, exactly ``E_max`` at the radius,
    and exclusion beyond it.
    """
    radius, beta, e_max = 2.0, 0.25, 4.0
    row_times = np.array([0.0, 1.0, 2.0, 5.0])
    obs_times = np.array([1.0, 2.0])
    row_coords = jnp.asarray(np.stack([row_times, np.zeros(4), np.zeros(4)], axis=1))
    obs_coords = jnp.asarray(np.stack([obs_times, np.zeros(2), np.zeros(2)], axis=1))

    localization = TemporalLocalization(
        temporal_radius=radius, tapering_beta=beta, max_inflation=e_max
    )
    assert localization.requires_coordinates is True
    assert localization.localizes_parameters is True

    # The strategy is coordinate-driven; the anomalies only fix the shapes.
    inflation = np.asarray(
        localization.inflation_factors(
            jnp.zeros((4, 8)),
            jnp.zeros((2, 8)),
            row_coords=row_coords,
            obs_coords=obs_coords,
        )
    )
    distance = np.abs(row_times[:, None] - obs_times[None, :])
    expected = np.asarray(
        taper_inflation(
            jnp.asarray(distance),
            truncation=radius,
            tapering_beta=beta,
            max_inflation=e_max,
        )
    )
    np.testing.assert_allclose(inflation, expected, rtol=1e-5)

    assert inflation[1, 0] == pytest.approx(1.0)  # zero lag: full weight
    assert 1.0 < inflation[0, 0] < e_max  # inside the radius: tapered
    assert inflation[0, 1] == pytest.approx(e_max, rel=1e-5)  # at the radius
    assert np.isinf(inflation[3]).all()  # beyond it: excluded


def test_temporal_block_grouping_shares_one_selection_per_parameter() -> None:
    """All knots of one parameter are updated with a single selection (§4.3).

    ``ParamAugmentation.group_ids`` is the same block convention the
    time-varying smoother uses, and ``resolve_row_inflation`` reduces a block
    to its per-observation minimum — so the knot closest to an observation
    decides the whole parameter's taper.
    """
    n_e, n_knots = 6, 3
    params = _trajectory_params(
        np.zeros((n_knots, n_e)), static=np.zeros(n_e)
    )  # theta_0..2 then gamma
    group_ids = ParamAugmentation(num_time_points=n_knots).group_ids(params)
    np.testing.assert_array_equal(np.asarray(group_ids), [0, 0, 0, 1])

    localization = TemporalLocalization(
        temporal_radius=2.0, tapering_beta=0.25, max_inflation=4.0, block_grouping=True
    )
    assert localization.block_grouping is True

    row_times = np.array([0.0, 1.0, 2.0, 0.0])  # knots at their segment starts
    obs_times = np.array([1.0, 2.0])
    inflation = localization.inflation_factors(
        jnp.zeros((4, n_e)),
        jnp.zeros((2, n_e)),
        row_coords=jnp.asarray(np.stack([row_times, np.zeros(4), np.zeros(4)], axis=1)),
        obs_coords=jnp.asarray(np.stack([obs_times, np.zeros(2), np.zeros(2)], axis=1)),
    )
    grouped = np.asarray(resolve_row_inflation(inflation, group_ids, None))

    raw = np.asarray(inflation)
    shared = raw[:n_knots].min(axis=0)
    for knot in range(n_knots):
        np.testing.assert_allclose(grouped[knot], shared, rtol=1e-6)
    # Every knot is at full weight for both observations, because one of them
    # is: the grouping is doing something the per-row factors do not.
    np.testing.assert_allclose(grouped[:n_knots], 1.0, rtol=1e-6)
    assert raw[0, 1] > 1.0
    # The static parameter is its own block and keeps its own factors.
    np.testing.assert_allclose(grouped[n_knots], raw[n_knots], rtol=1e-6)


@pytest.mark.parametrize(  # type: ignore[misc]
    "spacing,expected",
    [(1.0, [0.0, 1.0, 2.0]), (2.0, [0.0, 2.0, 4.0]), (0.5, [0.0, 0.5, 1.0])],
    ids=["cycle_grid", "coarse", "fine"],
)
def test_row_coordinates_are_the_knot_times_in_cycles(
    spacing: float, expected: list[float]
) -> None:
    """Knot ``j`` sits at ``knot_time / cycle_length``, fractional if need be.

    ``temporal_radius`` is measured in CYCLES whatever the knot spacing is, so
    the coordinates cannot be the knot INDEX once the two grids differ: a
    half-cycle grid would otherwise place its knots two cycles apart and a
    two-cycle grid would squeeze them into one, silently rescaling the taper.
    The middle case reduces to the old integers exactly.
    """
    from data_assimilation.filter_smoothing.base import knot_times

    n_e = 4
    params = _trajectory_params(
        np.zeros((3, n_e)), static=np.zeros(n_e), spacing=spacing
    )
    smoother = FilterSmoothingESMDA(
        num_steps=1,
        temporal_localization=TemporalLocalization(temporal_radius=2.0),
        **_smoothing_kwargs(_TrajectoryToyModel()),
    )

    update_mask, row_times = smoother._trajectory_row_layout(
        params, knot_times(params), _CYCLE_LENGTH
    )
    # Three trajectory rows then the static one, mirroring the flatten order.
    np.testing.assert_array_equal(np.asarray(update_mask), [True, True, True, False])
    np.testing.assert_allclose(np.asarray(row_times), expected + [0.0], atol=1e-9)

    row_coords, obs_coords, _, localize_mask = smoother._localization_plumbing(
        params,
        ParamAugmentation(num_time_points=3),
        update_mask,
        row_times,
        num_cycles=2,
        n_d=1,
    )
    assert row_coords is not None and obs_coords is not None
    np.testing.assert_allclose(
        np.asarray(row_coords)[:, 0], expected + [0.0], atol=1e-9
    )
    # The observation batches stay at their segment ends, in whole cycles.
    np.testing.assert_allclose(np.asarray(obs_coords)[:, 0], [1.0, 2.0], atol=1e-9)
    np.testing.assert_array_equal(np.asarray(localize_mask), np.asarray(update_mask))


# ---------------------------------------------------------------------------
# (6) Alpha schedule and common random numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[misc]
    "num_steps,alpha,expected",
    [(4, None, 4.0), (1, None, 1.0), (4, 4.0, 4.0), (2, 2.0, 2.0)],
)
def test_alpha_defaults_to_num_steps(
    num_steps: int, alpha: Optional[float], expected: float
) -> None:
    """Equal-weight schedule (Eq. 5), the same convention as ``_BaseESMDA``.

    ``sum_k 1/alpha_k = 1`` for the tempered updates to equal one Bayesian
    conditioning; a scalar alpha applied every iteration therefore has to be
    ``num_steps``.
    """
    smoother = FilterSmoothingESMDA(
        num_steps=num_steps, alpha=alpha, **_smoothing_kwargs(_TrajectoryToyModel())
    )
    assert float(smoother.alpha) == pytest.approx(expected)


@pytest.mark.parametrize("common_inner_noise", [True, False])  # type: ignore[misc]
def test_common_inner_noise_repeats_the_inner_realization(
    common_inner_noise: bool,
) -> None:
    """CRN (§9): one inner realization per window, reused across iterations.

    The forward model here ignores the parameters, so ``Theta -> D`` is
    constant and any difference between two iterations' recorded ``D`` is
    inner Monte Carlo noise — exactly the noise that dilutes the outer
    cross-covariances when it is redrawn. Cycle 0's block is pre-analysis and
    therefore identical either way; cycle 1's is downstream of the first
    analysis and separates the two settings.
    """
    n_e, num_cycles, num_steps = 8, 2, 2
    x0 = np.asarray(jax.random.normal(jax.random.PRNGKey(15), (n_e,)))
    knots = np.asarray(jax.random.normal(jax.random.PRNGKey(16), (num_cycles + 1, n_e)))
    obs_op = _RecordingObsOp(np.array([[1.0]]))

    FilterSmoothingESMDA(
        num_steps=num_steps,
        common_inner_noise=common_inner_noise,
        rng_key=jax.random.PRNGKey(17),
        **_smoothing_kwargs(
            _TrajectoryToyModel(param_effect=0.0, decay=0.9),
            observation_operator=obs_op,
        ),
    ).run(
        state=_scalar_state(x0),
        params=_trajectory_params(knots),
        observations=jnp.array([[1.0], [1.0]]),
    )

    calls = obs_op.calls
    assert len(calls) == (num_steps + 1) * num_cycles
    np.testing.assert_array_equal(calls[0], calls[num_cycles])
    if common_inner_noise:
        np.testing.assert_array_equal(calls[1], calls[num_cycles + 1])
    else:
        assert not np.array_equal(calls[1], calls[num_cycles + 1])


# ---------------------------------------------------------------------------
# (7) Per-cycle segment restriction
# ---------------------------------------------------------------------------


def _inner_filter(cycle_length: Optional[float] = _CYCLE_LENGTH) -> Any:
    from data_assimilation.filter_smoothing.base import _TrajectoryStateFilter

    return _TrajectoryStateFilter(
        observation_operator=_ToyObsOp(np.array([[1.0]])),
        forward_model=_TrajectoryToyModel(),
        C_D=jnp.array([0.01]),
        mode="state",
        cycle_length=cycle_length,
    )


def test_params_for_cycle_restricts_the_trajectory_to_the_segment() -> None:
    """Cycle ``k`` gets the trajectory over ``[k*dt, (k+1)*dt]``, not one knot.

    The knot grid here IS the cycle grid, which is the case the old
    piecewise-constant contract covered — and even here the segment is a
    two-knot ramp, on a segment-local clock, because that is what the backends
    interpolate and what the truth trajectory was generated with. A static
    parameter with no ``time`` dimension rides along untouched.
    """
    n_e, n_knots = 5, 3
    knots = np.asarray(jax.random.normal(jax.random.PRNGKey(18), (n_knots, n_e)))
    gamma = np.asarray(jax.random.normal(jax.random.PRNGKey(19), (n_e,)))
    params = _trajectory_params(knots, static=gamma)
    inner = _inner_filter()

    for cycle in range(n_knots - 1):
        segment = inner._params_for_cycle(cycle, params)
        # Segment-local clock: the schedule a backend receives is call-relative.
        np.testing.assert_allclose(
            np.asarray(segment["time"].values), [0.0, _CYCLE_LENGTH], atol=1e-9
        )
        np.testing.assert_allclose(
            np.asarray(segment["theta"].values),
            knots[cycle : cycle + 2],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(segment["gamma"].values), gamma, rtol=1e-6
        )

    assert inner._params_for_cycle(0, None) is None
    # A params Dataset without a trajectory is passed through by identity.
    static_only = xarray.Dataset(
        {"gamma": (("ensemble",), jnp.asarray(gamma))},
        coords={"ensemble": np.arange(n_e)},
    )
    assert inner._params_for_cycle(1, static_only) is static_only


def test_params_for_cycle_interpolates_inside_a_coarse_knot_interval() -> None:
    """Knots COARSER than a cycle: every segment gets its own interpolated ramp.

    Two knots 2 cycles apart, so neither cycle has a knot of its own — the old
    "knot k backs cycle k" reading has nothing to say here. Cycle 0 must ramp
    from the first knot to the interval's midpoint value and cycle 1 from there
    to the second knot, which is exactly the trajectory the truth generator
    would have handed the solver.
    """
    inner = _inner_filter()
    params = _trajectory_params(np.array([[0.0, 10.0], [4.0, 14.0]]), spacing=2.0)

    first = inner._params_for_cycle(0, params)
    np.testing.assert_allclose(np.asarray(first["time"].values), [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(first["theta"].values), [[0.0, 10.0], [2.0, 12.0]], rtol=1e-6
    )

    second = inner._params_for_cycle(1, params)
    np.testing.assert_allclose(np.asarray(second["time"].values), [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(second["theta"].values), [[2.0, 12.0], [4.0, 14.0]], rtol=1e-6
    )


def test_params_for_cycle_keeps_the_knots_inside_a_fine_segment() -> None:
    """Knots FINER than a cycle: the interior knot survives, at its local time.

    A half-cycle grid, so each segment brackets one interior knot. Collapsing
    the segment to its endpoints would silently smooth away exactly the
    resolution a finer ``time.seconds_per_knot`` was configured to buy.
    """
    inner = _inner_filter()
    params = _trajectory_params(
        np.array([[0.0], [1.0], [4.0], [9.0], [16.0]]), spacing=0.5
    )

    segment = inner._params_for_cycle(1, params)
    np.testing.assert_allclose(
        np.asarray(segment["time"].values), [0.0, 0.5, 1.0], atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(segment["theta"].values).ravel(), [4.0, 9.0, 16.0], rtol=1e-6
    )


def test_params_for_cycle_holds_a_single_knot_constant() -> None:
    """One knot is a legal (constant) trajectory: clamped at both ends."""
    inner = _inner_filter()
    params = _trajectory_params(np.array([[2.0, 3.0]]))

    for cycle in (0, 5):
        segment = inner._params_for_cycle(cycle, params)
        np.testing.assert_allclose(
            np.asarray(segment["time"].values), [0.0, _CYCLE_LENGTH], atol=1e-9
        )
        np.testing.assert_allclose(
            np.asarray(segment["theta"].values), [[2.0, 3.0], [2.0, 3.0]], rtol=1e-6
        )


def test_params_for_cycle_clamps_beyond_the_last_knot() -> None:
    """Past the trajectory's end the value is HELD, never extrapolated.

    A window whose cycles run past its knots (or a segment straddling the end)
    must freeze at the last knot: linear extrapolation of an AR(2) draw would
    walk the parameter off to an arbitrary value nothing constrains.
    """
    inner = _inner_filter()
    knots = np.array([[0.0], [1.0]])
    params = _trajectory_params(knots)

    # Segment 1 straddles the last knot: it opens at it and holds.
    straddling = inner._params_for_cycle(1, params)
    np.testing.assert_allclose(
        np.asarray(straddling["theta"].values).ravel(), [1.0, 1.0], rtol=1e-6
    )
    # ... and a segment entirely past the end is constant at the last knot.
    beyond = inner._params_for_cycle(4, params)
    np.testing.assert_allclose(
        np.asarray(beyond["theta"].values).ravel(), [1.0, 1.0], rtol=1e-6
    )


def test_params_for_cycle_needs_a_cycle_length_and_knot_times() -> None:
    """Both halves of "where is segment k on the trajectory?" are required.

    The knot axis is physical seconds, so neither the cycle length nor the
    ``time`` coordinate can be inferred; guessing either would silently re-time
    the whole trajectory rather than fail.
    """
    params = _trajectory_params(np.array([[0.0], [1.0]]))
    with pytest.raises(ValueError, match="cycle_length"):
        _inner_filter(cycle_length=None)._params_for_cycle(0, params)

    with pytest.raises(ValueError, match="'time' coordinate"):
        _inner_filter()._params_for_cycle(0, params.drop_vars("time"))


def test_base_filter_params_for_cycle_is_the_identity() -> None:
    """The hook is a no-op for every existing filter configuration (§3.1).

    Identity, not merely an equal Dataset: an ordinary run must not pay a copy
    or a re-index per cycle for a feature it does not use.
    """
    params = xarray.Dataset(
        {"a": (("ensemble",), jnp.zeros(4))}, coords={"ensemble": np.arange(4)}
    )
    enkf = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(np.array([[1.0]])),
        forward_model=_TrajectoryToyModel(),
        C_D=jnp.array([0.01]),
        mode="state",
    )
    assert enkf._params_for_cycle(0, params) is params
    assert enkf._params_for_cycle(3, params) is params
    assert enkf._params_for_cycle(0, None) is None
