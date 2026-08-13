"""Unit tests for the filter-smoothing hybrid (data_assimilation.filter_smoothing).

Three groups, all on toy in-memory forward models — no CFD solver:

* the pure trajectory helpers (``knot_times``, ``params_for_segment``,
  ``segment_bounds``, ``trajectory_values_at``), which own the segment
  geometry and are where an off-by-one in the knot placement would hide;
* the ESMDA ``final_forecast=False`` seam the hybrid rests on: it must skip
  exactly the posterior forward pass and change nothing else;
* the estimator itself — constructor validation, the exact reduction of the
  static-parameter path to a plain filter run, and the per-segment trajectory
  slicing of the dynamic path.
"""

import pathlib
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray
from data_assimilation.filter_smoothing import (
    FilterSmoothing,
    knot_times,
    params_for_segment,
    segment_bounds,
    trajectory_values_at,
)
from data_assimilation.filtering import EnsembleKalmanFilter
from data_assimilation.inflation import RTPS
from data_assimilation.smoothing.esmda import (
    ParameterESMDA,
    StateAndTimeVaryingParameterESMDA,
    StateESMDA,
    TimeVaryingParameterESMDA,
)

# One window: two filter cycles of two observation frames each. The frame times
# are on the WINDOW clock, which is the hybrid's contract, so the cycle
# boundaries fall at 5 and 10 seconds.
_NX = 2
_N_SENSORS = 2
_CYCLE_FRAMES = 2
_NUM_CYCLES = 2
_WINDOW_FRAMES = _CYCLE_FRAMES * _NUM_CYCLES
_CYCLE_SECONDS = 5.0
_FRAME_SECONDS = _CYCLE_SECONDS / _CYCLE_FRAMES
_N_E = 12
_OBS_VARIANCE = 0.04


# ---------------------------------------------------------------------------
# Toy pieces (same shapes as tests/test_filtering.py, one window's worth)
# ---------------------------------------------------------------------------


def _param_drive(params: Optional[xarray.Dataset], n_e: int) -> np.ndarray:
    """The scalar each member is forced with, from static OR trajectory params.

    A trajectory is reduced to its mean over the knots the caller handed this
    forecast — which is what makes the per-segment restriction observable in
    the forecast at all, without the toy having to interpolate anything.
    """
    if params is None:
        return np.zeros(n_e)
    values = params["a"]
    if "time" in values.dims:
        return np.asarray(values.mean(dim="time").values, dtype=float)
    return np.asarray(values.values, dtype=float)


class _ToyEnsembleModel:
    """Damped linear forecast over ``num_frames`` output frames, driven by ``a``.

    Records every ``run_ensemble`` call: the hybrid's contract is largely about
    *what each forecast is handed* (which slice of the trajectory, which warm
    start), so the spy is the assertion surface for the dynamic path.
    """

    save_on_disk = False
    results_dir: Optional[pathlib.Path] = None

    def __init__(self, num_frames: int, param_effect: float = 1.0) -> None:
        self.num_frames = num_frames
        self.param_effect = param_effect
        self.calls = 0
        self.params_seen: list[Optional[xarray.Dataset]] = []
        self.states_seen: list[Optional[xarray.Dataset]] = []

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> Optional[xarray.Dataset]:
        # ``Optional``: the on-disk twin below overrides this and returns None,
        # exactly as the real ensemble models do in save-on-disk mode.
        assert state is not None
        self.calls += 1
        # Deep copies: the caller keeps mutating/rebuilding these Datasets.
        self.params_seen.append(None if params is None else params.copy(deep=True))
        self.states_seen.append(state.copy(deep=True))

        x = np.asarray(state["u"].values, dtype=float)  # (N_e, nx)
        drive = _param_drive(params, x.shape[0])
        frames = []
        current = x
        for _ in range(self.num_frames):
            current = 0.9 * current + self.param_effect * drive[:, None]
            frames.append(current)
        values = np.stack(frames, axis=1)  # (N_e, T, nx)
        return xarray.Dataset(
            {"u": (("ensemble", "time", "x"), values)},
            coords={
                "ensemble": np.arange(x.shape[0]),
                "time": np.arange(1, self.num_frames + 1) * _FRAME_SECONDS,
                "x": np.arange(x.shape[1]),
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


class _OnDiskToyEnsembleModel(_ToyEnsembleModel):
    """Disk-writing twin of :class:`_ToyEnsembleModel`, for the staging path."""

    save_on_disk = True

    def __init__(self, num_frames: int, results_dir: pathlib.Path) -> None:
        super().__init__(num_frames)
        self.results_dir = results_dir

    def set_results_dir(self, results_dir: pathlib.Path) -> None:
        self.results_dir = results_dir

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> Optional[xarray.Dataset]:
        forecast = super().run_ensemble(state=state, params=params)
        assert forecast is not None and self.results_dir is not None
        for member in range(forecast.sizes["ensemble"]):
            forecast.isel(ensemble=member, drop=True).to_netcdf(
                self.results_dir / f"state_{member}.nc"
            )
        return None


class _TemporalToyObsOp:
    """``y_t = H x_t`` for every frame, as the labelled DataArray the ops return."""

    def __init__(self) -> None:
        self.H = np.eye(_N_SENSORS, _NX)

    def __call__(self, state: xarray.Dataset) -> xarray.DataArray:
        x = np.asarray(state["u"].values, dtype=float)  # (..., time, nx)
        values = np.einsum("...i,di->...d", x, self.H)
        dims = (
            ("ensemble", "time", "obs") if "ensemble" in state.dims else ("time", "obs")
        )
        return xarray.DataArray(
            values,
            dims=dims,
            coords={"time": np.asarray(state["time"].values, dtype=float)},
        )


def _obs_op() -> Any:
    """``Any``: the smoothers annotate the bare ``ObservationOperator`` type."""
    return _TemporalToyObsOp()


def _batch_times(cycle: int) -> np.ndarray:
    """Cycle ``k``'s frame times, on the WINDOW clock (seconds)."""
    return cycle * _CYCLE_SECONDS + np.arange(1, _CYCLE_FRAMES + 1) * _FRAME_SECONDS


def _observation_batches(
    seed: int = 3, num_cycles: int = _NUM_CYCLES
) -> list[xarray.DataArray]:
    """One labelled ("time", "obs") batch per filter cycle."""
    rng = np.random.default_rng(seed)
    return [
        xarray.DataArray(
            rng.normal(size=(_CYCLE_FRAMES, _N_SENSORS)),
            dims=("time", "obs"),
            coords={"time": _batch_times(k), "obs": np.arange(_N_SENSORS)},
        )
        for k in range(num_cycles)
    ]


def _initial_state(seed: int = 1) -> xarray.Dataset:
    rng = np.random.default_rng(seed)
    return xarray.Dataset(
        {"u": (("ensemble", "x"), rng.normal(size=(_N_E, _NX)))},
        coords={"ensemble": np.arange(_N_E), "x": np.arange(_NX)},
    )


def _static_prior(seed: int = 2) -> xarray.Dataset:
    rng = np.random.default_rng(seed)
    return xarray.Dataset(
        {"a": (("ensemble",), rng.normal(loc=1.0, scale=0.3, size=_N_E))},
        coords={"ensemble": np.arange(_N_E)},
    )


def _trajectory_prior(knots: np.ndarray, seed: int = 4) -> xarray.Dataset:
    """A knot trajectory ensemble on the given (window-clock) knot times."""
    rng = np.random.default_rng(seed)
    values = rng.normal(loc=1.0, scale=0.3, size=(knots.size, _N_E))
    return xarray.Dataset(
        {"a": (("time", "ensemble"), values)},
        coords={"time": knots, "ensemble": np.arange(_N_E)},
    )


def _static_smoother(model: Any, num_steps: int = 2, seed: int = 0) -> ParameterESMDA:
    return ParameterESMDA(
        observation_operator=_obs_op(),
        forward_model=model,
        C_D=jnp.diag(jnp.full(_WINDOW_FRAMES * _N_SENSORS, _OBS_VARIANCE)),
        num_steps=num_steps,
        rng_key=jax.random.PRNGKey(seed),
    )


def _dynamic_smoother(
    model: Any,
    num_knots: int,
    num_steps: int = 2,
    seed: int = 0,
    num_cycles: int = _NUM_CYCLES,
) -> TimeVaryingParameterESMDA:
    return TimeVaryingParameterESMDA(
        observation_operator=_obs_op(),
        forward_model=model,
        C_D=jnp.diag(jnp.full(num_cycles * _CYCLE_FRAMES * _N_SENSORS, _OBS_VARIANCE)),
        num_time_points=num_knots,
        num_steps=num_steps,
        rng_key=jax.random.PRNGKey(seed),
    )


def _filter(model: Any, mode: str, seed: int = 7) -> EnsembleKalmanFilter:
    return EnsembleKalmanFilter(
        observation_operator=_obs_op(),
        forward_model=model,
        C_D=jnp.full(_N_SENSORS, _OBS_VARIANCE),
        mode=mode,  # type: ignore[arg-type]
        # Joint mode refuses to run without spread maintenance; state mode is
        # indifferent to it, and keeping one construction keeps the two modes
        # comparable.
        inflation=RTPS(alpha=0.5) if mode == "joint" else None,
        rng_key=jax.random.PRNGKey(seed),
    )


def _window_observations(batches: list[xarray.DataArray]) -> xarray.DataArray:
    """What the hybrid hands the smoother (used by the seam tests directly)."""
    return xarray.concat(batches, dim="time", join="override")


# ---------------------------------------------------------------------------
# (a) knot_times
# ---------------------------------------------------------------------------


def test_knot_times_returns_the_knot_axis() -> None:
    knots = np.array([0.0, 2.5, 10.0])
    np.testing.assert_allclose(knot_times(_trajectory_prior(knots)), knots)


def test_knot_times_requires_a_time_coordinate() -> None:
    """A ``time`` dimension without values cannot locate a segment."""
    params = xarray.Dataset(
        {"a": (("time", "ensemble"), np.ones((2, _N_E)))},
        coords={"ensemble": np.arange(_N_E)},
    )
    with pytest.raises(ValueError, match="no 'time' coordinate"):
        knot_times(params)


def test_knot_times_rejects_a_non_monotonic_axis() -> None:
    params = _trajectory_prior(np.array([0.0, 10.0, 5.0]))
    with pytest.raises(ValueError, match="not strictly increasing"):
        knot_times(params)


# ---------------------------------------------------------------------------
# (b) params_for_segment
# ---------------------------------------------------------------------------


def test_params_for_segment_interpolates_endpoints_and_rebases_the_axis() -> None:
    """Knots coarser than the segment: two interpolated endpoints, local axis."""
    params = _trajectory_prior(np.array([0.0, 10.0]))
    a = np.asarray(params["a"].values)  # (2, N_e)

    segment = params_for_segment(params, 2.5, 7.5)

    np.testing.assert_allclose(segment.coords["time"].values, [0.0, 5.0])
    expected = np.stack([0.75 * a[0] + 0.25 * a[1], 0.25 * a[0] + 0.75 * a[1]])
    np.testing.assert_allclose(np.asarray(segment["a"].values), expected)
    # The member axis rides through untouched.
    np.testing.assert_array_equal(
        segment.coords["ensemble"].values, params.coords["ensemble"].values
    )


def test_params_for_segment_carries_strictly_interior_knots() -> None:
    """Knots finer than the segment keep their resolution; boundaries do not duplicate."""
    knots = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    params = _trajectory_prior(knots)
    a = np.asarray(params["a"].values)

    segment = params_for_segment(params, 3.0, 7.0)

    # Endpoints at 3 and 7, interior knots 4 and 6 — the knots ON no boundary
    # here, but 2 and 8 are outside and must not appear.
    np.testing.assert_allclose(segment.coords["time"].values, [0.0, 1.0, 3.0, 4.0])
    expected = np.stack([0.5 * a[1] + 0.5 * a[2], a[2], a[3], 0.5 * a[3] + 0.5 * a[4]])
    np.testing.assert_allclose(np.asarray(segment["a"].values), expected)


def test_params_for_segment_drops_knots_sitting_on_the_boundaries() -> None:
    """A boundary knot is already the interpolated endpoint — no duplicate time."""
    params = _trajectory_prior(np.array([0.0, 5.0, 10.0]))

    segment = params_for_segment(params, 0.0, 5.0)

    np.testing.assert_allclose(segment.coords["time"].values, [0.0, 5.0])
    np.testing.assert_allclose(
        np.asarray(segment["a"].values), np.asarray(params["a"].values)[:2]
    )


def test_params_for_segment_clamps_past_the_last_knot() -> None:
    """Outside the knot range the trajectory holds its end value."""
    params = _trajectory_prior(np.array([0.0, 4.0]))
    a = np.asarray(params["a"].values)

    segment = params_for_segment(params, 6.0, 10.0)

    np.testing.assert_allclose(segment.coords["time"].values, [0.0, 4.0])
    np.testing.assert_allclose(np.asarray(segment["a"].values), np.stack([a[1], a[1]]))


def test_params_for_segment_passes_static_variables_through() -> None:
    params = _trajectory_prior(np.array([0.0, 10.0]))
    params["c"] = (("ensemble",), np.linspace(0.0, 1.0, _N_E))

    segment = params_for_segment(params, 0.0, 5.0)

    assert "time" not in segment["c"].dims
    np.testing.assert_array_equal(
        np.asarray(segment["c"].values), np.asarray(params["c"].values)
    )


def test_params_for_segment_is_a_no_op_without_a_time_dimension() -> None:
    params = _static_prior()
    assert params_for_segment(params, 0.0, 5.0) is params


def test_params_for_segment_rejects_a_non_positive_segment() -> None:
    params = _trajectory_prior(np.array([0.0, 10.0]))
    with pytest.raises(ValueError, match="positive length"):
        params_for_segment(params, 5.0, 5.0)


# ---------------------------------------------------------------------------
# (c) segment_bounds
# ---------------------------------------------------------------------------


def test_segment_bounds_tile_the_window_from_zero() -> None:
    """Segment k ends at batch k's last frame; the first one starts at 0.0."""
    assert segment_bounds(_observation_batches()) == [(0.0, 5.0), (5.0, 10.0)]


def test_segment_bounds_handle_one_frame_per_cycle() -> None:
    """The production geometry (assimilate_every_n_step=1, one frame per cycle)."""
    batches = [
        xarray.DataArray(
            np.zeros((1, _N_SENSORS)),
            dims=("time", "obs"),
            coords={"time": np.array([(k + 1) * _CYCLE_SECONDS])},
        )
        for k in range(3)
    ]
    assert segment_bounds(batches) == [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]


def test_segment_bounds_reject_plain_arrays() -> None:
    with pytest.raises(ValueError, match="labelled"):
        segment_bounds([np.zeros((2, _N_SENSORS))])


def test_segment_bounds_reject_a_clock_reset_between_batches() -> None:
    """Cycle-local coordinates would put two segments in the same place."""
    batches = [
        xarray.DataArray(
            np.zeros((_CYCLE_FRAMES, _N_SENSORS)),
            dims=("time", "obs"),
            coords={"time": np.arange(1, _CYCLE_FRAMES + 1) * _FRAME_SECONDS},
        )
        for _ in range(2)
    ]
    with pytest.raises(ValueError, match="strictly increasing WINDOW clock"):
        segment_bounds(batches)


def test_segment_bounds_reject_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one filter cycle"):
        segment_bounds([])


# ---------------------------------------------------------------------------
# (d) trajectory_values_at
# ---------------------------------------------------------------------------


def test_trajectory_values_at_interpolates_per_member() -> None:
    params = _trajectory_prior(np.array([0.0, 10.0]))
    a = np.asarray(params["a"].values)

    values = trajectory_values_at(params, 2.5)

    assert "time" not in values.dims
    assert values["a"].dims == ("ensemble",)
    assert values["a"].shape == (_N_E,)
    np.testing.assert_allclose(
        np.asarray(values["a"].values), 0.75 * a[0] + 0.25 * a[1]
    )


def test_trajectory_values_at_clamps_outside_the_knot_range() -> None:
    params = _trajectory_prior(np.array([0.0, 4.0]))
    a = np.asarray(params["a"].values)
    np.testing.assert_allclose(
        np.asarray(trajectory_values_at(params, -1.0)["a"]), a[0]
    )
    np.testing.assert_allclose(
        np.asarray(trajectory_values_at(params, 99.0)["a"]), a[1]
    )


def test_trajectory_values_at_passes_static_variables_through() -> None:
    params = _trajectory_prior(np.array([0.0, 10.0]))
    params["c"] = (("ensemble",), np.linspace(0.0, 1.0, _N_E))

    values = trajectory_values_at(params, 5.0)

    np.testing.assert_array_equal(
        np.asarray(values["c"].values), np.asarray(params["c"].values)
    )
    assert trajectory_values_at(_static_prior(), 5.0) is not None


def test_trajectory_values_at_is_a_no_op_without_a_time_dimension() -> None:
    params = _static_prior()
    assert trajectory_values_at(params, 5.0) is params


# ---------------------------------------------------------------------------
# (e) The ESMDA ``final_forecast=False`` seam
# ---------------------------------------------------------------------------


def test_final_forecast_false_returns_params_alone_and_skips_the_last_forecast() -> (
    None
):
    """``num_steps`` forecasts, no posterior pass, ``num_steps`` pred-obs entries."""
    model = _ToyEnsembleModel(num_frames=_WINDOW_FRAMES)
    smoother = _static_smoother(model, num_steps=2)
    smoother.collect_obs_diagnostics = True

    result = smoother(
        state=_initial_state(),
        params=_static_prior(),
        observations=_window_observations(_observation_batches()),
        final_forecast=False,
    )

    assert isinstance(result, xarray.Dataset)  # not the (params, state) tuple
    assert "time" not in result.dims
    assert model.calls == 2
    assert len(smoother.pred_obs_history) == 2


def test_final_forecast_false_with_a_params_history_still_skips_it() -> None:
    model = _ToyEnsembleModel(num_frames=_WINDOW_FRAMES)
    smoother = _static_smoother(model, num_steps=3)

    history = smoother(
        state=_initial_state(),
        params=_static_prior(),
        observations=_window_observations(_observation_batches()),
        return_params_history=True,
        final_forecast=False,
    )

    assert isinstance(history, xarray.Dataset)
    # Entry 0 is the prior, then one per MDA step.
    assert history.sizes["esmda_step"] == 4
    assert model.calls == 3


def test_final_forecast_false_returns_the_same_params_as_the_full_run() -> None:
    """The seam skips a forecast, it does not perturb the MDA loop."""
    observations = _window_observations(_observation_batches())
    state, prior = _initial_state(), _static_prior()

    full = _static_smoother(_ToyEnsembleModel(num_frames=_WINDOW_FRAMES), seed=11)(
        state=state, params=prior, observations=observations
    )
    assert isinstance(full, tuple)
    skipped = _static_smoother(_ToyEnsembleModel(num_frames=_WINDOW_FRAMES), seed=11)(
        state=state, params=prior, observations=observations, final_forecast=False
    )
    assert isinstance(skipped, xarray.Dataset)

    np.testing.assert_array_equal(
        np.asarray(skipped["a"].values), np.asarray(full[0]["a"].values)
    )


def test_final_forecast_false_rejects_a_state_history() -> None:
    smoother = _static_smoother(_ToyEnsembleModel(num_frames=_WINDOW_FRAMES))
    with pytest.raises(ValueError, match="return_state_history requires"):
        smoother(
            state=_initial_state(),
            params=_static_prior(),
            observations=_window_observations(_observation_batches()),
            return_state_history=True,
            final_forecast=False,
        )


# ---------------------------------------------------------------------------
# (f) Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_a_state_bearing_smoother() -> None:
    model = _ToyEnsembleModel(num_frames=_WINDOW_FRAMES)
    smoother = StateESMDA(
        observation_operator=_obs_op(),
        forward_model=model,
        C_D=jnp.diag(jnp.full(_WINDOW_FRAMES * _N_SENSORS, _OBS_VARIANCE)),
    )
    with pytest.raises(ValueError, match="initial state as well"):
        # The annotation already forbids this one; the runtime check is what
        # protects a Hydra-instantiated smoother, which mypy never sees.
        FilterSmoothing(smoother=smoother, filter=_filter(model, "state"))  # type: ignore[arg-type]


def test_constructor_rejects_the_joint_state_and_dynamic_smoother() -> None:
    """It IS a ParameterESMDA through the MRO, so the check cannot be isinstance alone."""
    model = _ToyEnsembleModel(num_frames=_WINDOW_FRAMES)
    smoother = StateAndTimeVaryingParameterESMDA(
        observation_operator=_obs_op(),
        forward_model=model,
        C_D=jnp.diag(jnp.full(_WINDOW_FRAMES * _N_SENSORS, _OBS_VARIANCE)),
        num_time_points=2,
    )
    assert isinstance(smoother, ParameterESMDA)
    with pytest.raises(ValueError, match="initial state as well"):
        FilterSmoothing(smoother=smoother, filter=_filter(model, "state"))


def test_constructor_rejects_a_parameter_only_filter() -> None:
    model = _ToyEnsembleModel(num_frames=_WINDOW_FRAMES)
    parameter_filter = EnsembleKalmanFilter(
        observation_operator=_obs_op(),
        forward_model=model,
        C_D=jnp.full(_N_SENSORS, _OBS_VARIANCE),
        mode="parameter",
        inflation=RTPS(alpha=0.5),
    )
    with pytest.raises(ValueError, match="no state block"):
        FilterSmoothing(smoother=_static_smoother(model), filter=parameter_filter)


def test_run_rejects_unlabelled_observations() -> None:
    model = _ToyEnsembleModel(num_frames=_WINDOW_FRAMES)
    hybrid = FilterSmoothing(
        smoother=_static_smoother(model), filter=_filter(model, "state")
    )
    with pytest.raises(ValueError, match="labelled"):
        hybrid.run(
            state=_initial_state(),
            params=_static_prior(),
            observations=[np.zeros((_CYCLE_FRAMES, _N_SENSORS))],
        )


def test_run_rejects_a_single_window_dataarray() -> None:
    model = _ToyEnsembleModel(num_frames=_WINDOW_FRAMES)
    hybrid = FilterSmoothing(
        smoother=_static_smoother(model), filter=_filter(model, "state")
    )
    with pytest.raises(ValueError, match="one batch PER FILTER CYCLE"):
        hybrid.run(
            state=_initial_state(),
            params=_static_prior(),
            observations=_window_observations(_observation_batches()),
        )


# ---------------------------------------------------------------------------
# (g) The static path reduces EXACTLY to a plain filter run
# ---------------------------------------------------------------------------


def test_static_joint_path_is_bitwise_a_plain_enkf_run() -> None:
    """Joint mode + static theta ≡ ``EnsembleKalmanFilter.run`` over the cycles.

    The strongest property the hybrid claims: with no trajectory to place on
    the segments, the filter phase is a single ``filter.run`` and must be
    indistinguishable — bit for bit, same keys — from driving that filter
    directly with the ESMDA posterior. Anything the hybrid did *around* the
    call (re-numbering, staging, history slicing) would show up here.
    """
    batches = _observation_batches()
    state, prior = _initial_state(), _static_prior()

    hybrid = FilterSmoothing(
        smoother=_static_smoother(_ToyEnsembleModel(num_frames=_WINDOW_FRAMES)),
        filter=_filter(_ToyEnsembleModel(num_frames=_CYCLE_FRAMES), "joint"),
    )
    result = hybrid.run(
        state=state, params=prior, observations=batches, return_history=True
    )

    reference = _filter(_ToyEnsembleModel(num_frames=_CYCLE_FRAMES), "joint").run(
        state=state,
        params=result.esmda_params,
        observations=batches,
        return_history=True,
    )

    assert result.params is not None and reference.params is not None
    assert result.state is not None and reference.state is not None
    np.testing.assert_array_equal(
        np.asarray(result.params["a"].values), np.asarray(reference.params["a"].values)
    )
    np.testing.assert_array_equal(
        np.asarray(result.state["u"].values), np.asarray(reference.state["u"].values)
    )
    assert [d.cycle for d in result.diagnostics] == [0, 1]
    assert [d.innovation_chi2 for d in result.diagnostics] == [
        d.innovation_chi2 for d in reference.diagnostics
    ]
    # One entry per cycle, the ANALYSED params: the filter's prior entry is
    # dropped so params/state/diagnostics index the same cycles.
    assert result.params_history is not None
    assert result.params_history.sizes["cycle"] == _NUM_CYCLES
    assert result.state_history is not None
    assert result.state_history.sizes["cycle"] == _NUM_CYCLES
    assert result.applied_params_history is None
    assert result.esmda_params_history is not None


def test_state_mode_reports_no_filter_params() -> None:
    """In state mode the parameters only ride through the forecasts."""
    hybrid = FilterSmoothing(
        smoother=_static_smoother(_ToyEnsembleModel(num_frames=_WINDOW_FRAMES)),
        filter=_filter(_ToyEnsembleModel(num_frames=_CYCLE_FRAMES), "state"),
    )
    result = hybrid.run(
        state=_initial_state(),
        params=_static_prior(),
        observations=_observation_batches(),
        return_history=True,
    )

    assert result.params is None
    assert result.params_history is None
    assert result.state is not None
    assert "a" in result.esmda_params


def test_pred_obs_histories_are_accumulated_over_the_window() -> None:
    """The filter rebinds its lists per run; the hybrid keeps one entry per cycle."""
    filter_model = _ToyEnsembleModel(num_frames=_CYCLE_FRAMES)
    enkf = _filter(filter_model, "state")
    enkf.collect_pred_obs = True
    hybrid = FilterSmoothing(
        smoother=_static_smoother(_ToyEnsembleModel(num_frames=_WINDOW_FRAMES)),
        filter=enkf,
    )

    hybrid.run(
        state=_initial_state(),
        params=_static_prior(),
        observations=_observation_batches(),
    )

    assert len(hybrid.pred_obs_history) == _NUM_CYCLES
    assert len(hybrid.pred_obs_post_history) == _NUM_CYCLES
    assert len(hybrid.pred_obs_frames_history) == _NUM_CYCLES
    # Frame-major (T*N_obs, N_e), the filter's own layout.
    assert hybrid.pred_obs_history[0].shape == (_CYCLE_FRAMES * _N_SENSORS, _N_E)


# ---------------------------------------------------------------------------
# (h) The dynamic path: one segment per cycle
# ---------------------------------------------------------------------------


def test_dynamic_state_mode_forecasts_each_segment_with_its_own_slice() -> None:
    """Each segment's forecast receives the trajectory restricted to it.

    Two knots over the window and two segments, so every endpoint is a
    different interpolation: segment 0 gets ``[theta(0), theta(5)]`` on a local
    ``[0, 5]`` axis and segment 1 ``[theta(5), theta(10)]`` on the same local
    axis — the re-basing that makes the schedule consumable by a forward model
    configured for one cycle.
    """
    knots = np.array([0.0, 10.0])
    filter_model = _ToyEnsembleModel(num_frames=_CYCLE_FRAMES)
    hybrid = FilterSmoothing(
        smoother=_dynamic_smoother(
            _ToyEnsembleModel(num_frames=_WINDOW_FRAMES), num_knots=knots.size
        ),
        filter=_filter(filter_model, "state"),
    )

    result = hybrid.run(
        state=_initial_state(),
        params=_trajectory_prior(knots),
        observations=_observation_batches(),
        return_history=True,
    )

    theta = np.asarray(result.esmda_params["a"].values)  # (2 knots, N_e)
    midpoint = 0.5 * (theta[0] + theta[1])
    assert len(filter_model.params_seen) == _NUM_CYCLES

    for segment, expected in enumerate(
        (np.stack([theta[0], midpoint]), np.stack([midpoint, theta[1]]))
    ):
        seen = filter_model.params_seen[segment]
        assert seen is not None
        np.testing.assert_allclose(
            seen.coords["time"].values, [0.0, _CYCLE_SECONDS], atol=1e-9
        )
        np.testing.assert_allclose(
            np.asarray(seen["a"].values), expected, rtol=1e-6, atol=1e-8
        )

    # The window's own bookkeeping: one cycle per segment, globally numbered.
    assert [d.cycle for d in result.diagnostics] == [0, 1]
    assert result.state_history is not None
    assert result.state_history.sizes["cycle"] == _NUM_CYCLES
    assert result.params is None


def test_dynamic_state_mode_chains_the_warm_start_across_segments() -> None:
    """Segment k+1 starts from segment k's ANALYSED state, as one filter run would."""
    filter_model = _ToyEnsembleModel(num_frames=_CYCLE_FRAMES)
    hybrid = FilterSmoothing(
        smoother=_dynamic_smoother(
            _ToyEnsembleModel(num_frames=_WINDOW_FRAMES), num_knots=2
        ),
        filter=_filter(filter_model, "state"),
    )

    result = hybrid.run(
        state=_initial_state(),
        params=_trajectory_prior(np.array([0.0, 10.0])),
        observations=_observation_batches(),
        return_history=True,
    )

    assert result.state_history is not None
    analysed_first = np.asarray(result.state_history.isel(cycle=0)["u"].values)
    np.testing.assert_array_equal(
        np.asarray(filter_model.states_seen[1]["u"].values),  # type: ignore[index]
        analysed_first,
    )
    assert result.state is not None
    np.testing.assert_array_equal(
        np.asarray(result.state["u"].values),
        np.asarray(result.state_history.isel(cycle=-1)["u"].values),
    )


class _IdentityParameterESMDA(ParameterESMDA):
    """A smoother whose analysis is a pass-through: ``__call__`` returns the prior.

    Lets a test drive the PUBLIC ``FilterSmoothing.run`` path with an exactly
    chosen posterior trajectory — a real MDA update would move the knots
    individually, which is precisely what the constant-trajectory equivalence
    test below must avoid.
    """

    def __call__(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        observations: Optional[Any] = None,
        return_params_history: bool = False,
        return_state_history: bool = False,
        final_forecast: bool = True,
    ) -> xarray.Dataset:
        assert params is not None
        return params


def test_dynamic_constant_trajectory_is_bitwise_one_multi_cycle_filter_run() -> None:
    """The per-segment loop is a re-arrangement of the filter, not a new filter.

    ``_run_dynamic`` claims that chaining L single-segment ``filter.run`` calls
    is numerically identical to one L-cycle call (the rng stream continues and
    the warm starts chain). Pinned here bitwise: a CONSTANT trajectory makes
    every per-segment restriction carry the same per-member drive as the full
    trajectory (the toy reduces params to their knot mean, and with knots at
    ``[0, horizon]`` every segment weight is an exact binary fraction, so the
    interpolated knot values are bitwise the member values), so the hybrid's
    dynamic path against an identically seeded standalone filter run over the
    same batches must agree to the last bit — any rng-stream or warm-start
    divergence would show as O(1) differences.
    """
    batches = _observation_batches()
    horizon = _NUM_CYCLES * _CYCLE_SECONDS
    member_values = np.asarray(_static_prior()["a"].values, dtype=float)
    constant_trajectory = xarray.Dataset(
        {"a": (("time", "ensemble"), np.stack([member_values, member_values]))},
        coords={"time": np.array([0.0, horizon]), "ensemble": np.arange(_N_E)},
    )

    hybrid_model = _ToyEnsembleModel(num_frames=_CYCLE_FRAMES)
    hybrid = FilterSmoothing(
        smoother=_IdentityParameterESMDA(
            observation_operator=_obs_op(),
            forward_model=_ToyEnsembleModel(num_frames=_WINDOW_FRAMES),
            C_D=jnp.diag(jnp.full(_WINDOW_FRAMES * _N_SENSORS, _OBS_VARIANCE)),
            num_steps=2,
            rng_key=jax.random.PRNGKey(0),
        ),
        filter=_filter(hybrid_model, "state", seed=7),
    )
    result = hybrid.run(
        state=_initial_state(),
        params=constant_trajectory,
        observations=batches,
    )

    # Reference: ONE multi-cycle run of an identically seeded standalone
    # filter handed the SAME trajectory (state mode passes params through to
    # every forecast unmodified).
    ref_model = _ToyEnsembleModel(num_frames=_CYCLE_FRAMES)
    reference = _filter(ref_model, "state", seed=7).run(
        state=_initial_state(),
        params=constant_trajectory,
        observations=batches,
    )

    assert result.state is not None and reference.state is not None
    np.testing.assert_array_equal(
        np.asarray(result.state["u"].values),
        np.asarray(reference.state["u"].values),
    )
    # The forecasts themselves were bitwise identical too, segment by segment.
    for hybrid_state, ref_state in zip(hybrid_model.states_seen, ref_model.states_seen):
        np.testing.assert_array_equal(
            np.asarray(hybrid_state["u"].values),  # type: ignore[index]
            np.asarray(ref_state["u"].values),  # type: ignore[index]
        )


def test_dynamic_joint_mode_carries_the_correction_on_the_esmda_schedule() -> None:
    """Segment k's prior is ``e_k + c``, with ``c = posterior_{k-1} - e_{k-1}``."""
    knots = np.array([0.0, 10.0])
    filter_model = _ToyEnsembleModel(num_frames=_CYCLE_FRAMES)
    hybrid = FilterSmoothing(
        smoother=_dynamic_smoother(
            _ToyEnsembleModel(num_frames=_WINDOW_FRAMES), num_knots=knots.size
        ),
        filter=_filter(filter_model, "joint"),
    )

    result = hybrid.run(
        state=_initial_state(),
        params=_trajectory_prior(knots),
        observations=_observation_batches(),
        return_history=True,
    )

    theta = result.esmda_params
    # Segment midpoints on the window clock: 2.5 and 7.5.
    e_0 = np.asarray(trajectory_values_at(theta, 2.5)["a"].values)
    e_1 = np.asarray(trajectory_values_at(theta, 7.5)["a"].values)

    assert result.applied_params_history is not None
    applied = np.asarray(result.applied_params_history["a"].values)  # (cycle, N_e)
    assert applied.shape == (_NUM_CYCLES, _N_E)
    # The first segment runs on the bare schedule (correction still zero) ...
    np.testing.assert_allclose(applied[0], e_0, rtol=1e-6, atol=1e-8)
    # ... and the second on the schedule plus what cycle 0 learned.
    assert result.params_history is not None
    posterior_0 = np.asarray(result.params_history.isel(cycle=0)["a"].values)
    np.testing.assert_allclose(
        applied[1], e_1 + (posterior_0 - e_0), rtol=1e-6, atol=1e-8
    )
    # The forecasts really got those parameters, statically (a joint analysis
    # cannot consume a time dimension).
    for segment in range(_NUM_CYCLES):
        seen = filter_model.params_seen[segment]
        assert seen is not None and "time" not in seen.dims
        np.testing.assert_allclose(np.asarray(seen["a"].values), applied[segment])

    assert result.params is not None
    np.testing.assert_array_equal(
        np.asarray(result.params["a"].values),
        np.asarray(result.params_history.isel(cycle=-1)["a"].values),
    )


# ---------------------------------------------------------------------------
# (i) On-disk staging of the per-segment cycle directories
# ---------------------------------------------------------------------------


def _on_disk_hybrid(
    root: pathlib.Path, num_cycles: int
) -> tuple[FilterSmoothing, EnsembleKalmanFilter]:
    filter_model = _OnDiskToyEnsembleModel(_CYCLE_FRAMES, root)
    enkf = _filter(filter_model, "state")
    smoother = _dynamic_smoother(
        _ToyEnsembleModel(num_frames=num_cycles * _CYCLE_FRAMES),
        num_knots=2,
        num_cycles=num_cycles,
    )
    return FilterSmoothing(smoother=smoother, filter=enkf), enkf


def test_dynamic_path_renumbers_the_staged_cycle_directories(
    tmp_path: pathlib.Path,
) -> None:
    """Every single-cycle run writes ``cycle_0``; the window must end up 0..L-1.

    Without the staging root each segment would first DELETE the previous
    segment's state files (``_set_cycle_results_dir`` empties the directory it
    is pointed at) and the window would keep one cycle's forecast instead of L.
    """
    root = tmp_path / "filter"
    root.mkdir()
    hybrid, enkf = _on_disk_hybrid(root, num_cycles=_NUM_CYCLES)

    hybrid.run(
        state=_initial_state(),
        params=_trajectory_prior(np.array([0.0, 10.0])),
        observations=_observation_batches(),
    )

    assert sorted(p.name for p in root.iterdir()) == ["cycle_0", "cycle_1"]
    assert len(list((root / "cycle_1").glob("state_*.nc"))) == _N_E
    # The staging root is transient and the filter's own root is restored, so a
    # caller can keep driving the same filter (the next window) unchanged.
    assert not (root / "_segment_staging").exists()
    assert enkf.base_results_dir == root


def test_dynamic_path_applies_the_filters_own_pruning_semantics(
    tmp_path: pathlib.Path,
) -> None:
    """``prune_disk_cycles`` keeps ``cycle_0`` and the last cycle, drops the rest."""
    root = tmp_path / "filter"
    root.mkdir()
    hybrid, enkf = _on_disk_hybrid(root, num_cycles=3)
    enkf.prune_disk_cycles = True

    hybrid.run(
        state=_initial_state(),
        params=_trajectory_prior(np.array([0.0, 15.0])),
        observations=_observation_batches(num_cycles=3),
    )

    assert sorted(p.name for p in root.iterdir()) == ["cycle_0", "cycle_2"]
