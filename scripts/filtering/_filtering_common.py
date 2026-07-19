"""Shared post-processing helpers for the sequential-filtering pipeline.

``scripts/filtering/run_filtering.py`` runs the EnKF and saves the raw
artifacts; ``compute_filtering_metrics.py`` turns those into
``run_summary.yaml`` and ``make_filtering_figures.py`` draws the figures. This
module is the filtering analogue of ``scripts/esmda/_esmda_common.py``: it holds
the glue the metric and figure stages share so they are not duplicated.

The filter's time axis is the **cycle**: one full-weight analysis per forecast
segment, updating the end-of-segment state. So the per-cycle quantities here
(one analyzed state / parameter vector per cycle) play the role the per-window
rollout quantities play in the ESMDA pipeline, and the truth is compared at the
end-of-cycle frames. The heavy lifting (lazy truth access, sensor interpolation,
the sensor/parameter/state metrics) is reused verbatim from the ESMDA pipeline's
``scripts.esmda._esmda_common`` -- only the filter-specific reshaping (cycles ->
a time axis, selecting the truth's end-of-cycle frames) lives here.
"""

# mypy: ignore-errors
# Legacy untyped helper module, matching scripts/esmda/_esmda_common.py: it is
# type-checked transitively whenever a script importing it is committed. Waived
# wholesale rather than annotated piecemeal; drop this when typed.

from __future__ import annotations

import pathlib

import numpy as np
import xarray

from pyurbanair.utils.run_utils import add_velocity_magnitude

# Reuse the ESMDA pipeline's truth-access / sensor-series / scalar helpers
# unchanged (the filtering pipeline is a downstream consumer of the same
# machinery). ``_sensor_component_timeseries`` is the low-level per-sensor
# (u, v, w) interpolation shared by the truth and ensemble series builders.
from scripts.esmda._esmda_common import (  # noqa: F401  (re-exported)
    _sensor_component_timeseries,
    build_sensor_sets,
    load_run_config,
    open_truth,
    parameter_metric_summary,
    read_yaml,
    select_z_plane,
    sensor_magnitude,
    series_stats,
    streaming_state_rmse,
    vector_sensor_metrics,
    write_yaml,
)

# ---------------------------------------------------------------------------
# Cycle <-> frame bookkeeping
# ---------------------------------------------------------------------------


def end_of_cycle_indices(n_per_cycle: int, num_cycles: int) -> list[int]:
    """Truth-frame index of each cycle's last (end-of-segment) frame.

    Cycle ``c`` owns the contiguous half-open block
    ``[c*n_per_cycle, (c+1)*n_per_cycle)`` of truth frames; the analysis updates
    the state at that block's final frame, so this returns
    ``[(c+1)*n_per_cycle - 1 for c in range(num_cycles)]`` -- the frames the
    analyzed states are compared against.
    """
    return [(c + 1) * int(n_per_cycle) - 1 for c in range(int(num_cycles))]


def truth_end_of_cycle(ta: dict, n_frames: int | None = None) -> xarray.Dataset:
    """Lazily open the truth at its end-of-cycle frames (a ``time``-dimmed view).

    Mirrors ``open_truth`` (the multi-GB truth stays on disk), then selects only
    the one frame per cycle the analysis is scored against. When ``n_frames`` is
    given (e.g. the filter only saved the final analyzed frame), the last
    ``n_frames`` end-of-cycle frames are kept so they align with the available
    analyzed states.
    """
    end_idx = end_of_cycle_indices(ta["n_per_cycle"], ta["num_cycles"])
    if n_frames is not None:
        end_idx = end_idx[-int(n_frames) :]
    truth = open_truth(
        ta["true_state_path"],
        ta["n_total"],
        ta["x_offset"],
        ta["start_idx"],
        ta["t_offset"],
    )
    truth = truth.isel(time=end_idx)
    # Use an integer cycle-index time axis (0, 1, ...) so the truth and the
    # analyzed states share the same axis and pair up cycle-for-cycle; the
    # downstream metrics/plots align on it (the physical end-of-cycle times are
    # not needed, and are not carried consistently on the analyzed states).
    return truth.assign_coords(time=np.arange(truth.sizes["time"], dtype=float))


def load_analyzed_states(run_dir: pathlib.Path, ta: dict) -> xarray.Dataset:
    """The per-cycle analyzed ensemble states as a ``(time, ensemble, ...)`` view.

    Prefers ``state_history.nc`` (the analyzed end-of-cycle frame of every
    cycle, saved when ``run.save_history``); its ``cycle`` dimension is renamed
    to ``time`` and given the physical end-of-cycle times so it lines up with
    :func:`truth_end_of_cycle`. Falls back to ``posterior_state.nc`` (the final
    analyzed frame alone) as a single-frame time axis when the history was not
    saved, so the state/sensor diagnostics still work (over the last cycle only).
    """
    history_path = pathlib.Path(run_dir) / "state_history.nc"
    if history_path.exists():
        hist = xarray.open_dataset(history_path)
        # Promote the ``cycle`` dim to a ``time`` axis with an integer
        # cycle-index (matching :func:`truth_end_of_cycle`). The saved ``time``
        # coord is only the first cycle's scalar end-of-segment time (concat
        # kept it scalar), so it cannot index the cycles -- drop it.
        n = hist.sizes["cycle"]
        if "time" in hist.coords:
            hist = hist.drop_vars("time")
        hist = hist.rename({"cycle": "time"})
        return hist.assign_coords(time=np.arange(n, dtype=float))

    posterior = xarray.open_dataset(pathlib.Path(run_dir) / "posterior_state.nc")
    if "time" in posterior.coords:
        posterior = posterior.drop_vars("time")
    return posterior.expand_dims(time=[0.0])


def load_params_history(run_dir: pathlib.Path, ta: dict) -> xarray.Dataset:
    """The per-cycle analyzed params on a physical ``time`` axis for truth-alignment.

    ``params_history.nc`` stacks the prior (its first entry) then each cycle's
    analyzed params along ``cycle``. The parameter metric/figure code compares
    these against ``true_params`` by interpolating the truth onto the estimate's
    x-axis (see ``pyurbanair.plotting.compute_parameter_metrics``), so the
    estimate needs an x-axis in the truth's units (seconds) rather than a bare
    cycle index -- otherwise a *drifting* (time-varying) truth is sampled at the
    wrong times. Relabel ``cycle`` -> ``time`` with physical seconds: the prior
    sits at t=0 and cycle ``c``'s posterior at its end-of-segment time
    ``(c+1)*sim_time``. For a static (constant) truth the axis is immaterial (the
    truth interpolates to the same value everywhere), so this is backward
    compatible.
    """
    hist = xarray.open_dataset(pathlib.Path(run_dir) / "params_history.nc")
    n = int(hist.sizes["cycle"])
    sim_time = float(ta["sim_time"])
    # Entry 0 is the prior (t=0); entry i>=1 is cycle (i-1)'s posterior at its
    # end-of-segment time i*sim_time. So times[i] = i*sim_time.
    times = np.arange(n, dtype=float) * sim_time
    if "time" in hist.coords:
        hist = hist.drop_vars("time")
    return hist.rename({"cycle": "time"}).assign_coords(time=times)


# ---------------------------------------------------------------------------
# Per-cycle sensor series (truth vs analyzed ensemble at fixed points)
# ---------------------------------------------------------------------------


def truth_cycle_sensor_series(
    truth_end: xarray.Dataset, sensor_sets: dict, solver_name: str
) -> dict:
    """Truth per-component ``(u, v, w)`` sensor series over the end-of-cycle frames.

    ``truth_end`` is :func:`truth_end_of_cycle`; returns
    ``{name: DataArray(component, time, sensor)}``.
    """
    return {
        name: _sensor_component_timeseries(truth_end, ox, oy, oz, solver_name)
        for name, (ox, oy, oz) in sensor_sets.items()
    }


def ensemble_cycle_sensor_series(
    analyzed_states: xarray.Dataset, sensor_sets: dict, solver_name: str
) -> dict:
    """Analyzed-ensemble per-component ``(u, v, w)`` sensor series over cycles.

    ``analyzed_states`` is :func:`load_analyzed_states` (``time`` = cycle axis);
    returns ``{name: DataArray(component, ..., time, sensor)}`` keeping the
    ``ensemble`` axis. The truth/ensemble time axes are aligned per-sensor by the
    shared metric/plot code, so differing physical times are handled.
    """
    states = analyzed_states.load()
    return {
        name: _sensor_component_timeseries(states, ox, oy, oz, solver_name)
        for name, (ox, oy, oz) in sensor_sets.items()
    }


# ---------------------------------------------------------------------------
# Velocity magnitude mean/std of an ensemble state (for the field figures)
# ---------------------------------------------------------------------------


def ensemble_velocity_mean_std(state: xarray.Dataset):
    """``(vel_mean, vel_std)`` of the |U| field over the ``ensemble`` axis.

    The filter saves the raw ensemble state (no precomputed ``vel_mean`` /
    ``vel_std`` -- unlike the ESMDA pipeline's ``posterior_state_mean.nc``), so
    the field figures derive them here. Combines u/v/w by index (matching
    ``add_velocity_magnitude`` / the ESMDA streaming summary).
    """
    vel = add_velocity_magnitude(state)["vel_magnitude"]
    return vel.mean("ensemble"), vel.std("ensemble")


# ---------------------------------------------------------------------------
# Cycle diagnostics (the always-available filter health series)
# ---------------------------------------------------------------------------

_DIAGNOSTIC_FIELDS = (
    "obs_prior_rmse",
    "obs_posterior_rmse",
    "innovation_chi2",
    "state_spread_prior",
    "state_spread_posterior",
    "param_spread_prior",
    "param_spread_posterior",
)


def cycle_diagnostics_series(run_dir: pathlib.Path) -> dict:
    """Read ``cycle_diagnostics.yaml`` into ``{field: np.ndarray}`` over cycles.

    Missing/``None`` entries (e.g. the parameter spreads in ``mode='state'``)
    become NaN so :func:`series_stats` skips them. ``{}`` if the file is absent.
    """
    rows = read_yaml(pathlib.Path(run_dir) / "cycle_diagnostics.yaml")
    if not rows:
        return {}
    series = {}
    for field in _DIAGNOSTIC_FIELDS:
        vals = [
            (np.nan if row.get(field) is None else float(row[field])) for row in rows
        ]
        if not np.all(np.isnan(vals)):
            series[field] = np.asarray(vals, dtype=float)
    return series
