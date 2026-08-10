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
end-of-cycle frames. Lazy truth access and sensor interpolation are reused
verbatim from the ESMDA pipeline's ``scripts.esmda._esmda_common``, and the
sensor/parameter/state metrics come from ``evaluation.scores`` /
``evaluation.turbulence`` / ``evaluation.sensors`` -- only the filter-specific
reshaping (cycles -> a time axis, selecting the truth's end-of-cycle frames)
lives here.
"""

# mypy: ignore-errors
# Legacy untyped helper module, matching scripts/esmda/_esmda_common.py: it is
# type-checked transitively whenever a script importing it is committed. Waived
# wholesale rather than annotated piecemeal; drop this when typed.

from __future__ import annotations

import logging
import pathlib
from typing import NamedTuple

import numpy as np
import xarray

from pyurbanair.utils.run_utils import add_velocity_magnitude

# Reuse the ESMDA pipeline's truth-access / sensor-series / scalar helpers
# unchanged (the filtering pipeline is a downstream consumer of the same
# machinery). ``_sensor_component_timeseries`` is the low-level per-sensor
# (u, v, w) interpolation shared by the truth and ensemble series builders.
from scripts.esmda._esmda_common import (  # noqa: F401  (re-exported)
    _concat_sensor_pieces,
    _sensor_component_timeseries,
    _window_sensor_series,
    build_sensor_sets,
    load_run_config,
    open_truth,
    read_yaml,
    truth_sensor_series,
    write_yaml,
)

logger = logging.getLogger(__name__)

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
    x-axis (see ``evaluation.scores.compute_parameter_metrics``), so the
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
    become NaN so :func:`evaluation.scores.series_stats` skips them. ``{}`` if
    the file is absent.
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


# ---------------------------------------------------------------------------
# Cycle state sources: what plays the role of the ESMDA window state files
#
# The WP1.3-1.5 evaluation blocks (sensor statistics, mean fields, and the
# figures over them) are all reductions over the *ensemble states of one
# assimilation step*. The ESMDA pipeline has exactly one thing to reduce --
# ``windows/window_{w}_posterior_state.nc``, the whole window rolled out at
# ``time.output_frequency`` -- so its metric stage never has to choose. The
# filter has two, and which one exists depends on how the run was configured:
#
#   FORECAST (``run.ensemble_save_on_disk=true``). The filter's own on-disk mode
#     already writes every member's full forecast segment to
#     ``_ensemble_states/cycle_{k}/state_{m}.nc`` and never prunes it
#     (``prune_disk_cycles`` is left False by run_filtering.py). Cycle ``k``'s
#     forecast is the ensemble rolled out from cycle ``k-1``'s analysis under
#     cycle ``k-1``'s analyzed parameters, i.e. exactly the posterior rollout the
#     ESMDA window files hold -- and, being a forecast, a genuinely out-of-sample
#     verification object. Resolved in time, so the window statistics get a real
#     within-cycle variance and the mean fields get real resolved turbulence.
#
#   ANALYSIS (the default). Only ``state_history.nc`` exists: ONE analyzed frame
#     per cycle. Every reduction still works -- a cycle is then a one-frame
#     window -- but two things degrade, and both are recorded rather than hidden:
#     the per-cycle VARIANCE statistic is null (a ddof=1 variance of one sample),
#     and the "temporal" moments behind TKE / <u'w'> are taken across cycles
#     rather than within them, so they carry the analysis increments as if they
#     were turbulence. Fine for the mean field and the rank histogram of the mean
#     statistic; read the TKE panels as an upper bound.
#
# Everything downstream consumes :class:`CycleStates` and does not branch again:
# the source decides the sensor-series extraction, the truth pairing and the
# binning unit once, here.
# ---------------------------------------------------------------------------

_FORECAST_STATES_DIR = "_ensemble_states"

# Slack on the "does this frame fall inside its cycle" test, relative to the
# cycle length. The rebasing below lands a segment's first frame exactly on the
# cycle boundary in exact arithmetic; in IEEE double it can land a few ULP short,
# and dropping that frame would silently shorten one cycle's sample.
_CYCLE_EDGE_TOLERANCE = 1e-9


class CycleStates(NamedTuple):
    """Which per-cycle ensemble states the evaluation blocks reduce over.

    Attributes:
        kind: ``"forecast"`` or ``"analysis"`` (see the module comment above).
        bin_seconds: The unit :func:`evaluation.sensors.window_statistics` bins
            the extracted series by, so that one bin is one cycle. The physical
            cycle length for ``forecast`` (whose series carry real seconds); 1.0
            for ``analysis``, whose series carry an integer cycle index instead
            -- there is one frame per cycle, so a physical axis would buy
            nothing and would need the analyzed frames' times, which
            ``state_history.nc`` does not carry per cycle.
        num_cycles: Cycles available on the ENSEMBLE side, which is what the
            binning uses. Can be short of ``truth_access.yaml``'s count when the
            history was not saved and only the final analyzed frame survives.
        n_members: Ensemble size, read from metadata alone.
        paths: ``forecast`` only -- ``paths[k][m]`` is member ``m``'s forecast
            file for cycle ``k``. ``None`` for ``analysis``.
        description: One line naming the source and its caveat. It reaches the
            reader by three routes, and all three are needed: the log, the run
            summary's ``cycle_states`` block, and ``eval_fields.nc``'s
            ``moment_sampling`` attribute -- which is the one the figure stage
            reads, since S1 and F1 open that file and nothing else.
    """

    kind: str
    bin_seconds: float
    num_cycles: int
    n_members: int
    paths: list[list[pathlib.Path]] | None
    description: str


def _forecast_cycle_paths(
    run_dir: pathlib.Path, num_cycles: int
) -> list[list[pathlib.Path]] | None:
    """``paths[k][m]`` for the on-disk forecasts, or ``None`` if unusable.

    All-or-nothing across cycles and members: a partial set (a run killed
    mid-cycle, or one pruned by hand) would put different cycles on different
    ensemble sizes and different horizons in one statistic, which is worse than
    falling back to the analyzed frames.
    """
    root = pathlib.Path(run_dir) / _FORECAST_STATES_DIR
    if not root.is_dir():
        return None
    cycles = []
    for k in range(num_cycles):
        # ``state_10.nc`` must not sort before ``state_2.nc``: the member index
        # is what pairs a file with its parameter row, so the sort is numeric.
        # Anything in the directory that merely LOOKS like a member file without
        # carrying an integer index -- a hand-renamed partial write, a
        # ``state_final.nc`` -- is dropped rather than fed to ``int()``, which
        # would raise out of the sort key and take the whole stage down over a
        # stray file.
        members = []
        for path in (root / f"cycle_{k}").glob("state_*.nc"):
            try:
                members.append((int(path.stem.rsplit("_", 1)[1]), path))
            except ValueError:
                logger.info("Ignoring %s: it is not a numbered member state file", path)
        members = [path for _, path in sorted(members)]
        if not members:
            logger.info(
                "No forecast states for cycle %d under %s; the evaluation blocks "
                "fall back to the per-cycle analyzed frames",
                k,
                root,
            )
            return None
        cycles.append(members)
    sizes = {len(members) for members in cycles}
    if len(sizes) != 1:
        logger.warning(
            "The on-disk forecasts hold different member counts per cycle (%s); "
            "falling back to the per-cycle analyzed frames rather than scoring "
            "cycles against each other at different ensemble sizes",
            sorted(sizes),
        )
        return None
    return cycles


def cycle_state_source(run_dir: pathlib.Path, ta: dict) -> CycleStates:
    """Pick the per-cycle ensemble states to evaluate (see the module comment)."""
    run_dir = pathlib.Path(run_dir)
    num_cycles = int(ta["num_cycles"])
    sim_time = float(ta["sim_time"])

    paths = _forecast_cycle_paths(run_dir, num_cycles)
    if paths is not None:
        return CycleStates(
            kind="forecast",
            bin_seconds=sim_time,
            num_cycles=num_cycles,
            n_members=len(paths[0]),
            paths=paths,
            description=(
                "on-disk ensemble forecasts (_ensemble_states/cycle_*/state_*.nc): "
                "every frame of every cycle's forecast segment"
            ),
        )

    # ``load_analyzed_states`` is the authority on what the analysis path will
    # actually read, including its fallback to the final analyzed frame alone
    # when ``run.save_history`` was off -- so the cycle count is taken from the
    # view it produces rather than from a file this function assumes exists.
    analyzed = load_analyzed_states(run_dir, ta)
    n_available = int(analyzed.sizes["time"])
    has_members = "ensemble" in analyzed.dims
    n_members = int(analyzed.sizes["ensemble"]) if has_members else 0
    analyzed.close()
    if not has_members:
        # Every block built on this source is probabilistic -- window statistics,
        # rank counts, the ensemble spread behind the hit rate -- so an
        # ensemble-mean-only artifact has nothing for them to score. Raised with
        # the reason spelled out rather than surfacing as a bare KeyError on
        # ``sizes["ensemble"]``: the callers turn this into "these blocks are
        # absent, the rest of the summary stands", and the log line they print is
        # this message.
        raise ValueError(
            "the per-cycle analyzed states carry no 'ensemble' axis, so there is "
            "no ensemble to reduce (an ensemble-mean-only artifact)"
        )
    return CycleStates(
        kind="analysis",
        bin_seconds=1.0,
        num_cycles=n_available,
        n_members=n_members,
        paths=None,
        description=(
            "per-cycle analyzed frames (state_history.nc): ONE frame per cycle, so "
            "the per-cycle variance statistic is null and the TKE / <u'w'> moments "
            "are taken ACROSS cycles (they carry the analysis increments). Re-run "
            "with run.ensemble_save_on_disk=true for within-cycle turbulence"
        ),
    )


def _rebased_forecast_times(times: np.ndarray, cycle: int, sim_time: float):
    """A forecast segment's times on the run's global axis, and which to keep.

    The filter analyzes a segment's LAST frame, so that frame is the end of cycle
    ``k`` and belongs at ``(k+1)*sim_time - dt``; everything before it counts
    back at the segment's own cadence. Anchoring on the last frame rather than
    the first is what makes cycle 0 work: it is a cold start, so its segment
    carries the model's spin-up in front of the cycle proper, and anchoring on
    the first frame would push the whole cycle past its own boundary. The
    spin-up frames land before ``k*sim_time`` and are dropped here, which is the
    same horizon the truth's ``[k*n_per_cycle, (k+1)*n_per_cycle)`` block covers.
    """
    times = np.asarray(times, dtype=float)
    dt = float(np.median(np.diff(times))) if times.size > 1 else sim_time
    rebased = times - times[-1] + (cycle + 1) * sim_time - dt
    keep = rebased >= cycle * sim_time - _CYCLE_EDGE_TOLERANCE * sim_time
    return rebased, keep


def cycle_sensor_series(
    run_dir: pathlib.Path,
    ta: dict,
    source: CycleStates,
    sensor_sets: dict,
    on_member=None,
) -> dict:
    """Ensemble ``(component, ensemble, time, sensor)`` series over the cycles.

    The filtering counterpart of ``ensemble_sensor_series``, and it keeps that
    function's memory discipline for the same reason: one member's forecast
    segment at a time, never a whole cycle's ensemble. ``on_member`` is called as
    ``on_member(cycle, member, state)`` on the materialised slice, so the mean
    fields accumulate inside this pass rather than in a second one
    (master-plan invariant 2).

    The time axis is what :attr:`CycleStates.bin_seconds` promises: real seconds
    rebased onto the run's global clock for ``forecast``, an integer cycle index
    for ``analysis``.
    """
    solver_name = str(ta["assim_solver_name"])
    if source.kind == "analysis":
        # One lazily-opened file whose ``time`` axis is already the cycle index;
        # ``_window_sensor_series`` slices it member-at-a-time exactly as the
        # ESMDA extraction slices a window file.
        states = load_analyzed_states(run_dir, ta)
        return dict(
            _window_sensor_series(states, sensor_sets, solver_name, on_member, 0)
        )

    assert source.paths is not None  # invariant of kind == "forecast"
    sim_time = float(ta["sim_time"])
    pieces: dict = {name: [] for name in sensor_sets}
    for cycle, members in enumerate(source.paths):
        per_member: dict = {name: [] for name in sensor_sets}
        for member_index, path in enumerate(members):
            with xarray.open_dataset(path) as raw:
                if "time" not in raw.coords:
                    raise ValueError(
                        f"{path} carries no time coordinate, so its frames "
                        f"cannot be placed in cycle {cycle}"
                    )
                rebased, keep = _rebased_forecast_times(
                    raw["time"].values, cycle, sim_time
                )
                # ``expand_dims`` after the selection: the per-member files carry
                # no ensemble axis, and every downstream reduction (and the mean
                # field collector) is written against the ESMDA layout, which
                # does.
                member = (
                    raw[["u", "v", "w"]]
                    .assign_coords(time=rebased)
                    .isel(time=np.flatnonzero(keep))
                    .load()
                    .expand_dims(ensemble=[member_index])
                )
            for name, (ox, oy, oz) in sensor_sets.items():
                per_member[name].append(
                    _sensor_component_timeseries(member, ox, oy, oz, solver_name)
                )
            if on_member is not None:
                on_member(cycle, member_index, member)
        for name, parts in per_member.items():
            pieces[name].append(
                parts[0] if len(parts) == 1 else xarray.concat(parts, dim="ensemble")
            )
    return _concat_sensor_pieces(pieces)


def truth_cycle_statistics_series(
    ta: dict, source: CycleStates, sensor_sets: dict
) -> dict:
    """The truth sensor series matched to ``source``, frame for frame.

    Matching is the whole point: the window statistics compare a truth statistic
    against the members' statistics, so both have to be computed over the same
    frames of the same cycles. Against ``forecast`` states that is the truth's
    own ``[k*n_per_cycle, (k+1)*n_per_cycle)`` blocks on the physical axis;
    against ``analysis`` states it is the one end-of-cycle frame per cycle, on
    the same integer cycle index the analyzed frames carry.
    """
    solver_name = str(ta["truth_solver_name"])
    if source.kind == "analysis":
        truth_end = truth_end_of_cycle(ta, source.num_cycles)
        return truth_cycle_sensor_series(truth_end, sensor_sets, solver_name)
    return dict(
        truth_sensor_series(
            ta["true_state_path"],
            ta["n_total"],
            ta["x_offset"],
            ta["start_idx"],
            ta["t_offset"],
            sensor_sets,
            solver_name,
            source.num_cycles,
            int(ta["n_per_cycle"]),
        )
    )


def truth_field_pass(collector, ta: dict, source: CycleStates):
    """Stream the truth through a mean-field collector over the same frames.

    The counterpart of ``MeanFieldCollector.truth_pass`` for each source: the
    truth's contiguous per-cycle blocks for ``forecast``, and its end-of-cycle
    frames alone for ``analysis`` -- fed as one chunk, which the collector
    sub-chunks internally, because they are not contiguous in the truth and so
    cannot be expressed as its ``(n_chunks, chunk_frames)`` slicing.
    """
    if source.kind == "forecast":
        return collector.truth_pass(ta, source.num_cycles, int(ta["n_per_cycle"]))
    truth_end = truth_end_of_cycle(ta, source.num_cycles)
    if truth_end.sizes.get("time", 0):
        collector.add(0, 0, truth_end)
    truth_end.close()
    return collector
