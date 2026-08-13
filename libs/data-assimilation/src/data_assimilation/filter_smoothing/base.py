"""ESMDA parameter estimation over a window, sequential filtering inside it.

The hybrid of ``smoothing/`` and ``filtering/``: per assimilation window the
ESMDA smoother runs its normal MDA **parameter-estimation** loop (static or
time-varying), and then — instead of ESMDA's posterior forward pass — a
sequential filter runs over the same window with the ESMDA-estimated
parameters. The window's posterior *state* is therefore always a filtered
state, produced cycle by cycle, while the parameters are estimated by a
smoother that saw the whole window at once.

One ``run()`` call is one window:

1. **ESMDA phase.** The per-cycle observation batches are concatenated into one
   window DataArray and handed to the smoother with ``final_forecast=False``
   (see :meth:`~data_assimilation.smoothing.base.BaseSmoothing.__call__`): the
   MDA loop runs unchanged, but the posterior forward pass is skipped because
   the filter is about to produce that state itself. Output: ``theta``, either
   static ``(ensemble,)`` variables or a knot trajectory carrying ``time``.
2. **Filter phase**, over the SAME raw per-cycle observations — never the
   aggregated ones. Aggregation is a smoother-side choice and stays inside
   ESMDA's own ``aggregate_observations``; the filter assimilates every frame
   the operator produced (``filtering/base.py``, module docstring).

   * ``theta`` **static**: one ``filter.run(...)`` over the whole window, which
     is exactly how ``scripts/filtering/run_filtering.py`` drives the filter.
     Nothing hybrid-specific happens, and in joint mode the phase reduces
     *exactly* to a standard joint EnKF over those cycles.
   * ``theta`` **dynamic**: the filter's forward model is instantiated with a
     horizon of ONE cycle and every forecast starts at local time 0, and
     ``BaseFilter.run`` passes the *same* ``params`` object to every cycle — so
     a whole-window schedule would replay its first interval on every segment.
     The hybrid therefore loops the segments itself, calling ``filter.run``
     once per segment with that segment's observation batch:

     - ``mode="state"``: segment ``k`` forecasts with the trajectory
       *restricted to segment k* (:func:`params_for_segment`) — endpoints
       linearly interpolated, interior knots carried, the axis re-based to
       segment-local ``[0, seg_len]``, which is what the backends consume. The
       parameters ride through the analysis unmodified.
     - ``mode="joint"``: "correction on the ESMDA schedule". A correction ``c``
       (initially zero) is carried cycle to cycle; segment ``k``'s prior is
       ``e_k + c`` with ``e_k`` the trajectory evaluated at the segment
       MIDPOINT (:func:`trajectory_values_at`), and ``c`` is re-derived as
       ``posterior_k - e_k`` after the analysis. The midpoint is a fixed
       convention, not a knob: it is the best constant approximation of the
       schedule over the segment, and the filter needs a constant because a
       joint analysis estimates the parameter value *now*
       (``BaseFilter._check_static_params``). With a static ``theta`` this
       reduces to plain joint filtering, which is the property
       ``tests/test_filter_smoothing.py`` pins bitwise.

Segment geometry is never configured: it is derived at ``run()`` time from the
observation batches' own time coordinates (:func:`segment_bounds`), which live
on the WINDOW clock in seconds. Those coordinates are what the ESMDA phase
aggregates by; the filter itself is positional (``_cycle_observations``
transposes to ``("time", "obs")`` and takes ``.values``, in the batch's own
order, with no reference to the coordinate), so window-clock rather than
cycle-local coordinates change nothing on the filter side.

C_D stays per-instance: the smoother's is the window-aggregated ``(N_d, N_d)``
diagonal, the filter's a per-frame 1-D variance vector. The hybrid never builds
either — the run script does.
"""

import logging
import pathlib
import shutil
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import xarray
from data_assimilation.filtering.base import BaseFilter, CycleDiagnostics
from data_assimilation.smoothing.esmda import ParameterESMDA, StateAndParameterESMDA

logger = logging.getLogger(__name__)

# Relative slack of every knot-time comparison (is this knot strictly inside the
# segment?). The knot axis is built in float32 by the samplers, so an exact
# comparison against a float64 segment boundary would trip on representation
# alone.
_TIME_RTOL = 1e-6


def _time_tol(scale: float) -> float:
    """Absolute slack for a time comparison at the given scale."""
    return _TIME_RTOL * max(abs(float(scale)), 1.0)


def knot_times(params: xarray.Dataset) -> np.ndarray:
    """The trajectory's knot times, in PHYSICAL SECONDS on its own clock.

    The knot grid is a free choice (``time.seconds_per_knot``), independent of
    the cycle length, so the segment restriction below cannot infer the knots'
    positions from their index — it reads them off the ``time`` coordinate the
    samplers emit (``build_knot_times``). A ``time`` dimension carrying no
    coordinate values is therefore an error rather than something to fill in.
    """
    if "time" not in params.coords:
        raise ValueError(
            "The parameter trajectory has a 'time' dimension but no 'time' "
            "coordinate. Filter smoothing reads the knot times as physical "
            "seconds (the samplers set them from time.seconds_per_knot); "
            "without them a segment cannot be located on the trajectory."
        )
    times = np.asarray(params.coords["time"].values, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError(
            f"The trajectory's 'time' coordinate has shape {times.shape}; a "
            "1-D, non-empty knot axis is required."
        )
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError(
            "The trajectory's knot times are not strictly increasing: "
            f"{times.tolist()}. The segment restriction interpolates between "
            "consecutive knots, which needs an ordered grid."
        )
    return times


def _interpolate_knots(
    times: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """``np.interp`` of a time-leading array, broadcast over the trailing dims.

    ``values`` is ``(n_knots, ...)`` (typically ``(n_knots, N_e)`` — one
    trajectory per member), ``targets`` a 1-D array of times. Outside the knot
    range the value is CLAMPED to the nearest end knot, exactly as
    :func:`numpy.interp` does: a segment past the last knot holds it, and a
    single-knot trajectory is constant everywhere.
    """
    if times.size == 1:
        return np.repeat(values[:1], targets.size, axis=0)
    upper = np.clip(np.searchsorted(times, targets, side="left"), 1, times.size - 1)
    lower = upper - 1
    span = times[upper] - times[lower]
    weight = np.clip((targets - times[lower]) / span, 0.0, 1.0)
    weight = weight.reshape((targets.size,) + (1,) * (values.ndim - 1))
    return np.asarray(values[lower] * (1.0 - weight) + values[upper] * weight)


def params_for_segment(
    params: xarray.Dataset,
    t_start: float,
    t_end: float,
) -> xarray.Dataset:
    """The trajectory restricted to ``[t_start, t_end]`` of the window clock.

    The forecast of one segment is a call to a forward model configured for one
    cycle, and such a call is always RELATIVE: its schedule runs on a local
    ``[0, t_end - t_start]`` axis, which the backend then shifts onto its own
    continuing clock. So the restriction both selects and re-bases:

    * the two endpoints are linearly interpolated between the bracketing knots
      (clamped outside the knot range, see :func:`_interpolate_knots`), so the
      segment starts and ends exactly where the window trajectory does;
    * every knot STRICTLY inside the segment is carried through unchanged — a
      knot grid finer than the cycle keeps its resolution — while a knot sitting
      on a boundary is dropped, because the interpolated endpoint already
      carries it and a duplicate time would corrupt the backend's schedule;
    * the resulting ``time`` coordinate is segment-local, starting at 0.

    Variables without a ``time`` dimension are not part of the trajectory and
    pass through untouched, keeping their own coordinates. A ``params`` with no
    ``time`` dimension at all is returned as it is (nothing to restrict).
    """
    if "time" not in params.dims:
        return params

    start = float(t_start)
    end = float(t_end)
    if not end > start:
        raise ValueError(
            f"A segment must have positive length, got [{start}, {end}]. The "
            "bounds are on the window clock, in seconds "
            "(see segment_bounds)."
        )
    times = knot_times(params)
    length = end - start
    tol = _time_tol(length)
    interior = times[(times > start + tol) & (times < end - tol)]
    targets = np.concatenate(([start], interior, [end]))
    local = np.concatenate(([0.0], interior - start, [length]))

    data_vars: dict[str, xarray.DataArray] = {}
    for name, variable in params.data_vars.items():
        if "time" not in variable.dims:
            data_vars[str(name)] = variable
            continue
        ordered = variable.transpose("time", ...)
        segment = _interpolate_knots(times, np.asarray(ordered.values), targets).astype(
            variable.dtype, copy=False
        )
        data_vars[str(name)] = xarray.DataArray(
            segment,
            dims=ordered.dims,
            coords={
                str(dim): params.coords[dim]
                for dim in ordered.dims
                if dim != "time" and dim in params.coords
            },
        ).transpose(*variable.dims)

    return xarray.Dataset(data_vars, attrs=params.attrs).assign_coords(time=local)


def trajectory_values_at(params: xarray.Dataset, t: float) -> xarray.Dataset:
    """The trajectory evaluated at ONE time: a ``time``-dim-free Dataset.

    Every time-varying variable is linearly interpolated at ``t`` per ensemble
    member (clamped outside the knot range, as everywhere here) and loses its
    ``time`` dimension; static variables pass through. The result is therefore
    a plain ``(ensemble,)`` parameter Dataset — the only thing a joint filter
    analysis can consume, since it estimates the parameter value *now* and
    ``BaseFilter._check_static_params`` rejects a ``time`` dimension outright.

    A ``params`` with no ``time`` dimension is returned as it is, so the caller
    does not have to branch on static/dynamic before asking.
    """
    if "time" not in params.dims:
        return params

    times = knot_times(params)
    targets = np.asarray([float(t)])

    data_vars: dict[str, xarray.DataArray] = {}
    for name, variable in params.data_vars.items():
        if "time" not in variable.dims:
            data_vars[str(name)] = variable
            continue
        ordered = variable.transpose("time", ...)
        value = _interpolate_knots(times, np.asarray(ordered.values), targets)[
            0
        ].astype(variable.dtype, copy=False)
        dims = tuple(str(dim) for dim in ordered.dims[1:])
        data_vars[str(name)] = xarray.DataArray(
            value,
            dims=dims,
            coords={dim: params.coords[dim] for dim in dims if dim in params.coords},
        )

    return xarray.Dataset(data_vars, attrs=params.attrs)


def segment_bounds(
    observations: Sequence[xarray.DataArray],
) -> list[tuple[float, float]]:
    """Forecast-segment boundaries, in seconds on the WINDOW clock.

    One segment per filter cycle, read off the observations themselves rather
    than from a ``cycle_length`` knob: segment ``k`` ENDS at the last time
    coordinate of batch ``k`` — the time of the frame whose state the cycle's
    analysis updates — and STARTS where segment ``k-1`` ended (0.0 for the
    first batch, i.e. the window's own origin). Segments therefore tile the
    window exactly, by construction, whatever the frame cadence is.

    This is the only place the observations' time coordinates carry meaning for
    the filter phase; the filter itself consumes each batch positionally.
    """
    if not observations:
        raise ValueError(
            "observations must contain at least one filter cycle: the segment "
            "geometry is derived from the batches themselves."
        )

    bounds: list[tuple[float, float]] = []
    previous_end = 0.0
    for k, batch in enumerate(observations):
        if not isinstance(batch, xarray.DataArray):
            raise ValueError(
                f"observations[{k}] is a {type(batch).__name__}; the segment "
                'geometry needs labelled ("time", "obs") DataArrays whose time '
                "coordinate is the frame time in seconds on the window clock."
            )
        if "time" not in batch.coords:
            raise ValueError(
                f"observations[{k}] has no 'time' coordinate. The hybrid places "
                "each cycle on the window clock from the frame times; without "
                "them the trajectory cannot be restricted to a segment."
            )
        frame_times = np.asarray(batch.coords["time"].values, dtype=float)
        if frame_times.ndim != 1 or frame_times.size == 0:
            raise ValueError(
                f"observations[{k}] has a 'time' coordinate of shape "
                f"{frame_times.shape}; a 1-D, non-empty frame axis is required."
            )
        if frame_times.size > 1 and not np.all(np.diff(frame_times) > 0.0):
            raise ValueError(
                f"observations[{k}]'s frame times are not strictly increasing: "
                f"{frame_times.tolist()}. The filter assimilates a batch's "
                "frames in the order they are stored, so an unordered batch "
                "would place the cycle's end somewhere inside it."
            )
        end = float(frame_times[-1])
        if not end > previous_end:
            raise ValueError(
                f"observations[{k}] ends at t={end} s, which is not after the "
                f"previous cycle's end t={previous_end} s. The batches' time "
                "coordinates must be on one strictly increasing WINDOW clock "
                "(the first batch's frames come after the window origin 0.0)."
            )
        bounds.append((previous_end, end))
        previous_end = end
    return bounds


@dataclass
class FilterSmoothingResult:
    """Return value of :meth:`FilterSmoothing.run` (one assimilation window).

    ``esmda_params`` is the MDA posterior — the trajectory or static Dataset
    the ESMDA phase produced, in ESMDA's own schema, so the shared
    metric/figure stages read it unchanged. ``state`` is the filter's analyzed
    end-of-window frame (the warm start for the next window) and ``params`` the
    filter's own final parameters: ``None`` in ``mode="state"``, where the
    parameters only ride through the forecasts, and the final CORRECTED
    parameters in ``mode="joint"``.

    ``diagnostics`` holds one :class:`~data_assimilation.filtering.base.\
CycleDiagnostics` per filter cycle, renumbered 0..L-1 over the window (the
    per-segment ``filter.run`` calls of the dynamic path each number their one
    cycle 0).

    The histories are present only with ``return_history=True``:

    * ``esmda_params_history``: the smoother's own ``esmda_step``-concatenated
      trajectory, entry 0 the prior (``num_steps + 1`` entries).
    * ``params_history`` / ``state_history``: ``cycle``-concatenated, ONE ENTRY
      PER CYCLE, holding that cycle's ANALYSED values. Note the deliberate
      difference from ``FilterResult.params_history``, whose entry 0 is the
      prior: both hybrid paths drop it so the two histories index the same
      cycles as ``diagnostics``. ``params_history`` is ``None`` in
      ``mode="state"``.
    * ``applied_params_history``: joint mode with a dynamic trajectory only —
      the parameters each segment was actually forecast with, i.e.
      ``e_k + c_k`` before that cycle's analysis. It is what separates the
      ESMDA schedule from what the filter ran; ``None`` on the static path,
      where every cycle is forecast with ``esmda_params`` itself, and in state
      mode, where the applied parameters are per-segment trajectories of
      differing knot counts and do not stack.
    """

    esmda_params: xarray.Dataset
    state: Optional[xarray.Dataset]
    params: Optional[xarray.Dataset]
    diagnostics: list[CycleDiagnostics]
    esmda_params_history: Optional[xarray.Dataset] = None
    params_history: Optional[xarray.Dataset] = None
    applied_params_history: Optional[xarray.Dataset] = None
    state_history: Optional[xarray.Dataset] = None


class FilterSmoothing:
    """The hybrid estimator: an ESMDA instance driving a filter instance.

    Composition, not inheritance: both halves are the shipped, fully configured
    DA objects (built by the run script, with their own C_D, localization,
    inflation, on-disk knobs and PRNG keys), and this class only sequences
    them. Nothing here re-implements an analysis.

    Args:
        smoother: A parameter-only ESMDA instance —
            :class:`~data_assimilation.smoothing.esmda.ParameterESMDA` or its
            time-varying subclass. The state-bearing variants are rejected: the
            window's posterior state is the filter's job here, and letting
            ESMDA also update the initial condition would make two different
            estimators own the same quantity.
        filter: The sequential filter, in ``mode="state"`` or ``"joint"``.
            ``mode="parameter"`` is rejected — with no state block the filter
            phase would produce no posterior state at all, which is the one
            thing the hybrid asks it for.

    The filter's ``collect_pred_obs`` flag drives the same three histories
    here: each ``filter.run`` REBINDS its own lists (so a caller keeps the
    previous pass's), and the hybrid accumulates them across the per-segment
    calls into same-named attributes, rebound once per :meth:`run` call.
    """

    def __init__(self, smoother: ParameterESMDA, filter: BaseFilter) -> None:
        # ``StateAndTimeVaryingParameterESMDA`` inherits from
        # ``TimeVaryingParameterESMDA``, hence from ``ParameterESMDA``, so the
        # isinstance test below accepts it: the state-bearing branch has to be
        # rejected explicitly, and ``StateAndParameterESMDA`` is its root
        # (``StateESMDA`` and the joint dynamic variant both derive from it).
        if isinstance(smoother, StateAndParameterESMDA):
            raise ValueError(
                f"{type(smoother).__name__} estimates the window's initial "
                "state as well as the parameters, but filter smoothing gives "
                "the state to the sequential filter — the two would fight over "
                "it. Use esmda/smoother=static or =dynamic (ParameterESMDA / "
                "TimeVaryingParameterESMDA)."
            )
        if not isinstance(smoother, ParameterESMDA):
            raise ValueError(
                "smoother must be a parameter-only ESMDA instance "
                "(ParameterESMDA or TimeVaryingParameterESMDA), got "
                f"{type(smoother).__name__}."
            )
        if filter.mode not in ("state", "joint"):
            raise ValueError(
                f"The filter's mode={filter.mode!r} has no state block, so the "
                "filter phase would produce no posterior state — which is the "
                "whole reason it replaces ESMDA's posterior forecast. Use "
                "filtering.mode=state or =joint."
            )

        self.smoother = smoother
        # Shadowing the builtin is confined to this constructor's argument name,
        # which is the user-facing spec ("the filter"); nothing in this module
        # calls ``filter()``.
        self.filter = filter

        # Accumulated across the filter phase's calls when the filter records
        # them; rebound per ``run`` (see the class docstring).
        self.pred_obs_history: list[np.ndarray] = []
        self.pred_obs_post_history: list[np.ndarray] = []
        self.pred_obs_frames_history: list[Optional[xarray.DataArray]] = []

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_observations(observations: Any) -> list[xarray.DataArray]:
        """One labelled ``("time", "obs")`` DataArray per filter cycle.

        Stricter than either half alone, and deliberately so. The filter would
        accept plain ``(N_obs,)`` frame vectors and the smoother a
        pre-flattened array, but the hybrid needs the time COORDINATES twice
        over: the ESMDA phase aggregates the concatenated window by them
        (``AggregateObservations`` bins on the time coordinate) and the segment
        geometry is read off them. A plain array cannot supply either, and
        silently guessing a uniform cadence is exactly the kind of implicit
        geometry this design removes.
        """
        if observations is None:
            raise ValueError("observations must be provided.")
        if isinstance(observations, xarray.DataArray):
            raise ValueError(
                "observations is a single DataArray; filter smoothing consumes "
                'one batch PER FILTER CYCLE. Pass a list of ("time", "obs") '
                "DataArrays — the hybrid concatenates them itself for the "
                "ESMDA phase."
            )
        if not isinstance(observations, (list, tuple)):
            raise ValueError(
                "observations must be a list/tuple of per-cycle "
                '("time", "obs") DataArrays, got '
                f"{type(observations).__name__}. Plain arrays are rejected: "
                "the ESMDA phase aggregates by the time coordinate and the "
                "segment geometry is derived from it."
            )
        if not observations:
            raise ValueError("observations must contain at least one filter cycle.")
        for k, batch in enumerate(observations):
            if not isinstance(batch, xarray.DataArray):
                raise ValueError(
                    f"observations[{k}] is a {type(batch).__name__}; filter "
                    'smoothing needs labelled ("time", "obs") DataArrays with '
                    "frame times in seconds on the window clock."
                )
            if "time" not in batch.dims:
                raise ValueError(
                    f"observations[{k}] has dims {tuple(batch.dims)}; a "
                    '("time", "obs") batch is required.'
                )
        return list(observations)

    # ------------------------------------------------------------------
    # On-disk staging (dynamic path)
    # ------------------------------------------------------------------

    def _staging_dir(self) -> Optional[pathlib.Path]:
        """Where the per-segment ``filter.run`` calls write, or ``None``.

        Every single-cycle ``filter.run`` numbers its one cycle 0, and
        ``BaseFilter._set_cycle_results_dir`` EMPTIES the directory it is
        pointed at — so without staging each segment would write, and first
        delete, the same ``cycle_0/`` under the filter's results root. The
        segments are renumbered onto the window's global cycle index as they
        finish (:meth:`_collect_segment_dir`), which is the layout the
        downstream ``forecast`` state source expects. Same pattern as
        ``scripts/filtering/run_filtering.py``'s per-window staging.

        ``None`` in memory mode: nothing is written and nothing to stage.
        """
        if not self.filter.forward_model.save_on_disk:
            return None
        return pathlib.Path(self.filter.base_results_dir) / "_segment_staging"

    def _collect_segment_dir(
        self,
        segment: int,
        num_segments: int,
        root: pathlib.Path,
        staging: pathlib.Path,
    ) -> None:
        """Move one finished segment's ``cycle_0/`` onto the global index.

        Pruning is applied HERE rather than by the filter: with one cycle per
        ``run`` call the filter's own ``_prune_cycle_results_dir`` always sees
        ``cycle == num_cycles - 1`` and keeps the directory (it never prunes
        the final cycle). So the hybrid re-applies the filter's own semantics
        at the window level — the last segment is always kept, ``cycle_0``
        while ``keep_first_disk_cycle``, everything else dropped when
        ``prune_disk_cycles``.
        """
        source = staging / "cycle_0"
        if not source.is_dir():
            return
        target = root / f"cycle_{segment}"
        # A stale directory from an earlier run into the same output dir would
        # make ``rename`` fail (POSIX refuses to replace a non-empty dir).
        shutil.rmtree(target, ignore_errors=True)
        source.rename(target)

        if not self.filter.prune_disk_cycles:
            return
        if segment == num_segments - 1:
            return
        if segment == 0 and self.filter.keep_first_disk_cycle:
            return
        shutil.rmtree(target, ignore_errors=True)

    def _accumulate_pred_obs(self) -> None:
        """Append the filter's just-rebound pred-obs lists onto the hybrid's.

        Called after every ``filter.run``: the filter rebinds all three lists at
        entry, so what it holds now is that call's cycles alone and extending
        keeps the hybrid's lists one-entry-per-global-cycle.
        """
        if not self.filter.collect_pred_obs:
            return
        self.pred_obs_history.extend(self.filter.pred_obs_history)
        self.pred_obs_post_history.extend(self.filter.pred_obs_post_history)
        self.pred_obs_frames_history.extend(self.filter.pred_obs_frames_history)

    # ------------------------------------------------------------------
    # The window
    # ------------------------------------------------------------------

    def run(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        observations: Optional[Sequence[xarray.DataArray]] = None,
        *,
        return_history: bool = False,
    ) -> FilterSmoothingResult:
        """Assimilate one window: the MDA loop, then the filter over its cycles.

        Args:
            state: The window's initial state ensemble — the ESMDA phase's
                pinned initial condition (parameter-only variants never move
                it) and the first filter cycle's warm start. ``None`` is a
                legal cold start.
            params: The prior parameter ensemble: static ``(ensemble,)``
                variables, or a knot trajectory with ``time`` (seconds on the
                window clock) for the dynamic smoother.
            observations: One labelled ``("time", "obs")`` DataArray per filter
                cycle, with frame times in seconds on the WINDOW clock,
                strictly increasing across batches. They are used twice: the
                ESMDA phase assimilates their concatenation (aggregating by the
                time coordinate, if it is configured to), and the filter phase
                assimilates them raw, batch by batch.
            return_history: Collect the histories described in
                :class:`FilterSmoothingResult`.

        Returns:
            A :class:`FilterSmoothingResult`.
        """
        if params is None:
            # Every mode needs them: the smoother estimates them and the filter
            # forecasts with them. Caught here with the hybrid's own message
            # rather than deep inside the smoother's flattening.
            raise ValueError(
                "params must be provided: the ESMDA phase estimates the "
                "parameter ensemble and the filter phase forecasts with it."
            )
        batches = self._validate_observations(observations)

        # --- ESMDA phase ------------------------------------------------
        # ``join="override"``: the batches share the ``obs`` axis by
        # construction (one observation operator), and the default outer join
        # would silently union any axis that differs in the last bits and
        # NaN-pad the observations. Same convention as the neighbours.
        window_obs = xarray.concat(batches, dim="time", join="override")
        esmda_result = self.smoother(
            state=state,
            params=params,
            observations=window_obs,
            return_params_history=return_history,
            # The filter phase produces this window's posterior state, so the
            # smoother's posterior forward pass would be a wasted ensemble
            # forecast — and a misleading one, since its state is not what the
            # window returns. See ``_BaseESMDA._analysis``.
            final_forecast=False,
        )
        # ``final_forecast=False`` returns the params alone in both save modes,
        # so this is never the (params, state) tuple.
        assert isinstance(esmda_result, xarray.Dataset)
        esmda_params_history = esmda_result if return_history else None
        theta = (
            esmda_result.isel(esmda_step=-1, drop=True)
            if return_history
            else esmda_result
        )

        # --- Filter phase -----------------------------------------------
        self.pred_obs_history = []
        self.pred_obs_post_history = []
        self.pred_obs_frames_history = []

        dynamic = any("time" in theta[name].dims for name in theta.data_vars)
        if not dynamic:
            return self._run_static(
                theta, esmda_params_history, state, batches, return_history
            )
        return self._run_dynamic(
            theta, esmda_params_history, state, batches, return_history
        )

    def _run_static(
        self,
        theta: xarray.Dataset,
        esmda_params_history: Optional[xarray.Dataset],
        state: Optional[xarray.Dataset],
        batches: list[xarray.DataArray],
        return_history: bool,
    ) -> FilterSmoothingResult:
        """One ``filter.run`` over the whole window: the run_filtering path.

        Static parameters need no segment geometry — every cycle forecasts with
        the same Dataset, which is precisely what ``BaseFilter.run`` does with
        the ``params`` it is given. So there is nothing hybrid-specific left:
        the filter cycles, numbers, prunes and (in joint mode) updates exactly
        as a standalone filter would, and the phase is bitwise identical to
        running that filter directly on ``theta``.
        """
        result = self.filter.run(
            state=state,
            params=theta,
            observations=batches,
            return_history=return_history,
        )
        self._accumulate_pred_obs()

        joint = self.filter.mode == "joint"
        params_history: Optional[xarray.Dataset] = None
        if return_history and joint and result.params_history is not None:
            # Drop the filter's entry 0 (the prior) so this history indexes the
            # same cycles as ``diagnostics`` and ``state_history`` — see
            # FilterSmoothingResult.
            params_history = result.params_history.isel(cycle=slice(1, None))

        return FilterSmoothingResult(
            esmda_params=theta,
            state=result.state,
            params=result.params if joint else None,
            diagnostics=result.diagnostics,
            esmda_params_history=esmda_params_history,
            params_history=params_history,
            applied_params_history=None,
            state_history=result.state_history if return_history else None,
        )

    def _run_dynamic(
        self,
        theta: xarray.Dataset,
        esmda_params_history: Optional[xarray.Dataset],
        state: Optional[xarray.Dataset],
        batches: list[xarray.DataArray],
        return_history: bool,
    ) -> FilterSmoothingResult:
        """One ``filter.run`` per segment, with the trajectory placed on it.

        Chaining L single-segment calls is numerically identical to one L-cycle
        call, which is what makes this loop a re-arrangement of the filter
        rather than a different filter. Verified against ``BaseFilter.run``:
        it never resets ``self.rng_key`` (the key is only split, inside
        ``_analysis_cycle`` and the parameter evolution, and the split state
        persists on the instance across calls), and the analyzed
        ``result.state`` it returns is exactly the warm start the next cycle
        would have received — carried here as ``carry_state``. The only
        per-call reset is of the pred-obs histories, which is why they are
        accumulated after each call.
        """
        bounds = segment_bounds(batches)
        num_segments = len(bounds)
        joint = self.filter.mode == "joint"

        staging = self._staging_dir()
        root = pathlib.Path(self.filter.base_results_dir) if staging else None

        diagnostics: list[CycleDiagnostics] = []
        applied_history: list[xarray.Dataset] = []
        params_history: list[xarray.Dataset] = []
        state_history: list[xarray.Dataset] = []
        carry_state = state
        final_params: Optional[xarray.Dataset] = None
        # The joint correction, ``None`` until the first analysis produces one
        # (identically zero, but writing it as an explicit zero Dataset would
        # only add an arithmetic op to the first segment).
        correction: Optional[xarray.Dataset] = None

        if staging is not None:
            self.filter.base_results_dir = staging
        try:
            for segment, (t_start, t_end) in enumerate(bounds):
                midpoint = 0.5 * (t_start + t_end)
                schedule: Optional[xarray.Dataset] = None
                if joint:
                    # The segment's constant approximation of the schedule; the
                    # correction is what the filter has learned on top of it.
                    # The re-attach keeps `attrs` on the applied params: xarray
                    # binary ops drop them, while the two helpers above (and the
                    # state path) preserve them.
                    schedule = trajectory_values_at(theta, midpoint)
                    seg_params = (
                        schedule
                        if correction is None
                        else (schedule + correction).assign_attrs(schedule.attrs)
                    )
                else:
                    seg_params = params_for_segment(theta, t_start, t_end)

                result = self.filter.run(
                    state=carry_state,
                    params=seg_params,
                    observations=[batches[segment]],
                    # One cycle per call: the returned state/params ARE that
                    # cycle's analysed values, so the filter's own histories
                    # would only repeat them (with the prior prepended).
                    return_history=False,
                )
                self._accumulate_pred_obs()

                if joint:
                    assert result.params is not None and schedule is not None
                    correction = result.params - schedule
                    final_params = result.params

                carry_state = result.state
                # Renumber onto the window's global cycle index: every
                # single-cycle call numbered its own cycle 0.
                for diag in result.diagnostics:
                    diag.cycle = len(diagnostics)
                    diagnostics.append(diag)

                if return_history:
                    assert result.state is not None
                    state_history.append(result.state)
                    if joint:
                        assert result.params is not None and schedule is not None
                        applied_history.append(seg_params)
                        params_history.append(result.params)

                if staging is not None and root is not None:
                    self._collect_segment_dir(segment, num_segments, root, staging)

                logger.info(
                    "Filter-smoothing segment %d/%d assimilated ([%.4g, %.4g] s)",
                    segment,
                    num_segments,
                    t_start,
                    t_end,
                )
        finally:
            if staging is not None and root is not None:
                self.filter.base_results_dir = root
                shutil.rmtree(staging, ignore_errors=True)

        return FilterSmoothingResult(
            esmda_params=theta,
            state=carry_state,
            params=final_params if joint else None,
            diagnostics=diagnostics,
            esmda_params_history=esmda_params_history,
            params_history=(
                xarray.concat(params_history, dim="cycle", join="override")
                if params_history
                else None
            ),
            applied_params_history=(
                xarray.concat(applied_history, dim="cycle", join="override")
                if applied_history
                else None
            ),
            state_history=(
                xarray.concat(state_history, dim="cycle", join="override")
                if state_history
                else None
            ),
        )
