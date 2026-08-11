"""Moving (sliding) window orchestration of the filter-smoothing ESMDA.

Paper §8: instead of assimilating one window and stopping, slide it —
``[t_{n-L} ... t_n]`` becomes ``[t_{n-L+1} ... t_{n+1}]`` — so a long horizon is
processed as a sequence of overlapping windows. Three things are carried from
one window to the next, and they are the whole method:

1. **the state**, filtered forward by ``s`` cycles (the shift) — window
   ``w+1`` starts from window ``w``'s final-consistency-pass analysis *after*
   cycle ``s``, never from a re-run;
2. **the overlapping parameter knots**, which are already a posterior and
   become the next window's prior directly (no re-sampling: that is what makes
   this a smoother over the horizon rather than ``num_windows`` independent
   inversions);
3. **the leading edge**: as many new knots as left the window, appended by the
   parameter evolution model, per member, from the last carried knot.

All of that arithmetic is in SECONDS, not knot indices: the knot grid is
independent of the cycle grid (see
:mod:`~data_assimilation.filter_smoothing.base`), so the window advances by
``shift_seconds = s * cycle_length`` and the knots that fall before it are the
ones that leave. Each window's trajectory lives on its OWN clock starting at
``t = 0`` — the carried knots are re-based by ``-shift_seconds`` — because the
inner filter reads segment ``k`` at ``[k*cycle_length, (k+1)*cycle_length]``;
the finalized knots are put back on the horizon's clock when the result is
assembled. With one knot per cycle every step of this reduces to the positional
slicing it replaces.

Knots that leave the window are **finalized** (the paper's "fixed after leaving
the window") and recorded; concatenated in order they are the full-horizon
smoothed trajectory ensemble.

This is a pure orchestrator: it uses only
:class:`~data_assimilation.filter_smoothing.base.FilterSmoothingESMDA`'s public
API, and neither it nor ``filtering/base.py`` is modified. With
``num_windows=1`` the orchestrator is a transparent wrapper — the same PRNG
stream, the same history collection, the same Datasets by identity — so the
single-window path is bit-identical to calling ``run()`` directly.

Why ``ParameterEvolution`` and not ``ParameterTimeSeries.extrapolate`` for the
leading edge: ``extrapolate`` rebuilds a *whole* window prior from a posterior
(the non-overlapping regime ``run_esmda`` uses) and for e.g. AR(2) synthesizes
a fresh trajectory, discarding the carried members' posterior information.
``evolve`` is exactly the paper's per-member append, and it keeps
``data_assimilation`` free of ``pyurbanair`` imports.
"""

import logging
import time
from dataclasses import dataclass, replace
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import xarray
from data_assimilation.filter_smoothing.base import (
    FilterSmoothingESMDA,
    FilterSmoothingResult,
    IterationDiagnostics,
    knot_times,
)
from data_assimilation.filtering.parameter_evolution import (
    IdentityEvolution,
    ParameterEvolution,
)

logger = logging.getLogger(__name__)

# Folded into the smoother's key to derive the evolution stream. Folding
# (rather than splitting) reads the smoother's key WITHOUT advancing it, so the
# windows' own draws are untouched and window 0 is bit-identical to a plain
# ``run()``; the constant only has to be stable, its value is arbitrary.
_EVOLUTION_STREAM_TAG = 1

# Relative slack of the knot-grid arithmetic (uniform spacing, the shift's
# alignment with it, which knots leave the window). The knot axis is float32 on
# the way in, so an exact comparison would trip on representation alone.
_TIME_RTOL = 1e-4


@dataclass
class WindowDiagnostics:
    """What one window of the moving loop did.

    ``first_cycle``/``last_cycle`` are INCLUSIVE indices into the full horizon
    (window ``w`` consumes observation rows ``[w*s, w*s + L)``), so the spans of
    consecutive windows overlap by ``L - s`` cycles.
    ``iteration_diagnostics`` is that window's own outer-ESMDA record — see
    :class:`~data_assimilation.filter_smoothing.base.IterationDiagnostics`;
    reading ``obs_rmse`` across windows is how one checks that later windows
    are not steadily harder to fit than the first.
    """

    window: int
    first_cycle: int
    last_cycle: int
    iteration_diagnostics: list[IterationDiagnostics]
    window_time: Optional[float] = None


@dataclass
class MovingWindowResult:
    """Return value of :func:`run_moving_window`.

    ``params`` is the assembled full-horizon trajectory ensemble: each window's
    finalized knots in order, on the horizon's clock, concatenated along
    ``time``. Its knot count is ``(num_windows - 1) * (shift_seconds/dt) +
    n_knots`` — the horizon at the trajectory's own resolution, plus whatever
    the last window's trajectory carries beyond it (the trailing knot a
    windowed prior sampler emits rides along to the end). With one knot per
    cycle that is ``(num_windows - 1) * window_shift + n_knots``.

    ``state`` is the last window's filtered end state (Eq. 16 at the end of the
    horizon), ``window_results`` each window's own
    :class:`~data_assimilation.filter_smoothing.base.FilterSmoothingResult`
    with the heavy per-cycle/per-iteration histories stripped unless
    ``return_history`` was set, and ``window_diagnostics`` one
    :class:`WindowDiagnostics` per window.
    """

    params: xarray.Dataset
    state: Optional[xarray.Dataset]
    window_results: list[FilterSmoothingResult]
    window_diagnostics: list[WindowDiagnostics]


def run_moving_window(
    smoother: FilterSmoothingESMDA,
    *,
    state: Optional[xarray.Dataset] = None,
    params: Optional[xarray.Dataset] = None,
    observations: Optional[jnp.ndarray] = None,
    num_windows: int = 1,
    window_shift: int = 1,
    window_length: Optional[int] = None,
    evolution: Optional[ParameterEvolution] = None,
    rng_key: Optional[jax.Array] = None,
    return_history: bool = False,
) -> MovingWindowResult:
    """Assimilate a horizon of ``T`` cycles as ``num_windows`` sliding windows.

    Args:
        smoother: The :class:`~data_assimilation.filter_smoothing.base.\
FilterSmoothingESMDA` to run once per window. It is *not* reset between
            windows: its PRNG chain advances, so every window draws fresh
            perturbed observations (common random numbers are a *within*-window
            device).
        state: The initial state ensemble of window 0. ``None`` is a legal cold
            start; later windows always start from the carried state.
        params: The trajectory prior of window 0 — a Dataset with ``time``
            (knots, in seconds, on a UNIFORM grid starting at the window's
            origin) and ``ensemble`` dims, sampled over the FIRST WINDOW only,
            not over the horizon. Later windows' priors are built from it by
            the carry (the knots past ``shift_seconds``, re-based, plus as many
            evolved knots as left), so the knot count stays fixed.
        observations: Array of shape ``(T, N_d)`` over the full horizon, one
            batch per cycle. Window ``w`` consumes rows ``[w*s, w*s + L)``.
        num_windows: Number of windows. ``1`` is the plain single-window run.
        window_shift: ``s``, the number of cycles the window advances between
            runs, in ``[1, L]``. ``s = L`` gives non-overlapping windows.
        window_length: ``L``, cycles per window (the script's
            ``filter_smoothing.num_cycles``). ``None`` (default) SOLVES the
            horizon relation ``T = L + (num_windows - 1)*s`` for ``L``; passing
            it instead VALIDATES that relation, which is the louder failure
            mode — prefer passing it when the caller knows ``L``.
        evolution: Parameter evolution model appending the leading-edge knots.
            Defaults to
            :class:`~data_assimilation.filtering.parameter_evolution.\
IdentityEvolution` (persistence: the new knot equals the last carried one, and
            the trajectory's spread is maintained by the outer update alone).
        rng_key: PRNG key for the evolution draws. ``None`` (default) derives
            one from ``smoother.rng_key`` without disturbing it, and only when
            more than one window is actually run.
        return_history: Keep the per-window histories (per-iteration
            trajectories and the final pass's per-cycle Datasets) in
            ``window_results``. Off by default because they are ``L`` full
            state ensembles per window.

    Returns:
        A :class:`MovingWindowResult`.

    Raises:
        ValueError: On any inconsistency between the horizon, the window
            geometry and the prior's knot layout — including a knot grid the
            window's advance is not a whole number of knots of — all checked up
            front, before the first (expensive) window runs.
    """
    if params is None:
        raise ValueError(
            "params must be provided: the parameter trajectory ensemble is "
            "this method's control vector."
        )
    if observations is None:
        raise ValueError("observations must be provided.")
    obs_batches = jnp.asarray(observations)
    if obs_batches.ndim != 2:
        raise ValueError(
            "observations must have shape (num_cycles, N_d) — one batch per "
            f"cycle of the FULL horizon — got shape {obs_batches.shape}."
        )
    if num_windows < 1:
        raise ValueError(f"num_windows must be >= 1, got {num_windows}.")
    if window_shift < 1:
        raise ValueError(f"window_shift must be >= 1, got {window_shift}.")
    if "time" not in params.dims:
        raise ValueError(
            "params has no 'time' dimension: the moving window carries and "
            "extends a parameter TRAJECTORY. Sample a dynamic (time-varying) "
            "prior over the first window."
        )

    horizon = int(obs_batches.shape[0])
    num_cycles = _window_length(horizon, num_windows, window_shift, window_length)
    if window_shift > num_cycles:
        raise ValueError(
            f"window_shift={window_shift} exceeds the window length "
            f"L={num_cycles}: the window would skip cycles. Use "
            "window_shift=L for non-overlapping windows."
        )
    if int(params.sizes["time"]) < 1:
        raise ValueError("The trajectory prior has no knots to slide.")

    # The window's advance in SECONDS, and the knot grid it has to stay in phase
    # with. Both are needed only when the window actually moves, so a
    # single-window run never touches them — and never gains a failure mode.
    shift_seconds = 0.0
    knot_spacing = 0.0
    if num_windows > 1:
        cycle_length = getattr(smoother, "cycle_length", None)
        if cycle_length is None:
            raise ValueError(
                "The smoother has no cycle_length, so the window's advance "
                "cannot be expressed in seconds and the knots that leave it "
                "cannot be identified. Pass cycle_length= to "
                "FilterSmoothingESMDA."
            )
        shift_seconds = window_shift * float(cycle_length)
        knot_spacing = _knot_spacing(params)
        _check_alignment(shift_seconds, knot_spacing, window_shift, float(cycle_length))

    evolution = IdentityEvolution() if evolution is None else evolution
    # Only touch the PRNG when knots will actually be appended: with a single
    # window nothing is derived, so the smoother's key chain is exactly the one
    # a bare ``run()`` would see (the degenerate case is bit-identical).
    evolution_key: Optional[jax.Array] = None
    if num_windows > 1:
        evolution_key = (
            rng_key
            if rng_key is not None
            else jax.random.fold_in(smoother.rng_key, _EVOLUTION_STREAM_TAG)
        )

    window_state = state
    window_params = params
    finalized: list[xarray.Dataset] = []
    window_results: list[FilterSmoothingResult] = []
    window_diagnostics: list[WindowDiagnostics] = []

    for window in range(num_windows):
        first_cycle = window * window_shift
        is_last = window == num_windows - 1
        # The carried state is a MID-window analysis, which only the per-cycle
        # history holds; when the window shifts by its full length the carried
        # state is the pass's end state and no history is needed. That matters:
        # a history is L full state ensembles.
        needs_state_history = not is_last and window_shift < num_cycles

        started = time.perf_counter()
        result = smoother.run(
            state=window_state,
            params=window_params,
            observations=obs_batches[first_cycle : first_cycle + num_cycles],
            return_history=return_history or needs_state_history,
        )
        elapsed = time.perf_counter() - started

        # Every window runs on its own clock (knot times start at 0), so what it
        # finalizes has to be put back on the horizon's clock to be assembled.
        horizon_offset = window * shift_seconds

        if is_last:
            # Nothing follows, so the whole trajectory is finalized here —
            # including any trailing knot beyond the horizon.
            finalized.append(_on_horizon_clock(result.params, horizon_offset))
        else:
            # Extract the carry BEFORE the histories are dropped.
            window_state = _carried_state(result, window_shift, num_cycles)
            leaving, window_params, evolution_key = _advance_trajectory(
                result.params, shift_seconds, knot_spacing, evolution, evolution_key
            )
            finalized.append(_on_horizon_clock(leaving, horizon_offset))

        window_results.append(
            _strip_histories(result)
            if (needs_state_history and not return_history)
            else result
        )
        window_diagnostics.append(
            WindowDiagnostics(
                window=window,
                first_cycle=first_cycle,
                last_cycle=first_cycle + num_cycles - 1,
                iteration_diagnostics=result.iteration_diagnostics,
                window_time=elapsed,
            )
        )
        logger.info(
            "Moving window %d/%d (cycles %d-%d) completed in %.1f s",
            window + 1,
            num_windows,
            first_cycle,
            first_cycle + num_cycles - 1,
            elapsed,
        )

    return MovingWindowResult(
        # Concatenating a single piece would still rebuild the Dataset; the
        # single-window result must come back untouched.
        params=(
            finalized[0]
            if len(finalized) == 1
            # data_vars="minimal" + compat="override": only the trajectory
            # variables are concatenated along time, and the static ones are
            # taken from the first window as they are. The default
            # (data_vars="all") would instead give every static parameter a
            # spurious ``time`` dimension.
            else xarray.concat(
                finalized,
                dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="override",
                join="override",
            )
        ),
        state=window_results[-1].state,
        window_results=window_results,
        window_diagnostics=window_diagnostics,
    )


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------


def _window_length(
    horizon: int,
    num_windows: int,
    window_shift: int,
    window_length: Optional[int],
) -> int:
    """``L`` from the horizon relation ``T = L + (num_windows - 1)*s``.

    Solved for ``L`` when the caller did not supply it (the orchestrator has no
    other source: ``FilterSmoothingESMDA`` reads its cycle count from the
    observation batches it is handed), validated when it did.
    """
    span = (num_windows - 1) * window_shift
    if window_length is None:
        derived = horizon - span
        if derived < 1:
            raise ValueError(
                f"The horizon T={horizon} is too short for "
                f"num_windows={num_windows} windows shifted by "
                f"{window_shift}: it leaves L={derived} cycles per window. "
                "Simulate the truth over T = L + (num_windows-1)*window_shift "
                "cycles."
            )
        return derived
    if window_length < 1:
        raise ValueError(f"window_length must be >= 1, got {window_length}.")
    if horizon != window_length + span:
        raise ValueError(
            f"Horizon mismatch: observations cover T={horizon} cycles but "
            f"num_windows={num_windows} windows of L={window_length} cycles "
            f"shifted by {window_shift} cover "
            f"{window_length + span} = L + (num_windows-1)*window_shift."
        )
    return window_length


def _knot_spacing(params: xarray.Dataset) -> float:
    """The trajectory's (uniform) knot spacing ``dt``, in seconds.

    The sliding arithmetic needs ONE spacing: the appended leading-edge knots
    continue the grid at it, and the window's advance must be a multiple of it.
    A non-uniform grid has no such number — and cannot stay in phase with the
    windows either — so it is rejected here rather than silently re-gridded.
    """
    times = knot_times(params)
    if times.size < 2:
        raise ValueError(
            "The trajectory has a single knot, so its knot spacing — the step "
            "of the appended leading-edge knots — cannot be inferred. Sample "
            "the prior with at least two knots to move the window."
        )
    gaps = np.diff(times)
    spacing = float(np.mean(gaps))
    if not np.allclose(gaps, spacing, rtol=_TIME_RTOL, atol=_TIME_RTOL * spacing):
        raise ValueError(
            "The trajectory's knots are not evenly spaced: gaps "
            f"{np.unique(np.round(gaps, 6)).tolist()}. The moving window slides "
            "the grid by a fixed number of seconds and appends knots at its own "
            "spacing, both of which need a uniform grid (a sampler emits an "
            "uneven final interval when the window span is not a multiple of "
            "time.seconds_per_knot)."
        )
    return spacing


def _check_alignment(
    shift_seconds: float,
    knot_spacing: float,
    window_shift: int,
    cycle_length: float,
) -> None:
    """The window's advance must be a whole number of knots.

    Otherwise the knot grid drifts out of phase with the windows: window ``w``'s
    knots would sit at ``w*shift_seconds mod dt`` past its own segment
    boundaries, so "which knots leave the window" would change from window to
    window and the finalized pieces would no longer tile the horizon. Knots
    finer than a cycle always satisfy this; a knot grid COARSER than a cycle
    constrains ``window_shift`` instead (it must advance whole knots).
    """
    knots_per_shift = shift_seconds / knot_spacing
    if abs(knots_per_shift - round(knots_per_shift)) > _TIME_RTOL * max(
        round(knots_per_shift), 1
    ):
        raise ValueError(
            f"The window advances {shift_seconds:g}s (window_shift="
            f"{window_shift} cycles of {cycle_length:g}s) but the trajectory's "
            f"knots are {knot_spacing:g}s apart, which is "
            f"{knots_per_shift:.6g} knots per advance — not a whole number. The "
            "knot grid would drift out of phase with the windows: a different "
            "set of knots would leave the window each time and the finalized "
            "pieces would not tile the horizon. Choose window_shift so that "
            "window_shift * time.simulation_time is a multiple of "
            "time.seconds_per_knot (a knot spacing finer than one cycle always "
            "is)."
        )


def _on_horizon_clock(params: xarray.Dataset, offset: float) -> xarray.Dataset:
    """Put a window's knot times back on the horizon's clock.

    Window ``w`` runs on a clock re-based ``w`` times by the shift, so its
    finalized knots are ``offset = w * shift_seconds`` past the horizon's
    origin. Window 0 is returned BY IDENTITY (the single-window path must be
    bit-identical to a plain ``run()``).
    """
    if not offset:
        return params
    return params.assign_coords(
        time=np.asarray(params.coords["time"].values, dtype=float) + offset
    )


# ----------------------------------------------------------------------
# The carry
# ----------------------------------------------------------------------


def _carried_state(
    result: FilterSmoothingResult,
    window_shift: int,
    num_cycles: int,
) -> Optional[xarray.Dataset]:
    """The next window's initial ensemble: the analysis after cycle ``s``.

    Taken from the FINAL CONSISTENCY PASS (the one run with the window's
    posterior trajectory), never from an iteration's pass — the states an
    iteration produced were forecast with a trajectory that has since been
    updated.
    """
    if window_shift == num_cycles:
        # The last entry of the history is the pass's end state, which is what
        # ``result.state`` already is.
        return result.state
    history = result.final_pass.state_history
    if history is None:
        raise RuntimeError(
            "The final pass collected no state history, so the analysis after "
            f"cycle {window_shift} cannot be carried to the next window. This "
            "means return_history was overridden on the smoother."
        )
    return history.isel(cycle=window_shift - 1, drop=True)


def _advance_trajectory(
    posterior: xarray.Dataset,
    shift_seconds: float,
    knot_spacing: float,
    evolution: ParameterEvolution,
    rng_key: Optional[jax.Array],
) -> tuple[xarray.Dataset, xarray.Dataset, Optional[jax.Array]]:
    """Slide the trajectory by ``shift_seconds``: finalize, carry, extend.

    Returns ``(finalized, next_prior, rng_key)``. The knots before the shift
    LEAVE the window and are finalized on this window's own clock (the caller
    puts them back on the horizon's); the rest are carried verbatim — they are
    already a posterior and are the next window's prior — with their times
    re-based by ``-shift_seconds`` so the next window's clock again starts at
    its first segment. As many new knots as left are appended by chaining
    ``evolution.evolve`` from the last carried knot, one fresh subkey each, at
    times continuing the grid (``t_last + i*dt``); the knot count is therefore
    invariant across windows. With one knot per cycle the split is exactly the
    positional ``[:s]`` / ``[s:]`` it generalizes.

    Member pairing is positional throughout: the ensemble index is never
    reordered, so member ``i`` of the carried state and of the carried
    trajectory remain the same member.

    Static (no-``time``) variables are carried through untouched and are never
    handed to the evolution model — they are not part of the trajectory, and a
    random-walk model would otherwise walk them.
    """
    times = knot_times(posterior)
    rebased = times - shift_seconds
    # The knots the window slid past. Strictly before the shift: a knot landing
    # exactly on the new window's origin is still inside it (it is that
    # window's first segment boundary), which is what keeps the finalized
    # pieces a tiling rather than an overlap.
    num_finalized = int(
        np.count_nonzero(rebased < -_TIME_RTOL * max(shift_seconds, 1.0))
    )
    if num_finalized < 1:
        raise ValueError(
            f"The window advanced {shift_seconds:g}s but no knot left it (the "
            f"first knot is at t={times[0]:g}s). The trajectory would grow "
            "instead of sliding; check that the prior's knot grid starts at "
            "the window's own origin."
        )

    finalized = posterior.isel(time=slice(0, num_finalized))
    carried = posterior.isel(time=slice(num_finalized, None)).assign_coords(
        time=rebased[num_finalized:]
    )
    trajectory_vars = [
        str(name) for name in posterior.data_vars if "time" in posterior[name].dims
    ]
    # Continue the grid from the last INCOMING knot on the re-based clock, which
    # is the last carried one whenever any knot is carried at all (and is
    # negative when the shift consumed the whole trajectory, so the appended
    # knots still open at the new window's origin).
    new_times = rebased[-1] + np.arange(1, num_finalized + 1) * knot_spacing

    # ``drop=True``: the scalar ``time`` coordinate must not survive into the
    # evolved knots (the evolution models rebuild the Dataset from these coords).
    knot = posterior[trajectory_vars].isel(time=-1, drop=True)
    knots: list[xarray.Dataset] = []
    for _ in range(num_finalized):
        if rng_key is None:
            raise RuntimeError(
                "No PRNG key was derived for the evolution model although "
                "knots must be appended. This is an internal inconsistency."
            )
        rng_key, subkey = jax.random.split(rng_key)
        knot = evolution.evolve(knot, subkey)
        knots.append(knot)

    data_vars: dict[str, xarray.DataArray] = {}
    for name in posterior.data_vars:
        variable = posterior[name]
        if name not in trajectory_vars:
            data_vars[str(name)] = carried[name]
            continue
        appended = xarray.DataArray(
            np.stack([np.asarray(entry[name].values) for entry in knots], axis=0),
            dims=("time",) + tuple(str(dim) for dim in knots[0][name].dims),
        ).assign_coords(time=new_times)
        # ``join="override"`` keeps the carried block's coordinates (ensemble)
        # for the appended knots, which carry none of their own.
        data_vars[str(name)] = xarray.concat(
            [carried[name], appended], dim="time", join="override"
        ).transpose(*variable.dims)

    return finalized, xarray.Dataset(data_vars, attrs=posterior.attrs), rng_key


def _strip_histories(result: FilterSmoothingResult) -> FilterSmoothingResult:
    """Drop the histories collected only to extract the mid-window state.

    Copies rather than mutates: the returned record shares every other field
    (params, state, diagnostics) with the original *by identity*.
    """
    return replace(
        result,
        params_history=None,
        final_pass=replace(result.final_pass, params_history=None, state_history=None),
    )
