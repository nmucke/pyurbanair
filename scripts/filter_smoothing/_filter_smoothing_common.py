"""Shared post-processing helpers for the filter-smoothing pipeline.

``scripts/filter_smoothing/run_filter_smoothing.py`` runs the method and saves
the raw artifacts; ``compute_filter_smoothing_metrics.py`` turns those into
``run_summary.yaml`` and ``make_filter_smoothing_figures.py`` draws the
figures. This module is the filter-smoothing analogue of
``scripts/filtering/_filtering_common.py`` -- and it is deliberately a thin one,
because the two runs' per-cycle artifacts are the same artifacts: the inner
state filter IS the EnKF, so ``state_history.nc``, ``cycle_diagnostics.yaml``
and the optional ``_ensemble_states/cycle_*/`` tree are written by the same
code. Everything the filtering pipeline built on them is re-exported below
rather than copied.

Two things are genuinely new, and they are what lives here:

**The moving window.** With ``filter_smoothing.num_windows = W > 1`` the run
assimilates ``T = L + (W-1)*s`` cycles, but every window's inner filter writes
its cycles into the SAME ``_ensemble_states/cycle_0 … cycle_{L-1}`` directories
and the final pass overwrites ``state_history.nc`` / ``cycle_diagnostics.yaml``
whole. So the per-cycle artifacts on disk are the LAST window's ``L`` cycles,
covering global cycles ``[(W-1)*s, T)`` -- while ``truth_access.yaml``'s
``num_cycles`` is ``T``, the horizon the truth spans. Handing that ``T`` to the
filtering machinery is wrong in both directions: ``_forecast_cycle_paths``
would look for ``cycle_{L} … cycle_{T-1}``, find nothing and silently fall back
to the analyzed frames, and the truth would be binned over the whole horizon
against ``L`` cycles of ensemble. :func:`window_layout` reads the layout back
off the run's own records and :func:`final_window_truth_access` re-bases the
truth-access view onto the last window, after which every helper below behaves
exactly as it does on a single-window filtering run -- and the global cycle
indices the outputs are labelled with come from :func:`global_cycle_indices`.

**The outer ESMDA loop.** ``iteration_diagnostics.yaml`` (one record per outer
iteration of the last window) and ``window_diagnostics.yaml`` (one record per
window, carrying that window's own iteration records) have no filtering
counterpart at all; :func:`iteration_diagnostics_series` and
:func:`window_iteration_series` read them into the plain arrays the metric and
figure stages consume.

One artifact of the shared layout is *not* reusable, and it is a trap rather
than a gap: ``params_history.nc`` here is the inner pass's per-cycle parameter
record, and the inner pass is state-only, so every one of its ``cycle`` entries
is the same whole trajectory (with its own ``time`` knot dim) repeated.
``scripts.filtering._filtering_common.load_params_history`` renames ``cycle``
-> ``time`` and would collide with that knot dim; it is deliberately NOT
re-exported. The parameter side of this pipeline reads ``posterior_params.nc``
(the smoothed trajectory) and ``params_iterations.nc`` (that trajectory at every
outer iteration) instead.
"""

# mypy: ignore-errors
# Legacy untyped helper module, matching scripts/filtering/_filtering_common.py
# and scripts/esmda/_esmda_common.py: it is type-checked transitively whenever a
# script importing it is committed. Waived wholesale rather than annotated
# piecemeal; drop this when typed.

from __future__ import annotations

import logging
import pathlib
from typing import NamedTuple

import numpy as np
import xarray

# The whole per-cycle evaluation contract, reused unchanged: the filter
# smoothing run dir holds the same per-cycle artifacts the filtering pipeline
# reduces over, so the choice between the on-disk forecasts and the analyzed
# frames, the sensor extraction, the truth pairing and the binning unit are all
# already decided -- once ``ta`` describes ONE window (see
# :func:`final_window_truth_access`).
from scripts.filtering._filtering_common import (  # noqa: F401  (re-exported)
    CycleStates,
    build_sensor_sets,
    cycle_diagnostics_series,
    cycle_sensor_series,
    cycle_state_source,
    end_of_cycle_indices,
    ensemble_cycle_sensor_series,
    ensemble_velocity_mean_std,
    load_analyzed_states,
    load_run_config,
    read_yaml,
    truth_cycle_sensor_series,
    truth_cycle_statistics_series,
    truth_end_of_cycle,
    truth_field_pass,
    write_yaml,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Where the surviving per-cycle artifacts sit on the run's cycle axis
# ---------------------------------------------------------------------------


class WindowLayout(NamedTuple):
    """The moving window's geometry, as the finished run recorded it.

    Attributes:
        num_windows: ``W`` -- fixed-lag windows the run drove. 1 is the single
            fixed window (the default), where every field below degenerates to
            the plain filtering reading.
        window_length: ``L`` -- cycles per window, i.e. the number of
            observation batches one outer ESMDA update assimilates at once (how
            many trajectory KNOTS it touches is a separate number, set by
            ``time.seconds_per_knot``). Also the number of per-cycle artifacts
            that survive on disk, whatever ``W`` is.
        window_shift: ``s`` -- cycles the window slid between windows.
        total_cycles: ``T = L + (W-1)*s`` -- the assimilated horizon, which is
            what ``truth_access.yaml``'s ``num_cycles`` counts.
        first_cycle: Global cycle index of the LAST window's cycle 0, i.e.
            ``(W-1)*s``. The one number the whole mapping turns on: the on-disk
            ``cycle_k`` is the run's cycle ``first_cycle + k``.
        spans: ``(first, last)`` inclusive global cycle span of every window, in
            order. Consecutive spans overlap by ``L - s`` cycles.
        source: Which record the layout was read from, verbatim into
            ``run_summary.yaml``. A run dir that carries no layout at all is
            read as a single window and says so, rather than being assumed.
    """

    num_windows: int
    window_length: int
    window_shift: int
    total_cycles: int
    first_cycle: int
    spans: list[tuple[int, int]]
    source: str


def _spans_from(num_windows: int, window_length: int, window_shift: int):
    """Every window's inclusive global cycle span, from the three scalars."""
    return [
        (w * window_shift, w * window_shift + window_length - 1)
        for w in range(num_windows)
    ]


def window_layout(run_dir: pathlib.Path, ta: dict | None = None) -> WindowLayout:
    """Read back the moving window's geometry (see :class:`WindowLayout`).

    ``run_info.yaml``'s ``configuration`` is the primary record: the runner
    writes ``num_cycles`` (= ``L``), ``num_windows``, ``window_shift``,
    ``total_cycles`` and ``final_pass_first_cycle`` on every run, single-window
    included. ``window_diagnostics.yaml`` -- written only when the window
    actually moved -- carries the orchestrator's OWN per-window spans, so it
    wins where the two disagree: it is what the windows did, not what the config
    asked for.

    A run dir with neither (a hand-made fixture, or an artifact set predating
    the layout keys) is read as ONE window over ``ta``'s cycle count. That is
    the only assumption this function makes and it is named in ``source``, so a
    reader of ``run_summary.yaml`` can tell an assumed layout from a recorded
    one -- the alternative, degrading silently, is exactly what mis-maps the
    on-disk cycles.
    """
    run_dir = pathlib.Path(run_dir)
    ta = ta or {}
    configuration = (read_yaml(run_dir / "run_info.yaml") or {}).get(
        "configuration"
    ) or {}

    fallback_cycles = int(ta.get("num_cycles") or 0)
    length = int(configuration.get("num_cycles") or fallback_cycles or 1)
    num_windows = int(configuration.get("num_windows") or 1)
    shift = int(configuration.get("window_shift") or length)
    total = int(configuration.get("total_cycles") or fallback_cycles or length)
    first = configuration.get("final_pass_first_cycle")
    first = (num_windows - 1) * shift if first is None else int(first)
    source = (
        "run_info.yaml (configuration.num_windows / window_shift / "
        "final_pass_first_cycle)"
    )
    if not configuration:
        source = (
            "assumed a single window: run_info.yaml records no window layout, so "
            "the per-cycle artifacts are read as the whole horizon"
        )

    # The horizon equation the runner itself derives ``total_cycles`` from. A
    # violation means the three scalars cannot all be true, and mapping cycles
    # through a broken layout is worse than not mapping them: fall back to the
    # single-window reading and say so.
    if length + (num_windows - 1) * shift != total:
        logger.warning(
            "Inconsistent window layout in run_info.yaml: num_cycles=%d, "
            "num_windows=%d, window_shift=%d do not add up to total_cycles=%d. "
            "Reading the run as a single window over %d cycle(s)",
            length,
            num_windows,
            shift,
            total,
            total,
        )
        num_windows, length, shift, first = 1, total, total, 0
        source = (
            "single window assumed: run_info.yaml's window layout is internally "
            "inconsistent (num_cycles + (num_windows-1)*window_shift != "
            "total_cycles)"
        )

    spans = _spans_from(num_windows, length, shift)

    rows = read_yaml(run_dir / "window_diagnostics.yaml")
    recorded = [
        (int(row["first_cycle"]), int(row["last_cycle"]))
        for row in (rows or [])
        if isinstance(row, dict) and "first_cycle" in row and "last_cycle" in row
    ]
    if recorded:
        if recorded != spans:
            logger.warning(
                "window_diagnostics.yaml's spans %s disagree with the layout "
                "derived from run_info.yaml %s; using the orchestrator's own "
                "record",
                recorded,
                spans,
            )
        spans = recorded
        num_windows = len(spans)
        first = spans[-1][0]
        length = spans[-1][1] - spans[-1][0] + 1
        total = spans[-1][1] + 1
        shift = spans[1][0] - spans[0][0] if len(spans) > 1 else length
        source = "window_diagnostics.yaml (the orchestrator's per-window spans)"

    if fallback_cycles and fallback_cycles != total:
        # ``truth_access.yaml`` counts the truth blocks over the full horizon, so
        # the two are the same number by construction. A mismatch means one of
        # the records was written by a different run than the other (a rerun into
        # a populated results dir), and every cycle index below would be off.
        logger.warning(
            "truth_access.yaml spans %d cycle(s) but the window layout spans "
            "%d; the run directory mixes artifacts from two runs and the cycle "
            "indices below follow the window layout",
            fallback_cycles,
            total,
        )

    return WindowLayout(
        num_windows=int(num_windows),
        window_length=int(length),
        window_shift=int(shift),
        total_cycles=int(total),
        first_cycle=int(first),
        spans=spans,
        source=source,
    )


def global_cycle_indices(layout: WindowLayout) -> list[int]:
    """Global cycle index of each per-cycle artifact that survived on disk.

    ``[first_cycle, first_cycle + window_length)`` -- the last window's span.
    Every per-cycle number the metric and figure stages report is labelled with
    this, so a multi-window run's diagnostics are never read as if they started
    at cycle 0.
    """
    return [layout.first_cycle + k for k in range(layout.window_length)]


def final_window_truth_access(ta: dict, layout: WindowLayout) -> dict:
    """``truth_access.yaml`` re-based onto the window whose cycles are on disk.

    The truth spans all ``T`` cycles; the per-cycle artifacts are the last
    window's ``L``. Shifting the lazy view's ``start_idx`` by the window's first
    cycle (and its ``t_offset`` by that many cycle lengths, so the window opens
    at ``t = 0`` on the axis :func:`evaluation.sensors.window_masks` bins by)
    makes the returned dict describe exactly that window -- and therefore makes
    every helper re-exported from the filtering pipeline correct here without
    any of them knowing that windows exist.

    Reduces to ``dict(ta)`` with a trimmed ``n_total`` when ``first_cycle`` is
    0, which is every single-window run.
    """
    if not ta:
        return {}
    n_per_cycle = int(ta["n_per_cycle"])
    sim_time = float(ta["sim_time"])
    first = int(layout.first_cycle)
    dropped = first * n_per_cycle
    window_ta = dict(ta)
    window_ta["num_cycles"] = int(layout.window_length)
    window_ta["start_idx"] = int(ta["start_idx"]) + dropped
    window_ta["n_total"] = int(ta["n_total"]) - dropped
    window_ta["t_offset"] = float(ta["t_offset"]) + first * sim_time
    return window_ta


# ---------------------------------------------------------------------------
# The outer ESMDA loop's records (no filtering counterpart)
# ---------------------------------------------------------------------------

# The scalar fields of ``data_assimilation.filter_smoothing.IterationDiagnostics``
# that carry a value worth a series. ``iteration`` and ``alpha`` are the record's
# own labels, not measurements, so they are read separately where needed.
_ITERATION_FIELDS = (
    "obs_rmse",
    "innovation_chi2",
    "param_spread_prior",
    "param_spread_posterior",
)


def _iteration_series(rows) -> dict:
    """``{field: np.ndarray}`` over the outer iterations of one window."""
    series: dict = {}
    for field in _ITERATION_FIELDS:
        values = [
            (np.nan if row.get(field) is None else float(row[field]))
            for row in rows
            if isinstance(row, dict)
        ]
        if values and not np.all(np.isnan(values)):
            series[field] = np.asarray(values, dtype=float)
    return series


def iteration_diagnostics_series(run_dir: pathlib.Path) -> dict:
    """``iteration_diagnostics.yaml`` as ``{field: np.ndarray}`` over iterations.

    One entry per outer ESMDA iteration of the window whose artifacts survived
    (the last one). ``obs_rmse`` is measured on the stacked window system BEFORE
    that iteration's update, so it is the loop's cost function and should fall;
    the parameter spreads collapsing across iterations is the classic ESMDA
    over-fitting signature. ``{}`` when the file is absent.
    """
    return _iteration_series(
        read_yaml(pathlib.Path(run_dir) / "iteration_diagnostics.yaml") or []
    )


def window_iteration_series(run_dir: pathlib.Path) -> list:
    """Per-window ``(window_index, span, {field: array}, window_time)`` records.

    From ``window_diagnostics.yaml``, which exists only for a run that actually
    moved its window. ``[]`` otherwise -- and that absence is not a degradation:
    a single-window run's one record is ``iteration_diagnostics.yaml`` itself.
    """
    records = []
    for row in read_yaml(pathlib.Path(run_dir) / "window_diagnostics.yaml") or []:
        if not isinstance(row, dict):
            continue
        window_time = row.get("window_time")
        records.append(
            {
                "window": int(row.get("window", len(records))),
                "first_cycle": int(row["first_cycle"]),
                "last_cycle": int(row["last_cycle"]),
                "series": _iteration_series(row.get("iteration_diagnostics") or []),
                "window_seconds": (None if window_time is None else float(window_time)),
            }
        )
    return records


def load_iteration_trajectories(run_dir: pathlib.Path):
    """``params_iterations.nc`` -- the trajectory ensemble at every outer step.

    ``(esmda_step, time, ensemble)`` with step 0 the window's prior and the last
    step the converged trajectory; written under ``run.save_history`` (the
    default). ``None`` when it is absent, which costs the convergence blocks
    built on it and nothing else.
    """
    path = pathlib.Path(run_dir) / "params_iterations.nc"
    if not path.exists():
        logger.info(
            "No params_iterations.nc in %s (run.save_history was off); the "
            "per-iteration trajectory blocks are omitted",
            run_dir,
        )
        return None
    return xarray.open_dataset(path)
