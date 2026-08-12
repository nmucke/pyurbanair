"""Compute the metrics for a finished filter-smoothing run and write ``run_summary.yaml``.

Second stage of the three-script single-run filter-smoothing pipeline (see
``scripts/run_filter_smoothing_pipeline.sh``), mirroring
compute_filtering_metrics.py:

  1. scripts/filter_smoothing/run_filter_smoothing.py -- runs the method, saves
                                       the raw artifacts + truth_access.yaml +
                                       iteration_diagnostics.yaml +
                                       cycle_diagnostics.yaml + run_info.yaml
                                       (+ window_diagnostics.yaml when the
                                       window moved).
  2. scripts/filter_smoothing/compute_filter_smoothing_metrics.py (THIS) --
                                       reads those, computes the trajectory /
                                       state / sensor / convergence metrics and
                                       writes run_summary.yaml.
  3. scripts/filter_smoothing/make_filter_smoothing_figures.py -- reads those,
                                       draws the figures.

``run_summary.yaml`` is the run_info (configuration + timing) saved by
run_filter_smoothing.py, augmented with:

  * ``metrics_version``      -- estimator-semantics marker shared with the ESMDA
                                and filtering pipelines; 2 = fair (``M(M-1)``)
                                pairwise scores.
  * ``window_layout``        -- the moving window's geometry as the run recorded
                                it, and -- the point of the block -- WHICH
                                global cycles the per-cycle artifacts on disk
                                cover. Every window's inner filter rewrites the
                                same ``cycle_0 … cycle_{L-1}`` directories, so
                                what survives is the LAST window's ``L`` cycles
                                out of a ``T``-cycle horizon; the blocks below
                                are all scored over that span and labelled with
                                its global indices. ``source`` names the record
                                the layout came from, so an assumed layout is
                                never mistaken for a recorded one.
  * ``iteration_metrics``    -- the outer ESMDA loop's convergence: summary
                                stats of the per-iteration windowed obs-space
                                RMSE, innovation chi2 and trajectory spreads,
                                plus the prior->posterior reduction in each.
  * ``window_metrics``       -- moving-window runs only: the same per iteration,
                                per window, so a run whose later windows are
                                steadily harder to fit shows it here rather than
                                in a single averaged number.
  * ``filter_diagnostics``   -- summary stats of the INNER filter's per-cycle
                                innovation chi2 and obs-space prior/posterior
                                RMSE, over the final consistency pass.
  * ``parameter_metrics``    -- per-parameter RMSE/CRPS of the smoothed
                                TRAJECTORY vs the truth trajectory (+ skill vs
                                the prior) and its calibration.
  * ``trajectory_metrics``   -- the same per KNOT rather than reduced: the error
                                and the ensemble spread at every knot, which is
                                what says whether the smoothing tightened the
                                trajectory where the truth actually drifts.
  * ``ensemble_health``      -- duplicate-member counts of the trajectory
                                ensemble, run-wide and per outer iteration
                                (from params_iterations.nc).
  * ``state_metrics``        -- |U| field RMSE of the final pass's analyzed
                                end-of-cycle states vs the truth's matching
                                end-of-cycle frames.
  * ``sensor_metrics``       -- per sensor set, the full-vector (u, v, w) RMSE
                                and energy score (multivariate CRPS) over the
                                window's cycles.
  * ``cycle_states`` /
    ``sensor_statistics`` /
    ``field_metrics``        -- the per-cycle ensemble-state reductions, computed
                                by the filtering pipeline's own block
                                (``scripts.filtering.compute_filtering_metrics``)
                                because the artifacts they reduce are the same
                                artifacts. Which states those are -- the on-disk
                                forecasts under ``run.ensemble_save_on_disk=true``
                                or the analyzed frames otherwise -- is recorded
                                in ``cycle_states`` exactly as it is there.

Usage::

    python scripts/filter_smoothing/compute_filter_smoothing_metrics.py \
        --run-dir <filter smoothing output dir>
"""

import argparse
import logging
import pathlib
import sys

import numpy as np
import xarray

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from evaluation.scores import (
    METRICS_VERSION,
    compute_parameter_bundles,
    compute_parameter_metrics,
    ensemble_uniqueness,
    parameter_metric_summary,
    series_stats,
    vector_sensor_metrics,
)
from evaluation.turbulence import streaming_state_rmse

# The per-cycle ensemble-state reductions are shared, not duplicated: the inner
# filter writes the same artifacts ``compute_filtering_metrics`` already reduces,
# and its block is run-layout-agnostic once ``ta`` describes one window. The
# parameter flattening comes from the ESMDA stage for the same reason. Both are
# the established cross-package pattern here.
from scripts.esmda.compute_esmda_metrics import _flatten_parameter_members
from scripts.filter_smoothing._filter_smoothing_common import (
    build_sensor_sets,
    cycle_diagnostics_series,
    ensemble_cycle_sensor_series,
    final_window_truth_access,
    global_cycle_indices,
    iteration_diagnostics_series,
    load_analyzed_states,
    load_iteration_trajectories,
    load_run_config,
    read_yaml,
    truth_cycle_sensor_series,
    truth_end_of_cycle,
    window_iteration_series,
    window_layout,
    write_yaml,
)
from scripts.filtering.compute_filtering_metrics import _cycle_evaluation_blocks

logger = logging.getLogger(__name__)


def _nullable(values: np.ndarray) -> list[float | None]:
    """A per-knot float list with the non-finite entries as ``None``.

    YAML has no NaN literal that round-trips, and a diverged member leaves a
    whole knot non-finite; ``None`` keeps the list indexable by knot instead of
    dropping the entry and sliding every later one.
    """
    return [
        float(v) if np.isfinite(v) else None
        for v in np.asarray(values, dtype=float).ravel()
    ]


def _reduction(series: np.ndarray) -> float | None:
    """``1 - final/first`` of a convergence series, or ``None`` if unusable.

    Positive means the iterations improved it. The outer loop's ``obs_rmse`` is
    measured *before* each iteration's update, so its first entry is the prior's
    fit and its last is the fit of the second-to-last trajectory -- the loop's
    own cost function, and the one number that says whether the ESMDA iterations
    were worth their forecasts.
    """
    finite = np.asarray(series, dtype=float)
    if finite.size < 2 or not np.isfinite(finite[0]) or not np.isfinite(finite[-1]):
        return None
    if finite[0] == 0.0:
        return None
    return float(1.0 - finite[-1] / finite[0])


def _iteration_block(series: dict) -> dict | None:
    """``{field: series_stats, field_reduction: float}`` over the outer loop."""
    if not series:
        return None
    block: dict = {"num_iterations": int(len(next(iter(series.values()))))}
    for field, values in series.items():
        block[field] = series_stats(values)
        reduction = _reduction(values)
        if reduction is not None:
            block[f"{field}_reduction"] = reduction
    return block


def _window_block(run_dir: pathlib.Path) -> list | None:
    """One convergence record per window, or ``None`` for a single-window run.

    ``window_diagnostics.yaml`` is written only when the window moved, so its
    absence is a property of the run rather than a missing input: a single
    window's convergence IS ``iteration_metrics``. Each record keeps its own
    global cycle span, because the whole reason to read these separately is to
    see whether a window late on the horizon is harder to fit than window 0.
    """
    records = window_iteration_series(run_dir)
    if not records:
        return None
    block = []
    for record in records:
        entry: dict = {
            "window": record["window"],
            "first_cycle": record["first_cycle"],
            "last_cycle": record["last_cycle"],
            "window_seconds": record["window_seconds"],
        }
        iterations = _iteration_block(record["series"])
        if iterations is not None:
            entry.update(iterations)
        block.append(entry)
    return block


def _trajectory_metrics(
    posterior_params: xarray.Dataset,
    true_params: xarray.Dataset,
    prior_params: xarray.Dataset | None,
) -> dict:
    """Per-knot error and spread of the smoothed trajectory (metrics doc §3).

    ``parameter_metrics`` reduces these to scalars; this keeps them per knot,
    which is the resolution the method is about -- a trajectory estimate is only
    interesting knot by knot, and a run whose middle knots are sharp while its
    leading edge is still at prior width looks identical in the reduced numbers.
    One entry per knot of the trajectory (``time.seconds_per_knot`` sets how
    many that is, independently of the cycle count), and the ``knot`` column is
    their times in seconds — short enough that it costs the summary nothing.

    The prior half is present only when the prior is sampled on the SAME knot
    grid as the posterior. Under a moving window it is not, by design: the prior
    is sampled over the first window while the posterior spans the full horizon,
    and every later window's prior is the previous window's posterior rather
    than a sampler draw. ``evaluation.scores`` already refuses to broadcast
    across that mismatch; ``prior_comparable`` records which case ran so an
    absent prior column is never read as a prior that scored badly.
    """
    metrics = compute_parameter_metrics(posterior_params, true_params, prior_params)
    bundles = compute_parameter_bundles(posterior_params, true_params, prior_params)
    if not metrics:
        return {}

    block: dict = {}
    for name, entry in metrics.items():
        record: dict = {
            "knot": _nullable(entry["x"]),
            "rmse": _nullable(entry["rmse"]),
            "crps": _nullable(entry["crps"]),
            "prior_comparable": bool("prior_rmse" in entry),
        }
        if "prior_rmse" in entry:
            record["prior_rmse"] = _nullable(entry["prior_rmse"])
        for key in ("posterior_std", "prior_std", "contraction_ratio", "z_score"):
            values = bundles.get(name, {}).get(key)
            if values is not None:
                record[key] = _nullable(values)
        block[name] = record
    return block


def _ensemble_health(
    run_dir: pathlib.Path, posterior_params: xarray.Dataset
) -> dict | None:
    """Duplicate-member counts of the trajectory ensemble, run-wide and per step.

    The filter-smoothing counterpart of the ESMDA and filtering stages' blocks,
    and it exists for the same reason: a failure policy that replaces a diverged
    member with a copy of a surviving one leaves an ensemble whose nominal size
    ``M`` overstates its degrees of freedom, so the fair scores' ``M(M-1)``
    corrections are too generous.

    The per-step axis is the OUTER ESMDA iteration, not the cycle -- which is
    where this differs from the filtering stage and why it is not reused from
    there. ``params_history.nc`` here is the *inner* pass's per-cycle parameter
    record, and the inner pass is state-only, so all of its cycle entries are
    the same trajectory repeated; counting them would report the same number
    ``L+1`` times and call it a per-cycle series. ``params_iterations.nc`` is
    the artifact that actually varies (entry 0 the window's prior, then one per
    outer update), and ``run.save_history`` may have suppressed it -- which
    costs the list alone, with a log line, never the block.

    Returns ``None`` when even the final trajectory cannot be reduced.
    """
    try:
        health = ensemble_uniqueness(_flatten_parameter_members(posterior_params))
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Cannot assess ensemble health: %s", exc)
        return None

    if health["n_unique"] < health["n_members"]:
        logger.warning(
            "Duplicate members in the smoothed trajectory: %s/%s unique -- the "
            "fair scores' M(M-1) corrections overstate the ensemble's spread",
            health["n_unique"],
            health["n_members"],
        )

    per_iteration: list[int | None] = []
    trajectories = load_iteration_trajectories(run_dir)
    if trajectories is not None:
        try:
            for step in range(int(trajectories.sizes.get("esmda_step", 0))):
                try:
                    step_health = ensemble_uniqueness(
                        _flatten_parameter_members(trajectories.isel(esmda_step=step))
                    )
                except (ValueError, KeyError) as exc:
                    logger.warning(
                        "Skipping ensemble health for outer iteration %d: %s",
                        step,
                        exc,
                    )
                    per_iteration.append(None)
                    continue
                per_iteration.append(int(step_health["n_unique"]))
        except (OSError, ValueError, KeyError) as exc:
            # A history truncated by a killed job: it costs the list, not the
            # block and not the stage.
            logger.warning("Cannot read params_iterations.nc: %s", exc)
        finally:
            trajectories.close()

    minimum, median = health["min_pairwise"], health["median_pairwise"]
    return {
        "n_members": int(health["n_members"]),
        "n_unique": int(health["n_unique"]),
        # Entry 0 is the window's PRIOR trajectory, matching
        # params_iterations.nc's own ``esmda_step`` axis; the remaining entries
        # are the successive outer updates.
        "n_unique_per_iteration": per_iteration,
        "min_over_median_pairwise": (
            float(minimum / median)
            if minimum is not None and median is not None and median > 0
            else None
        ),
    }


def compute_metrics(run_dir: pathlib.Path) -> None:
    """Compute every run metric from the saved artifacts and write run_summary.yaml."""
    run_dir = pathlib.Path(run_dir)
    cfg = load_run_config(run_dir)
    ta = read_yaml(run_dir / "truth_access.yaml")

    # Seed the summary from the run metadata/timing saved by
    # run_filter_smoothing.py.
    summary = read_yaml(run_dir / "run_info.yaml")
    summary["metrics_version"] = METRICS_VERSION

    # --- Where the surviving per-cycle artifacts sit on the horizon ----------
    # Read FIRST: every truth-facing block below is scored over the window this
    # resolves, and the per-cycle numbers are labelled with its global indices.
    layout = window_layout(run_dir, ta)
    cycles = global_cycle_indices(layout)
    summary["window_layout"] = {
        "num_windows": layout.num_windows,
        "window_length": layout.window_length,
        "window_shift": layout.window_shift,
        "total_cycles": layout.total_cycles,
        # The mapping itself, spelled out rather than left to be re-derived: the
        # per-cycle artifacts on disk are cycle_0 … cycle_{L-1} of the LAST
        # window, which are the run's cycles ``evaluated_cycles``. Every window
        # overwrote the ones before it, so no earlier window's states survive.
        "final_pass_first_cycle": layout.first_cycle,
        "evaluated_cycles": cycles,
        "window_spans": [list(span) for span in layout.spans],
        "source": layout.source,
        "note": (
            "the trajectory blocks (parameter_metrics / trajectory_metrics) cover "
            "the FULL horizon; every state, sensor and cycle block covers the last "
            "window's cycles listed in evaluated_cycles, because each window's "
            "inner filter rewrites the same per-cycle artifacts"
        ),
    }
    # The truth-access view re-based onto that window. Handed to every helper
    # below in place of ``ta``, which is what lets the filtering pipeline's
    # machinery be reused verbatim: it never learns that windows exist.
    window_ta = final_window_truth_access(ta, layout)

    # --- Outer ESMDA convergence (always available) --------------------------
    iterations = _iteration_block(iteration_diagnostics_series(run_dir))
    if iterations is not None:
        summary["iteration_metrics"] = iterations
    else:
        logger.info(
            "No iteration_diagnostics.yaml in %s; the outer-loop convergence "
            "block is omitted",
            run_dir,
        )
    windows = _window_block(run_dir)
    if windows is not None:
        summary["window_metrics"] = windows

    # --- Inner filter health (always available) ------------------------------
    diag = cycle_diagnostics_series(run_dir)
    if diag:
        # Labelled with the GLOBAL cycle indices: on a moving-window run these
        # are the last window's cycles, and a reader who took them for cycles
        # 0…L-1 would place the whole series at the wrong end of the horizon.
        block: dict = {"cycles": cycles}
        block.update({field: series_stats(values) for field, values in diag.items()})
        summary["filter_diagnostics"] = block

    # --- Parameter TRAJECTORY (always present: this method estimates one) -----
    posterior_path = run_dir / "posterior_params.nc"
    if posterior_path.exists():
        posterior_params = xarray.open_dataset(posterior_path)
        prior_path = run_dir / "prior_params.nc"
        prior_params = xarray.open_dataset(prior_path) if prior_path.exists() else None
        true_params = xarray.open_dataset(run_dir / "true_params.nc")
        summary["parameter_metrics"] = parameter_metric_summary(
            posterior_params, true_params, prior_params
        )
        trajectory = _trajectory_metrics(posterior_params, true_params, prior_params)
        if trajectory:
            summary["trajectory_metrics"] = trajectory
        health = _ensemble_health(run_dir, posterior_params)
        if health is not None:
            summary["ensemble_health"] = health

    # The state and sensor metrics both open the (potentially multi-GB) truth.
    # If the truth-access record is missing (e.g. an older run) there is nothing
    # to compare against, so write what we have and stop.
    if not window_ta:
        write_yaml(summary, run_dir / "run_summary.yaml")
        print(f"Saved run summary in {run_dir / 'run_summary.yaml'}")
        return

    # --- State field |U| RMSE ------------------------------------------------
    # The final consistency pass's analyzed end-of-cycle states against the
    # truth's matching frames. ``truth_end_of_cycle`` takes the LAST
    # ``n_frames`` end-of-cycle frames of the window, so a run that saved no
    # history (one analyzed frame) still pairs with the right cycle.
    analyzed_states = load_analyzed_states(run_dir, window_ta)
    n_frames = int(analyzed_states.sizes["time"])
    truth_end = truth_end_of_cycle(window_ta, n_frames)
    rmse = streaming_state_rmse(truth_end, analyzed_states)
    summary["state_metrics"] = {
        "vel_magnitude_rmse": series_stats(rmse),
        # Which of the horizon's cycles those frames are, for the same reason
        # the diagnostics carry it.
        "cycles": cycles[-n_frames:],
    }

    # --- Sensors: full-vector (u, v, w) RMSE + energy score ------------------
    sensor_sets = build_sensor_sets(cfg)
    truth_series = truth_cycle_sensor_series(
        truth_end, sensor_sets, window_ta["truth_solver_name"]
    )
    ensemble_series = ensemble_cycle_sensor_series(
        analyzed_states, sensor_sets, window_ta["assim_solver_name"]
    )

    # Invariant 3, as in the filtering stage: the energy score is a
    # probabilistic one and needs the members, so an ensemble-mean-only state
    # artifact must cost its own sensor set rather than the whole stage -- which
    # would take the trajectory, health, convergence and state blocks with it.
    scorable = {
        name: coords
        for name, coords in sensor_sets.items()
        if "ensemble" in ensemble_series[name].dims
    }
    for name in sensor_sets:
        if name not in scorable:
            logger.warning(
                "No ensemble dimension in the %s sensor series -- the sensor "
                "metrics need the members, so that set is omitted",
                name,
            )

    sensor_metrics = {}
    for name, (sx, sy, sz) in scorable.items():
        m = vector_sensor_metrics(truth_series[name], ensemble_series[name])
        sensor_metrics[name] = {
            "num_sensors": int(np.asarray(sx).size),
            "velocity_vector_rmse": series_stats(m["rmse"]),
            "velocity_vector_energy_score": series_stats(m["energy_score"]),
        }
    summary["sensor_metrics"] = sensor_metrics

    # --- Cycle-state reductions: statistics + mean fields --------------------
    # Reused wholesale from the filtering pipeline (see the import comment): the
    # artifacts are the same artifacts, and ``window_ta`` has already told it
    # which cycles they are. It writes eval_fields.nc beside run_summary.yaml
    # exactly as it does there, which is what the figure stage reads.
    blocks = _cycle_evaluation_blocks(run_dir, window_ta, sensor_sets)
    states = blocks.get("cycle_states")
    if isinstance(states, dict):
        # The block names its source and its cycle COUNT; the horizon indices
        # are this pipeline's business, so they are added here rather than by
        # widening the shared block's contract. Counted from the END of the
        # window, not the start: a run that saved no history falls back to the
        # final analyzed frame alone, which is the window's LAST cycle -- the
        # same alignment ``truth_end_of_cycle(ta, n_frames)`` makes on the truth
        # side, so the two never label the same frame differently.
        count = states.get("num_cycles")
        scored = count if isinstance(count, int) and 0 < count <= len(cycles) else 0
        labelled = dict(states)
        labelled["cycles"] = cycles[len(cycles) - scored :] if scored else list(cycles)
        blocks["cycle_states"] = labelled
    summary.update(blocks)

    write_yaml(summary, run_dir / "run_summary.yaml")
    print(f"Saved run summary in {run_dir / 'run_summary.yaml'}")


def main() -> None:
    # Every reason a block degraded or was skipped is a ``logger`` call in this
    # module, in ``_filter_smoothing_common`` and in the reused filtering /
    # evaluation code; with no handler on the root logger a stage run standalone
    # printed only its final line and none of them. Configured here, at the
    # entry point (never in ``compute_metrics``) so importing this module -- the
    # tests do -- cannot reconfigure anyone's logging.
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--run-dir",
        type=pathlib.Path,
        required=True,
        help=(
            "The filter-smoothing run output directory written by "
            "scripts/filter_smoothing/run_filter_smoothing.py."
        ),
    )
    args = ap.parse_args()
    if not args.run_dir.exists():
        raise SystemExit(f"run dir not found: {args.run_dir}")
    compute_metrics(args.run_dir)


if __name__ == "__main__":
    main()
