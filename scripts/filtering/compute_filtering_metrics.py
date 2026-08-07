"""Compute the metrics for a finished filtering run and write ``run_summary.yaml``.

Second stage of the three-script single-run filtering pipeline (see
``scripts/run_filtering_pipeline.sh``), mirroring compute_esmda_metrics.py:

  1. scripts/filtering/run_filtering.py            -- runs the filter, saves the
                                       raw artifacts + truth_access.yaml +
                                       cycle_diagnostics.yaml + run_info.yaml.
  2. scripts/filtering/compute_filtering_metrics.py (THIS) -- reads those,
                                       computes the parameter / state / sensor /
                                       filter-diagnostic metrics and writes
                                       run_summary.yaml.
  3. scripts/filtering/make_filtering_figures.py   -- reads those, draws figures.

``run_summary.yaml`` is the run_info (configuration + timing) saved by
run_filtering.py, augmented with:

  * ``metrics_version``     -- estimator-semantics marker shared with the ESMDA
                               pipeline; 2 = fair (``M(M-1)``) pairwise scores
                               (see scripts/esmda/compute_esmda_metrics.py).
  * ``parameter_metrics``   -- per-parameter RMSE/CRPS of the final analyzed
                               ensemble vs truth (+ skill vs prior) and its
                               calibration (z-score, normalized error,
                               contraction ratio). Absent in ``mode='state'``
                               (no parameters estimated).
  * ``ensemble_health``     -- duplicate-member counts of the analyzed parameter
                               ensemble, run-wide and per cycle
                               (``n_unique_per_cycle``, from params_history.nc),
                               plus the min/median pairwise-distance ratio.
                               Absent in ``mode='state'``.
  * ``filter_diagnostics``  -- summary stats of the per-cycle innovation chi2 and
                               observation-space prior/posterior RMSE.
  * ``state_metrics``       -- |U| field RMSE of the analyzed end-of-cycle states
                               vs the truth's end-of-cycle frames.
  * ``sensor_metrics``      -- per sensor set, the full-vector (u, v, w) RMSE and
                               energy score (multivariate CRPS) over the cycles.
  * ``cycle_states``        -- which per-cycle ensemble states the three blocks
                               below reduced over (``kind`` = ``forecast`` or
                               ``analysis``, its cycle/member counts and a
                               one-line ``description`` of the caveat). The
                               filter has two possible stand-ins for the ESMDA
                               pipeline's per-window posterior rollouts and the
                               weaker one degrades the variance / TKE numbers,
                               so which one ran is recorded rather than implied.
  * ``sensor_statistics``   -- per sensor set, the per-cycle mean and variance of
                               u/v/w/|U| scored as the verification object (fair
                               CRPS, z-score, rank counts, identifiability), the
                               statistics-space counterpart of ``sensor_metrics``:
                               a pointwise time-series error mostly measures
                               turbulent phase, which no parameter estimate
                               controls. Posterior half only -- the filter saves
                               no comparable prior rollout. (Its ``n_windows``
                               key is the shared library's name for the bin
                               count, which here is the cycle count.)
  * ``field_metrics``       -- the VDI 3783/9 hit rate ``q`` of the time-mean
                               velocity field over evenly spaced z-slabs, with
                               the absolute tolerance ``W`` block-bootstrapped
                               from the truth's own sampling error. The reduced
                               fields behind it (mean, TKE, ``<u'w'>``, on the
                               slabs and at the station columns) are written
                               beside it as ``eval_fields.nc`` for the figure
                               stage. No prior half, same reason as above.

The time axis throughout is the cycle: the truth is compared at each cycle's
end-of-segment frame (see scripts/filtering/_filtering_common.py).

Usage::

    python scripts/filtering/compute_filtering_metrics.py --run-dir <filtering output dir>
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
    ensemble_uniqueness,
    parameter_metric_summary,
    series_stats,
    vector_sensor_metrics,
)
from evaluation.turbulence import streaming_state_rmse

# The WP1.3-1.5 machinery (the mean-field collector, the station columns, the
# hit-rate block and the window-statistics scoring loop) is shared, not
# duplicated: ``compute_esmda_metrics`` already owns it and it is entirely
# run-layout-agnostic once the per-member states are handed to it. Cross-package
# imports between the two script packages are the established pattern here --
# ``scripts/filtering/_filtering_common.py`` is built on the same move.
from scripts.esmda.compute_esmda_metrics import (
    MeanFieldCollector,
    _flatten_parameter_members,
    _mean_field_block,
    _sensor_statistics,
    station_columns,
)
from scripts.filtering._filtering_common import (
    CycleStates,
    build_sensor_sets,
    cycle_diagnostics_series,
    cycle_sensor_series,
    cycle_state_source,
    ensemble_cycle_sensor_series,
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


def _per_cycle_uniqueness(history: xarray.Dataset) -> list[int | None]:
    """Unique-member counts of each cycle's analyzed parameters.

    ``params_history.nc``'s ``cycle`` axis is ``[prior, cycle 0 posterior,
    cycle 1 posterior, ...]`` -- the filter stacks the ensemble it started from
    in front of the analyses -- so entry 0 is the **prior** and is skipped here:
    counting it would report ``num_cycles + 1`` cycles and put the prior's
    duplicate count at index 0, where every consumer expects cycle 0's.

    A slot that cannot be reduced becomes ``None`` rather than being dropped, so
    the list stays indexable by cycle (the same discipline the ESMDA stage's
    per-window list follows).
    """
    per_cycle: list[int | None] = []
    for entry in range(1, int(history.sizes.get("cycle", 0))):
        try:
            health = ensemble_uniqueness(
                _flatten_parameter_members(history.isel(cycle=entry))
            )
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping ensemble health for cycle %d: %s", entry - 1, exc)
            per_cycle.append(None)
            continue
        per_cycle.append(int(health["n_unique"]))
        if health["n_unique"] < health["n_members"]:
            logger.warning(
                "Duplicate analyzed parameter members in cycle %d: %s/%s unique",
                entry - 1,
                health["n_unique"],
                health["n_members"],
            )
    return per_cycle


def _ensemble_health(
    run_dir: pathlib.Path, posterior_params: xarray.Dataset
) -> dict[str, object] | None:
    """Duplicate-member counts for the final analyzed ensemble and each cycle.

    The filtering counterpart of the ESMDA stage's ``_ensemble_health``, and it
    exists for the same reason: a failure policy that replaces a diverged member
    with a copy of a surviving one leaves an ensemble whose nominal size ``M``
    overstates its degrees of freedom, so the fair scores' ``M(M-1)``
    corrections are too generous. The two counts answer different questions --
    two members cloned in one cycle may diverge again in a later one, and only
    the per-cycle counts see that -- hence both.

    Where ESMDA reduces ``windows/window_*_posterior_params.nc``, the filter
    writes no per-cycle parameter files at all; the per-cycle ensembles live
    stacked in ``params_history.nc`` (see :func:`_per_cycle_uniqueness`), which
    ``run.save_history`` may have suppressed. That absence costs the per-cycle
    list alone -- an empty list beside the run-wide counts, with a log line --
    never the block, and an unreadable file never the metric stage.

    Returns ``None`` when even the final analyzed ensemble cannot be reduced.
    """
    try:
        health = ensemble_uniqueness(_flatten_parameter_members(posterior_params))
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Cannot assess ensemble health: %s", exc)
        return None

    if health["n_unique"] < health["n_members"]:
        logger.warning(
            "Duplicate analyzed parameter members: %s/%s unique -- the fair "
            "scores' M(M-1) corrections overstate the ensemble's spread",
            health["n_unique"],
            health["n_members"],
        )

    per_cycle: list[int | None] = []
    history_path = run_dir / "params_history.nc"
    if not history_path.exists():
        logger.info(
            "No params_history.nc, so there are no per-cycle uniqueness counts "
            "(run.save_history was off); the run-wide counts stand"
        )
    else:
        try:
            with xarray.open_dataset(history_path) as history:
                per_cycle = _per_cycle_uniqueness(history)
        except (OSError, ValueError, KeyError) as exc:
            # A history truncated by a killed job. Invariant 3: it costs the
            # per-cycle list, not the block and not the stage.
            logger.warning("Cannot read %s: %s", history_path.name, exc)

    minimum, median = health["min_pairwise"], health["median_pairwise"]
    return {
        "n_members": int(health["n_members"]),
        "n_unique": int(health["n_unique"]),
        "n_unique_per_cycle": per_cycle,
        "min_over_median_pairwise": (
            float(minimum / median)
            if minimum is not None and median is not None and median > 0
            else None
        ),
    }


def _solid_state_path(
    run_dir: pathlib.Path, source: CycleStates
) -> pathlib.Path | None:
    """A state file to read the backend's obstacle indicator from.

    The sensor pass hands the collector ``ds[["u", "v", "w"]]`` slices, so the
    ``blanking`` variable pylbm writes beside the state never reaches it that
    way; the ESMDA stage reads it separately from one window file and this does
    the same from one file of whichever source ran. Handed over unconditionally
    rather than guessed at: a file with no indicator is a no-op (the hit rate
    falls back to the truth's zero-variance rule and records which ran in
    ``solid_cell_source``), while a file that *has* one is the difference
    between scoring fluid cells and diluting ``q`` toward the built-up fraction.
    """
    if source.kind == "forecast" and source.paths:
        return pathlib.Path(source.paths[0][0])
    history_path = run_dir / "state_history.nc"
    return history_path if history_path.exists() else None


def _cycle_evaluation_blocks(
    run_dir: pathlib.Path, ta: dict, sensor_sets: dict
) -> dict[str, object]:
    """``cycle_states`` + ``sensor_statistics`` + ``field_metrics`` (WP1.3-1.5).

    One pass over the per-cycle ensemble states serves all three: the sensor
    series the statistics are computed from, and -- through ``on_member`` -- the
    mean-field accumulators, so the fields cost no second read of the state
    files (master-plan invariant 2). The truth is streamed twice, once for the
    sensor series and once for the fields, because the fields can only be
    sampled onto the region the ensemble pass fixes.

    Returns ``{}`` when the source cannot be resolved at all; each block
    individually degrades to absent-with-a-log-line rather than taking its
    neighbours down.
    """
    try:
        source = cycle_state_source(run_dir, ta)
    except (OSError, ValueError, KeyError) as exc:
        # Everything below reduces over per-cycle ensemble states; with no
        # source there is nothing to reduce. The parameter / state / sensor
        # blocks the caller already computed are unaffected.
        logger.warning("No per-cycle ensemble states to evaluate: %s", exc)
        return {}

    logger.info("Cycle states: %s", source.description)
    blocks: dict[str, object] = {
        # Recorded, not implied. The two sources answer the same questions with
        # materially different fidelity -- the analysis source has one frame per
        # cycle, so its per-cycle variance is null and its TKE / <u'w'> moments
        # are taken across cycles rather than within them -- and a reader of this
        # file has no other way to tell which numbers they are looking at.
        "cycle_states": {
            "kind": str(source.kind),
            "num_cycles": int(source.num_cycles),
            "n_members": int(source.n_members),
            "description": str(source.description),
        }
    }

    station_x, station_y, station_set = station_columns(sensor_sets)
    field_collector = MeanFieldCollector(
        ta["assim_solver_name"],
        station_x,
        station_y,
        station_set,
        # ``CycleStates.n_members``, not ``_ensemble_size`` of the state files:
        # the forecast source's per-member files carry no ``ensemble`` axis at
        # all, so counting them would report 1 and switch the accumulators'
        # memory-budget stride off on exactly the largest ensembles.
        n_members=int(source.n_members),
        solid_state_path=_solid_state_path(run_dir, source),
    )
    ensemble_series = cycle_sensor_series(
        run_dir, ta, source, sensor_sets, on_member=field_collector.add
    )

    # Invariant 3, as in the ESMDA stage: every score in the statistics block is
    # a probabilistic one and needs the members, so a sensor set whose states
    # carry no ``ensemble`` axis costs its own entry rather than the block.
    scorable = {
        name: coords
        for name, coords in sensor_sets.items()
        if "ensemble" in ensemble_series[name].dims
    }
    for name in sensor_sets:
        if name not in scorable:
            logger.warning(
                "No ensemble dimension in the %s cycle sensor series -- the "
                "sensor statistics need the members, so that set is omitted",
                name,
            )

    if scorable:
        truth_series = truth_cycle_statistics_series(ta, source, scorable)
        # The same scoring the ESMDA stage runs, binned by the cycle instead of
        # the window: ``source.bin_seconds`` is its ``sim_time`` (the physical
        # cycle length for the forecast source, 1.0 for the analysis source
        # whose series carry an integer cycle index) and ``source.num_cycles``
        # its ``num_windows``, so one bin is one cycle either way.
        #
        # No prior half, and that is a property of the filter rather than an
        # omission: it saves no prior rollout. Cycle 0's forecast IS a prior
        # rollout -- it runs under the sampled prior parameters -- but it covers
        # ONE cycle against the posterior's ``num_cycles``, and scoring those
        # against each other is the "two different horizons in one skill score"
        # that the field block's ``_comparable_prior`` refuses outright. So the
        # prior statistics and their sampling floor are both None.
        blocks["sensor_statistics"] = _sensor_statistics(
            scorable,
            truth_series,
            ensemble_series,
            None,
            int(source.num_cycles),
            float(source.bin_seconds),
        )

    # --- Mean fields: hit rate + eval_fields.nc (WP1.4) ----------------------
    truth_collector = MeanFieldCollector(
        ta["truth_solver_name"],
        station_x,
        station_y,
        station_set,
        target=field_collector.target,
        keep_samples=True,
    )
    if field_collector.target is not None and not field_collector.failed:
        # Only once the ensemble pass has fixed the region: the truth is scored
        # on the assimilation grid, so without a target there is nothing to
        # interpolate onto, and re-reading the truth for a posterior that failed
        # mid-pass would buy nothing. ``_mean_field_block`` drops the block in
        # both cases.
        truth_field_pass(truth_collector, ta, source)
    field_metrics = _mean_field_block(
        run_dir,
        field_collector,
        truth_collector,
        None,  # no prior collector -- same reason as the sensor prior above
        # The physical span the accumulated frames cover, in BOTH sources: the
        # analysis source's frames are the end-of-cycle ones, carried on a cycle
        # index rather than on seconds, but they still sample this stretch of
        # the run and that is what figure F1 annotates.
        time_span=(0.0, int(source.num_cycles) * float(ta["sim_time"])),
        # ...and the span alone is exactly what would mislead: under the default
        # source those frames are ONE per cycle, so the TKE and <u'w'> the same
        # file carries are an across-cycle variance rather than resolved
        # turbulence, and nothing in the numbers says so. ``run_summary.yaml``
        # already records the caveat, but the figure stage reads
        # ``eval_fields.nc`` and not the summary, so it travels with the data.
        moment_sampling=source.description,
        # Whether those frames leave GAPS in the span above, which is the half a
        # figure may act on -- and it is the source's kind, not the presence of
        # the line beside it. Both sources name their frames; only ``analysis``
        # is sparse (one analyzed frame per cycle). ``forecast`` accumulates
        # every frame of every cycle's forecast segment, and the segments tile
        # the run, so its moments ARE a continuous time average and labelling
        # them otherwise would mislabel the better-configured of the two runs
        # -- with the same words, so a reader who learned to discount them here
        # would discount them on the run that needs them.
        moment_sampling_is_sparse=source.kind != "forecast",
    )
    if field_metrics is not None:
        # ``n_windows`` is the number of distinct chunk indices the accumulators
        # were fed, which is the cycle count only for the forecast source: the
        # analysis source arrives as ONE chunk (the whole cycle axis is a single
        # lazily-opened file), so it reports 1 however many cycles ran. Rather
        # than overwrite a count that is true to what the collector saw, the
        # horizon is recorded beside it.
        field_metrics["n_cycles"] = int(source.num_cycles)
        blocks["field_metrics"] = field_metrics
    return blocks


def compute_metrics(run_dir: pathlib.Path) -> None:
    """Compute every run metric from the saved artifacts and write run_summary.yaml."""
    cfg = load_run_config(run_dir)
    ta = read_yaml(run_dir / "truth_access.yaml")

    # Seed the summary from the run metadata/timing saved by run_filtering.py.
    summary = read_yaml(run_dir / "run_info.yaml")
    summary["metrics_version"] = METRICS_VERSION

    # --- Filter health diagnostics (always available) -----------------------
    # Summarize the per-cycle innovation chi2 and observation-space RMSE that the
    # filter recorded; these need no truth access and exist for every mode.
    diag = cycle_diagnostics_series(run_dir)
    if diag:
        summary["filter_diagnostics"] = {
            field: series_stats(values) for field, values in diag.items()
        }

    # --- Parameters (present unless mode='state') ---------------------------
    posterior_path = run_dir / "posterior_params.nc"
    if posterior_path.exists():
        posterior_params = xarray.open_dataset(posterior_path)
        prior_params = xarray.open_dataset(run_dir / "prior_params.nc")
        true_params = xarray.open_dataset(run_dir / "true_params.nc")
        # The posterior is the filter's final scalar estimate (end of run). Label
        # it with the run's end time so a drifting (time-varying) truth is
        # compared at that time rather than at t=0 (the truth is interpolated
        # onto the estimate's x-axis; see compute_parameter_metrics). A static
        # truth is constant, so the label is a no-op for it.
        if ta and "time" not in posterior_params.dims:
            final_time = float(ta["sim_time"]) * int(ta["num_cycles"])
            posterior_params = posterior_params.expand_dims(time=[final_time])
        summary["parameter_metrics"] = parameter_metric_summary(
            posterior_params, true_params, prior_params
        )
        health = _ensemble_health(run_dir, posterior_params)
        if health is not None:
            summary["ensemble_health"] = health

    # The state and sensor metrics both open the (potentially multi-GB) truth.
    # If the truth-access record is missing (e.g. an older run) there is nothing
    # to compare against, so write what we have and stop.
    if not ta:
        write_yaml(summary, run_dir / "run_summary.yaml")
        print(f"Saved run summary in {run_dir / 'run_summary.yaml'}")
        return

    # --- State field |U| RMSE -----------------------------------------------
    # Compare each cycle's analyzed ensemble-mean state against the truth's
    # end-of-cycle frame, streaming over a few z-slices (interpolating onto the
    # assim grid if the coords differ). One RMSE per cycle.
    analyzed_states = load_analyzed_states(run_dir, ta)
    n_frames = int(analyzed_states.sizes["time"])
    truth_end = truth_end_of_cycle(ta, n_frames)
    rmse = streaming_state_rmse(truth_end, analyzed_states)
    summary["state_metrics"] = {"vel_magnitude_rmse": series_stats(rmse)}

    # --- Sensors: full-vector (u, v, w) RMSE + energy score ------------------
    # Truth and ensemble are interpolated at the same physical points on their
    # own grids, so the metric is grid-independent. One value per cycle.
    sensor_sets = build_sensor_sets(cfg)
    truth_series = truth_cycle_sensor_series(
        truth_end, sensor_sets, ta["truth_solver_name"]
    )
    ensemble_series = ensemble_cycle_sensor_series(
        analyzed_states, sensor_sets, ta["assim_solver_name"]
    )

    # Invariant 3, and it has to be checked HERE rather than inside the scoring:
    # the energy score is a probabilistic one and needs the members, so
    # ``vector_sensor_metrics`` raises on a series with no ``ensemble`` axis --
    # before the library's own guard is ever reached. An ensemble-mean-only
    # state artifact must cost its own sensor set, not the whole stage: without
    # this the parameter, health, diagnostic and state blocks already computed
    # above would go down with it and no run_summary.yaml would be written at
    # all. Mirrors the same guard in compute_esmda_metrics.compute_metrics.
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

    # --- Cycle-state reductions: statistics + mean fields (WP1.3-1.5) --------
    # Everything above scores the ONE analyzed frame per cycle. These blocks
    # instead reduce over whichever per-cycle ensemble states the run actually
    # holds -- the full forecast segments when it was run with
    # ``run.ensemble_save_on_disk=true``, the analyzed frames otherwise -- which
    # is what ``cycle_state_source`` decides, once, on their behalf. Their own
    # extraction pass, deliberately: it carries the mean-field collector and, on
    # the forecast source, covers frames the blocks above never see.
    summary.update(_cycle_evaluation_blocks(run_dir, ta, sensor_sets))

    write_yaml(summary, run_dir / "run_summary.yaml")
    print(f"Saved run summary in {run_dir / 'run_summary.yaml'}")


def main() -> None:
    # Every reason a block degraded or was skipped is a ``logger`` call in this
    # module, in ``_filtering_common`` and in ``evaluation``; with no handler on
    # the root logger a stage run standalone printed only its final line and none
    # of them. Configured here, at the entry point (never in ``compute_metrics``)
    # so importing this module -- the tests do -- cannot reconfigure anyone's
    # logging. Mirrors ``compute_esmda_metrics.main``.
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
        help="The filtering run output directory written by scripts/filtering/run_filtering.py.",
    )
    args = ap.parse_args()
    if not args.run_dir.exists():
        raise SystemExit(f"run dir not found: {args.run_dir}")
    compute_metrics(args.run_dir)


if __name__ == "__main__":
    main()
