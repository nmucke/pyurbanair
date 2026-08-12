"""Draw the figures for a finished filter-smoothing run.

Third stage of the three-script single-run filter-smoothing pipeline (see
``scripts/run_filter_smoothing_pipeline.sh``), mirroring
make_filtering_figures.py:

  1. scripts/filter_smoothing/run_filter_smoothing.py -- runs the method, saves
                                       the raw artifacts + truth_access.yaml +
                                       run_info.yaml.
  2. scripts/filter_smoothing/compute_filter_smoothing_metrics.py -- reads
                                       those, writes run_summary.yaml (+
                                       eval_fields.nc).
  3. scripts/filter_smoothing/make_filter_smoothing_figures.py (THIS) -- reads
                                       those, draws the figures.

Writes, into the run directory:

  * ``parameter_evolution.png``    -- the prior and smoothed posterior parameter
                                      TRAJECTORIES against the truth trajectory,
                                      plus the per-cycle |U| RMSE.
  * ``parameter_error.png``        -- per-parameter posterior error per knot.
  * ``parameter_marginals.png``    -- prior vs posterior marginal per parameter.
  * ``iteration_convergence.png``  -- the outer ESMDA loop's obs RMSE /
                                      innovation chi2 / trajectory spread vs
                                      iteration, one line per window.
  * ``rollout_animation.mp4``      -- analyzed ensemble-mean |U| field vs the
                                      truth, one frame per cycle.
  * ``final_state_with_obs.png``   -- final analyzed |U| field with the sensors.
  * ``sensor_timeseries_<set>.png`` -- truth vs ensemble |U| at each sensor set.
  * ``station_profiles.png``       -- mean-velocity / TKE profiles at the sensor
                                      columns (needs ``eval_fields.nc``).
  * ``mean_slices.png``            -- time-mean field slices, truth vs posterior
                                      vs difference (same file).
  * ``sensor_fans.png``            -- sensor quantile fans with the observations.
  * ``rank_histogram.png``         -- rank histogram of the cycle statistics
                                      (needs ``run_summary.yaml``).

**Two time axes, and they are not the same one.** The parameter figures are
drawn over the smoothed trajectory, which spans the FULL ``T``-cycle horizon.
Every state / sensor figure is drawn over the per-cycle artifacts, which are the
LAST window's ``L`` cycles -- each window's inner filter rewrote them. Which
global cycles those are is resolved by
``scripts.filter_smoothing._filter_smoothing_common.window_layout`` and printed
by this stage, and the sensor fan's cycle boundaries are placed on the horizon's
clock rather than at 0 so the two figures can be read side by side.

As in the filtering stage, ``station_profiles.png`` and ``mean_slices.png`` need
``eval_fields.nc`` and ``rank_histogram.png`` needs ``run_summary.yaml``'s
``sensor_statistics`` block -- all written by
``compute_filter_smoothing_metrics.py`` -- and each degrades to a printed skip
line when it is absent. A skip never costs the figures after it. This script
owns all the run-dir layout, config and YAML knowledge; the figure functions in
``evaluation.figures`` are handed opened datasets and plain dicts.

Usage::

    python scripts/filter_smoothing/make_filter_smoothing_figures.py \
        --run-dir <filter smoothing output dir>
"""

import argparse
import logging
import pathlib
import sys

import numpy as np
import xarray
from omegaconf import OmegaConf

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from evaluation.figures import (
    plot_final_state_with_obs,
    plot_iteration_convergence,
    plot_mean_slices,
    plot_parameter_error,
    plot_parameter_marginals,
    plot_rank_histogram,
    plot_rollout_time_evolution,
    plot_sensor_fans,
    plot_sensor_timeseries,
    plot_station_profiles,
)
from evaluation.scores import compute_sensor_metrics
from evaluation.sensors import sensor_magnitude
from evaluation.turbulence import select_z_plane, streaming_state_rmse

from pyurbanair.config.hydra_helpers import create_observation_points
from pyurbanair.utils.animation_utils import animate_rollout_state
from pyurbanair.utils.run_utils import add_velocity_magnitude
from scripts.filter_smoothing._filter_smoothing_common import (
    build_sensor_sets,
    cycle_sensor_series,
    cycle_state_source,
    ensemble_velocity_mean_std,
    final_window_truth_access,
    global_cycle_indices,
    iteration_diagnostics_series,
    load_analyzed_states,
    load_run_config,
    read_yaml,
    truth_cycle_statistics_series,
    truth_end_of_cycle,
    window_iteration_series,
    window_layout,
)
from scripts.filtering.make_filtering_figures import _rank_counts, _reference_velocity


def _note_skipped(output_path: pathlib.Path, written: pathlib.Path | None) -> None:
    """Say so when a figure no-oped on inputs that were present but unusable.

    The figures return the path they wrote, or ``None`` when their inputs were
    empty or degenerate (a 2-member smoke ensemble has no meaningful KDE). The
    reason is logged by the figure itself; this only makes the *absence* visible
    in the stage's own print-based output, so a missing PNG is never a silent
    one.
    """
    if written is None:
        print(f"Skipped {output_path.name}: its inputs were absent or degenerate")


def _convergence_series(run_dir: pathlib.Path) -> tuple[list, list]:
    """``(per_window series, window labels)`` for D4.

    ``window_diagnostics.yaml`` is the multi-window record and carries every
    window's own iterations; a single-window run has only
    ``iteration_diagnostics.yaml``, which IS that one window's record. Reading
    the moving-window file first rather than adding its windows to the
    single-window series is what keeps the last window from being drawn twice --
    ``iteration_diagnostics.yaml`` is the LAST window's, not a run-wide average.
    """
    records = window_iteration_series(run_dir)
    if records:
        return (
            [{k: list(v) for k, v in record["series"].items()} for record in records],
            [record["window"] for record in records],
        )
    series = iteration_diagnostics_series(run_dir)
    if not series:
        return [], []
    return [{k: list(v) for k, v in series.items()}], [0]


def make_figures(run_dir: pathlib.Path) -> None:
    """Draw every figure from the saved artifacts in ``run_dir``."""
    run_dir = pathlib.Path(run_dir)
    cfg = load_run_config(run_dir)
    ta = read_yaml(run_dir / "truth_access.yaml")
    # The physical length of one forecast segment, i.e. one cycle.
    sim_time = float(ta["sim_time"])

    # Which cycles of the horizon the per-cycle artifacts are. Printed rather
    # than left implicit: on a moving-window run every state and sensor figure
    # below covers this span and not the whole run, and no PNG says so.
    layout = window_layout(run_dir, ta)
    cycles = global_cycle_indices(layout)
    window_ta = final_window_truth_access(ta, layout)
    if layout.num_windows > 1:
        print(
            f"Moving window: {layout.num_windows} window(s) of "
            f"{layout.window_length} cycle(s) shifting by {layout.window_shift} "
            f"over {layout.total_cycles} cycle(s). The per-cycle artifacts are "
            f"the last window's, i.e. cycles {cycles[0]}-{cycles[-1]}; the "
            "parameter figures cover the whole horizon."
        )

    true_params = xarray.open_dataset(run_dir / "true_params.nc")

    # The final pass's analyzed end-of-cycle ensemble states (time = cycle axis)
    # and the truth's matching end-of-cycle frames, both scoped to the window
    # whose cycles survived. Opened lazily; the plots pull only the z-plane /
    # z-slices they need.
    analyzed_states = load_analyzed_states(run_dir, window_ta)
    n_frames = int(analyzed_states.sizes["time"])
    truth_end = truth_end_of_cycle(window_ta, n_frames)
    obs_x, obs_y, obs_z = create_observation_points(cfg.obs)

    # Per-cycle state |U| RMSE (streamed over a few z-slices).
    rmse = streaming_state_rmse(truth_end, analyzed_states)

    # --- Parameter TRAJECTORY figures ---------------------------------------
    # The smoothed trajectory, its prior and the truth trajectory on one axis.
    # Unlike the filtering pipeline there is no params_history to read: the
    # inner pass is state-only, so its per-cycle parameter record is the same
    # trajectory repeated. The prior and posterior datasets ARE the figure.
    posterior_params_path = run_dir / "posterior_params.nc"
    prior_params_path = run_dir / "prior_params.nc"
    if posterior_params_path.exists():
        posterior_params = xarray.open_dataset(posterior_params_path)
        prior_params = (
            xarray.open_dataset(prior_params_path)
            if prior_params_path.exists()
            else None
        )
        # Alternating shading of the blocks each window FINALIZED (window ``w``
        # fixes the knots in ``[w*s*sim_time, (w+1)*s*sim_time)`` seconds on
        # leaving -- however many knots that is, since the knot grid is set by
        # time.seconds_per_knot and need not be the cycle grid), which is the
        # only window structure that is unambiguous on the trajectory's time
        # axis -- the window spans themselves overlap by ``L - s`` cycles and
        # would shade over each other. ``None`` for a single window, whose knots
        # were all finalized at once.
        finalized_edges = (
            [
                min(w * layout.window_shift, layout.total_cycles) * sim_time
                for w in range(layout.num_windows)
            ]
            + [layout.total_cycles * sim_time]
            if layout.num_windows > 1
            else None
        )
        plot_rollout_time_evolution(
            esmda_params=posterior_params,
            true_params=true_params,
            esmda_state=None,
            true_state=None,
            output_path=run_dir / "parameter_evolution.png",
            prior_params=prior_params,
            window_edges=finalized_edges,
            # The RMSE panel is on its own index axis ("Time step"), and those
            # steps are the cycles listed above -- not necessarily 0..L-1.
            rmse=rmse,
        )
        plot_parameter_error(
            esmda_params=posterior_params,
            true_params=true_params,
            output_path=run_dir / "parameter_error.png",
            window_edges=finalized_edges,
        )
        marginals = run_dir / "parameter_marginals.png"
        _note_skipped(
            marginals,
            plot_parameter_marginals(
                posterior_params=posterior_params,
                true_params=true_params,
                output_path=marginals,
                prior_params=prior_params,
            ),
        )
        posterior_params.close()
        if prior_params is not None:
            prior_params.close()
    else:
        print(f"No posterior_params.nc in {run_dir}; skipping the parameter figures")

    # --- Outer ESMDA convergence ---------------------------------------------
    # The one figure with no filtering counterpart: what the outer loop did per
    # iteration, per window. Its inputs are the diagnostics YAMLs the runner
    # always writes, so it needs neither the truth nor the metric stage.
    per_window, window_labels = _convergence_series(run_dir)
    if per_window:
        convergence = run_dir / "iteration_convergence.png"
        _note_skipped(
            convergence,
            plot_iteration_convergence(
                per_window, convergence, window_labels=window_labels
            ),
        )
    else:
        print(
            f"No iteration_diagnostics.yaml in {run_dir}; skipping "
            "iteration_convergence.png"
        )

    # --- Field figures ------------------------------------------------------
    # Analyzed ensemble mean/std of |U| over the cycles (z-plane only), animated
    # against the truth's end-of-cycle frames.
    analyzed_plane = select_z_plane(analyzed_states, z_level=0).load()
    mean_vel, std_vel = ensemble_velocity_mean_std(analyzed_plane)
    truth_end_plane = select_z_plane(truth_end, z_level=0)
    animate_rollout_state(
        true_state=truth_end_plane,
        mean_vel=mean_vel,
        std_vel=std_vel,
        output_path=run_dir / "rollout_animation.mp4",
        z_level=0,
    )

    # Final analyzed state (last cycle of the last window) with the sensor
    # locations, sharing the colour scale with the truth's final end-of-cycle
    # frame.
    final_mean = mean_vel.isel(time=-1)
    final_std = std_vel.isel(time=-1)
    true_final = add_velocity_magnitude(truth_end_plane.isel(time=-1))["vel_magnitude"]
    plot_final_state_with_obs(
        mean_vel=final_mean,
        std_vel=final_std,
        output_path=run_dir / "final_state_with_obs.png",
        true_vel=true_final,
        obs_x=obs_x,
        obs_y=obs_y,
        z_level=0,
    )

    # --- Sensor time series (truth vs ensemble |U| at each sensor set) -------
    # Which per-cycle ensemble states are extracted -- every frame of every
    # member's forecast segment when the run saved them, one analyzed frame per
    # cycle otherwise -- and the truth frames they are paired against are decided
    # ONCE, by the filtering pipeline's contract on ``window_ta``; nothing below
    # branches on the kind again except where a LABEL would otherwise lie.
    sensor_sets = build_sensor_sets(cfg)
    source = cycle_state_source(run_dir, window_ta)
    print(f"Cycle states: {source.description}")
    truth_series = truth_cycle_statistics_series(window_ta, source, sensor_sets)
    ensemble_series = cycle_sensor_series(run_dir, window_ta, source, sensor_sets)

    fan_truth: dict[str, np.ndarray] = {}
    fan_ensemble: dict[str, np.ndarray] = {}
    fan_times: np.ndarray | None = None
    shared_time_axis = True
    for name, (sx, sy, sz) in sensor_sets.items():
        true_magnitude = sensor_magnitude(truth_series[name])
        ensemble_magnitude = sensor_magnitude(ensemble_series[name])
        # ``compute_sensor_metrics`` is the library's own aligner -- it
        # interpolates the truth onto the ensemble's time axis per sensor, so
        # the fan and the time-series figure draw the identical truth curve on
        # the identical axis rather than two near-copies of it.
        aligned = compute_sensor_metrics(true_magnitude, ensemble_magnitude)
        plot_sensor_timeseries(
            true_sensor=true_magnitude,
            ensemble_sensor=ensemble_magnitude,
            output_path=run_dir / f"sensor_timeseries_{name}.png",
            title=f"State at {name} sensors",
            sensor_x=sx,
            sensor_y=sy,
            sensor_z=sz,
        )
        fan_truth[name] = aligned["truth"]
        fan_ensemble[name] = aligned["members"]
        # S5 draws every sensor set against ONE x-axis, so the sets have to
        # agree on it. This only guards the case where they stop agreeing:
        # passing the last set's clock for all of them would silently slide the
        # other columns.
        if fan_times is None:
            fan_times = aligned["time"]
        elif fan_times.shape != aligned["time"].shape or not np.allclose(
            fan_times, aligned["time"]
        ):
            shared_time_axis = False

    if not shared_time_axis:
        print(
            "Sensor sets do not share one time axis; sensor_fans.png falls back "
            "to a frame index rather than drawing them against each other's clock"
        )
        fan_times = None

    # The fan's x-axis, and what it MEANS -- the same two-source reasoning as the
    # filtering stage, plus one thing only this pipeline has: the window offset.
    # ``window_ta`` re-based the window to open at t=0, so both sources' series
    # start there whatever window they came from. ``plot_sensor_fans`` labels the
    # axis "Time [s]", and on a moving-window run the honest seconds are the
    # HORIZON's, so the window's own offset is added back here. Under
    # ``analysis`` the series carry an integer cycle index instead (one frame per
    # cycle, and state_history.nc records no physical time), so cycle ``k``'s
    # frame is converted to its end-of-segment time first, exactly as there.
    offset = float(layout.first_cycle) * sim_time
    cycle_edges: list[float] | None = None
    if (
        source.kind == "analysis"
        and fan_times is not None
        and fan_times.size == source.num_cycles
    ):
        fan_times = (np.arange(source.num_cycles, dtype=float) + 1.0) * sim_time
    elif source.kind == "analysis":
        fan_times = None
    if fan_times is not None:
        fan_times = np.asarray(fan_times, dtype=float) + offset
        if source.num_cycles > 1:
            cycle_edges = list(
                offset
                + np.linspace(0.0, sim_time * source.num_cycles, source.num_cycles + 1)
            )

    # --- Evaluation figures --------------------------------------------------
    # Each of these is skipped, with a printed line, when the artifact it reads
    # is absent, and a skip never costs the figures after it: ``eval_fields.nc``
    # and the ``sensor_statistics`` block are both written by
    # scripts/filter_smoothing/compute_filter_smoothing_metrics.py.
    #
    # Canopy height for S1's ``z/H`` axis and its roof line, when the case
    # carries one. Optional and absent from every shipped case today (the
    # geometry is an STL, not a config scalar), so this is simply not passed
    # when it is missing -- CLAUDE.md's no-op-when-absent rule. Note the path:
    # the ``case`` group is ``# @package _global_``, so its keys land at the
    # config root.
    building_height = OmegaConf.select(cfg, "geometry.building_height", default=None)

    eval_fields_path = run_dir / "eval_fields.nc"
    if eval_fields_path.exists():
        with xarray.open_dataset(eval_fields_path) as fields:
            # Which frames the moments were reduced over, recorded by the metric
            # stage that reduced them. Read off the FILE rather than recomputed
            # from ``source``: the two can disagree (a run whose forecasts were
            # pruned after the metrics ran), and the honest label is the one
            # describing the numbers actually in the file.
            moment_sampling = fields.attrs.get("moment_sampling")
            sampling_note = None if moment_sampling is None else str(moment_sampling)
            sampling_is_sparse = bool(
                int(fields.attrs.get("moment_sampling_is_sparse", 0) or 0)
            )
            profiles = run_dir / "station_profiles.png"
            _note_skipped(
                profiles,
                plot_station_profiles(
                    fields,
                    profiles,
                    u_ref=_reference_velocity(true_params),
                    building_height=(
                        None if building_height is None else float(building_height)
                    ),
                    sampling_note=sampling_note,
                    sampling_is_sparse=sampling_is_sparse,
                ),
            )
            slices = run_dir / "mean_slices.png"
            _note_skipped(
                slices,
                plot_mean_slices(
                    fields,
                    slices,
                    sampling_note=sampling_note,
                    sampling_is_sparse=sampling_is_sparse,
                ),
            )
    else:
        print(
            f"No eval_fields.nc in {run_dir}; skipping station_profiles.png and "
            "mean_slices.png (re-run "
            "scripts/filter_smoothing/compute_filter_smoothing_metrics.py on this "
            "run dir to write it)"
        )

    # The observation error the run actually assimilated with: this is the key
    # run_filter_smoothing.py builds C_D from, and it lives under
    # ``filter_smoothing``, not ``filtering`` or ``esmda``. ``None`` when the
    # saved config predates it, which S5 degrades on (no envelope) rather than
    # inventing a width.
    obs_error_std = OmegaConf.select(
        cfg, "filter_smoothing.obs_error_std", default=None
    )
    # What the fan draws is the other thing the state source decides: the
    # analysis source's series ARE the analyzed states, but the forecast
    # source's are every frame of each cycle's forecast segment -- the PRIOR
    # side of each analysis. Same figure, opposite side of the filter, so the
    # legend follows the source.
    fan_label = "Forecast" if source.kind == "forecast" else "Posterior"
    fans = run_dir / "sensor_fans.png"
    _note_skipped(
        fans,
        plot_sensor_fans(
            fan_truth,
            fan_ensemble,
            fans,
            times=fan_times,
            obs_error_std=None if obs_error_std is None else float(obs_error_std),
            window_edges=cycle_edges,
            fan_label=fan_label,
        ),
    )

    rank_counts = _rank_counts(read_yaml(run_dir / "run_summary.yaml"))
    if rank_counts:
        histogram = run_dir / "rank_histogram.png"
        _note_skipped(histogram, plot_rank_histogram(rank_counts, histogram))
    else:
        print(
            f"No sensor_statistics rank counts in {run_dir / 'run_summary.yaml'}; "
            "skipping rank_histogram.png (run "
            "scripts/filter_smoothing/compute_filter_smoothing_metrics.py on this "
            "run dir first)"
        )

    print(f"Saved figures in {run_dir}")


def main() -> None:
    # Every "why" behind a skipped figure is a ``logger.info`` inside
    # ``evaluation.figures`` (and inside the cycle-state contract); with no
    # handler on the root logger the operator gets the bare "Skipped x.png" line
    # and none of the reasons. Configured here, at the entry point, rather than
    # in ``make_figures`` -- importing this module (the tests do) must not
    # reconfigure anyone's logging.
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
    make_figures(args.run_dir)


if __name__ == "__main__":
    main()
