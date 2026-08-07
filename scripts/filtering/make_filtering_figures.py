"""Draw the figures for a finished filtering run.

Third stage of the three-script single-run filtering pipeline (see
``scripts/run_filtering_pipeline.sh``), mirroring make_esmda_figures.py:

  1. scripts/filtering/run_filtering.py            -- runs the filter, saves the
                                       raw artifacts + truth_access.yaml + run_info.yaml.
  2. scripts/filtering/compute_filtering_metrics.py -- reads those, writes run_summary.yaml.
  3. scripts/filtering/make_filtering_figures.py   (THIS) -- reads those, draws the figures.

Writes, into the run directory:

  * ``parameter_evolution.png``    -- parameter trajectories over cycles (prior ->
                                      each cycle's posterior) + per-cycle |U| RMSE.
  * ``parameter_error.png``        -- per-parameter posterior error over cycles.
  * ``rollout_animation.mp4``      -- analyzed ensemble-mean |U| field vs the truth,
                                      one frame per cycle.
  * ``final_state_with_obs.png``   -- final analyzed |U| field with the sensors.
  * ``sensor_timeseries_<set>.png`` -- truth vs ensemble |U| at each sensor set.
  * ``parameter_marginals.png``    -- prior vs posterior marginal per parameter.
  * ``station_profiles.png``       -- mean-velocity / TKE profiles at the sensor
                                      columns (needs ``eval_fields.nc``).
  * ``mean_slices.png``            -- time-mean field slices, truth vs prior vs
                                      posterior vs difference (same file).
  * ``sensor_fans.png``            -- sensor quantile fans with the observations.
  * ``rank_histogram.png``         -- rank histogram of the cycle statistics
                                      (needs ``run_summary.yaml``).

The parameter figures are skipped in ``mode='state'`` (no parameters estimated).
The time axis throughout is the cycle (see scripts/filtering/_filtering_common.py).

Of the last five (the WP1.5 evaluation figures, mirroring the ESMDA stage's),
three read an artifact an old run dir may not have -- ``station_profiles.png``
and ``mean_slices.png`` need ``eval_fields.nc`` and ``rank_histogram.png`` needs
``run_summary.yaml``'s ``sensor_statistics`` block, all three written by
``scripts/filtering/compute_filtering_metrics.py`` -- and each degrades to a
printed skip line when it is absent, so a run whose metric stage predates them
still gets every figure it can support. A skip never costs the figures after it.
The other two read artifacts every run dir has: the parameter datasets
(``parameter_marginals.png``, itself skipped in ``mode='state'``) and the sensor
series extracted here (``sensor_fans.png``). This script owns all the run-dir
layout, config and YAML knowledge; the figure functions in ``evaluation.figures``
are handed opened datasets and plain dicts (master-plan invariant 5).

The ESMDA suite's ``probe_spectra.png`` has no filtering counterpart today: it
needs the high-rate probe records only a dedicated solver rerun script writes,
and the filtering pipeline has no such script.

Usage::

    python scripts/filtering/make_filtering_figures.py --run-dir <filtering output dir>
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
from scripts.filtering._filtering_common import (
    build_sensor_sets,
    cycle_sensor_series,
    cycle_state_source,
    ensemble_velocity_mean_std,
    load_analyzed_states,
    load_params_history,
    load_run_config,
    read_yaml,
    truth_cycle_statistics_series,
    truth_end_of_cycle,
)


def _note_skipped(output_path: pathlib.Path, written: pathlib.Path | None) -> None:
    """Say so when a figure no-oped on inputs that were present but unusable.

    The WP1.5 figures return the path they wrote, or ``None`` when their inputs
    were empty or degenerate (a 2-member smoke ensemble has no meaningful KDE, a
    ``mode='state'`` run has no parameter to draw a marginal of). The reason is
    logged by the figure itself; this only makes the *absence* visible in the
    stage's own print-based output, so a missing PNG is never a silent one.
    """
    if written is None:
        print(f"Skipped {output_path.name}: its inputs were absent or degenerate")


def _reference_velocity(true_params: xarray.Dataset) -> float | None:
    """The truth's inflow speed, for normalizing the S1 profiles by ``U_ref``.

    ``velocity_magnitude`` is the run's free-stream inflow parameter (see
    ``conf/params/static.yaml``), so the truth's value is the natural reference;
    a truth carrying several knots is averaged over them, since one profile
    figure covers the whole run. ``None`` -- which the figure reads as "plot in
    m/s" -- when the case does not carry that parameter at all (a
    pressure-gradient driven run) or when its value is not a usable positive
    number.
    """
    if "velocity_magnitude" not in true_params:
        return None
    values = np.asarray(true_params["velocity_magnitude"].values, dtype=float)
    # Filtered before the mean rather than with ``nanmean``: an all-non-finite
    # parameter is a documented null path here, not something to warn about.
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    u_ref = float(finite.mean())
    return u_ref if u_ref > 0.0 else None


def _rank_counts(summary: dict) -> dict:
    """``{set: {half: {statistic: counts}}}`` out of ``run_summary.yaml``.

    The YAML shape is this script's business, not the figure's (invariant 5):
    ``sensor_statistics[set]`` holds the ensemble sizes beside a ``posterior``
    (and, where the run scored one, a ``prior``) mapping of statistic name to its
    scored entry, of which figure D1 wants only ``rank_counts``. Anything that is
    not shaped like that -- an older summary whose sets carry no halves, a half
    whose entries predate the rank counts -- is dropped rather than guessed at,
    and an empty result is the caller's cue to skip the figure.
    """
    counts: dict = {}
    for name, block in (summary.get("sensor_statistics") or {}).items():
        if not isinstance(block, dict):
            continue
        halves = {}
        for half in ("prior", "posterior"):
            entries = block.get(half)
            if not isinstance(entries, dict):
                continue
            per_statistic = {
                statistic: entry["rank_counts"]
                for statistic, entry in entries.items()
                if isinstance(entry, dict) and entry.get("rank_counts")
            }
            if per_statistic:
                halves[half] = per_statistic
        if halves:
            counts[name] = halves
    return counts


def make_figures(run_dir: pathlib.Path) -> None:
    """Draw every figure from the saved artifacts in ``run_dir``."""
    cfg = load_run_config(run_dir)
    ta = read_yaml(run_dir / "truth_access.yaml")
    # The physical length of one forecast segment, i.e. one cycle. Only the S5
    # cycle boundaries need it, and only on a source whose series carry seconds.
    sim_time = float(ta["sim_time"])

    true_params = xarray.open_dataset(run_dir / "true_params.nc")

    # The analyzed end-of-cycle ensemble states (time = cycle axis) and the
    # truth's matching end-of-cycle frames. Opened lazily; the plots pull only
    # the z-plane / z-slices they need.
    analyzed_states = load_analyzed_states(run_dir, ta)
    n_frames = int(analyzed_states.sizes["time"])
    truth_end = truth_end_of_cycle(ta, n_frames)
    obs_x, obs_y, obs_z = create_observation_points(cfg.obs)

    # Per-cycle state |U| RMSE (streamed over a few z-slices).
    rmse = streaming_state_rmse(truth_end, analyzed_states)

    # --- Parameter figures (present unless mode='state') --------------------
    history_path = run_dir / "params_history.nc"
    if history_path.exists():
        # params_history holds the prior as its first cycle entry followed by
        # each cycle's posterior, so it reads as the full prior->posterior
        # trajectory on its own (no separate prior overlay needed). Loaded on a
        # physical ``time`` axis so a drifting (time-varying) truth aligns with
        # each cycle's end-of-segment time.
        params_history = load_params_history(run_dir, ta)
        plot_rollout_time_evolution(
            esmda_params=params_history,
            true_params=true_params,
            esmda_state=None,
            true_state=None,
            output_path=run_dir / "parameter_evolution.png",
            prior_params=None,
            window_edges=None,
            rmse=rmse,
        )
        plot_parameter_error(
            esmda_params=params_history,
            true_params=true_params,
            output_path=run_dir / "parameter_error.png",
            window_edges=None,
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

    # Final analyzed state (last cycle) with the sensor locations, sharing the
    # colour scale with the truth's final end-of-cycle frame.
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
    # Truth and ensemble are interpolated at the same physical points on their
    # own grids, so the figures are grid-independent. Which per-cycle ensemble
    # states are extracted -- every frame of every member's forecast segment
    # when the run saved them, one analyzed frame per cycle otherwise -- and the
    # truth frames they are paired against are decided ONCE, by the contract in
    # _filtering_common; nothing below branches on the kind again except where
    # an axis label would otherwise lie (see the fan edges).
    sensor_sets = build_sensor_sets(cfg)
    source = cycle_state_source(run_dir, ta)
    print(f"Cycle states: {source.description}")
    truth_series = truth_cycle_statistics_series(ta, source, sensor_sets)
    # Extracted one member at a time by the contract, so a full-ensemble state
    # file is never held whole -- and this is the figure stage's ONLY pass over
    # them, which is why the fan series below are collected inside the same loop
    # rather than re-extracted (master-plan invariant 2).
    ensemble_series = cycle_sensor_series(run_dir, ta, source, sensor_sets)

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
        # ``plot_sensor_timeseries`` recomputes the same alignment internally
        # from its two DataArray arguments, so the call runs twice per sensor
        # set; it is cheap enough not to be worth widening that signature for.
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
        # agree on it. They do today -- all of them come out of the same cycle
        # state files, so they share the assimilation grid's frames -- and this
        # only guards the case where they stop agreeing: passing the last set's
        # clock for all of them would silently slide the other columns.
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

    # The one place the state source has to be re-read, because it decides what
    # the fan's x-axis MEANS. Under ``forecast`` the series carry real seconds on
    # the run's global clock, so the axis is a time and the cycle boundaries sit
    # at ``k * sim_time``. Under ``analysis`` they carry an integer CYCLE INDEX
    # instead -- one frame per cycle, and ``state_history.nc`` does not record
    # the analyzed frames' physical times -- while ``plot_sensor_fans`` labels
    # any axis it is handed "Time [s]". Handing it the raw indices would print
    # seconds that are not seconds.
    #
    # But those indices are not unlabelled frames either: the filter analyzes the
    # END of each forecast segment, so cycle ``k``'s analyzed frame is the state
    # at ``(k+1) * sim_time``. Rebasing onto that is what ``load_params_history``
    # already does for the parameter axis in this same pipeline, and it makes the
    # "Time [s]" label true, puts the two sources on one physical clock, and
    # leaves the cycle boundaries meaning what they mean everywhere else. So the
    # index is converted rather than discarded -- and only falls back to the
    # figure's own frame axis if the sets disagreed on a clock (``fan_times`` is
    # already None) or the axis is not the one frame per cycle assumed here.
    cycle_edges: list[float] | None = None
    if (
        source.kind == "analysis"
        and fan_times is not None
        and fan_times.size == source.num_cycles
    ):
        fan_times = (np.arange(source.num_cycles, dtype=float) + 1.0) * sim_time
    elif source.kind == "analysis":
        fan_times = None
    if fan_times is not None and source.num_cycles > 1:
        cycle_edges = list(
            np.linspace(0.0, sim_time * source.num_cycles, source.num_cycles + 1)
        )

    # --- WP1.5 evaluation figures --------------------------------------------
    # Each of these is skipped, with a printed line, when the artifact it reads
    # is absent, and a skip never costs the figures after it (master-plan
    # invariant 3): ``eval_fields.nc`` and the ``sensor_statistics`` block are
    # both written by scripts/filtering/compute_filtering_metrics.py, and the
    # parameter datasets do not exist at all in ``mode='state'``. All of the
    # layout and config knowledge lives here -- the figures are handed opened
    # datasets, numpy arrays and plain dicts, so ``libs/evaluation`` stays a leaf.
    posterior_params_path = run_dir / "posterior_params.nc"
    if posterior_params_path.exists():
        # Same gate as the parameter figures above: ``mode='state'`` estimates no
        # parameters, so it writes no posterior to draw a marginal of. The prior
        # is optional in the same spirit -- the figure drops that half of each
        # panel rather than failing when it is missing.
        prior_params_path = run_dir / "prior_params.nc"
        with xarray.open_dataset(posterior_params_path) as posterior_params:
            prior_params = (
                xarray.open_dataset(prior_params_path)
                if prior_params_path.exists()
                else None
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
            if prior_params is not None:
                prior_params.close()
    else:
        print(
            f"No posterior_params.nc in {run_dir}; skipping parameter_marginals.png "
            "(filtering.mode=state estimates no parameters)"
        )

    # Canopy height for S1's ``z/H`` axis and its roof line, when the case
    # carries one. Optional and absent from every shipped case today (the
    # geometry is an STL, not a config scalar), so this is read the same way as
    # ``filtering.obs_error_std`` and simply not passed when it is missing --
    # CLAUDE.md's no-op-when-absent rule. Note the path: the ``case`` group is
    # ``# @package _global_``, so its keys land at the config root and the canopy
    # scalar belongs beside the rest of the geometry, not under a ``case:`` node.
    building_height = OmegaConf.select(cfg, "geometry.building_height", default=None)

    eval_fields_path = run_dir / "eval_fields.nc"
    if eval_fields_path.exists():
        # The averaging window, the stride, the fluid mask and the station labels
        # all travel inside this file, so neither figure reopens anything else --
        # and neither re-streams the cycle states the file exists to replace.
        with xarray.open_dataset(eval_fields_path) as fields:
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
                ),
            )
            slices = run_dir / "mean_slices.png"
            _note_skipped(slices, plot_mean_slices(fields, slices))
    else:
        print(
            f"No eval_fields.nc in {run_dir}; skipping station_profiles.png and "
            "mean_slices.png (re-run scripts/filtering/compute_filtering_metrics.py "
            "on this run dir to write it)"
        )

    # The observation error the run actually assimilated with: this is the key
    # run_filtering.py builds C_D from, and it lives under ``filtering``, not
    # ``esmda``. ``None`` when the saved config predates it, which S5 degrades on
    # (no envelope) rather than inventing a width. The realized noisy
    # observations are not persisted, so the band S5 draws is the clean truth
    # +/- sigma_o and the figure says so itself.
    obs_error_std = OmegaConf.select(cfg, "filtering.obs_error_std", default=None)
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
            "scripts/filtering/compute_filtering_metrics.py on this run dir first)"
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
        help="The filtering run output directory written by scripts/filtering/run_filtering.py.",
    )
    args = ap.parse_args()
    if not args.run_dir.exists():
        raise SystemExit(f"run dir not found: {args.run_dir}")
    make_figures(args.run_dir)


if __name__ == "__main__":
    main()
