"""Draw the figures for a finished ESMDA run.

Third stage of the three-script single-run ESMDA pipeline (see
``scripts/run_esmda_pipeline.sh``):

  1. scripts/esmda/run_esmda.py            -- runs the assimilation, saves the raw
                                       artifacts + truth_access.yaml + run_info.yaml.
  2. scripts/esmda/compute_esmda_metrics.py -- reads those, writes run_summary.yaml.
  3. scripts/esmda/make_esmda_figures.py   (THIS) -- reads those, draws the figures.

Writes, into the run directory:

  * ``rollout_time_evolution.png`` -- parameter trajectories + state |U| RMSE.
  * ``parameter_error.png``        -- per-parameter posterior error over time.
  * ``rollout_animation.mp4``      -- ensemble-mean |U| field vs the truth.
  * ``final_state_with_obs.png``   -- final |U| field with the sensor locations.
  * ``sensor_timeseries_<set>.png`` -- truth vs ensemble |U| at each sensor set.
  * ``parameter_marginals.png``    -- prior vs posterior marginal per parameter.
  * ``station_profiles.png``       -- mean-velocity / TKE profiles at the sensor
                                      columns (needs ``eval_fields.nc``).
  * ``mean_slices.png``            -- time-mean field slices, truth vs prior vs
                                      posterior vs difference (same file).
  * ``sensor_fans.png``            -- sensor quantile fans with the observations.
  * ``probe_spectra.png``          -- premultiplied energy spectra at the probes
                                      (needs the high-rate probe records).
  * ``rank_histogram.png``         -- rank histogram of the window statistics
                                      (needs ``run_summary.yaml``).
  * ``data_mismatch_decay.png``    -- per-member O_N vs ESMDA iteration against
                                      the 1/2 target band (needs the per-window
                                      observation-space files).

Of the WP1.5 five, three read an artifact an old run dir may not have --
``station_profiles.png`` and ``mean_slices.png`` need ``eval_fields.nc``,
``rank_histogram.png`` needs ``run_summary.yaml``'s ``sensor_statistics`` block
-- and each degrades to a printed skip line when it is absent, so a run whose
metric stage predates those still gets every figure it can support. The other
two read artifacts every run dir has: the parameter datasets
(``parameter_marginals.png``) and the sensor series extracted above
(``sensor_fans.png``). ``probe_spectra.png`` (phase 3) is skipped the same way,
and is skipped on almost every run dir: it needs the high-rate probe records only
an explicit ``scripts/esmda/run_probe_series.py`` rerun writes.
``data_mismatch_decay.png`` (phase 2) likewise skips on any run dir written
before WP2.1 or with ``esmda.save_obs_diagnostics=false``. This script owns
all the run-dir layout, config and YAML knowledge; the figure functions in
``evaluation.figures`` are handed opened datasets and plain dicts (master-plan
invariant 5).

Honors ``run.skip_viz`` (set in the saved config): a no-op when true, mirroring
the old in-script behaviour.

Also runs on a **windowed filtering** run directory, which writes these same
artifacts in the ESMDA schema (``scripts/run_filtering_pipeline.sh`` points it
at ``<run dir>/esmda_view/`` so the two families' same-named outputs never
collide). Everything that differs there is read defensively rather than
branched on: a missing ``run.skip_viz`` / ``is_dynamic``, the observation-error
scalar under its own config key, and the D3 step-axis label (see
``_d3_step_label``).

Usage::

    python scripts/esmda/make_esmda_figures.py --run-dir <esmda output dir>
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
    plot_data_mismatch_decay,
    plot_final_state_with_obs,
    plot_mean_slices,
    plot_parameter_error,
    plot_parameter_marginals,
    plot_rank_histogram,
    plot_rollout_time_evolution,
    plot_sensor_fans,
    plot_sensor_timeseries,
    plot_spectra,
    plot_station_profiles,
)
from evaluation.scores import compute_sensor_metrics
from evaluation.sensors import sensor_magnitude
from evaluation.turbulence import select_z_plane, streaming_state_rmse

from pyurbanair.config.hydra_helpers import create_observation_points
from pyurbanair.utils.animation_utils import animate_rollout_state
from pyurbanair.utils.run_utils import add_velocity_magnitude
from scripts.esmda._esmda_common import (
    analyzed_truth_frames,
    build_sensor_sets,
    ensemble_sensor_series,
    load_run_config,
    obs_diagnostics_bundle,
    open_truth,
    probe_spectra_bundle,
    read_yaml,
    truth_sensor_series,
)


def _note_skipped(output_path: pathlib.Path, written: pathlib.Path | None) -> None:
    """Say so when a figure no-oped on inputs that were present but unusable.

    The WP1.5 figures return the path they wrote, or ``None`` when their inputs
    were empty or degenerate (a 2-member smoke ensemble has no meaningful KDE,
    a run with no sensors has no station columns). The reason is logged by the
    figure itself; this only makes the *absence* visible in the stage's own
    print-based output, so a missing PNG is never a silent one.
    """
    if written is None:
        print(f"Skipped {output_path.name}: its inputs were absent or degenerate")


def _reference_velocity(true_params: xarray.Dataset) -> float | None:
    """The truth's inflow speed, for normalizing the S1 profiles by ``U_ref``.

    ``velocity_magnitude`` is the run's free-stream inflow parameter (see
    ``conf/params/static.yaml``), so the truth's value is the natural reference;
    a time-varying truth is averaged over its knots, since one profile figure
    covers the whole run. ``None`` -- which the figure reads as "plot in m/s" --
    when the case does not carry that parameter at all (a pressure-gradient
    driven run) or when its value is not a usable positive number.
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
    (and, under ``run.save_prior_state``, a ``prior``) mapping of statistic name
    to its scored entry, of which figure D1 wants only ``rank_counts``. Anything
    that is not shaped like that -- an older summary whose sets carry no halves,
    a half whose entries predate the rank counts -- is dropped rather than
    guessed at, and an empty result is the caller's cue to skip the figure.
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


def _d3_step_label(run_dir: pathlib.Path) -> str:
    """What D3's step axis counts on this run, out of ``run_info.yaml``.

    The observation-space files are written by the ESMDA runner and by the
    windowed sequential filter, and their ``esmda_step`` axis is positional in
    both -- prior first, posterior last -- which is why every reader of it is
    length-agnostic. What differs is the meaning of a step: ESMDA's are the
    iterations of one tempered solve, while the filter's two rows are the prior
    and posterior of its per-cycle analyses, pooled over the window's cycles. So
    the axis is labelled from the runner that wrote the dir (``configuration``
    carries ``smoother`` for ESMDA and ``filter`` for the filter), and anything
    unrecognised keeps the ESMDA label -- this stage's own default.
    """
    configuration = read_yaml(run_dir / "run_info.yaml").get("configuration") or {}
    if configuration.get("filter") and not configuration.get("smoother"):
        return "filter analysis"
    return "ESMDA iteration"


def make_figures(run_dir: pathlib.Path) -> None:
    """Draw every figure from the saved artifacts in ``run_dir``."""
    cfg = load_run_config(run_dir)
    # ``select``, not ``cfg.run.skip_viz``: this stage also runs on the
    # ESMDA-schema artifacts a windowed FILTERING run writes, and a filtering
    # config saved before that knob was added to conf/run_filtering.yaml carries
    # no ``run.skip_viz``. Absent means "draw them", which is what such a run dir
    # wants; an ESMDA config always has the key, so this is a no-op for them.
    if bool(OmegaConf.select(cfg, "run.skip_viz", default=False)):
        print(f"run.skip_viz is set; no figures generated for {run_dir}")
        return

    ta = read_yaml(run_dir / "truth_access.yaml")
    num_windows = int(ta["num_windows"])
    sim_time = float(ta["sim_time"])
    # Defaulted rather than indexed for the same reason: every run dir the ESMDA
    # runner wrote records it, and a run dir that does not gets the conservative
    # reading (no window shading on the parameter plots) instead of a KeyError
    # that would cost every figure below.
    is_dynamic = bool(ta.get("is_dynamic", False))

    posterior_params = xarray.open_dataset(run_dir / "posterior_params.nc")
    prior_params = xarray.open_dataset(run_dir / "prior_params.nc")
    true_params = xarray.open_dataset(run_dir / "true_params.nc")
    posterior_state = xarray.open_dataset(run_dir / "posterior_state_mean.nc")

    # Open the (potentially multi-GB) truth lazily. The plots below each pull
    # only the slice they need: a single z-plane for the animation/final state
    # and a few z-slices for the streamed error curve. Both of those pair the
    # truth with ``posterior_state_mean.nc`` FRAME BY FRAME, so the truth is
    # reduced to the frames that file holds -- a no-op unless the run dir says
    # its state frames are strided (``analyzed_truth_frames``). The sensor
    # series below re-open the truth in full: they align on the physical time
    # axis, not on frame index.
    true_state = analyzed_truth_frames(
        open_truth(
            ta["true_state_path"],
            ta["n_total"],
            ta["x_offset"],
            ta["start_idx"],
            ta["t_offset"],
        ),
        ta,
    )
    obs_x, obs_y, obs_z = create_observation_points(cfg.obs)

    # Truth reduced to the z=0 horizontal plane (single layer kept), so the
    # animation and final-state plot never load the full 3-D velocity field.
    true_state_plane = select_z_plane(true_state, z_level=0)
    true_vel = add_velocity_magnitude(true_state_plane)["vel_magnitude"]

    # State |U| RMSE series for the rollout plot (streamed over a few z-slices).
    rmse = streaming_state_rmse(true_state, posterior_state)

    # Boundaries between assimilation windows on the (rebased) global time axis,
    # used to lightly shade alternating windows in the parameter plot. Gated on
    # ``is_dynamic`` because a STATIC parameter's x-axis is a window index, not a
    # time -- ``_concat_windows`` only rebases onto ``w * sim_time`` when the
    # parameters are dynamic -- so shading it at ``w * sim_time`` would mark the
    # wrong places.
    window_edges = (
        list(np.linspace(0.0, sim_time * num_windows, num_windows + 1))
        if is_dynamic and num_windows > 1
        else None
    )
    # S5's x-axis is physical time on EVERY run: ``ensemble_sensor_series``
    # rebases window ``w`` onto ``w * sim_time`` regardless of whether the
    # parameters are dynamic, so a static multi-window run (conf/run_esmda.yaml
    # documents one) has real window boundaries to mark even though its
    # parameter plots get none. Hence a second, dynamics-independent edge list.
    sensor_window_edges = (
        list(np.linspace(0.0, sim_time * num_windows, num_windows + 1))
        if num_windows > 1
        else None
    )

    plot_rollout_time_evolution(
        esmda_params=posterior_params,
        true_params=true_params,
        esmda_state=None,
        true_state=None,
        output_path=run_dir / "rollout_time_evolution.png",
        prior_params=prior_params,
        window_edges=window_edges,
        rmse=rmse,
    )
    plot_parameter_error(
        esmda_params=posterior_params,
        true_params=true_params,
        output_path=run_dir / "parameter_error.png",
        window_edges=window_edges,
    )
    animate_rollout_state(
        true_state=true_state_plane,
        mean_vel=posterior_state["vel_mean"],
        std_vel=posterior_state["vel_std"],
        output_path=run_dir / "rollout_animation.mp4",
        z_level=0,
    )
    plot_final_state_with_obs(
        mean_vel=posterior_state["vel_mean"],
        std_vel=posterior_state["vel_std"],
        output_path=run_dir / "final_state_with_obs.png",
        true_vel=true_vel,
        obs_x=obs_x,
        obs_y=obs_y,
        z_level=0,
    )

    # Sensor time series: true vs ensemble |U| at the assimilation sensors and at
    # a held-out validation set. The truth and ensemble are interpolated at the
    # same physical points on their own grids, so the figures are grid-independent.
    # Extracted one window at a time to keep the truth/full ensemble out of memory.
    sensor_sets = build_sensor_sets(cfg)
    truth_series = truth_sensor_series(
        ta["true_state_path"],
        ta["n_total"],
        ta["x_offset"],
        ta["start_idx"],
        ta["t_offset"],
        sensor_sets,
        ta["truth_solver_name"],
        num_windows,
        int(ta["n_per_window"]),
    )
    state_paths = [
        run_dir / "windows" / f"window_{w}_posterior_state.nc"
        for w in range(num_windows)
    ]
    ensemble_series = ensemble_sensor_series(
        state_paths,
        sensor_sets,
        ta["assim_solver_name"],
        sim_time,
    )
    # The |U| series the S5 fans draw, collected here rather than re-extracted:
    # this is the only pass over the window state files the figure stage makes
    # (master-plan invariant 2).
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
        # from its two DataArray arguments (a WP0.2 signature this WP does not
        # widen), so the call runs twice per sensor set. Measured at 0.06 s on a
        # real run -- not worth a signature change to remove.
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
        # agree on it. They do today -- all of them are extracted from the same
        # window state files, so they share the assimilation grid's frames -- and
        # this only guards the case where they stop agreeing: passing the last
        # set's clock for all of them would silently slide the other columns.
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

    # --- WP1.5 evaluation figures --------------------------------------------
    # Each of these is skipped, with a printed line, when the artifact it reads
    # is absent, and a skip never costs the figures after it (master-plan
    # invariant 3): ``eval_fields.nc`` only exists for runs whose metric stage
    # ran post-WP1.4, the ``sensor_statistics`` block only post-WP1.3, and the
    # prior halves of both only under ``run.save_prior_state``. All of the layout
    # and config knowledge lives here -- the figures are handed opened datasets,
    # numpy arrays and plain dicts, so ``libs/evaluation`` stays a leaf.
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

    # Canopy height for S1's ``z/H`` axis and its roof line, when the case
    # carries one. Optional and absent from every shipped case today (the
    # geometry is an STL, not a config scalar), so this is read the same way as
    # ``esmda.obs_error_std`` and simply not passed when it is missing --
    # CLAUDE.md's no-op-when-absent rule. Note the path: the ``case`` group is
    # ``# @package _global_``, so its keys land at the config root and the
    # canopy scalar belongs beside the rest of the geometry, not under a
    # ``case:`` node (there is none).
    building_height = OmegaConf.select(cfg, "geometry.building_height", default=None)

    eval_fields_path = run_dir / "eval_fields.nc"
    if eval_fields_path.exists():
        # The averaging window, the stride, the fluid mask and the station
        # labels all travel inside this file, so neither figure reopens anything
        # else -- and neither re-streams the window states the file exists to
        # replace.
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
            "mean_slices.png (re-run scripts/esmda/compute_esmda_metrics.py on "
            "this run dir to write it)"
        )

    # The observation error the run actually assimilated with: this is the key
    # run_esmda.py builds C_D from. ``None`` when the saved config predates it,
    # which S5 degrades on (no envelope) rather than inventing a width. The
    # realized noisy observations are not persisted before WP2.1, so the band
    # S5 draws is the clean truth +/- sigma_o and the figure says so itself.
    #
    # Two keys, one number: the same scalar lives under whichever entry point
    # wrote the run dir, and this stage runs on both (a windowed filtering run
    # writes these artifacts too). The order is the config's own -- an ESMDA run
    # has only the first -- so nothing changes for them, and a filtering run
    # gets the envelope rather than a bare fan.
    # ``make_filtering_figures.py`` already reads its own key this way.
    obs_error_std = next(
        (
            value
            for key in (
                "esmda.obs_error_std",
                "filtering.obs_error_std",
            )
            if (value := OmegaConf.select(cfg, key, default=None)) is not None
        ),
        None,
    )
    fans = run_dir / "sensor_fans.png"
    _note_skipped(
        fans,
        plot_sensor_fans(
            fan_truth,
            fan_ensemble,
            fans,
            times=fan_times,
            obs_error_std=None if obs_error_std is None else float(obs_error_std),
            window_edges=sensor_window_edges,
        ),
    )

    # S4 needs the high-rate probe records an explicit ``run_probe_series.py``
    # rerun writes; no assimilation produces them, so absence is the normal case
    # and costs this figure alone. The bundle is the same reduction the metric
    # stage scores, so the LSD annotated on a panel is the number in
    # ``run_summary.yaml`` before its median over sensors.
    spectra = probe_spectra_bundle(run_dir)
    if spectra is None:
        # Printed, not only logged: this stage's contract is that every skipped
        # figure says so on stdout, and the *reason* is the log line the bundle
        # already emitted.
        print(
            f"No usable probe records in {run_dir}; skipping probe_spectra.png "
            "(run scripts/esmda/run_probe_series.py on this run dir first)"
        )
    else:
        spectrum_figure = run_dir / "probe_spectra.png"
        _note_skipped(
            spectrum_figure,
            plot_spectra(
                spectra["f"],
                spectra["truth"],
                spectra["posterior"],
                spectrum_figure,
                prior=spectra["prior"],
                truth_halves=spectra["truth_halves"],
                variance=spectra["variance"],
                f_cutoff=spectra["f_cutoff"],
                sensor_sets=spectra["sensor_sets"],
            ),
        )

    # D3 reads the same bundle the metric stage scores, so the boxes and
    # ``run_summary.yaml``'s ``esmda_diagnostics`` come off one reduction. Absent
    # on any run dir written before WP2.1 or with
    # ``esmda.save_obs_diagnostics=false`` -- the bundle logs the reason.
    mismatch = obs_diagnostics_bundle(run_dir)
    if mismatch is None:
        print(
            f"No observation-space files in {run_dir}; skipping "
            "data_mismatch_decay.png (rerun the assimilation with "
            "esmda.save_obs_diagnostics=true to write them)"
        )
    else:
        decay = run_dir / "data_mismatch_decay.png"
        _note_skipped(
            decay,
            plot_data_mismatch_decay(
                mismatch["per_window"],
                decay,
                num_observations=mismatch["num_observations"],
                window_indices=mismatch["window_indices"],
                step_label=_d3_step_label(run_dir),
            ),
        )

    rank_counts = _rank_counts(read_yaml(run_dir / "run_summary.yaml"))
    if rank_counts:
        histogram = run_dir / "rank_histogram.png"
        _note_skipped(histogram, plot_rank_histogram(rank_counts, histogram))
    else:
        print(
            f"No sensor_statistics rank counts in {run_dir / 'run_summary.yaml'}; "
            "skipping rank_histogram.png (run scripts/esmda/compute_esmda_metrics.py "
            "on this run dir first)"
        )

    print(f"Saved figures in {run_dir}")


def main() -> None:
    # Every "why" behind a skipped figure is a ``logger.info`` inside
    # ``evaluation.figures``; with no handler on the root logger the operator
    # gets the bare "Skipped x.png" line and none of the reasons. Configured
    # here, at the entry point, rather than in ``make_figures`` -- importing
    # this module (the tests do) must not reconfigure anyone's logging.
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
        help="The ESMDA run output directory written by scripts/esmda/run_esmda.py.",
    )
    args = ap.parse_args()
    if not args.run_dir.exists():
        raise SystemExit(f"run dir not found: {args.run_dir}")
    make_figures(args.run_dir)


if __name__ == "__main__":
    main()
