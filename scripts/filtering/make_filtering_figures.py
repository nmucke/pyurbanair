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

The parameter figures are skipped in ``mode='state'`` (no parameters estimated).
The time axis throughout is the cycle (see scripts/filtering/_filtering_common.py).

Usage::

    python scripts/filtering/make_filtering_figures.py --run-dir <filtering output dir>
"""

import argparse
import pathlib
import sys

import xarray

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from pyurbanair.config.hydra_helpers import create_observation_points
from pyurbanair.plotting import (
    plot_final_state_with_obs,
    plot_parameter_error,
    plot_rollout_time_evolution,
    plot_sensor_timeseries,
)
from pyurbanair.utils.animation_utils import animate_rollout_state
from pyurbanair.utils.run_utils import add_velocity_magnitude
from scripts.filtering._filtering_common import (
    build_sensor_sets,
    ensemble_cycle_sensor_series,
    ensemble_velocity_mean_std,
    load_analyzed_states,
    load_run_config,
    read_yaml,
    select_z_plane,
    sensor_magnitude,
    streaming_state_rmse,
    truth_cycle_sensor_series,
    truth_end_of_cycle,
)


def make_figures(run_dir: pathlib.Path) -> None:
    """Draw every figure from the saved artifacts in ``run_dir``."""
    cfg = load_run_config(run_dir)
    ta = read_yaml(run_dir / "truth_access.yaml")

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
        # trajectory on its own (no separate prior overlay needed).
        params_history = xarray.open_dataset(history_path)
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
    # One point per cycle; truth and ensemble interpolated at the same physical
    # points on their own grids, so the figures are grid-independent.
    sensor_sets = build_sensor_sets(cfg)
    truth_series = truth_cycle_sensor_series(
        truth_end, sensor_sets, ta["truth_solver_name"]
    )
    ensemble_series = ensemble_cycle_sensor_series(
        analyzed_states, sensor_sets, ta["assim_solver_name"]
    )
    for name, (sx, sy, sz) in sensor_sets.items():
        plot_sensor_timeseries(
            true_sensor=sensor_magnitude(truth_series[name]),
            ensemble_sensor=sensor_magnitude(ensemble_series[name]),
            output_path=run_dir / f"sensor_timeseries_{name}.png",
            title=f"State at {name} sensors",
            sensor_x=sx,
            sensor_y=sy,
            sensor_z=sz,
        )

    print(f"Saved figures in {run_dir}")


def main() -> None:
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
