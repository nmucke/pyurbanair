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

Honors ``run.skip_viz`` (set in the saved config): a no-op when true, mirroring
the old in-script behaviour.

Usage::

    python scripts/esmda/make_esmda_figures.py --run-dir <esmda output dir>
"""

import argparse
import pathlib
import sys

import numpy as np
import xarray

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from evaluation.figures import (
    plot_final_state_with_obs,
    plot_parameter_error,
    plot_rollout_time_evolution,
    plot_sensor_timeseries,
)
from evaluation.sensors import sensor_magnitude
from evaluation.turbulence import select_z_plane, streaming_state_rmse

from pyurbanair.config.hydra_helpers import create_observation_points
from pyurbanair.utils.animation_utils import animate_rollout_state
from pyurbanair.utils.run_utils import add_velocity_magnitude
from scripts.esmda._esmda_common import (
    build_sensor_sets,
    ensemble_sensor_series,
    load_run_config,
    open_truth,
    read_yaml,
    truth_sensor_series,
)


def make_figures(run_dir: pathlib.Path) -> None:
    """Draw every figure from the saved artifacts in ``run_dir``."""
    cfg = load_run_config(run_dir)
    if cfg.run.skip_viz:
        print(f"run.skip_viz is set; no figures generated for {run_dir}")
        return

    ta = read_yaml(run_dir / "truth_access.yaml")
    num_windows = int(ta["num_windows"])
    sim_time = float(ta["sim_time"])
    is_dynamic = bool(ta["is_dynamic"])

    posterior_params = xarray.open_dataset(run_dir / "posterior_params.nc")
    prior_params = xarray.open_dataset(run_dir / "prior_params.nc")
    true_params = xarray.open_dataset(run_dir / "true_params.nc")
    posterior_state = xarray.open_dataset(run_dir / "posterior_state_mean.nc")

    # Open the (potentially multi-GB) truth lazily. The plots below each pull
    # only the slice they need: a single z-plane for the animation/final state
    # and a few z-slices for the streamed error curve.
    true_state = open_truth(
        ta["true_state_path"],
        ta["n_total"],
        ta["x_offset"],
        ta["start_idx"],
        ta["t_offset"],
    )
    obs_x, obs_y, obs_z = create_observation_points(cfg.obs)

    # Truth reduced to the z=0 horizontal plane (single layer kept), so the
    # animation and final-state plot never load the full 3-D velocity field.
    true_state_plane = select_z_plane(true_state, z_level=0)
    true_vel = add_velocity_magnitude(true_state_plane)["vel_magnitude"]

    # State |U| RMSE series for the rollout plot (streamed over a few z-slices).
    rmse = streaming_state_rmse(true_state, posterior_state)

    # Boundaries between assimilation windows on the (rebased) global time axis,
    # used to lightly shade alternating windows in the parameter plot.
    window_edges = (
        list(np.linspace(0.0, sim_time * num_windows, num_windows + 1))
        if is_dynamic and num_windows > 1
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
        help="The ESMDA run output directory written by scripts/esmda/run_esmda.py.",
    )
    args = ap.parse_args()
    if not args.run_dir.exists():
        raise SystemExit(f"run dir not found: {args.run_dir}")
    make_figures(args.run_dir)


if __name__ == "__main__":
    main()
