"""Run the uDALES forward model with time-varying inflow parameters.

This script demonstrates running a simulation where inflow_angle and
velocity_magnitude vary smoothly over the simulation interval using
uDALES time-dependent nudging.
"""

import argparse
import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pyudales.forward_model import ForwardModel as UDALESForwardModel
from pyudales.utils.grid_utils import interpolate_grid

from pyurbanair.utils.animation_utils import animate_state
from pyurbanair.utils.run_utils import add_velocity_magnitude, extract_2d_slice
from scripts import config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run uDALES with time-varying inflow parameters."
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip plotting and animation outputs.",
    )
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=None,
        help="Override results directory for model outputs.",
    )
    parser.add_argument(
        "--use-true-params",
        action="store_true",
        help="Use TRUE_PARAMS values (same as run_forward_model.py) instead "
        "of default constants.  Useful for diagnosing whether divergence "
        "is caused by parameter values vs the scalar/time-varying mechanism.",
    )
    args = parser.parse_args()

    # Build time-varying parameters
    # Time is relative to the simulation interval (not including spinup).
    # apply_time_varying_inflow internally prepends a constant plateau for spinup.
    sim_time = config.TIME["simulation_time"]
    n_snapshots = 2
    time_seconds = np.linspace(0, sim_time, n_snapshots)

    if args.use_true_params:
        angle = config.TRUE_PARAMS["inflow_angle"]
        vel = config.TRUE_PARAMS["velocity_magnitude"]
        print(f"Using TRUE_PARAMS: angle={angle}, vel={vel}")
    else:
        angle = -15.0
        vel = 3.0

    params = xr.Dataset(
        data_vars={
            "inflow_angle": ("time", np.linspace(angle, angle, n_snapshots)),
            "velocity_magnitude": ("time", np.full(n_snapshots, vel)),
            "pressure_gradient_magnitude": config.TRUE_PARAMS[
                "pressure_gradient_magnitude"
            ],
        },
        coords={"time": time_seconds},
    )

    # Create the forward model with time-varying params
    model_kwargs = config.model_args("pyudales")
    model_kwargs["params"] = params
    model_kwargs["nudging_config"] = {"tnudge": 10.0, "nnudge": 0}
    if args.results_dir is not None:
        model_kwargs["results_dir"] = pathlib.Path(args.results_dir)

    fm = UDALESForwardModel(**model_kwargs)
    config.prepare_forward_model("pyudales", fm)
    config.clean_forward_model_outputs("pyudales", fm)

    state = fm.run_single()
    if state is None:
        state = fm.get_states()
    state = add_velocity_magnitude(state)

    print("Model: pyudales (time-varying inflow)")
    print(f"Dims: {dict(state.sizes)}")
    print(f"Vars: {list(state.data_vars)}")
    print(
        f"Inflow angle: {params['inflow_angle'].values[0]:.1f} -> {params['inflow_angle'].values[-1]:.1f} deg"
    )
    print(
        f"Velocity magnitude: {params['velocity_magnitude'].values[0]:.1f} -> {params['velocity_magnitude'].values[-1]:.1f} m/s"
    )

    if not args.skip_viz:
        out_dir = config.BASE_RESULTS_DIR / "forward_model" / "pyudales_time_varying"
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_var = "vel_magnitude" if "vel_magnitude" in state.data_vars else "u"
        plot_2d = extract_2d_slice(state[plot_var], z_level=0)
        plt.figure(figsize=(6, 5))
        plt.imshow(plot_2d, origin="lower")
        plt.colorbar(label=plot_var)
        plt.title(f"pyudales time-varying - {plot_var} (last time, mid z)")
        plt.tight_layout()
        plt.savefig(out_dir / "field_snapshot.png")
        plt.close()

        state = interpolate_grid(state)

        animate_state(
            state=state,
            output_path=out_dir / "state_animation.mp4",
            z_level=0,
        )
        print(f"Saved visualization outputs in {out_dir}")

        plt.figure()
        for idx in [10, 40, 70]:
            u_at_left_end = state.u.values[:, 1, idx, 1]
            v_at_left_end = state.v.values[:, 1, idx, 1]
            inflow_angle_from_state = (
                np.arctan2(v_at_left_end, u_at_left_end) * 180.0 / np.pi
            )
            plt.plot(inflow_angle_from_state, label=f"idx={idx}")
        plt.legend()
        plt.title("inflow angle from state")
        plt.xlabel("time")
        plt.ylabel("inflow angle")
        plt.show()


if __name__ == "__main__":
    main()
