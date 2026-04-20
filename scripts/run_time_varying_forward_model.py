"""Run a forward model with time-varying inflow parameters.

Supports both pyudales (via time-dependent nudging) and pylbm (via
``uvel_time.dat``).  Use ``--model`` to select which solver to run.
"""

import argparse
import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pyurbanair.utils.animation_utils import animate_state
from pyurbanair.utils.run_utils import add_velocity_magnitude, extract_2d_slice

from scripts import config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a forward model with time-varying inflow parameters."
    )
    parser.add_argument(
        "--model",
        choices=["pyudales", "pylbm"],
        default="pylbm",
        help="Which solver to use (default: pyudales).",
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

    model_name: str = args.model

    # Build time-varying parameters
    # Time is relative to the simulation interval (not including spinup).
    # The forward model internally prepends a constant plateau for spinup.
    sim_time = config.TIME["simulation_time"]
    n_snapshots = 100
    time_seconds = np.linspace(0, sim_time, n_snapshots)

    if args.use_true_params:
        angle = config.TRUE_PARAMS["inflow_angle"]
        vel = config.TRUE_PARAMS["velocity_magnitude"]
        print(f"Using TRUE_PARAMS: angle={angle}, vel={vel}")
    else:
        angle = -45.0
        vel = 3.0

        x = np.linspace(angle, -angle, n_snapshots)

        def sigmoid(x, center, width, min_val, max_val):
            return min_val + (max_val - min_val) / (1 + np.exp(-(x - center) / width))

        inflow_vec = sigmoid(x, center=-30.0, width=5.0, min_val=angle, max_val=-angle)

    data_vars: dict = {
        "inflow_angle": ("time", inflow_vec),
        "velocity_magnitude": ("time", np.full(n_snapshots, vel)),
    }

    # pressure_gradient_magnitude is only relevant for pyudales
    if model_name == "pyudales":
        data_vars["pressure_gradient_magnitude"] = config.TRUE_PARAMS[
            "pressure_gradient_magnitude"
        ]

    params = xr.Dataset(data_vars=data_vars, coords={"time": time_seconds})

    model_name = args.model
    forward_model = config.create_forward_model(
        model_name=model_name,
        results_dir=(
            pathlib.Path(args.results_dir) if args.results_dir is not None else None
        ),
    )
    config.prepare_forward_model(model_name=model_name, forward_model=forward_model)
    config.clean_forward_model_outputs(
        model_name=model_name, forward_model=forward_model
    )

    # For pylbm, params are passed to run_single; for pyudales they were
    # passed at init and run_single uses the stored params.
    if model_name == "pylbm":
        state = forward_model.run_single(params=params)
    else:
        state = forward_model.run_single()
        if state is None:
            state = forward_model.get_states()

    state = add_velocity_magnitude(state)

    print(f"Model: {model_name} (time-varying inflow)")
    print(f"Dims: {dict(state.sizes)}")
    print(f"Vars: {list(state.data_vars)}")
    print(
        f"Inflow angle: {params['inflow_angle'].values[0]:.1f} -> "
        f"{params['inflow_angle'].values[-1]:.1f} deg"
    )
    print(
        f"Velocity magnitude: {params['velocity_magnitude'].values[0]:.1f} -> "
        f"{params['velocity_magnitude'].values[-1]:.1f} m/s"
    )

    if not args.skip_viz:
        out_dir = (
            config.BASE_RESULTS_DIR / "forward_model" / f"{model_name}_time_varying"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_var = "vel_magnitude" if "vel_magnitude" in state.data_vars else "u"
        plot_2d = extract_2d_slice(state[plot_var], z_level=0)
        plt.figure(figsize=(6, 5))
        plt.imshow(plot_2d, origin="lower")
        plt.colorbar(label=plot_var)
        plt.title(f"{model_name} time-varying - {plot_var} (last time, mid z)")
        plt.tight_layout()
        plt.savefig(out_dir / "field_snapshot.png")
        plt.close()

        if model_name == "pyudales":
            from pyudales.utils.grid_utils import interpolate_grid

            state = interpolate_grid(state)

        animate_state(
            state=state,
            output_path=out_dir / "state_animation.mp4",
            z_level=0,
        )
        print(f"Saved visualization outputs in {out_dir}")


if __name__ == "__main__":
    main()
