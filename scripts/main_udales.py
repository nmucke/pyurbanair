import os
import pathlib
import pdb

import matplotlib.pyplot as plt
import numpy as np
import xarray
from animation import animate_state
from pyudales.forward_model import ForwardModel
from pyudales.utils.forward_model_utils import create_new_forward_model

# Directory settings
# MATLAB_BIN = "/Applications/MATLAB_R2025b.app/bin/matlab"
MATLAB_BIN = "/opt/sw/matlab-2023b/bin/matlab"
CASE_DIR = "examples/udales/experiments/xie_and_castro"
EXPERIMENT_NAME = "999"
RESULTS_DIR = pathlib.Path(".temp/udales")

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Compute ressources
NCPU = 4

# Forward model settings
FIXED_INPUT = {
    "save_only_last_timestep": False,
    "output_frequency": 2.0,
    "ncpu": NCPU,
    "matlab_bin": MATLAB_BIN,
    "case_dir": CASE_DIR,
    "experiment_name": EXPERIMENT_NAME,
    "verbose": False,
    "nx": 120,
    "ny": 120,
    "nz": 8,
    "bounds": ((0, 160), (0, 160), (0, 40)),
    # "results_dir": RESULTS_DIR,
}


def main() -> None:

    forward_model = ForwardModel(**FIXED_INPUT)  # type: ignore[arg-type]
    params = xarray.Dataset(
        data_vars={
            "inflow_angle": -45,
            "velocity_magnitude": 3,
            "pressure_gradient_magnitude": 0.0041912 * 10,
        },
    )
    forward_model.run_preprocessing(python_or_matlab="python")
    state = forward_model(params=params)
    if state is None:
        raise RuntimeError("Expected in-memory state from forward_model run.")

    vel_magnitude = np.sqrt(state.u.values**2 + state.v.values**2 + state.w.values**2)
    # Add vel_magnitude as a data variable in state
    state = state.assign(vel_magnitude=(("time", "zm", "yt", "xt"), vel_magnitude))

    animate_state(
        state=state,
        output_path=pathlib.Path("figures/udales_animation.mp4"),
        z_level=0,
        vmin={"u": -3.0, "v": -2.0, "w": -2.0, "pres": 0.0, "vel_magnitude": 0.0},
        vmax={"u": 3.0, "v": 2.0, "w": 2.0, "pres": 1.0, "vel_magnitude": 3.0},
    )

    print(vel_magnitude.shape)
    print(vel_magnitude.min(), vel_magnitude.max())
    plt.figure()
    plt.imshow(vel_magnitude[0, 0, :, :])
    plt.colorbar()
    plt.savefig("figures/vel_magnitude_udales.png")
    plt.close()
    # plt.show()


if __name__ == "__main__":
    main()
