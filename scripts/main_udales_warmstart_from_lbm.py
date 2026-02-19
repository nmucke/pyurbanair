import os
import pathlib
import pdb
import shutil

import matplotlib.pyplot as plt
import numpy as np
import xarray
from animation import animate_ensemble_state, animate_state
from pyudales.rollout_forward_model import RolloutForwardModel

# Directory settings
# MATLAB_BIN = "/Applications/MATLAB_R2025b.app/bin/matlab"
MATLAB_BIN = "/opt/sw/matlab-2023b/bin/matlab"
EXPERIMENT_DIR = "examples/udales/experiments/xie_and_castro"
EXPERIMENT_NAME = "300"
RESULTS_DIR = pathlib.Path(".temp/udales")

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Compute resources
# NOTE: For warm start from xarray, use NCPU=1 to avoid domain decomposition issues.
# uDALES uses different decomposition schemes (z-pencil vs standard) for different
# arrays, which makes multi-processor warm start from Python-generated files complex.
NCPU = 1

# Forward model settings
FIXED_INPUT = {
    "save_only_last_timestep": False,
    "output_frequency": 1.0,
    "ncpu": NCPU,
    "matlab_bin": MATLAB_BIN,
    "case_dir": EXPERIMENT_DIR,
    "experiment_name": EXPERIMENT_NAME,
    "verbose": False,
    # "results_dir": RESULTS_DIR,
}


def main() -> None:

    if os.path.exists(".temp"):
        shutil.rmtree(".temp")

    forward_model = RolloutForwardModel(**FIXED_INPUT)
    params = xarray.Dataset(
        data_vars={
            "inflow_angle": -45,
            "velocity_magnitude": 5,
            "pressure_gradient_magnitude": 0.0041912,
        },
    )
    forward_model.run_preprocessing(python_or_matlab="python")

    udales_state = forward_model(params=params)

    state = xarray.open_dataset("esmda_init_conditions/lbm/state_0.nc").load()
    state.u.values = state.u.values * 75
    state.v.values = state.v.values * 75
    state.w.values = state.w.values * 75

    state1 = forward_model(state=state, params=params)
    params = xarray.Dataset(
        data_vars={
            "inflow_angle": -45,
            "velocity_magnitude": 10.0,
            "pressure_gradient_magnitude": 0.0041912,
        },
    )
    state2 = forward_model(state=state1, params=params)
    state3 = forward_model(state=state2, params=params)

    state = xarray.concat([state1, state2, state3], dim="time")

    # Add vel_magnitude as a data variable in state
    vel_magnitude = np.sqrt(state.u.values**2 + state.v.values**2 + state.w.values**2)
    state = state.assign(vel_magnitude=(("time", "zm", "yt", "xt"), vel_magnitude))
    animate_state(
        state=state,
        output_path=pathlib.Path("figures/udales_animation.mp4"),
        z_level=0,
    )

    print(vel_magnitude.shape)
    print(vel_magnitude.min(), vel_magnitude.max())
    plt.figure()
    # Select last ensemble member, first time step, all yt/xt
    plt.imshow(vel_magnitude[-1, 0, 0, :, :])
    plt.colorbar()
    plt.savefig("figures/vel_magnitude_udales.png")
    plt.close()
    # plt.show()


if __name__ == "__main__":
    main()
