import os
import pathlib
import pdb

import matplotlib.pyplot as plt
import numpy as np
import xarray
from pyudales.forward_model import ForwardModel

# Directory settings
# MATLAB_BIN = "/Applications/MATLAB_R2025b.app/bin/matlab"
MATLAB_BIN = "/opt/sw/matlab-2023b/bin/matlab"
EXPERIMENT_DIR = "examples/udales/experiments/300"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Compute ressources
NCPU = 4

# Forward model settings
FIXED_INPUT = {
    "save_only_last_timestep": True,
    "ncpu": NCPU,
    "matlab_bin": MATLAB_BIN,
    "experiment_dir": EXPERIMENT_DIR,
    "verbose": False,
}


def main() -> None:

    forward_model = ForwardModel(**FIXED_INPUT)  # type: ignore[arg-type]
    params = xarray.Dataset(
        data_vars={
            "inflow_angle": 15,
            "velocity_magnitude": 3,
            "pressure_gradient_magnitude": 0.0041912,
        },
    )
    forward_model.run_preprocessing(python_or_matlab="matlab")
    state = forward_model(params=params)
    # state = xarray.load_dataset(".temp/lbm/out005000.nc")

    vel_magnitude = np.sqrt(state.u.values**2 + state.v.values**2 + state.w.values**2)
    print(vel_magnitude.shape)
    print(vel_magnitude.min(), vel_magnitude.max())
    plt.figure()
    plt.imshow(vel_magnitude[0, 0, :, :])
    plt.colorbar()
    plt.savefig("figures/vel_magnitude_udales.png")
    plt.show()


if __name__ == "__main__":
    main()
