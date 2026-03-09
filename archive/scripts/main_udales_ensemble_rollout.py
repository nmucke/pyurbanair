import os
import pathlib
import shutil
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray
from animation import animate_ensemble_state, animate_state
from pyudales.ensemble_forward_model import EnsembleForwardModel
from pyudales.rollout_forward_model import RolloutForwardModel

# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# Random seed
SEED = 42

# Directory settings
# MATLAB_BIN = "/Applications/MATLAB_R2025b.app/bin/matlab"
MATLAB_BIN = "/opt/sw/matlab-2023b/bin/matlab"
CASE_DIR = "examples/udales/experiments/xie_and_castro"
EXPERIMENT_NAME = "999"
RESULTS_DIR = pathlib.Path(".temp/udales")

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

ENSEMBLE_SIZE = 4

# Compute ressources
NCPU_PER_PROCESS = 1
NUM_PARALLEL_PROCESSES = 4

# Forward model settings
FIXED_INPUT = {
    "save_only_last_timestep": False,
    "output_frequency": 2.0,
    "ncpu": NCPU_PER_PROCESS,
    "matlab_bin": MATLAB_BIN,
    "case_dir": CASE_DIR,
    "experiment_name": EXPERIMENT_NAME,
    "verbose": False,
    "results_dir": RESULTS_DIR,
}


def main() -> None:

    if os.path.exists(".temp"):
        shutil.rmtree(".temp")

    ##### Setup parameter ensemble #####
    inflow_angle_range = jnp.linspace(-45, 45, ENSEMBLE_SIZE)
    velocity_magnitude_range = jnp.ones(ENSEMBLE_SIZE) * 1.0

    params_ensemble = xarray.Dataset(
        data_vars={
            "inflow_angle": ("ensemble", inflow_angle_range),
            "velocity_magnitude": ("ensemble", velocity_magnitude_range),
        },
        coords={"ensemble": jnp.arange(len(inflow_angle_range))},
    )

    forward_model = RolloutForwardModel(**FIXED_INPUT)
    forward_model.run_preprocessing(python_or_matlab="python")

    ensemble_forward_model = EnsembleForwardModel(
        forward_model=forward_model,
        ensemble_size=ENSEMBLE_SIZE,
        num_parallel_processes=NUM_PARALLEL_PROCESSES,
        num_cpus_per_process=NCPU_PER_PROCESS,
    )
    state1 = ensemble_forward_model.run_ensemble(params=params_ensemble)

    state0 = xarray.load_dataset(RESULTS_DIR / "state_0.nc")
    v = state0.v.values
    v[-1, ...] = v[-1, ...] * 2.0
    state0.v.values = v
    state0.to_netcdf(RESULTS_DIR / "state_0.nc")

    state2 = ensemble_forward_model.run_ensemble(
        state=pathlib.Path(RESULTS_DIR), params=params_ensemble
    )
    params_ensemble = xarray.Dataset(
        data_vars={
            "inflow_angle": ("ensemble", jnp.flip(inflow_angle_range)),
            "velocity_magnitude": ("ensemble", jnp.ones(ENSEMBLE_SIZE) * 4.0),
        },
        coords={"ensemble": jnp.arange(len(inflow_angle_range))},
    )
    state3 = ensemble_forward_model.run_ensemble(
        state=pathlib.Path(RESULTS_DIR), params=params_ensemble
    )
    # state4 = ensemble_forward_model.run_ensemble(state=state3, params=params_ensemble)

    if forward_model.save_on_disk:
        state = ensemble_forward_model.get_states()
    else:
        state = xarray.concat([state1, state2, state3], dim="time")

    vel_magnitude = np.sqrt(state.u.values**2 + state.v.values**2 + state.w.values**2)
    # Add vel_magnitude as a data variable in state
    state = state.assign(
        vel_magnitude=(("ensemble", "time", "zm", "yt", "xt"), vel_magnitude)
    )

    animate_ensemble_state(
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
    plt.show()


if __name__ == "__main__":
    main()
