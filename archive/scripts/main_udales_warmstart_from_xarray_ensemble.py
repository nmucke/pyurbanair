import os
import pathlib
import pdb
import shutil

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray
from animation import animate_ensemble_state, animate_state
from pyudales.ensemble_forward_model import EnsembleForwardModel
from pyudales.rollout_forward_model import RolloutForwardModel

# Directory settings
# MATLAB_BIN = "/Applications/MATLAB_R2025b.app/bin/matlab"
MATLAB_BIN = "/opt/sw/matlab-2023b/bin/matlab"
EXPERIMENT_DIR = "examples/udales/experiments/xie_and_castro"
EXPERIMENT_NAME = "300"
RESULTS_DIR = pathlib.Path(".temp/udales")
ENSEMBLE_SIZE = 3
NUM_PARALLEL_PROCESSES = 1
NCPU_PER_PROCESS = 1

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
    "results_dir": RESULTS_DIR,
}


def main() -> None:

    if os.path.exists(".temp"):
        shutil.rmtree(".temp")

    forward_model = RolloutForwardModel(**FIXED_INPUT)

    inflow_angle_range = jnp.linspace(-45, 45, ENSEMBLE_SIZE)
    velocity_magnitude_range = jnp.ones(ENSEMBLE_SIZE) * 1.0
    params = xarray.Dataset(
        data_vars={
            "inflow_angle": ("ensemble", inflow_angle_range),
            "velocity_magnitude": ("ensemble", velocity_magnitude_range),
        },
        coords={"ensemble": jnp.arange(len(inflow_angle_range))},
    )

    forward_model.run_preprocessing(python_or_matlab="python")

    ensemble_forward_model = EnsembleForwardModel(
        forward_model=forward_model,
        ensemble_size=ENSEMBLE_SIZE,
        num_parallel_processes=NUM_PARALLEL_PROCESSES,
        num_cpus_per_process=NCPU_PER_PROCESS,
    )

    state = ensemble_forward_model.run_ensemble(params=params)
    # state_new = state1.copy() if state1 is not None else None
    # state2 = ensemble_forward_model.run_ensemble(state=state_new, params=params)
    # state3 = ensemble_forward_model.run_ensemble(state=state2, params=params)

    if forward_model.save_on_disk:
        state = ensemble_forward_model.get_states()
    # else:
    #     state = xarray.concat([state1, state2, state3], dim="time")

    # Add vel_magnitude as a data variable in state
    # Note: state has dimensions (ensemble, time, zm, yt, xt) after concat
    vel_magnitude = np.sqrt(state.u.values**2 + state.v.values**2 + state.w.values**2)  # type: ignore[union-attr]
    state = state.assign(  # type: ignore[union-attr]
        vel_magnitude=(("ensemble", "time", "zm", "yt", "xt"), vel_magnitude)
    )
    animate_ensemble_state(
        state=state,  # Pass the full xarray Dataset
        output_path=pathlib.Path("figures/udales_animation.mp4"),
        z_level=0,
        vmin={"u": -3.0, "v": -2.0, "w": -2.0, "pres": 0.0, "vel_magnitude": 0.0},
        vmax={"u": 3.0, "v": 2.0, "w": 2.0, "pres": 1.0, "vel_magnitude": 3.0},
    )

    print(vel_magnitude.shape)
    print(vel_magnitude.min(), vel_magnitude.max())
    plt.figure()
    # Select last ensemble member, first time step, all yt/xt
    plt.imshow(vel_magnitude[-1, 0, 0, :, :])
    plt.colorbar()
    plt.savefig("figures/vel_magnitude_udales.png")
    plt.show()


if __name__ == "__main__":
    main()
