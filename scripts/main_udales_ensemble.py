import os
import pathlib
import pdb
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray
from animation import animate_state
from pyudales.ensemble_forward_model import EnsembleForwardModel
from pyudales.forward_model import ForwardModel
from pyudales.utils.forward_model_utils import create_new_forward_model

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

ENSEMBLE_SIZE = 12

# Compute ressources
NCPU_PER_PROCESS = 4
NUM_PARALLEL_PROCESSES = 2

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

    ##### Setup parameter ensemble #####
    rng_key = jax.random.PRNGKey(SEED)

    rng_key, subkey = jax.random.split(rng_key)
    inflow_angle_range = jax.random.normal(subkey, (ENSEMBLE_SIZE,)) * 8

    rng_key, subkey = jax.random.split(rng_key)
    velocity_magnitude_range = jax.random.normal(subkey, (ENSEMBLE_SIZE,)) * 1 + 4.0
    velocity_magnitude_range = jnp.maximum(velocity_magnitude_range, 0.1)

    params_ensemble = xarray.Dataset(
        data_vars={
            "inflow_angle": ("ensemble", inflow_angle_range),
            "velocity_magnitude": ("ensemble", velocity_magnitude_range),
        },
        coords={"ensemble": jnp.arange(len(inflow_angle_range))},
    )

    forward_model = ForwardModel(**FIXED_INPUT)  # type: ignore[arg-type]
    forward_model.run_preprocessing(python_or_matlab="python")

    ensemble_forward_model = EnsembleForwardModel(
        forward_model=forward_model,
        ensemble_size=ENSEMBLE_SIZE,
        # results_dir=pathlib.Path(RESULTS_DIR),
        num_parallel_processes=NUM_PARALLEL_PROCESSES,
        num_cpus_per_process=NCPU_PER_PROCESS,
    )
    t1 = time.time()
    state = ensemble_forward_model.run_ensemble(params=params_ensemble)
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")

    import pdb

    pdb.set_trace()

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
    plt.show()


if __name__ == "__main__":
    main()
