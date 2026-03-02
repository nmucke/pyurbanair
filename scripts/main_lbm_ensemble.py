import os
import pathlib
import pdb
import shutil
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray
from animation import animate_ensemble_state, animate_state
from pylbm.ensemble_forward_model import EnsembleForwardModel

# from pylbm.compile_program import compile_lbm
from pylbm.forward_model import ForwardModel

ENSEMBLE_SIZE = 4
NUM_PARALLEL_PROCESSES = 1
NCPU_PER_PROCESS = 1
SEED = 42


def main() -> None:

    if os.path.exists(".temp"):
        shutil.rmtree(".temp")

    stl_path = pathlib.Path("examples/lbm/experiments/xie_castro_2008_STL.stl")

    forward_model = ForwardModel(
        stl_path=stl_path,
        nx=160,
        ny=160,
        nz=8,
        simulation_time=100 * 0.0538,
        bounds=((0, 160), (0, 160), (0, 100)),
        output_frequency=10 * 0.0538,
    )
    forward_model.compile()

    ensemble_forward_model = EnsembleForwardModel(
        forward_model=forward_model,
        ensemble_size=ENSEMBLE_SIZE,
        num_parallel_processes=NUM_PARALLEL_PROCESSES,
        num_cpus_per_process=NCPU_PER_PROCESS,
    )

    params = xarray.Dataset(
        data_vars={
            "inflow_angle": ("ensemble", jnp.linspace(-40, 40, ENSEMBLE_SIZE)),
            "velocity_magnitude": ("ensemble", jnp.ones(ENSEMBLE_SIZE) * 10),
        },
        coords={"ensemble": jnp.arange(ENSEMBLE_SIZE)},
    )

    t1 = time.time()
    state = ensemble_forward_model.run_ensemble(params=params)
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")
    if state is None:
        raise RuntimeError("Expected in-memory ensemble state.")

    vel_magnitude = np.sqrt(state.u.values**2 + state.v.values**2 + state.w.values**2)
    state = state.assign(
        vel_magnitude=(("ensemble", "time", "z", "y", "x"), vel_magnitude)
    )

    # Remove the blanking data_var from the xarray.Dataset if present
    if "blanking" in state.data_vars:
        state = state.drop_vars("blanking")

    if "rho" in state.data_vars:
        state = state.drop_vars("rho")

    state = state.drop_vars("v")
    state = state.drop_vars("w")

    animate_ensemble_state(
        state=state,
        output_path=pathlib.Path("figures/lbm_ensemble_animation.mp4"),
        z_level=1,
    )

    vel_magnitude = state.v.values
    print(vel_magnitude.shape)
    print(vel_magnitude.min(), vel_magnitude.max())
    plt.figure()
    plt.imshow(vel_magnitude[0, 1, :, :])
    # plt.imshow(state.blanking.values[0, 1, :, :])
    plt.colorbar()
    plt.savefig("figures/vel_magnitude_lbm.png")
    plt.show()


if __name__ == "__main__":
    main()
