import pathlib
import pdb

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pylbm
import xarray
from animation import animate_ensemble_state, animate_state
from pylbm.ensemble_forward_model import EnsembleForwardModel
from pylbm.rollout_forward_model import RolloutForwardModel

ENSEMBLE_SIZE = 3
NUM_PARALLEL_PROCESSES = 3
NUM_CPUS_PER_PROCESS = 1


def main() -> None:
    stl_path = pathlib.Path("examples/lbm/experiments/xie_castro_2008_STL.stl")
    # stl_path = pathlib.Path("examples/lbm/experiments/geom.STL")

    forward_model = RolloutForwardModel(
        stl_path=stl_path,
        nx=120,
        ny=120,
        nz=8,
        simulation_time=1000 * 0.0538,
        bounds=((0, 160), (0, 160), (0, 40)),
        output_frequency=10 * 0.0538,
        results_dir=pathlib.Path(".temp/results"),
    )
    params = xarray.Dataset(
        data_vars={
            "inflow_angle": ("ensemble", jnp.linspace(-40, 40, ENSEMBLE_SIZE)),
            "velocity_magnitude": ("ensemble", jnp.ones(ENSEMBLE_SIZE) * 10),
        },
    )
    # save_on_disk is applied automatically when results_dir is passed above

    ensemble_forward_model = EnsembleForwardModel(
        forward_model=forward_model,
        ensemble_size=ENSEMBLE_SIZE,
        num_parallel_processes=NUM_PARALLEL_PROCESSES,
        num_cpus_per_process=NUM_CPUS_PER_PROCESS,
    )

    # Run 3 rollout steps (cold → warm → warm)
    states = []
    for _ in range(3):
        state = ensemble_forward_model.run_ensemble(params=params)
        states.append(state)

    # state = ensemble_forward_model.get_states()
    # state = xarray.concat(states, dim="time", join="override")
    state = ensemble_forward_model.get_states()

    vel_magnitude = np.sqrt(state.u.values**2 + state.v.values**2 + state.w.values**2)
    state = state.assign(
        vel_magnitude=(("ensemble", "time", "z", "y", "x"), vel_magnitude)
    )

    # Remove the blanking data_var from the xarray.Dataset if present
    if "blanking" in state.data_vars:
        state = state.drop_vars("blanking")

    if "rho" in state.data_vars:
        state = state.drop_vars("rho")

    animate_ensemble_state(
        state=state,
        output_path=pathlib.Path("figures/lbm_animation.mp4"),
        z_level=1,
        # vmin={"u": -3.0, "v": -2.0, "w": -2.0, "pres": 0.0, "vel_magnitude": 0.0},
        # vmax={"u": 3.0, "v": 2.0, "w": 2.0, "pres": 1.0, "vel_magnitude": 3.0},
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
