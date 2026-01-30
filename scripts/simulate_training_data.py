import os
import pathlib
import pdb
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import xarray
from animation import animate_ensemble_state, animate_state
from pyudales.ensemble_forward_model import EnsembleForwardModel
from pyudales.forward_model import ForwardModel
from pyudales.utils.forward_model_utils import create_new_forward_model
from pyudales.utils.grid_utils import interpolate_grid

np.random.seed(1)

# Directory settings
# MATLAB_BIN = "/Applications/MATLAB_R2025b.app/bin/matlab"
MATLAB_BIN = "/opt/sw/matlab-2023b/bin/matlab"
CASE_DIR = "examples/udales/experiments/xie_and_castro_training_data"
EXPERIMENT_NAME = "999"
RESULTS_DIR = pathlib.Path(".temp/udales")

NUM_TRAIN_DATA = 500
BATCH_SIZE = 2

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Compute ressources
NCPU_PER_PROCESS = 16
NUM_PARALLEL_PROCESSES = 1

# Forward model settings
FIXED_INPUT = {
    "save_only_last_timestep": False,
    "output_frequency": 2.0,
    "ncpu": NCPU_PER_PROCESS,
    "matlab_bin": MATLAB_BIN,
    "case_dir": CASE_DIR,
    "experiment_name": EXPERIMENT_NAME,
    "verbose": False,
    # "results_dir": RESULTS_DIR,
}


def main() -> None:

    for i in range(2):
        if os.path.exists(".temp"):
            shutil.rmtree(".temp")

        random_initial_condition_args = {
            "irandom": i,
            "randqt": 2.5e-4,
            "randthl": 1.0,
            "randu": 2.0,
        }
        forward_model = ForwardModel(
            **FIXED_INPUT,
            random_initial_condition_args=random_initial_condition_args,
            results_dir=pathlib.Path(f"training_data/sim_{i}"),
        )  # type: ignore[arg-type]
        params = xarray.Dataset(
            data_vars={
                "inflow_angle": 0.0,
                "velocity_magnitude": 3,
                "pressure_gradient_magnitude": 0.0041912,
            },
        )
        forward_model.run_preprocessing(python_or_matlab="python")

        state = forward_model(params=params)
        state = interpolate_grid(state)

        state_save = state.isel(zt=1)
        state_save.to_netcdf(f"training_data/sim_{i}.nc")

    #     state = xarray.open_dataset(f"training_data/sim_{i}.nc").load()


    #     vel_magnitude = np.sqrt(state.u.values**2 + state.v.values**2 + state.w.values**2)
    #     # Add vel_magnitude as a data variable in state
    #     state = state.assign(vel_magnitude=(("time", "yt", "xt"), vel_magnitude))

    #     states.append(state)

    # state = xarray.concat(states, dim="ensemble", join="override")

    # diff = state.isel(ensemble=-1) - state.isel(ensemble=0)
    # # print(diff.mean())
    # # print(diff.std())


    # state = xarray.concat(states + [diff], dim="ensemble", join="override")


    # animate_ensemble_state(
    #     state=state,
    #     output_path=pathlib.Path("figures/udales_animation.mp4"),
    #     z_level=1,
    #     vmin={"u": -3.0, "v": -2.0, "w": -2.0, "pres": 0.0, "vel_magnitude": 0.0},
    #     vmax={"u": 3.0, "v": 2.0, "w": 2.0, "pres": 1.0, "vel_magnitude": 3.0},
    # )

    # plt.figure()
    # plt.imshow(vel_magnitude[-1, 1, :, :])
    # plt.colorbar()
    # plt.savefig("figures/vel_magnitude_udales.png")
    # plt.show()

    # t1 = time.time()

    # # for i in range(NUM_TRAIN_DATA // BATCH_SIZE):

    # if os.path.exists(".temp"):
    #     shutil.rmtree(".temp")

    # forward_model = ForwardModel(**FIXED_INPUT)  # type: ignore[arg-type]
    # forward_model.run_preprocessing(python_or_matlab="python")

    # inflow_angle_vec = np.random.uniform(low=-45, high=45, size=BATCH_SIZE)
    # print(inflow_angle_vec)
    # params_ensemble = xarray.Dataset(
    #     data_vars={
    #         "inflow_angle": ("ensemble", inflow_angle_vec),
    #         "velocity_magnitude": ("ensemble", np.ones(BATCH_SIZE) * 3),
    #     },
    #     coords={"ensemble": np.arange(BATCH_SIZE)},
    # )
    # ensemble_forward_model = EnsembleForwardModel(
    #     forward_model=forward_model,
    #     ensemble_size=BATCH_SIZE,
    #     num_parallel_processes=NUM_PARALLEL_PROCESSES,
    #     num_cpus_per_process=NCPU_PER_PROCESS,
    # )

    # t1 = time.time()
    # state_parallel = ensemble_forward_model.run_ensemble(params=params_ensemble)
    # t2 = time.time()
    # print(f"Time taken: {t2 - t1} seconds")

    # ensemble_forward_model = EnsembleForwardModel(
    #     forward_model=forward_model,
    #     ensemble_size=BATCH_SIZE,
    #     num_parallel_processes=1,
    #     num_cpus_per_process=NCPU_PER_PROCESS,
    # )

    # t1 = time.time()
    # state_sequential = ensemble_forward_model.run_ensemble(params=params_ensemble)
    # t2 = time.time()
    # print(f"Time taken: {t2 - t1} seconds")

    # #     diff = state_parallel - state_sequential


    #     for sim_idx in range(BATCH_SIZE):
    #         state = xarray.open_dataset(f"{RESULTS_DIR}/state_{sim_idx}.nc")

    #         state = interpolate_grid(state)

    #         state = state.isel(zt=1)

    #         state = state.assign_coords({"inflow_angle": inflow_angle_vec[sim_idx]})

    #         # state.to_netcdf(f"training_data/sim_{i*BATCH_SIZE + sim_idx}.nc")

    # t2 = time.time()
    # print(f"Time taken: {t2 - t1} seconds")


if __name__ == "__main__":
    main()
