import os
import pathlib
import pdb
import shutil
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import xarray
from animation import animate_ensemble_state, animate_state
from data_assimilation.observation_operator import (
    ObservationOperator,
    TemporalObservationOperator,
)
from data_assimilation.smoothing.esmda import StateAndParameterESMDA
from pyudales.ensemble_forward_model import EnsembleForwardModel
from pyudales.forward_model import ForwardModel
from pyudales.rollout_forward_model import RolloutForwardModel

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel
from pyurbanair.utils.state_utils import get_velocity_magnitude_field


def get_ensemble_mean_field(
    output: tuple | None, esmda: StateAndParameterESMDA
) -> xarray.Dataset:
    """Get the ensemble mean field from the output of the ESMDA."""
    if isinstance(output, tuple):
        params = output[0]
        ensemble_mean_field = output[1]
        ensemble_mean_field = ensemble_mean_field.mean(dim="ensemble")
    else:
        params = output
        ensemble_mean_field = []
        for i in range(NUM_ESMDA_STEPS + 1):
            esmda_step = esmda.get_state(step=i, ensemble_member=0)
            for j in range(1, ENSEMBLE_SIZE):
                esmda_state = esmda.get_state(step=i, ensemble_member=j)
                for var in esmda_step.data_vars:
                    esmda_step[var].values = (
                        esmda_step[var].values + esmda_state[var].values
                    )
            for var in esmda_step.data_vars:
                esmda_step[var].values /= ENSEMBLE_SIZE
            ensemble_mean_field.append(esmda_step)
        ensemble_mean_field = xarray.concat(
            ensemble_mean_field, dim="esmda_step", join="override"
        )

    return ensemble_mean_field, params


def get_ensemble_std_field(
    output: tuple | None, esmda: StateAndParameterESMDA
) -> xarray.Dataset:
    """Get the ensemble std field from the output of the ESMDA."""
    if isinstance(output, tuple):
        params = output[0]
        ensemble_std_field = output[1]
        ensemble_std_field = ensemble_std_field.std(dim="ensemble")
    else:
        params = output
        ensemble_std_field = []
        for i in range(NUM_ESMDA_STEPS + 1):
            esmda_step = esmda.get_state(step=i, ensemble_member=0)
            for j in range(1, ENSEMBLE_SIZE):
                esmda_state = esmda.get_state(step=i, ensemble_member=j)
                for var in esmda_step.data_vars:
                    esmda_step[var].values = (
                        esmda_step[var].values + esmda_state[var].values
                    )
            for var in esmda_step.data_vars:
                esmda_step[var].values /= ENSEMBLE_SIZE
            ensemble_std_field.append(esmda_step)
        ensemble_std_field = xarray.concat(
            ensemble_std_field, dim="esmda_step", join="override"
        )

    return ensemble_std_field, params


NUM_ASSIMILATION_WINDOWS = 3

# Random seed
SEED = 42

Z_PLOT_LEVEL = 1

# Directory settings
# MATLAB_BIN = "/Applications/MATLAB_R2025b.app/bin/matlab"
MATLAB_BIN = "/opt/sw/matlab-2023b/bin/matlab"
EXPERIMENT_DIR = "examples/udales/experiments/xie_and_castro"
EXPERIMENT_NAME = "999"
RESULTS_DIR = ".temp/udales"
TEMP_DIR = None  # "/scratch/ntmucke/pyudales"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Initialize states
INIT_STATES_DIR = pathlib.Path("esmda_init_conditions/udales")

# Compute ressources
NCPU_PER_PROCESS = 1
NUM_PARALLEL_PROCESSES = 4

# True parameters
TRUE_PRESSURE_GRADIENT_MAGNITUDE = 0.0041912
TRUE_VELOCITY_MAGNITUDE = 3.0
TRUE_ANGLE = 10.0

# Data assimilation settings
ENSEMBLE_SIZE = 4
NUM_ESMDA_STEPS = 2
ALPHA = 1 / NUM_ESMDA_STEPS

# Observation settings
# OBS_IDS_X = [40, 50, 90, 120, 80, 20, 50, 90]
# OBS_IDS_Y = [30, 60, 90, 120, 20, 60, 90, 50]
# OBS_IDS_Z = [1, 1, 1, 1, 1, 1, 1, 1]
# OBS_STATES = ["u", "v", "w"]
# NUM_OBS = len(OBS_IDS_X) * len(OBS_STATES)


OBS_X = jnp.linspace(10, 150, 4)
OBS_Y = jnp.linspace(10, 150, 4)
OBS_X, OBS_Y = jnp.meshgrid(OBS_X, OBS_Y)
OBS_X = OBS_X.flatten()
OBS_Y = OBS_Y.flatten()
OBS_Z = jnp.full(len(OBS_X), 2.0)
OBS_STATES = ["u", "v", "w"]
NUM_OBS = len(OBS_X) * len(OBS_STATES)

# Observation error settings
OBS_ERROR_STD = 0.1
C_D = jnp.diag(OBS_ERROR_STD**2 * jnp.ones(NUM_OBS))

# Forward model settings
FIXED_INPUT = {
    "save_only_last_timestep": False,
    "output_frequency": 2.0,
    "ncpu": NCPU_PER_PROCESS,
    "matlab_bin": MATLAB_BIN,
    "case_dir": EXPERIMENT_DIR,
    "verbose": False,
    "temp_dir": TEMP_DIR,
    "experiment_name": EXPERIMENT_NAME,
}


def main() -> None:
    """Main function."""

    if os.path.exists(".temp"):
        shutil.rmtree(".temp")

    ##### Setup parameter ensemble #####
    rng_key = jax.random.PRNGKey(SEED)

    true_params = xarray.open_dataset(INIT_STATES_DIR / f"params.nc").isel(ensemble=0)
    true_state = xarray.open_dataset(INIT_STATES_DIR / f"state_{0}.nc").isel(time=-1)

    forward_model = RolloutForwardModel(**FIXED_INPUT)
    forward_model.run_preprocessing(python_or_matlab="python")

    forward_model.set_results_dir(pathlib.Path(RESULTS_DIR))
    ensemble_forward_model = EnsembleForwardModel(
        forward_model=forward_model,
        ensemble_size=ENSEMBLE_SIZE,
        num_parallel_processes=NUM_PARALLEL_PROCESSES,
        num_cpus_per_process=NCPU_PER_PROCESS,
    )
    forward_model.set_results_dir(None)

    init_states = [
        xarray.open_dataset(INIT_STATES_DIR / f"state_{i}.nc").isel(
            time=slice(-1, None)
        )
        for i in range(ENSEMBLE_SIZE)
    ]
    init_states = xarray.concat(init_states, dim="ensemble", join="override")
    init_params = xarray.open_dataset(INIT_STATES_DIR / f"params.nc").isel(
        ensemble=slice(0, ENSEMBLE_SIZE)
    )

    ##### Setup observations #####
    observation_operator = ObservationOperator(
        obs_x=OBS_X,
        obs_y=OBS_Y,
        obs_z=OBS_Z,
        obs_states=OBS_STATES,
        solver_name="udales",
    )
    if FIXED_INPUT["output_frequency"] is not None:
        observation_operator = TemporalObservationOperator(
            observation_operator=observation_operator,
            mode="mean",
        )

    esmda = StateAndParameterESMDA(
        observation_operator=observation_operator,
        forward_model=ensemble_forward_model,
        C_D=C_D,
        num_steps=NUM_ESMDA_STEPS,
        alpha=ALPHA,
        rng_key=rng_key,
    )

    true_state_list = []
    true_obs_list = []
    true_params_list = [true_params]
    esmda_state_list = []
    esmda_params_list = [init_params]
    esmda_obs_list = []
    for i in range(NUM_ASSIMILATION_WINDOWS):

        ##### Perturb true parameters #####
        rng_key, subkey = jax.random.split(rng_key)
        vel_magnitude_perturbation = jax.random.normal(subkey) * 0.0

        rng_key, subkey = jax.random.split(rng_key)
        angle_perturbation = jax.random.normal(subkey) * 0.0

        perturbed_velocity_magnitude = (
            true_params.velocity_magnitude.values + vel_magnitude_perturbation
        )
        perturbed_inflow_angle = true_params.inflow_angle.values + angle_perturbation

        true_params = xarray.Dataset(
            data_vars={
                "velocity_magnitude": ([], perturbed_velocity_magnitude),
                "inflow_angle": ([], perturbed_inflow_angle),
            },
        )

        ##### Run true simulation #####
        true_state = forward_model(params=true_params, state=true_state)
        true_state_list.append(true_state.isel(zt=1, zm=1))
        true_params_list.append(true_params)

        true_obs = observation_operator(true_state)
        rng_key, subkey = jax.random.split(rng_key)
        true_obs = true_obs + jnp.sqrt(C_D) @ jax.random.normal(subkey, true_obs.shape)
        true_obs_list.append(true_obs)

        init_params, init_states = esmda(
            state=init_states.isel(time=-1),  # type: ignore[attr-defined]
            params=init_params,
            observations=true_obs,
            return_params_history=False,
            return_state_history=False,
        )
        esmda_obs_list.append(observation_operator(init_states))

        esmda_state_list.append(init_states.isel(zt=1, zm=1))  # type: ignore[attr-defined]
        esmda_params_list.append(init_params)

    true_state_list = xarray.concat(true_state_list, dim="time", join="override")
    true_params_list = xarray.concat(true_params_list, dim="time", join="override")
    esmda_state_list = xarray.concat(esmda_state_list, dim="time", join="override")
    esmda_params_list = xarray.concat(esmda_params_list, dim="time", join="override")

    os.makedirs("esmda_results", exist_ok=True)

    true_state_list.to_netcdf("esmda_results/true_state.nc")  # type: ignore[attr-defined]
    true_params_list.to_netcdf("esmda_results/true_params.nc")  # type: ignore[attr-defined]
    esmda_state_list.to_netcdf("esmda_results/esmda_state.nc")  # type: ignore[attr-defined]
    esmda_params_list.to_netcdf("esmda_results/esmda_params.nc")  # type: ignore[attr-defined]

    # esmda_obs_list = xarray.concat(esmda_obs_list, dim="time", join="override")
    # true_obs_list = xarray.concat(true_obs_list, dim="time", join="override")
    # esmda_obs_list.to_netcdf("esmda_results/esmda_obs.nc")
    # true_obs_list.to_netcdf("esmda_results/true_obs.nc")

    esmda_mean = esmda_state_list.mean(dim="ensemble")  # type: ignore[attr-defined]
    esmda_std = esmda_state_list.std(dim="ensemble")  # type: ignore[attr-defined]

    true_velocity_field = get_velocity_magnitude_field(true_state_list)
    mean_velocity_field = get_velocity_magnitude_field(esmda_mean)
    std_velocity_field = get_velocity_magnitude_field(esmda_std)

    velocity_magnitude_error = jnp.abs(true_velocity_field - mean_velocity_field)

    rmse = jnp.sqrt(jnp.mean(velocity_magnitude_error**2, axis=(1, 2)))

    nsteps, nx, ny = true_velocity_field.shape
    rmse_time = jnp.arange(nsteps)

    xarray_to_animate = xarray.Dataset(
        data_vars={
            "true_state": (
                ["ensemble", "time", "yt", "xt"],
                true_velocity_field.reshape(1, nsteps, ny, nx),
            ),
            "esmda_mean": (
                ["ensemble", "time", "yt", "xt"],
                mean_velocity_field.reshape(1, nsteps, ny, nx),
            ),
            "esmda_std": (
                ["ensemble", "time", "yt", "xt"],
                std_velocity_field.reshape(1, nsteps, ny, nx),
            ),
            "velocity_magnitude_error": (
                ["ensemble", "time", "yt", "xt"],
                velocity_magnitude_error.reshape(1, nsteps, ny, nx),
            ),
        },
        coords={
            "ensemble": jnp.arange(1),
            "time": jnp.arange(nsteps),
            "yt": jnp.arange(ny),
            "xt": jnp.arange(nx),
        },
    )
    animate_ensemble_state(
        state=xarray_to_animate,
        output_path=pathlib.Path("figures/udales_esmda_animation.mp4"),
        z_level=0,
        vmin={"u": -3.0, "v": -2.0, "w": -2.0, "pres": 0.0, "vel_magnitude": 0.0},
        vmax={"u": 3.0, "v": 2.0, "w": 2.0, "pres": 1.0, "vel_magnitude": 3.0},
    )

    esmda_mean_params = esmda_params_list.mean(dim="ensemble")  # type: ignore[attr-defined]
    esmda_std_params = esmda_params_list.std(dim="ensemble")  # type: ignore[attr-defined]
    true_mean_params = true_params_list.mean(dim="ensemble")  # type: ignore[attr-defined]

    inflow_angle_mean = esmda_mean_params.inflow_angle.values
    inflow_angle_std = esmda_std_params.inflow_angle.values
    velocity_magnitude_mean = esmda_mean_params.velocity_magnitude.values
    velocity_magnitude_std = esmda_std_params.velocity_magnitude.values
    true_inflow_angle_mean = true_mean_params.inflow_angle.values
    true_velocity_magnitude_mean = true_mean_params.velocity_magnitude.values
    timesteps = (
        esmda_mean_params.time.values
        if "time" in esmda_mean_params.dims
        else jnp.arange(len(inflow_angle_mean))
    )

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(
        timesteps, inflow_angle_mean, label="ESMDA Mean", linewidth=2, color="tab:blue"
    )
    plt.fill_between(
        timesteps,
        inflow_angle_mean - 2 * inflow_angle_std,
        inflow_angle_mean + 2 * inflow_angle_std,
        color="tab:blue",
        alpha=0.3,
        label="ESMDA ±2 Std",
    )
    plt.plot(
        timesteps, true_inflow_angle_mean, label="True", linewidth=2, color="tab:red"
    )
    plt.xlabel("Time Window")
    plt.ylabel("Inflow Angle")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(
        timesteps,
        velocity_magnitude_mean,
        label="ESMDA Mean",
        linewidth=2,
        color="tab:blue",
    )
    plt.fill_between(
        timesteps,
        velocity_magnitude_mean - 2 * velocity_magnitude_std,
        velocity_magnitude_mean + 2 * velocity_magnitude_std,
        color="tab:blue",
        alpha=0.3,
        label="ESMDA ±2 Std",
    )
    plt.plot(
        timesteps,
        true_velocity_magnitude_mean,
        label="True",
        linewidth=2,
        color="tab:red",
    )
    plt.xlabel("Time Window")
    plt.ylabel("Velocity Magnitude")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(rmse_time, rmse, label="RMSE", linewidth=2, color="tab:blue")
    plt.xlabel("Time")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(pathlib.Path("figures/rollout_udales_esmda_params.pdf"))
    plt.close()

    # true_velocity_field = get_velocity_magnitude_field(true_state)


if __name__ == "__main__":
    main()
