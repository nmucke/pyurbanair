import os
import pathlib
import pdb
import shutil
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import xarray
from data_assimilation.observation_operator import (
    ObservationOperator,
    TemporalObservationOperator,
)
from data_assimilation.smoothing.esmda import StateAndParameterESMDA
from pylbm.ensemble_forward_model import EnsembleForwardModel
from pylbm.rollout_forward_model import RolloutForwardModel as LBMRolloutForwardModel
from pyudales.rollout_forward_model import (
    RolloutForwardModel as UDALESRolloutForwardModel,
)

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
        num_timesteps = []
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
            num_timesteps.append(esmda_step.time.shape[0])

        num_timesteps = min(num_timesteps)
        out = []
        for state in ensemble_mean_field:
            state_time = state.time.shape[0]
            diff_time = abs(num_timesteps - state_time)
            out.append(state.isel(time=slice(diff_time, state_time + 1)))
        ensemble_mean_field = xarray.concat(out, dim="esmda_step", join="override")

    return ensemble_mean_field, params


# Random seed
SEED = 42

Z_PLOT_LEVEL = 1

RESULTS_DIR = ".temp/lbm"

# Initialize states
INIT_STATES_DIR = pathlib.Path("esmda_init_conditions/lbm")

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Compute ressources
NCPU_PER_PROCESS = 1
NUM_PARALLEL_PROCESSES = 25

# True parameters
TRUE_VELOCITY_MAGNITUDE = 10.0
TRUE_ANGLE = 10.0

# Data assimilation settings
ENSEMBLE_SIZE = 300
NUM_ESMDA_STEPS = 2
ALPHA = 1 / NUM_ESMDA_STEPS


OBS_X = [13, 45.6, 94.3, 108.9, 87.3, 20.0, 52.6, 90.0, 60.0, 75.0, 75.0]
OBS_Y = [30.6, 52.7, 92.9, 108.0, 10.0, 90.0, 10.0, 50.0, 80.0, 90.0, 60.0]
OBS_Z = [2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8]
OBS_STATES = ["u", "v", "w"]
NUM_OBS = len(OBS_X) * len(OBS_STATES)

# Observation error settings
OBS_ERROR_STD = 0.01
C_D = jnp.diag(OBS_ERROR_STD**2 * jnp.ones(NUM_OBS))

# Forward model settings
udales_time = 10
lbm_time_steps = int(udales_time / 0.0538)
lbm_output_frequency = int(lbm_time_steps / 50)

# Forward model settings
LBM_FIXED_INPUT = {
    "stl_path": "examples/lbm/experiments/xie_castro_2008_STL.stl",
    "nx": 120,
    "ny": 120,
    "nz": 8,
    "num_timesteps": lbm_time_steps,
    "bounds": ((0, 160), (0, 160), (0, 40)),
    "verbose": False,
    "output_frequency": lbm_output_frequency,
}

# Forward model settings
UDALES_FIXED_INPUT = {
    "save_only_last_timestep": False,
    "output_frequency": 1.0,
    "ncpu": 1,
    "matlab_bin": "/opt/sw/matlab-2023b/bin/matlab",
    "case_dir": "examples/udales/experiments/xie_and_castro",
    "verbose": False,
    "experiment_name": "999",
}


def main() -> None:
    """Main function."""

    if os.path.exists(".temp"):
        shutil.rmtree(".temp")

    ##### Setup parameter ensemble #####
    rng_key = jax.random.PRNGKey(SEED)

    true_params = xarray.Dataset(
        data_vars={
            "inflow_angle": TRUE_ANGLE,
            "velocity_magnitude": TRUE_VELOCITY_MAGNITUDE,
        },
    )
    udales_forward_model = UDALESRolloutForwardModel(**UDALES_FIXED_INPUT)
    udales_forward_model.run_preprocessing()
    # udales_forward_model = LBMRolloutForwardModel(**LBM_FIXED_INPUT)

    ##### Run true simulation #####
    true_state = udales_forward_model(params=true_params)
    true_state = udales_forward_model(params=true_params, state=true_state)
    true_state = true_state / 75.0

    true_velocity_field = get_velocity_magnitude_field(true_state)

    lbm_forward_model = LBMRolloutForwardModel(
        **LBM_FIXED_INPUT,
        cuda=True,
        results_dir=pathlib.Path(RESULTS_DIR),
    )

    ##### Setup observations #####
    observation_operator = ObservationOperator(
        obs_x=OBS_X,
        obs_y=OBS_Y,
        obs_z=OBS_Z,
        obs_states=OBS_STATES,
        solver_name="pylbm",
    )
    if LBM_FIXED_INPUT["output_frequency"] is not None:
        observation_operator = TemporalObservationOperator(
            observation_operator=observation_operator,
            mode="mean",
        )
    true_obs = observation_operator(true_state)
    rng_key, subkey = jax.random.split(rng_key)
    true_obs = true_obs + jnp.sqrt(C_D) @ jax.random.normal(subkey, true_obs.shape)

    ensemble_forward_model = EnsembleForwardModel(
        forward_model=lbm_forward_model,
        ensemble_size=ENSEMBLE_SIZE,
        num_parallel_processes=NUM_PARALLEL_PROCESSES,
        num_cpus_per_process=NCPU_PER_PROCESS,
    )
    init_states = [
        xarray.open_dataset(INIT_STATES_DIR / f"state_{i}.nc").isel(time=-1)
        for i in range(ENSEMBLE_SIZE)
    ]
    init_states = xarray.concat(init_states, dim="ensemble", join="override")
    init_params = xarray.open_dataset(INIT_STATES_DIR / f"params.nc").isel(
        ensemble=slice(0, ENSEMBLE_SIZE)
    )

    ##### Run ESMDA #####
    t1 = time.time()

    esmda = StateAndParameterESMDA(
        observation_operator=observation_operator,
        forward_model=ensemble_forward_model,
        C_D=C_D,
        num_steps=NUM_ESMDA_STEPS,
        alpha=ALPHA,
        rng_key=rng_key,
    )
    output = esmda(
        state=init_states,
        params=init_params,
        observations=true_obs,
        return_params_history=True,
        return_state_history=True,
    )
    t2 = time.time()
    print(f"ESMDA time: {t2 - t1:.2f} seconds")

    # Get ESMDA ensemble mean field and parameters
    ensemble_mean_field, params = get_ensemble_mean_field(output, esmda)

    mean_velocity_field = get_velocity_magnitude_field(ensemble_mean_field)

    mean_velocity_field = mean_velocity_field[:, 2:]

    rmse = [
        jnp.sqrt(jnp.mean((mean_velocity_field[i] - true_velocity_field) ** 2)).item()
        for i in range(NUM_ESMDA_STEPS + 1)
    ]

    # If 'time' is in the dimensions of ensemble_mean_field, select the last time step
    if "time" in ensemble_mean_field.dims:
        rmse_init = [
            jnp.sqrt(
                jnp.mean((mean_velocity_field[i, 0] - true_velocity_field[0]) ** 2)
            ).item()
            for i in range(NUM_ESMDA_STEPS + 1)
        ]
        mean_velocity_field_init = mean_velocity_field[:, 0]
        true_velocity_field_init = true_velocity_field[0]

        mean_velocity_field = mean_velocity_field[:, -1]
        true_velocity_field = true_velocity_field[-1]

    ##### Plot results #####
    fig, axes = plt.subplots(
        NUM_ESMDA_STEPS + 1, 8, figsize=(8 * 4, 4 * (NUM_ESMDA_STEPS + 1))
    )

    hist_args = lambda i: {
        "bins": 25,
        "alpha": 0.5,
        "label": f"Step {i+1}",
    }
    im_args = {
        "vmin": true_velocity_field[1, :, :].min(),
        "vmax": true_velocity_field[1, :, :].max(),
    }
    angle_axvline_args = {
        "x": TRUE_ANGLE,
        "color": "red",
        "linewidth": 3,
        "label": "True",
    }
    velocity_axvline_args = {
        "x": TRUE_VELOCITY_MAGNITUDE,
        "color": "red",
        "linewidth": 3,
        "label": "True",
    }
    for i in range(NUM_ESMDA_STEPS + 1):
        im = axes[i, 3].imshow(mean_velocity_field[i, Z_PLOT_LEVEL, :, :], **im_args)
        im = axes[i, 4].imshow(true_velocity_field[Z_PLOT_LEVEL, :, :], **im_args)
        im = axes[i, 5].imshow(
            mean_velocity_field[i, Z_PLOT_LEVEL, :, :]
            - true_velocity_field[Z_PLOT_LEVEL, :, :],
            **im_args,
        )

        axes[i, 1].scatter(OBS_X, OBS_Y, color="red")
        axes[i, 0].scatter(OBS_X, OBS_Y, color="red")

        if i == 0:
            axes[i, 3].set_title("Ens mean end time")
            axes[i, 4].set_title("True end time")
            axes[i, 0].set_title(f"Ens mean init cond")
            axes[i, 1].set_title(f"True init cond")
            axes[i, 6].set_title("Angle distribution")
            if "velocity_magnitude" in params:
                axes[i, 7].set_title("Velocity magnitude distribution")

        axes[i, 5].set_title(f"End time RMSE: {rmse[i]:.4f}")

        im = axes[i, 0].imshow(
            mean_velocity_field_init[i, Z_PLOT_LEVEL, :, :], **im_args
        )
        im = axes[i, 1].imshow(true_velocity_field_init[Z_PLOT_LEVEL, :, :], **im_args)
        im = axes[i, 2].imshow(
            mean_velocity_field_init[i, Z_PLOT_LEVEL, :, :]
            - true_velocity_field_init[Z_PLOT_LEVEL, :, :],
            **im_args,
        )
        # fig.colorbar(im, ax=axes[i, 5])
        axes[i, 2].set_title(f"Init RMSE: {rmse_init[i]:.4f}")

        axes[i, 6].hist(params.inflow_angle.isel(esmda_step=i).values, **hist_args(i))
        axes[i, 6].set_xlim(-25, 25)
        axes[i, 6].axvline(**angle_axvline_args)
        axes[i, 6].axvline(
            jnp.mean(params.inflow_angle.isel(esmda_step=i).values),
            color="black",
            linestyle="--",
            label="ESMDA Mean",
            linewidth=3,
        )
        axes[i, 6].legend()

        if "velocity_magnitude" in params:
            axes[i, 7].hist(
                params.velocity_magnitude.isel(esmda_step=i).values, **hist_args(i)
            )
            axes[i, 7].set_xlim(0, 25)
            axes[i, 7].axvline(**velocity_axvline_args)
            axes[i, 7].axvline(
                jnp.mean(params.velocity_magnitude.isel(esmda_step=i).values),
                color="black",
                linestyle="--",
                label="ESMDA Mean",
                linewidth=3,
            )
            axes[i, 7].legend()

    plt.savefig(
        os.path.join(
            FIGURES_DIR, f"esmda_results_lbm_with_udales_data_{NUM_ESMDA_STEPS}.pdf"
        )
    )
    plt.close()
    # plt.show()


if __name__ == "__main__":
    main()
