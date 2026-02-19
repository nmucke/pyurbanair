import os
import pathlib
import pdb
import shutil

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import xarray
from data_assimilation.observation_operator import ObservationOperator
from data_assimilation.smoothing.esmda import ParameterESMDA
from pylbm.ensemble_forward_model import EnsembleForwardModel
from pylbm.forward_model import ForwardModel

from pyurbanair.utils.state_utils import get_velocity_magnitude_field

Z_PLOT_LEVEL = 1


def get_ensemble_mean_field(
    output: tuple | None, esmda: ParameterESMDA
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


# Random seed
SEED = 42

# Directory settings
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# True parameters
TRUE_VELOCITY_MAGNITUDE = 10.0
TRUE_ANGLE = 10.0

NUM_PARALLEL_PROCESSES = 32

# Data assimilation settings
ENSEMBLE_SIZE = 200
NUM_ESMDA_STEPS = 2
ALPHA = 1 / NUM_ESMDA_STEPS

# Observation settings
# OBS_IDS_X = [40, 50, 90, 120, 80, 20, 50, 90]
# OBS_IDS_Y = [30, 60, 90, 120, 20, 60, 90, 50]
# OBS_IDS_Z = [1, 1, 1, 1, 1, 1, 1, 1]
# OBS_STATES = ["u", "v", "w"]
# NUM_OBS = len(OBS_IDS_X) * len(OBS_STATES)


OBS_X = [43, 51.6, 94.3, 110.9, 87.3, 20.0, 52.6, 90.0]
OBS_Y = [30.6, 62.7, 92.9, 108.0, 20.0, 60.0, 90.0, 50.0]
OBS_Z = [2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8]
OBS_STATES = ["u", "v", "w"]
NUM_OBS = len(OBS_X) * len(OBS_STATES)

# Observation error settings
OBS_ERROR_STD = 0.01
C_D = jnp.diag(OBS_ERROR_STD**2 * jnp.ones(NUM_OBS))

# Forward model settings
FIXED_INPUT = {
    "stl_path": "examples/lbm/experiments/xie_castro_2008_STL.stl",
    "nx": 120,
    "ny": 120,
    "nz": 8,
    "num_timesteps": 50,
    "bounds": ((0, 160), (0, 160), (0, 40)),
    "verbose": True,
    "output_frequency": 50,
}


def main() -> None:
    """Main function."""

    if os.path.exists(".temp"):
        shutil.rmtree(".temp")

    ##### Setup parameter ensemble #####
    rng_key = jax.random.PRNGKey(SEED)

    rng_key, subkey = jax.random.split(rng_key)
    inflow_angle_range = jax.random.normal(subkey, (ENSEMBLE_SIZE,)) * 8

    rng_key, subkey = jax.random.split(rng_key)
    velocity_magnitude_range = jax.random.normal(subkey, (ENSEMBLE_SIZE,)) * 1 + 7.0
    velocity_magnitude_range = jnp.maximum(velocity_magnitude_range, 0.1)

    params_ensemble = xarray.Dataset(
        data_vars={
            "inflow_angle": ("ensemble", inflow_angle_range),
            "velocity_magnitude": ("ensemble", velocity_magnitude_range),
        },
        coords={"ensemble": jnp.arange(len(inflow_angle_range))},
    )

    ##### Setup forward model #####
    true_params = xarray.Dataset(
        data_vars={
            "inflow_angle": TRUE_ANGLE,
            "velocity_magnitude": TRUE_VELOCITY_MAGNITUDE,
        },
    )
    forward_model = ForwardModel(**FIXED_INPUT)  # type: ignore[arg-type]

    ##### Run true simulation #####
    true_state = forward_model(params=true_params)
    true_velocity_field = get_velocity_magnitude_field(true_state)

    ##### Setup observations #####
    observation_operator = ObservationOperator(
        obs_x=OBS_X,
        obs_y=OBS_Y,
        obs_z=OBS_Z,
        obs_states=OBS_STATES,
        solver_name="pylbm",
    )
    true_obs = observation_operator(true_state.isel(time=-1))
    rng_key, subkey = jax.random.split(rng_key)

    true_obs = true_obs + jnp.sqrt(C_D) @ jax.random.normal(subkey, true_obs.shape)

    forward_model.apply_save_on_disk(results_dir=pathlib.Path(".temp/lbm"))
    ensemble_forward_model = EnsembleForwardModel(
        forward_model=forward_model,
        ensemble_size=ENSEMBLE_SIZE,
        # results_dir=pathlib.Path(RESULTS_DIR),
        num_parallel_processes=NUM_PARALLEL_PROCESSES,
        num_cpus_per_process=1,
    )

    ##### Run ESMDA #####
    esmda = ParameterESMDA(
        observation_operator=observation_operator,
        forward_model=ensemble_forward_model,
        C_D=C_D,
        num_steps=NUM_ESMDA_STEPS,
        alpha=ALPHA,
        rng_key=rng_key,
        results_dir=pathlib.Path(".temp/lbm"),
    )
    output = esmda(
        params=params_ensemble,
        observations=true_obs,
        return_params_history=True,
        return_state_history=True,
    )

    # Get ESMDA ensemble mean field and parameters
    ensemble_mean_field, params = get_ensemble_mean_field(output, esmda)

    mean_velocity_field = get_velocity_magnitude_field(ensemble_mean_field)

    rmse = [
        jnp.sqrt(jnp.mean((mean_velocity_field[i] - true_velocity_field) ** 2)).item()
        for i in range(NUM_ESMDA_STEPS + 1)
    ]

    # If 'time' is in the dimensions of ensemble_mean_field, select the last time step
    if "time" in ensemble_mean_field.dims:
        mean_velocity_field = mean_velocity_field[:, -1]
        true_velocity_field = true_velocity_field[-1]

        ##### Plot results #####
    fig, axes = plt.subplots(
        NUM_ESMDA_STEPS + 1, 5, figsize=(16, 4 * (NUM_ESMDA_STEPS + 1))
    )

    hist_args = lambda i: {
        "bins": 25,
        "alpha": 0.5,
        "label": f"Step {i+1}",
    }
    im_args = {
        "vmin": true_velocity_field[1, :, :].min(),
        "vmax": true_velocity_field[1, :, :].max(),
        "extent": [0, 160, 0, 160],
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
        im = axes[i, 0].imshow(mean_velocity_field[i, Z_PLOT_LEVEL, :, :], **im_args)
        im = axes[i, 1].imshow(true_velocity_field[Z_PLOT_LEVEL, :, :], **im_args)
        im = axes[i, 2].imshow(
            mean_velocity_field[i, Z_PLOT_LEVEL, :, :]
            - true_velocity_field[Z_PLOT_LEVEL, :, :],
            **im_args,
        )

        axes[i, 3].hist(params.inflow_angle.isel(esmda_step=i).values, **hist_args(i))
        axes[i, 3].set_xlim(-15, 15)
        axes[i, 3].axvline(**angle_axvline_args)
        axes[i, 3].axvline(
            jnp.mean(params.inflow_angle.isel(esmda_step=i).values),
            color="black",
            linestyle="--",
            label="ESMDA Mean",
            linewidth=3,
        )
        axes[i, 3].legend()

        if "velocity_magnitude" in params:
            axes[i, 4].hist(
                params.velocity_magnitude.isel(esmda_step=i).values, **hist_args(i)
            )
            axes[i, 4].set_xlim(5, 15)
            axes[i, 4].axvline(**velocity_axvline_args)
            axes[i, 4].axvline(
                jnp.mean(params.velocity_magnitude.isel(esmda_step=i).values),
                color="black",
                linestyle="--",
                label="ESMDA Mean",
                linewidth=3,
            )
            axes[i, 4].legend()

        fig.colorbar(im, ax=axes[i, 0])
        fig.colorbar(im, ax=axes[i, 1])
        fig.colorbar(im, ax=axes[i, 2])

        axes[i, 1].scatter(OBS_X, OBS_Y, color="red")
        axes[i, 0].scatter(OBS_X, OBS_Y, color="red")

        if i == 0:
            axes[i, 0].set_title("Ensemble mean")
            axes[i, 1].set_title("True")
            axes[i, 3].set_title("Angle distribution")
            if "velocity_magnitude" in params:
                axes[i, 4].set_title("Velocity magnitude distribution")

        axes[i, 2].set_title(f"RMSE: {rmse[i]:.4f}")
    plt.savefig(os.path.join(FIGURES_DIR, f"esmda_results_lbm_{NUM_ESMDA_STEPS}.pdf"))
    plt.close()

    # # Get ESMDA ensemble mean field and parameters
    # ensemble_mean_field, params = get_ensemble_mean_field(output=output, esmda=esmda)

    # mean_velocity_field = get_velocity_magnitude_field(ensemble_mean_field)

    # # ensemble_mean_field = state.mean(dim="ensemble")
    # # velocity_field = get_velocity_magnitude_field(ensemble_mean_field)
    # # velocity_field = velocity_field[:, 0]

    # # If 'time' is in the dimensions of ensemble_mean_field, select the last time step
    # if "time" in ensemble_mean_field.dims:
    #     mean_velocity_field = mean_velocity_field[:, -1]
    #     true_velocity_field = true_velocity_field[-1]

    # rmse = [
    #     jnp.sqrt(
    #         jnp.mean((velocity_field[i] - true_velocity_field[1, :, :]) ** 2)
    #     ).item()
    #     for i in range(NUM_ESMDA_STEPS + 1)
    # ]

    # ##### Plot results #####
    # fig, axes = plt.subplots(
    #     NUM_ESMDA_STEPS + 1, 5, figsize=(16, 4 * (NUM_ESMDA_STEPS + 1))
    # )

    # hist_args = lambda i: {
    #     "bins": 25,
    #     "alpha": 0.5,
    #     "label": f"Step {i+1}",
    # }
    # im_args = {
    #     "vmin": true_velocity_field[1, :, :].min(),
    #     "vmax": true_velocity_field[1, :, :].max(),
    #     "extent": [0, 160, 0, 160],
    # }
    # angle_axvline_args = {"x": TRUE_ANGLE, "color": "red", "linewidth": 3}
    # velocity_axvline_args = {
    #     "x": TRUE_VELOCITY_MAGNITUDE,
    #     "color": "red",
    #     "linewidth": 3,
    # }
    # for i in range(NUM_ESMDA_STEPS + 1):
    #     im = axes[i, 0].imshow(velocity_field[i, 1, :, :], **im_args)
    #     im = axes[i, 1].imshow(true_velocity_field[1, :, :], **im_args)
    #     im = axes[i, 2].imshow(
    #         velocity_field[i, 1, :, :] - true_velocity_field[1, :, :], **im_args
    #     )

    #     axes[i, 3].hist(params.inflow_angle.isel(esmda_step=i).values, **hist_args(i))
    #     axes[i, 3].set_xlim(-15, 15)
    #     axes[i, 3].axvline(**angle_axvline_args)
    #     axes[i, 3].legend()

    #     if "velocity_magnitude" in params:
    #         axes[i, 4].hist(
    #             params.velocity_magnitude.isel(esmda_step=i).values, **hist_args(i)
    #         )
    #         axes[i, 4].set_xlim(0, 8)
    #         axes[i, 4].axvline(**velocity_axvline_args)
    #         axes[i, 4].legend()

    #     fig.colorbar(im, ax=axes[i, 0])
    #     fig.colorbar(im, ax=axes[i, 1])
    #     fig.colorbar(im, ax=axes[i, 2])

    #     axes[i, 1].scatter(OBS_X, OBS_Y, color="red")
    #     axes[i, 0].scatter(OBS_X, OBS_Y, color="red")

    #     if i == 0:
    #         axes[i, 0].set_title("Ensemble mean")
    #         axes[i, 1].set_title("True")
    #         axes[i, 3].set_title("Angle distribution")
    #         if "velocity_magnitude" in params:
    #             axes[i, 4].set_title("Velocity magnitude distribution")

    #     axes[i, 2].set_title(f"RMSE: {rmse[i]:.4f}")
    # plt.savefig(os.path.join(FIGURES_DIR, "esmda_results_lbm.pdf"))
    # plt.show()


if __name__ == "__main__":
    main()
