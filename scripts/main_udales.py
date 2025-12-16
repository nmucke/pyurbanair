import os
import pdb

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import xarray
from data_assimilation.observation_operator import ObservationOperator
from data_assimilation.smoothing.esmda import ESMDA
from pyudales.forward_model import ForwardModel

from pyurbanair.utils.state_utils import get_velocity_magnitude_field

# Random seed
SEED = 42

# Directory settings
MATLAB_BIN = "/Applications/MATLAB_R2025b.app/bin/matlab"
EXPERIMENT_DIR = "examples/udales/experiments/300"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Compute ressources
NCPU = 4

# True parameters
TRUE_PRESSURE_GRADIENT_MAGNITUDE = 0.0041912
TRUE_VELOCITY_MAGNITUDE = 3.0
TRUE_ANGLE = 10.0

# Data assimilation settings
ENSEMBLE_SIZE = 100
NUM_ESMDA_STEPS = 2
ALPHA = 1 / NUM_ESMDA_STEPS

# Observation settings
OBS_IDS_X = [40, 50, 90, 120, 80, 20, 50, 90]
OBS_IDS_Y = [30, 60, 90, 120, 20, 60, 90, 50]
OBS_IDS_Z = [1, 1, 1, 1, 1, 1, 1, 1]
OBS_STATES = ["u", "v", "w"]
NUM_OBS = len(OBS_IDS_X) * len(OBS_STATES)

# Observation error settings
OBS_ERROR_STD = 0.01
C_D = jnp.diag(OBS_ERROR_STD**2 * jnp.ones(NUM_OBS))

# Forward model settings
FIXED_INPUT = {
    "save_only_last_timestep": True,
    "ncpu": NCPU,
    "matlab_bin": MATLAB_BIN,
    "experiment_dir": EXPERIMENT_DIR,
    "verbose": False,
}


def main() -> None:
    """Main function."""

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

    ##### Setup forward model #####
    true_params = xarray.Dataset(
        data_vars={
            "inflow_angle": TRUE_ANGLE,
            "velocity_magnitude": TRUE_VELOCITY_MAGNITUDE,
            "pressure_gradient_magnitude": TRUE_PRESSURE_GRADIENT_MAGNITUDE,
        },
    )
    forward_model = ForwardModel(**FIXED_INPUT)
    forward_model.run_preprocessing()

    ##### Run true simulation #####
    true_state = forward_model(params=true_params)
    true_velocity_field = get_velocity_magnitude_field(true_state)
    true_velocity_field = true_velocity_field[0]

    ##### Setup observations #####
    observation_operator = ObservationOperator(
        OBS_IDS_X, OBS_IDS_Y, OBS_IDS_Z, OBS_STATES
    )
    true_obs = observation_operator(true_state)
    rng_key, subkey = jax.random.split(rng_key)
    true_obs = true_obs + jnp.sqrt(C_D) @ jax.random.normal(subkey, true_obs.shape)

    ##### Run ESMDA #####
    esmda = ESMDA(
        observation_operator=observation_operator,
        forward_model=forward_model,
        C_D=C_D,
        num_steps=NUM_ESMDA_STEPS,
        alpha=ALPHA,
        rng_key=rng_key,
    )
    params, state = esmda(
        params=params_ensemble,
        observations=true_obs,
        return_params_history=True,
        return_state_history=True,
    )

    ensemble_mean_field = state.mean(dim="ensemble")
    velocity_field = get_velocity_magnitude_field(ensemble_mean_field)
    velocity_field = velocity_field[:, 0]

    rmse = [
        jnp.sqrt(
            jnp.mean((velocity_field[i] - true_velocity_field[1, :, :]) ** 2)
        ).item()
        for i in range(NUM_ESMDA_STEPS + 1)
    ]

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
    }
    angle_axvline_args = {"x": TRUE_ANGLE, "color": "red", "linewidth": 3}
    velocity_axvline_args = {
        "x": TRUE_VELOCITY_MAGNITUDE,
        "color": "red",
        "linewidth": 3,
    }
    for i in range(NUM_ESMDA_STEPS + 1):
        im = axes[i, 0].imshow(velocity_field[i, 1, :, :], **im_args)
        im = axes[i, 1].imshow(true_velocity_field[1, :, :], **im_args)
        im = axes[i, 2].imshow(
            velocity_field[i, 1, :, :] - true_velocity_field[1, :, :], **im_args
        )

        axes[i, 3].hist(params.inflow_angle.isel(esmda_step=i).values, **hist_args(i))
        axes[i, 3].set_xlim(-15, 15)
        axes[i, 3].axvline(**angle_axvline_args)
        axes[i, 3].legend()

        if "velocity_magnitude" in params:
            axes[i, 4].hist(
                params.velocity_magnitude.isel(esmda_step=i).values, **hist_args(i)
            )
            axes[i, 4].set_xlim(0, 8)
            axes[i, 4].axvline(**velocity_axvline_args)
            axes[i, 4].legend()

        fig.colorbar(im, ax=axes[i, 0])
        fig.colorbar(im, ax=axes[i, 1])
        fig.colorbar(im, ax=axes[i, 2])

        axes[i, 1].scatter(OBS_IDS_X, OBS_IDS_Y, color="red")
        axes[i, 0].scatter(OBS_IDS_X, OBS_IDS_Y, color="red")

        if i == 0:
            axes[i, 0].set_title("Ensemble mean")
            axes[i, 1].set_title("True")
            axes[i, 3].set_title("Angle distribution")
            if "velocity_magnitude" in params:
                axes[i, 4].set_title("Velocity magnitude distribution")

        axes[i, 2].set_title(f"RMSE: {rmse[i]:.4f}")
    plt.savefig(os.path.join(FIGURES_DIR, "esmda_results.pdf"))
    plt.show()


if __name__ == "__main__":
    main()
