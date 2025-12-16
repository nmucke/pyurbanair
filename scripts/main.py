import os
import pdb

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import xarray
from data_assimilation.observation_operator import ObservationOperator
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
NUM_ESMDA_STEPS = 5
ALPHA = 1 / NUM_ESMDA_STEPS

# Observation settings
OBS_IDS_X = [40, 50, 90, 120]  # , 80, 20, 50, 90]
OBS_IDS_Y = [30, 60, 90, 120]  # , 20, 60, 90, 50]
OBS_IDS_Z = [1, 1, 1, 1]  # , 1, 1, 1, 1]
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


def esmda_step(
    params: xarray.Dataset,
    obs: jnp.ndarray,
    pred_obs: jnp.ndarray,
    alpha: float,
    C_D: jnp.ndarray,
    rng_key: jax.random.PRNGKey,
) -> xarray.Dataset:
    """Perform one ESMDA assimilation step.

    Args:
        params: Forecasted parameters for each ensemble member as xarray.Dataset
            with data variables for each parameter and 'ensemble' dimension.
            Shape: [N_e] for each parameter variable, where N_e is ensemble size.
        obs: Observed data d_obs, shape [N_d] where N_d is number of observations
        pred_obs: Predicted observations G(m^f_j) for each ensemble member, shape [N_e, N_d]
        alpha: Scaling factor α_i for this assimilation step
        C_D: Measurement error covariance matrix, shape [N_d, N_d] or [N_d] for diagonal
        rng_key: Random number generator key
    Returns:
        Updated parameters m^a_j as xarray.Dataset with same structure as input params
    """
    obs = jnp.asarray(obs)
    pred_obs = jnp.asarray(pred_obs).T
    C_D = jnp.asarray(C_D)

    # Extract parameter names and values
    param_names = list(params.data_vars.keys())
    N_e = params.sizes["ensemble"]  # Number of ensemble members
    N_p = len(param_names)  # Number of parameters
    N_d = len(obs)  # Number of observations

    # Extract parameters as array of shape [N_p, N_e]
    params_array = [params[param_name].values for param_name in param_names]
    params_array = jnp.array(params_array)  # Shape: [N_p, N_e]

    # Compute ensemble means
    params_mean = jnp.mean(params_array, axis=1)  # Shape: [N_p]
    pred_obs_mean = jnp.mean(pred_obs, axis=1)  # Shape: [N_d]

    # Compute deviations from means
    params_dev = params_array - params_mean[:, None]  # Shape: [N_p, N_e]
    pred_obs_dev = pred_obs - pred_obs_mean[:, None]  # Shape: [N_d, N_e]

    # Compute cross-covariance C^f_MD between model parameters and data
    # C^f_MD = (1/(N_e-1)) * sum_j (m^f_j - m^f_mean) * (G(m^f_j) - G_mean)^T
    C_MD = jnp.dot(params_dev, pred_obs_dev.T) / (N_e - 1)  # Shape: [N_p, N_d]

    # Compute auto-covariance C^f_DD of the data
    # C^f_DD = (1/(N_e-1)) * sum_j (G(m^f_j) - G_mean) * (G(m^f_j) - G_mean)^T
    C_DD = jnp.dot(pred_obs_dev, pred_obs_dev.T) / (N_e - 1)  # Shape: [N_d, N_d]

    # C_D is diagonal, use diagonal matrix for C_D_sqrt
    C_D_sqrt = jnp.sqrt(C_D)  # Ensure positive

    # Initialize updated parameters
    params_updated = jnp.zeros_like(params_array)  # Shape: [N_e, N_p]

    # Generate random noise
    rng_key, subkey = jax.random.split(rng_key)
    Z = jax.random.normal(subkey, (N_d, N_e))

    # Generate perturbed observations
    perturbed_obs = obs[:, None] + jnp.sqrt(alpha) * (C_D_sqrt @ Z)  # Shape: [N_d, N_e]

    # Compute innovation: d_j - G(m^f_j)
    innovation = perturbed_obs - pred_obs  # Shape: [N_d, N_e]

    # Compute (C^f_DD + α_i * C_D)
    C_DD_alpha = C_DD + alpha * C_D

    # Solve (C^f_DD + α_i * C_D) * x = innovation for x
    try:
        x = jnp.linalg.solve(C_DD_alpha, innovation)
    except jnp.linalg.LinAlgError:
        # If solve fails, use least squares
        x = jnp.linalg.lstsq(C_DD_alpha, innovation, rcond=None)[0]

    # Update parameters: m^a_j = m^f_j + C^f_MD * x
    # C_MD is [N_p, N_d], x is [N_d], so C_MD @ x is [N_p]
    params_updated = params_array + C_MD @ x

    # Reconstruct xarray.Dataset with updated parameters
    updated_data_vars = {}
    for i, param_name in enumerate(param_names):
        updated_data_vars[param_name] = ("ensemble", params_updated[i, :])

    return (
        xarray.Dataset(
            data_vars=updated_data_vars,
            coords=params.coords,
        ),
        rng_key,
    )


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
    params_history = params_ensemble.copy()
    velocity_field_history = []
    rmse = []
    for i in range(NUM_ESMDA_STEPS):
        states = forward_model.run_ensemble(params=params_ensemble)
        pred_obs = observation_operator(states)
        params_ensemble, rng_key = esmda_step(
            params_ensemble, true_obs, pred_obs, ALPHA, C_D, rng_key
        )
        params_history = xarray.concat(
            [params_history, params_ensemble], dim="esmda_step", join="override"
        )

        ensemble_mean_field = states.mean(dim="ensemble")
        velocity_field = get_velocity_magnitude_field(ensemble_mean_field)
        velocity_field_history.append(velocity_field[0])

        rmse.append(jnp.sqrt(jnp.mean((velocity_field - true_velocity_field) ** 2)))

    states = forward_model.run_ensemble(params=params_ensemble)

    ensemble_mean_field = states.mean(dim="ensemble")
    velocity_field = get_velocity_magnitude_field(ensemble_mean_field)
    velocity_field_history.append(velocity_field[0])

    rmse.append(jnp.sqrt(jnp.mean((velocity_field - true_velocity_field) ** 2)))

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
        im = axes[i, 0].imshow(velocity_field_history[i][1, :, :], **im_args)
        im = axes[i, 1].imshow(true_velocity_field[1, :, :], **im_args)
        im = axes[i, 2].imshow(
            velocity_field_history[i][1, :, :] - true_velocity_field[1, :, :], **im_args
        )

        axes[i, 3].hist(params_history.inflow_angle.values[i], **hist_args(i))
        axes[i, 3].set_xlim(-15, 15)
        axes[i, 3].axvline(**angle_axvline_args)
        axes[i, 3].legend()

        if "velocity_magnitude" in params_history:
            axes[i, 4].hist(params_history.velocity_magnitude.values[i], **hist_args(i))
            axes[i, 4].set_xlim(0, 6)
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
            if "velocity_magnitude" in params_history:
                axes[i, 4].set_title("Velocity magnitude distribution")

        axes[i, 2].set_title(f"RMSE: {rmse[i]:.4f}")
    plt.savefig(os.path.join(FIGURES_DIR, "esmda_results.pdf"))
    plt.show()


if __name__ == "__main__":
    main()
