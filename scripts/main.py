import pdb

import matplotlib.pyplot as plt
import numpy as np
import xarray
from pyudales.forward_model import ForwardModel

# Directory settings
MATLAB_BIN = "/Applications/MATLAB_R2025b.app/bin/matlab"
EXPERIMENT_DIR = "examples/udales/experiments/300"

# Fixed parameters
VELOCITY_MAGNITUDE = 3.0
PRESSURE_GRADIENT_MAGNITUDE = 0.0041912
ANGLE_RANGE = np.random.randn(5) * 8
TRUE_ANGLE = 10.0
OBS_IDS_X = [40, 50, 90, 120, 80, 20, 50, 90]
OBS_IDS_Y = [30, 60, 90, 120, 20, 60, 90, 50]
NUM_OBS = len(OBS_IDS_X)

NUM_ESMDA_STEPS = 3
ALPHA = 1 / NUM_ESMDA_STEPS
U_ERROR_STD = 0.05
V_ERROR_STD = 0.05
W_ERROR_STD = 0.05
C_D = np.diag(
    np.concatenate(
        [
            U_ERROR_STD**2 * np.ones(NUM_OBS),
            V_ERROR_STD**2 * np.ones(NUM_OBS),
            W_ERROR_STD**2 * np.ones(NUM_OBS),
        ]
    ),
)


FIXED_INPUT = {
    "save_only_last_timestep": True,
    "ncpu": 4,
    "matlab_bin": MATLAB_BIN,
    "experiment_dir": EXPERIMENT_DIR,
}

class ObservationOperator:
    def __init__(self, obs_ids_x: list[int], obs_ids_y: list[int]):
        self.obs_ids_x = obs_ids_x
        self.obs_ids_y = obs_ids_y
        self.num_obs = len(obs_ids_x)

    def _observation_one_state(self, state: xarray.Dataset) -> xarray.Dataset:
        """Apply observation operator to one state.
        
        Args:
            state: xarray Dataset.
            
        Returns:
            Vector of shape (num_obs * 3) where num_obs = NUM_OBS.
        """
        u = state.u.values
        v = state.v.values
        w = state.w.values
        u_obs = np.zeros((NUM_OBS))
        v_obs = np.zeros((NUM_OBS))
        w_obs = np.zeros((NUM_OBS))
        for i in range(NUM_OBS):
            u_obs[i] = u[0, 1, OBS_IDS_Y[i], OBS_IDS_X[i]]
            v_obs[i] = v[0, 1, OBS_IDS_Y[i], OBS_IDS_X[i]]
            w_obs[i] = w[0, 1, OBS_IDS_Y[i], OBS_IDS_X[i]]
        return np.concatenate([u_obs, v_obs, w_obs], axis=0)
    
    def _observation_ensemble(self, states: xarray.Dataset) -> np.ndarray:
        """Apply observation operator to each ensemble member.
        
        Args:
            states: xarray Dataset with ensemble dimension.
            
        Returns:
            Matrix of shape (ensemble, num_obs) where num_obs = NUM_OBS * 3.
        """
        ensemble_size = states.sizes['ensemble']
        num_obs = NUM_OBS * 3  # u, v, w for each observation point
        obs_matrix = np.zeros((ensemble_size, num_obs))
        
        for i in range(ensemble_size):
            obs_matrix[i, :] = self._observation_one_state(
                states.isel(ensemble=i)
            )

        return obs_matrix

    def __call__(self, state: xarray.Dataset) -> np.ndarray:
        if "ensemble" in state.dims:
            return self._observation_ensemble(state)
        else:
            return self._observation_one_state(state)


# def observation_operator(state: xarray.Dataset) -> xarray.Dataset:
#     """Observation operator."""
#     u = state.u.values
#     v = state.v.values
#     w = state.w.values

#     u_obs = np.zeros((NUM_OBS))
#     v_obs = np.zeros((NUM_OBS))
#     w_obs = np.zeros((NUM_OBS))

#     if "ensemble" in state.dims:


#     return np.concatenate([u_obs, v_obs, w_obs], axis=0)


def simulate_ensemble(
    forward_model: ForwardModel,
    angles: np.ndarray,
) -> list[xarray.Dataset]:
    """
    Simulate an ensemble of forward model runs for a given set of angles.

    Args:
        forward_model: The forward model instance.
        angles: A list of angles to simulate.

    Returns:
        A list of ensemble states.
    """
    ensemble_states: list[xarray.Dataset] = []
    for angle in angles:
        forward_model._apply_inflow_settings(
            inflow_angle=angle,
            velocity_magnitude=VELOCITY_MAGNITUDE,
            pressure_gradient_magnitude=PRESSURE_GRADIENT_MAGNITUDE,
        )
        ensemble_states.append(forward_model.run())
    return ensemble_states


def get_velocity_magnitude_field(state: xarray.Dataset) -> np.ndarray:
    """Get the velocity magnitude field from a state."""
    u = state.u.values
    v = state.v.values
    w = state.w.values
    return np.sqrt(u**2 + v**2 + w**2)


def get_ensemble_mean_field(states: list[xarray.Dataset]) -> xarray.Dataset:
    """Get the ensemble mean field from a list of states."""
    u = np.stack([state.u.values for state in states], axis=0)
    v = np.stack([state.v.values for state in states], axis=0)
    w = np.stack([state.w.values for state in states], axis=0)
    return xarray.Dataset(
        data_vars={
            "u": (["time", "z", "y", "x"], np.mean(u, axis=0)),
            "v": (["time", "z", "y", "x"], np.mean(v, axis=0)),
            "w": (["time", "z", "y", "x"], np.mean(w, axis=0)),
        },
        coords={
            "time": [0],
            "zt": states[0].zt.values,
            "yt": states[0].yt.values,
            "xt": states[0].xt.values,
            "zm": states[0].zm.values,
            "ym": states[0].ym.values,
            "xm": states[0].xm.values,
        },
    )


def esmda_step(
    params: xarray.Dataset,
    obs: np.ndarray,
    pred_obs: np.ndarray,
    alpha: float,
    C_D: np.ndarray,
) -> np.ndarray:
    """Perform one ESMDA assimilation step.

    Args:
        angles: Forecasted angles for each ensemble member, shape [N_e]
        obs: Observed data d_obs, shape [N_d] where N_d is number of observations
        pred_obs: Predicted observations G(m^f_j) for each ensemble member, shape [N_e, N_d]
        alpha: Scaling factor α_i for this assimilation step
        C_D: Measurement error covariance matrix, shape [N_d, N_d] or [N_d] for diagonal

    Returns:
        Updated angles m^a_j for each ensemble member, shape [N_e]
    """
    params = [params.variables[var] for var in params.data_vars]
    params = np.asarray(params).squeeze()
    obs = np.asarray(obs)
    pred_obs = np.asarray(pred_obs)
    C_D = np.asarray(C_D)

    N_e = len(params)  # Number of ensemble members
    N_d = len(obs)  # Number of observations

    # Ensure pred_obs has correct shape
    if pred_obs.shape != (N_e, N_d):
        raise ValueError(
            f"pred_obs must have shape ({N_e}, {N_d}), got {pred_obs.shape}"
        )

    # Handle C_D: if 1D, assume it's diagonal variances
    if C_D.ndim == 1:
        if len(C_D) != N_d:
            raise ValueError(f"If C_D is 1D, it must have length {N_d}, got {len(C_D)}")
        C_D = np.diag(C_D)
    elif C_D.shape != (N_d, N_d):
        raise ValueError(
            f"C_D must have shape ({N_d}, {N_d}) or ({N_d},), got {C_D.shape}"
        )

    # Compute ensemble means
    params_mean = np.mean(params)
    pred_obs_mean = np.mean(pred_obs, axis=0)  # Mean over ensemble members

    # Compute deviations from means
    params_dev = params - params_mean  # Shape: [N_e]
    pred_obs_dev = pred_obs - pred_obs_mean  # Shape: [N_e, N_d]

    # Compute cross-covariance C^f_MD between model parameters (angles) and data
    # C^f_MD = (1/(N_e-1)) * sum_j (m^f_j - m^f_mean) * (G(m^f_j) - G_mean)^T
    # For scalar m (one angle), C_MD should be shape [1, N_d] or [N_d] as a vector
    C_MD = (1.0 / (N_e - 1)) * np.dot(params_dev, pred_obs_dev)  # Shape: [N_d]

    # Compute auto-covariance C^f_DD of the data
    # C^f_DD = (1/(N_e-1)) * sum_j (G(m^f_j) - G_mean) * (G(m^f_j) - G_mean)^T
    C_DD = (1.0 / (N_e - 1)) * np.dot(pred_obs_dev.T, pred_obs_dev)  # Shape: [N_d, N_d]

    # Compute Cholesky decomposition of C_D for generating perturbations
    # For diagonal C_D, this is more efficient
    if np.allclose(C_D, np.diag(np.diag(C_D))):
        # C_D is diagonal, use simpler computation
        C_D_diag = np.diag(C_D)
        C_D_sqrt_diag = np.sqrt(np.maximum(C_D_diag, 1e-10))  # Ensure positive
        C_D_sqrt = np.diag(C_D_sqrt_diag)
    else:
        # Full covariance matrix
        try:
            C_D_sqrt = np.linalg.cholesky(C_D)  # C_D = C_D_sqrt @ C_D_sqrt.T
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(C_D)
            eigenvals = np.maximum(eigenvals, 1e-10)  # Ensure positive
            C_D_sqrt = eigenvecs @ np.diag(np.sqrt(eigenvals))

    # Initialize updated angles
    params_updated = np.zeros_like(params)

    # Process each ensemble member
    for j in range(N_e):
        # Equation 10: Generate perturbed observations d_j
        # d_j = d_obs + sqrt(alpha) * C_D^{1/2} * z
        # where z ~ N(0, I)
        z = np.random.randn(N_d)
        d_j = obs + np.sqrt(alpha) * (C_D_sqrt @ z)

        # Equation 9: Update ensemble member
        # m^a_j = m^f_j + C^f_MD * (C^f_DD + α_i * C_D)^{-1} * (d_j - G(m^f_j))

        # Compute innovation: d_j - G(m^f_j)
        innovation = d_j - pred_obs[j, :]  # Shape: [N_d]

        # Compute (C^f_DD + α_i * C_D)
        C_DD_alpha = C_DD + alpha * C_D

        # Solve (C^f_DD + α_i * C_D) * x = innovation for x
        try:
            x = np.linalg.solve(C_DD_alpha, innovation)
        except np.linalg.LinAlgError:
            # If solve fails, use least squares
            x = np.linalg.lstsq(C_DD_alpha, innovation, rcond=None)[0]

        # Update angle: m^a_j = m^f_j + C^f_MD * x
        params_updated[j] = params[j] + np.dot(C_MD, x)

    return params_updated


def main() -> None:
    """Main function."""

    true_params = xarray.Dataset(
        data_vars={
            "inflow_angle": TRUE_ANGLE,
            "velocity_magnitude": VELOCITY_MAGNITUDE,
            "pressure_gradient_magnitude": PRESSURE_GRADIENT_MAGNITUDE,
        },
    )
    forward_model = ForwardModel(**FIXED_INPUT)  # type: ignore[arg-type]
    forward_model.run_preprocessing()
    true_state = forward_model(true_params)

    observation_operator = ObservationOperator(OBS_IDS_X, OBS_IDS_Y)
    true_obs = observation_operator(true_state)
    true_obs = true_obs + np.sqrt(C_D) @ np.random.randn(true_obs.shape[0])
    
    true_velocity_field = get_velocity_magnitude_field(true_state)
    true_velocity_field = true_velocity_field[0]

    params = xarray.Dataset(
        data_vars={"inflow_angle": ("ensemble", ANGLE_RANGE)},
        coords={"ensemble": np.arange(len(ANGLE_RANGE))},
    )
    params_history = params.copy()
    velocity_field_history = []
    rmse = []
    for i in range(NUM_ESMDA_STEPS):
        states = forward_model.run_ensemble(params)
        pred_obs = observation_operator(states)
        params = esmda_step(params, true_obs, pred_obs, ALPHA, C_D)
        params = xarray.Dataset(
            data_vars={"inflow_angle": ("ensemble", params)},
            coords={"ensemble": np.arange(len(ANGLE_RANGE))},
        )
        params_history = xarray.concat([params_history, params], dim="esmda_step")

        ensemble_mean_field = states.mean(dim="ensemble")
        velocity_field = get_velocity_magnitude_field(ensemble_mean_field)
        velocity_field_history.append(velocity_field[0])

        rmse.append(np.sqrt(np.mean((velocity_field - true_velocity_field) ** 2)))

    states = forward_model.run_ensemble(params)

    ensemble_mean_field = states.mean(dim="ensemble")
    velocity_field = get_velocity_magnitude_field(ensemble_mean_field)
    velocity_field_history.append(velocity_field[0])

    rmse.append(np.sqrt(np.mean((velocity_field - true_velocity_field) ** 2)))

    fig, axes = plt.subplots(
        NUM_ESMDA_STEPS + 1, 4, figsize=(16, 4 * (NUM_ESMDA_STEPS + 1))
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
    axvline_args = {"x": TRUE_ANGLE, "color": "red", "linewidth": 3}
    for i in range(NUM_ESMDA_STEPS + 1):
        im = axes[i, 0].imshow(velocity_field_history[i][1, :, :], **im_args)
        im = axes[i, 1].imshow(true_velocity_field[1, :, :], **im_args)
        im = axes[i, 2].imshow(
            velocity_field_history[i][1, :, :] - true_velocity_field[1, :, :], **im_args
        )

        axes[i, 3].hist(params_history.inflow_angle.values[i], **hist_args(i))
        axes[i, 3].set_xlim(-15, 15)
        axes[i, 3].axvline(**axvline_args)
        axes[i, 3].legend()

        fig.colorbar(im, ax=axes[i, 0])
        fig.colorbar(im, ax=axes[i, 1])
        fig.colorbar(im, ax=axes[i, 2])

        axes[i, 1].scatter(OBS_IDS_X, OBS_IDS_Y, color="red")
        axes[i, 0].scatter(OBS_IDS_X, OBS_IDS_Y, color="red")

        if i == 0:
            axes[i, 0].set_title("Ensemble mean")
            axes[i, 1].set_title("True")
            axes[i, 3].set_title("Angle distribution")

        axes[i, 2].set_title(f"RMSE: {rmse[i]:.4f}")
    plt.savefig("esmda_results.pdf")
    plt.show()


if __name__ == "__main__":
    main()
