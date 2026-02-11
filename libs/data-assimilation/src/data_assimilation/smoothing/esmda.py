import os
import pathlib
import pdb
import shutil
from typing import Optional

import jax
import jax.numpy as jnp
import xarray
from data_assimilation.observation_operator import ObservationOperator
from data_assimilation.smoothing.base import BaseSmoothing

from pyurbanair.base_forward_model import BaseForwardModel


class ParameterESMDA(BaseSmoothing):
    """Parameter ESMDA smoothing."""

    def __init__(
        self,
        observation_operator: ObservationOperator,
        forward_model: BaseForwardModel,
        C_D: jnp.ndarray,
        num_steps: int = 3,
        alpha: Optional[float] = None,
        rng_key: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(42),
        results_dir: Optional[pathlib.Path] = None,
    ) -> None:
        """Initialize the Parameter ESMDA smoothing."""
        super().__init__(observation_operator, forward_model)

        self.results_dir = results_dir
        if self.results_dir is not None or self.forward_model.save_on_disk:
            if self.results_dir is None:
                self.results_dir = pathlib.Path(".temp/esmda")
                os.makedirs(self.results_dir, exist_ok=True)

            self.save_on_disk = True
            for i in range(num_steps + 1):
                os.makedirs(self.results_dir / f"step_{i}", exist_ok=True)
        else:
            self.save_on_disk = False

        if alpha is None:
            self.alpha = 1 / num_steps
        else:
            self.alpha = alpha
        self.C_D = C_D
        self.C_D_sqrt = jnp.sqrt(self.C_D)
        self.rng_key = rng_key
        self.num_steps = num_steps

    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
    ) -> xarray.Dataset:
        """Perform one ESMDA assimilation step."""

        obs = jnp.asarray(obs)

        pred_obs = self._observation_step(
            state=state, results_dir=self.forward_model.results_dir
        )
        pred_obs = jnp.asarray(pred_obs).T

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

        # Initialize updated parameters
        params_updated = jnp.zeros_like(params_array)  # Shape: [N_e, N_p]

        # Generate random noise
        self.rng_key, subkey = jax.random.split(self.rng_key)
        Z = jax.random.normal(subkey, (N_d, N_e))

        # Generate perturbed observations
        perturbed_obs = obs[:, None] + jnp.sqrt(self.alpha) * (
            self.C_D_sqrt @ Z
        )  # Shape: [N_d, N_e]

        # Compute innovation: d_j - G(m^f_j)
        innovation = perturbed_obs - pred_obs  # Shape: [N_d, N_e]

        # Compute (C^f_DD + α_i * C_D)
        C_DD_alpha = C_DD + self.alpha * self.C_D

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

        return xarray.Dataset(
            data_vars=updated_data_vars,
            coords=params.coords,
        )

    def get_state(
        self,
        ensemble_member: int,
        step: int,
    ) -> xarray.Dataset:
        """Get the state from the results directory."""
        return xarray.open_dataset(
            self.results_dir / f"step_{step}" / f"state_{ensemble_member}.nc"  # type: ignore[operator]
        )

    def _save_states_to_disk(self, step: int) -> None:
        """Save the states to disk."""

        src_dir = self.forward_model.results_dir
        for f in pathlib.Path(src_dir).iterdir():
            if f.suffix == ".nc":
                target = self.results_dir / f"step_{step}" / f"{f.name}"  # type: ignore[operator]
                shutil.move(str(f), str(target))

    def _analysis(
        self,
        params: xarray.Dataset,
        observations: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
        return_params_history: bool = False,
        return_state_history: bool = False,
    ) -> xarray.Dataset:
        """Perform the ESMDA analysis."""
        if return_params_history:
            params_history = [params]
        if return_state_history:
            state_history = []
        for i in range(self.num_steps):
            state = self._forecast_step(state=state, params=params)

            params = self._one_step(params=params, obs=observations, state=state)
            if return_params_history:
                params_history.append(params)

            if return_state_history:
                if self.forward_model.save_on_disk:
                    self._save_states_to_disk(step=i)
                else:
                    state_history.append(state)

            print(f"ESMDA step {i} completed")

        if return_params_history:
            params_history = xarray.concat(
                params_history, dim="esmda_step", join="override"
            )

        if return_state_history:
            state = self._forecast_step(state=state, params=params)
            if self.forward_model.save_on_disk:
                self._save_states_to_disk(step=self.num_steps)
            else:
                state_history.append(state)
                state_history = xarray.concat(
                    state_history, dim="esmda_step", join="override"
                )

        if self.save_on_disk:
            if return_params_history:
                return params_history
            return params
        if return_params_history and return_state_history:
            return params_history, state_history
        elif return_params_history:
            return params_history
        elif return_state_history:
            return state_history
        else:
            return params


class StateAndParameterESMDA(BaseSmoothing):
    """State and Parameter ESMDA smoothing."""

    def __init__(
        self,
        observation_operator: ObservationOperator,
        forward_model: BaseForwardModel,
        C_D: jnp.ndarray,
        num_steps: int = 3,
        alpha: Optional[float] = None,
        rng_key: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(42),
        results_dir: Optional[pathlib.Path] = None,
    ) -> None:
        """Initialize the State and Parameter ESMDA smoothing."""
        super().__init__(observation_operator, forward_model)

        self.results_dir = results_dir
        if self.results_dir is not None or self.forward_model.save_on_disk:
            if self.results_dir is None:
                self.results_dir = pathlib.Path(".temp/esmda")
                os.makedirs(self.results_dir, exist_ok=True)

            self.save_on_disk = True
            for i in range(num_steps + 1):
                os.makedirs(self.results_dir / f"step_{i}", exist_ok=True)
        else:
            self.save_on_disk = False

        if alpha is None:
            self.alpha = 1 / num_steps
        else:
            self.alpha = alpha
        self.C_D = C_D
        self.C_D_sqrt = jnp.sqrt(self.C_D)
        self.rng_key = rng_key
        self.num_steps = num_steps

    def _get_states(
        self,
        state: Optional[xarray.Dataset] = None,
        results_dir: Optional[pathlib.Path] = None,
    ) -> jnp.ndarray:
        """Get the states from the results directory."""
        if state is not None:
            states = state.isel(time=0)
        elif results_dir is not None:
            states = [
                xarray.open_dataset(f).isel(time=-1)
                for f in pathlib.Path(results_dir).iterdir()
            ]
            states = xarray.concat(states, dim="ensemble", join="override")
        return states

    def _create_empty_state(self, state: xarray.Dataset) -> xarray.Dataset:
        """Create an empty state."""
        return xarray.Dataset(
            {
                var: (
                    state[var].dims,
                    jnp.empty(state[var].shape, dtype=state[var].dtype),
                )
                for var in state.data_vars
            },
            coords={coord: state.coords[coord] for coord in state.coords},
        )

    def _flatten_state(self, state: xarray.Dataset) -> jnp.ndarray:
        """Flatten the state xarray dataset into a JAX array.

        The output array will have the shape (degrees_of_freedom, ensemble_size).

        Args:
            state (xarray.Dataset): The state ensemble, where each data variable
                                    is expected to have an 'ensemble' dimension.

        Returns:
            jnp.ndarray: A JAX array of shape (degrees_of_freedom, ensemble_size).
        """
        flat_vars = []
        # Sorting ensures a consistent order for flattening and unflattening
        for var_name in sorted(state.data_vars):
            variable = state[var_name]

            # Transpose to bring 'ensemble' to the first position, then flatten
            # all other dimensions. The '...' is a placeholder for all other dimensions.
            flat_var = variable.transpose("ensemble", ...).values.reshape(
                variable.sizes["ensemble"], -1
            )

            # Transpose to get (degrees_of_freedom_for_var, ensemble_size)
            flat_vars.append(flat_var.T)

        return jnp.concatenate(flat_vars, axis=0)

    def _unflatten_state(
        self,
        states_array: jnp.ndarray,
        state_template: xarray.Dataset,
    ) -> xarray.Dataset:
        """Unflatten a JAX array into an xarray dataset.

        This function is the inverse of _flatten_state.

        Args:
            states_array (jnp.ndarray): JAX array of shape (degrees_of_freedom, ensemble_size).
            state_template (xarray.Dataset): A template xarray dataset with the desired
                output structure, including coordinates and dimension information.

        Returns:
            xarray.Dataset: The unflattened state ensemble.
        """
        new_data_vars = {}
        ensemble_size = states_array.shape[1]

        current_pos = 0
        # Sorting ensures a consistent order, matching _flatten_state
        for var_name in sorted(state_template.data_vars):
            template_var = state_template[var_name]

            # Determine the size of the flattened variable (excluding the ensemble dimension)
            var_flat_size = template_var.size // template_var.sizes["ensemble"]

            # Extract data chunk for the current variable
            flat_var_chunk = states_array[current_pos : current_pos + var_flat_size, :]
            current_pos += var_flat_size

            # Reshape the data
            # 1. Transpose chunk to (ensemble_size, var_flat_size)
            # 2. Reshape to (ensemble_size, *shape_without_ensemble)
            dims_no_ensemble = [d for d in template_var.dims if d != "ensemble"]
            shape_no_ensemble = [template_var.sizes[d] for d in dims_no_ensemble]
            data = flat_var_chunk.T.reshape((ensemble_size, *shape_no_ensemble))

            # Create a DataArray with the correct dimensions to allow for transposing
            new_dims_order = ["ensemble"] + dims_no_ensemble
            data_array = xarray.DataArray(data, dims=new_dims_order)

            # Transpose to match the original dimension order of the template variable
            # and add it to our dictionary for the new dataset
            new_data_vars[var_name] = data_array.transpose(*template_var.dims)

        return xarray.Dataset(new_data_vars, coords=state_template.coords)

    def _one_step(
        self,
        params: xarray.Dataset,
        state: xarray.Dataset,
        obs: jnp.ndarray,
    ) -> xarray.Dataset:
        """Perform one ESMDA assimilation step."""

        obs = jnp.asarray(obs)

        pred_obs = self._observation_step(
            state=state,
            results_dir=(
                self.forward_model.results_dir
                if self.forward_model.save_on_disk
                else None
            ),
        )
        pred_obs = jnp.asarray(pred_obs).T

        # Get the states from the results directory
        states_array = self._get_states(
            state=state,
            results_dir=(
                self.forward_model.results_dir
                if self.forward_model.save_on_disk
                else None
            ),
        )

        # Create a copy of states_array with same sizes and data_vars but no values
        state_template = self._create_empty_state(states_array)

        states_array = self._flatten_state(states_array)

        # Extract parameter names and values
        param_names = list(params.data_vars.keys())
        state_names = list(state.data_vars.keys())
        N_e = params.sizes["ensemble"]  # Number of ensemble members
        N_p = len(param_names)  # Number of parameters
        N_s = states_array.shape[0]  # Number of state variables
        N_d = len(obs)  # Number of observations

        # Extract parameters as array of shape [N_p, N_e]
        params_array = [params[param_name].values for param_name in param_names]
        params_array = jnp.array(params_array)  # Shape: [N_p, N_e]

        # Concatenate params_array and states_array
        states_array = jnp.concatenate([states_array, params_array], axis=0)

        # Compute ensemble means
        # params_mean = jnp.mean(params_array, axis=1)  # Shape: [N_p]
        states_mean = jnp.mean(states_array, axis=1)  # Shape: [N_s]
        pred_obs_mean = jnp.mean(pred_obs, axis=1)  # Shape: [N_d]

        # Compute deviations from means
        # params_dev = params_array - params_mean[:, None]  # Shape: [N_p, N_e]
        states_dev = states_array - states_mean[:, None]  # Shape: [N_s, N_e]
        pred_obs_dev = pred_obs - pred_obs_mean[:, None]  # Shape: [N_d, N_e]

        # Compute cross-covariance C^f_MD between model state and parameters and data
        # C^f_MD = (1/(N_e-1)) * sum_j (m^f_j - m^f_mean) * (G(m^f_j) - G_mean)^T
        C_MD = jnp.dot(states_dev, pred_obs_dev.T) / (N_e - 1)  # Shape: [N_p, N_d]

        # Compute auto-covariance C^f_DD of the data
        # C^f_DD = (1/(N_e-1)) * sum_j (G(m^f_j) - G_mean) * (G(m^f_j) - G_mean)^T
        C_DD = jnp.dot(pred_obs_dev, pred_obs_dev.T) / (N_e - 1)  # Shape: [N_d, N_d]

        # Generate random noise
        self.rng_key, subkey = jax.random.split(self.rng_key)
        Z = jax.random.normal(subkey, (N_d, N_e))

        # Generate perturbed observations
        perturbed_obs = obs[:, None] + jnp.sqrt(self.alpha) * (
            self.C_D_sqrt @ Z
        )  # Shape: [N_d, N_e]

        # Compute innovation: d_j - G(m^f_j)
        innovation = perturbed_obs - pred_obs  # Shape: [N_d, N_e]

        # Compute (C^f_DD + α_i * C_D)
        C_DD_alpha = C_DD + self.alpha * self.C_D

        # Solve (C^f_DD + α_i * C_D) * x = innovation for x
        try:
            x = jnp.linalg.solve(C_DD_alpha, innovation)
        except jnp.linalg.LinAlgError:
            # If solve fails, use least squares
            x = jnp.linalg.lstsq(C_DD_alpha, innovation, rcond=None)[0]

        # Update parameters: m^a_j = m^f_j + C^f_MD * x
        # C_MD is [N_p, N_d], x is [N_d], so C_MD @ x is [N_p]
        states_array = states_array + C_MD @ x

        # Extract updated parameters
        params_updated = states_array[N_s:, :]
        states_array = states_array[:N_s, :]

        # Reconstruct xarray.Dataset with updated parameters
        updated_data_vars = {}
        for i, param_name in enumerate(param_names):
            updated_data_vars[param_name] = ("ensemble", params_updated[i, :])

        states_array = self._unflatten_state(states_array, state_template)

        # import matplotlib.pyplot as plt

        # from pyurbanair.utils.state_utils import get_velocity_magnitude_field

        # velocity_field = get_velocity_magnitude_field(states_array)

        # plt.figure()
        # plt.imshow(velocity_field[0, 1, :, :])
        # plt.colorbar()
        # plt.show()

        # pdb.set_trace()
        return (
            states_array,
            xarray.Dataset(
                data_vars=updated_data_vars,
                coords=params.coords,
            ),
        )

    def get_state(
        self,
        ensemble_member: int,
        step: int,
    ) -> xarray.Dataset:
        """Get the state from the results directory."""
        return xarray.open_dataset(
            self.results_dir / f"step_{step}" / f"state_{ensemble_member}.nc"  # type: ignore[operator]
        )

    def _save_states_to_disk(self, step: int) -> None:
        """Save the states to disk."""

        src_dir = self.forward_model.results_dir
        for f in pathlib.Path(src_dir).iterdir():
            if f.suffix == ".nc":
                target = self.results_dir / f"step_{step}" / f"{f.name}"  # type: ignore[operator]
                shutil.move(str(f), str(target))

    def _analysis(
        self,
        params: xarray.Dataset,
        observations: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
        return_params_history: bool = False,
        return_state_history: bool = False,
    ) -> xarray.Dataset:
        """Perform the ESMDA analysis."""
        if return_params_history:
            params_history = [params]
        if return_state_history:
            state_history = []
        for i in range(self.num_steps):
            state = self._forecast_step(state=state, params=params)

            if return_state_history:
                if self.forward_model.save_on_disk:
                    self._save_states_to_disk(step=i)
                else:
                    state_history.append(state)

            state, params = self._one_step(params=params, obs=observations, state=state)

            if return_params_history:
                params_history.append(params)

            print(f"ESMDA step {i} completed")

        state = self._forecast_step(state=state, params=params)

        if return_state_history:
            if self.forward_model.save_on_disk:
                self._save_states_to_disk(step=i)
            else:
                state_history.append(state)
                state_history = xarray.concat(
                    state_history, dim="esmda_step", join="override"
                )

        if return_params_history:
            params_history = xarray.concat(
                params_history, dim="esmda_step", join="override"
            )

        if self.save_on_disk:
            if return_params_history:
                return params_history
            return params
        if return_params_history and return_state_history:
            return params_history, state_history
        elif return_params_history:
            return params_history
        elif return_state_history:
            return state_history
        else:
            return params, state
