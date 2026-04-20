import os
import pathlib
from abc import abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
import xarray
from data_assimilation.observation_operator import ObservationOperator
from data_assimilation.smoothing.base import BaseSmoothing
from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel


class _BaseESMDA(BaseSmoothing):
    """Shared ESMDA logic for parameter-only and joint state-parameter variants."""

    def __init__(
        self,
        observation_operator: ObservationOperator,
        forward_model: BaseEnsembleForwardModel,
        C_D: jnp.ndarray,
        num_steps: int = 3,
        alpha: Optional[float] = None,
        rng_key: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(42),
    ) -> None:
        super().__init__(observation_operator, forward_model)

        self.alpha = num_steps if alpha is None else alpha
        self.C_D = C_D
        self.C_D_sqrt = jnp.sqrt(self.C_D)
        self.rng_key = rng_key
        self.num_steps = num_steps

        if self.forward_model.save_on_disk:
            self.base_results_dir = self.forward_model.results_dir
            for i in range(num_steps + 1):
                step_dir = self.base_results_dir / f"step_{i}"
                os.makedirs(step_dir, exist_ok=True)
                for state_file in step_dir.glob("state_*.nc"):
                    state_file.unlink(missing_ok=True)

    def _set_step_results_dir(self, step: int) -> None:
        """Point the forward model's results directory at the given step."""
        if self.forward_model.save_on_disk:
            self.forward_model.set_results_dir(self.base_results_dir / f"step_{step}")

    def get_state(self, ensemble_member: int, step: int) -> xarray.Dataset:
        """Get the state for one ensemble member at a given step."""
        return xarray.open_dataset(
            self.base_results_dir / f"step_{step}" / f"state_{ensemble_member}.nc"
        )

    def _compute_kalman_update(
        self,
        augmented: jnp.ndarray,
        pred_obs: jnp.ndarray,
        obs: jnp.ndarray,
        N_e: int,
    ) -> jnp.ndarray:
        """Compute the ESMDA Kalman update for an augmented state vector.

        Args:
            augmented: Array of shape (N_aug, N_e) — the augmented state
                (parameters only, or state + parameters).
            pred_obs: Array of shape (N_d, N_e) — predicted observations.
            obs: Array of shape (N_d,) — true observations.
            N_e: Ensemble size.

        Returns:
            Updated augmented array of the same shape.
        """
        N_d = obs.shape[0]

        aug_mean = jnp.mean(augmented, axis=1, keepdims=True)
        pred_obs_mean = jnp.mean(pred_obs, axis=1, keepdims=True)

        aug_dev = augmented - aug_mean
        pred_obs_dev = pred_obs - pred_obs_mean

        C_MD = jnp.dot(aug_dev, pred_obs_dev.T) / (N_e - 1)
        C_DD = jnp.dot(pred_obs_dev, pred_obs_dev.T) / (N_e - 1)

        self.rng_key, subkey = jax.random.split(self.rng_key)
        Z = jax.random.normal(subkey, (N_d, N_e))
        perturbed_obs = obs[:, None] + jnp.sqrt(self.alpha) * (self.C_D_sqrt @ Z)

        innovation = perturbed_obs - pred_obs
        C_DD_alpha = C_DD + self.alpha * self.C_D

        try:
            x = jnp.linalg.solve(C_DD_alpha, innovation)
        except jnp.linalg.LinAlgError:
            x = jnp.linalg.lstsq(C_DD_alpha, innovation, rcond=None)[0]

        return augmented + C_MD @ x

    @abstractmethod
    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
    ) -> tuple[Optional[xarray.Dataset], xarray.Dataset]:
        """Perform one ESMDA assimilation step.

        Returns:
            (updated_state_or_None, updated_params)
        """
        raise NotImplementedError

    def _analysis(
        self,
        params: xarray.Dataset,
        observations: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
        return_params_history: bool = False,
        return_state_history: bool = False,
    ) -> xarray.Dataset | tuple[xarray.Dataset, xarray.Dataset]:
        """Perform the ESMDA analysis loop."""
        params_history: list[xarray.Dataset] = [params] if return_params_history else []
        state_history: list[xarray.Dataset] = []

        for i in range(self.num_steps):
            self._set_step_results_dir(i)

            state = self._forecast_step(state=state, params=params)

            if return_state_history and not self.forward_model.save_on_disk:
                state_history.append(state)

            state, params = self._one_step(
                params=params,
                obs=observations,
                state=state,
            )
            if return_params_history:
                params_history.append(params)

            print(f"ESMDA step {i} completed")

        # Final forecast with updated params
        self._set_step_results_dir(self.num_steps)
        state = self._forecast_step(state=state, params=params)

        if return_state_history and not self.forward_model.save_on_disk:
            state_history.append(state)

        # Build return values
        result_params = (
            xarray.concat(params_history, dim="esmda_step", join="override")
            if return_params_history
            else params
        )

        if self.forward_model.save_on_disk:
            return result_params

        if return_state_history:
            result_state = xarray.concat(
                state_history, dim="esmda_step", join="override"
            )
            return result_params, result_state

        return result_params, state


class ParameterESMDA(_BaseESMDA):
    """Parameter-only ESMDA smoothing."""

    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
    ) -> tuple[Optional[xarray.Dataset], xarray.Dataset]:
        obs = jnp.asarray(obs)
        param_names = list(params.data_vars.keys())
        N_e = params.sizes["ensemble"]

        pred_obs = self._observation_step(
            state=state,
            results_dir=(
                self.forward_model.results_dir
                if self.forward_model.save_on_disk
                else None
            ),
        )
        pred_obs = jnp.asarray(pred_obs).T

        if pred_obs.ndim != 2 or pred_obs.shape[1] != N_e:
            raise ValueError(
                "Predicted observations shape does not match ensemble size. "
                f"Expected second dimension {N_e}, got {pred_obs.shape}. "
                "This usually indicates stale or unexpected files in the ESMDA "
                "results step directory."
            )

        params_array = jnp.array([params[name].values for name in param_names])
        params_updated = self._compute_kalman_update(params_array, pred_obs, obs, N_e)

        updated_data_vars = {
            name: ("ensemble", params_updated[i, :])
            for i, name in enumerate(param_names)
        }
        return None, xarray.Dataset(data_vars=updated_data_vars, coords=params.coords)


class TimeVaryingParameterESMDA(ParameterESMDA):
    """Parameter-only ESMDA for time-varying inflow parameters.

    Each time point of each parameter is treated as an independent scalar
    parameter during the Kalman update.  The flattening groups all time
    points of one parameter together (e.g. ``inflow_angle_0``, …,
    ``inflow_angle_{N_t-1}``, ``velocity_magnitude_0``, …) so that
    different physical quantities are never mixed.
    """

    def __init__(
        self,
        observation_operator: ObservationOperator,
        forward_model: BaseEnsembleForwardModel,
        C_D: jnp.ndarray,
        num_time_points: int,
        time_coords: jnp.ndarray,
        num_steps: int = 3,
        alpha: Optional[float] = None,
        rng_key: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(42),
    ) -> None:
        super().__init__(
            observation_operator=observation_operator,
            forward_model=forward_model,
            C_D=C_D,
            num_steps=num_steps,
            alpha=alpha,
            rng_key=rng_key,
        )
        self.num_time_points = num_time_points
        self.time_coords = time_coords

    def _flatten_time_varying_params(self, params: xarray.Dataset) -> xarray.Dataset:
        """Flatten ``(time, ensemble)`` params to scalar ``(ensemble,)`` vars.

        Each variable with a ``time`` dimension is expanded into
        ``{name}_0``, ``{name}_1``, … so that all time points of one
        parameter are contiguous.  Variables without a ``time`` dimension
        are passed through unchanged.
        """
        flat_data_vars: dict = {}
        for name in params.data_vars:
            if "time" in params[name].dims:
                for t_idx in range(self.num_time_points):
                    flat_data_vars[f"{name}_{t_idx}"] = (
                        "ensemble",
                        jnp.asarray(params[name].isel(time=t_idx).values),
                    )
            else:
                flat_data_vars[name] = (
                    "ensemble",
                    jnp.asarray(params[name].values),
                )
        return xarray.Dataset(
            data_vars=flat_data_vars,
            coords={"ensemble": params.coords["ensemble"]},
        )

    def _unflatten_params(
        self,
        flat_params: xarray.Dataset,
        original_params: xarray.Dataset,
    ) -> xarray.Dataset:
        """Reverse :meth:`_flatten_time_varying_params`."""
        data_vars: dict = {}
        for name in original_params.data_vars:
            if "time" in original_params[name].dims:
                time_slices = [
                    jnp.asarray(flat_params[f"{name}_{t_idx}"].values)
                    for t_idx in range(self.num_time_points)
                ]
                data_vars[name] = (
                    ("time", "ensemble"),
                    jnp.stack(time_slices, axis=0),
                )
            else:
                data_vars[name] = flat_params[name]
        return xarray.Dataset(data_vars=data_vars, coords=original_params.coords)

    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
    ) -> tuple[Optional[xarray.Dataset], xarray.Dataset]:
        flat_params = self._flatten_time_varying_params(params)
        _, updated_flat = super()._one_step(flat_params, obs, state)
        updated_params = self._unflatten_params(updated_flat, params)
        return None, updated_params


class StateAndParameterESMDA(_BaseESMDA):
    """Joint state and parameter ESMDA smoothing."""

    def _get_states(
        self,
        state: Optional[xarray.Dataset] = None,
        results_dir: Optional[pathlib.Path] = None,
    ) -> xarray.Dataset:
        """Get ensemble states, selecting the first timestep."""
        if state is not None:
            return state.isel(time=0)
        if results_dir is not None:
            state_files = self._get_sorted_state_files(pathlib.Path(results_dir))
            if not state_files:
                raise FileNotFoundError(
                    f"No state_*.nc files found in results directory: {results_dir}"
                )
            states = [xarray.open_dataset(f).isel(time=-1) for f in state_files]
            return xarray.concat(states, dim="ensemble", join="override")
        raise ValueError("Either state or results_dir must be provided.")

    def _flatten_state(self, state: xarray.Dataset) -> jnp.ndarray:
        """Flatten ensemble state to (degrees_of_freedom, ensemble_size)."""
        flat_vars = []
        for var_name in sorted(state.data_vars):
            variable = state[var_name]
            flat_var = variable.transpose("ensemble", ...).values.reshape(
                variable.sizes["ensemble"], -1
            )
            flat_vars.append(flat_var.T)
        return jnp.concatenate(flat_vars, axis=0)

    def _unflatten_state(
        self,
        states_array: jnp.ndarray,
        state_template: xarray.Dataset,
    ) -> xarray.Dataset:
        """Unflatten a (degrees_of_freedom, ensemble_size) array back to xarray."""
        new_data_vars = {}
        ensemble_size = states_array.shape[1]
        current_pos = 0

        for var_name in sorted(state_template.data_vars):
            template_var = state_template[var_name]
            var_flat_size = template_var.size // template_var.sizes["ensemble"]
            flat_var_chunk = states_array[current_pos : current_pos + var_flat_size, :]
            current_pos += var_flat_size

            dims_no_ensemble = [d for d in template_var.dims if d != "ensemble"]
            shape_no_ensemble = [template_var.sizes[d] for d in dims_no_ensemble]
            data = flat_var_chunk.T.reshape((ensemble_size, *shape_no_ensemble))

            new_dims_order = ["ensemble"] + dims_no_ensemble
            data_array = xarray.DataArray(data, dims=new_dims_order)
            new_data_vars[var_name] = data_array.transpose(*template_var.dims)

        return xarray.Dataset(new_data_vars, coords=state_template.coords)

    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
    ) -> tuple[xarray.Dataset, xarray.Dataset]:
        obs = jnp.asarray(obs)
        param_names = list(params.data_vars.keys())
        N_e = params.sizes["ensemble"]

        pred_obs = self._observation_step(
            state=state,
            results_dir=(
                self.forward_model.results_dir
                if self.forward_model.save_on_disk
                else None
            ),
        )
        pred_obs = jnp.asarray(pred_obs).T

        states_array = self._get_states(
            state=state,
            results_dir=(
                self.forward_model.results_dir
                if self.forward_model.save_on_disk
                else None
            ),
        )
        state_template = xarray.Dataset(
            {
                var: (
                    states_array[var].dims,
                    jnp.empty(states_array[var].shape, dtype=states_array[var].dtype),
                )
                for var in states_array.data_vars
            },
            coords={coord: states_array.coords[coord] for coord in states_array.coords},
        )

        states_flat = self._flatten_state(states_array)
        N_s = states_flat.shape[0]

        params_array = jnp.array([params[name].values for name in param_names])
        augmented = jnp.concatenate([states_flat, params_array], axis=0)

        augmented = self._compute_kalman_update(augmented, pred_obs, obs, N_e)

        # Split back into state and params
        updated_states = self._unflatten_state(augmented[:N_s, :], state_template)
        params_updated = augmented[N_s:, :]

        updated_data_vars = {
            name: ("ensemble", params_updated[i, :])
            for i, name in enumerate(param_names)
        }
        return updated_states, xarray.Dataset(
            data_vars=updated_data_vars,
            coords=params.coords,
        )
