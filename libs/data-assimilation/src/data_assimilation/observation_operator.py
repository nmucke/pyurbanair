"""Observation operator for the data assimilation."""

import numpy as np
import xarray


class ObservationOperator:
    """Observation operator for the data assimilation."""

    def __init__(
        self,
        obs_ids_x: list[int],
        obs_ids_y: list[int],
        obs_ids_z: list[int],
        obs_states: list[str],
    ):
        """Initialize the observation operator."""
        self.obs_ids_x = obs_ids_x
        self.obs_ids_y = obs_ids_y
        self.obs_ids_z = obs_ids_z
        self.obs_states = obs_states

        self.num_obs = len(obs_ids_x) * len(obs_states)

    def _observation_one_state(self, state: xarray.Dataset) -> xarray.Dataset:
        """Apply observation operator to one state.

        Args:
            state: xarray Dataset.

        Returns:
            Vector of shape (num_obs * 3) where num_obs = NUM_OBS.
        """
        obs_values = np.zeros((self.num_obs))
        for i, state_var in enumerate(self.obs_states):
            obs_values[i] = state[state_var].values[
                :, self.obs_ids_z[i], self.obs_ids_y[i], self.obs_ids_x[i]
            ]
        return obs_values

    def _observation_ensemble(self, states: xarray.Dataset) -> np.ndarray:
        """Apply observation operator to each ensemble member.

        Args:
            states: xarray Dataset with ensemble dimension.

        Returns:
            Matrix of shape (ensemble, num_obs) where num_obs = NUM_OBS * 3.
        """
        ensemble_size = states.sizes["ensemble"]
        obs_matrix = np.zeros((ensemble_size, self.num_obs))

        for i in range(ensemble_size):
            obs_matrix[i, :] = self._observation_one_state(states.isel(ensemble=i))

        return obs_matrix

    def __call__(self, state: xarray.Dataset) -> np.ndarray:
        if "ensemble" in state.dims:
            return self._observation_ensemble(state)
        else:
            return self._observation_one_state(state)
