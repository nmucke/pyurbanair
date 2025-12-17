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
        solver_name: str = "pylbm",
    ):
        """Initialize the observation operator."""
        self.obs_ids_x = obs_ids_x
        self.obs_ids_y = obs_ids_y
        self.obs_ids_z = obs_ids_z
        self.obs_states = obs_states

        self.num_sensors = len(obs_ids_x)
        self.num_obs = len(obs_ids_x) * len(obs_states)

        if solver_name == "udales":
            self.dim_mapping = {
                "u": {"z": "zt", "y": "yt", "x": "xm"},
                "v": {"z": "zt", "y": "ym", "x": "xt"},
                "w": {"z": "zm", "y": "yt", "x": "xt"},
            }
        elif solver_name == "pylbm":
            self.dim_mapping = {
                "u": {"z": "z", "y": "y", "x": "x"},
                "v": {"z": "z", "y": "y", "x": "x"},
                "w": {"z": "z", "y": "y", "x": "x"},
            }
        else:
            raise ValueError(f"Solver {solver_name} not supported.")

    def _observation_single(self, state: xarray.Dataset) -> np.ndarray:
        """Apply observation operator to one state.

        Args:
            state: xarray Dataset.

        Returns:
            Vector of shape (num_obs) where num_obs = num_sensors * num_states.
        """
        # Map dimension names for each variable
        # u: (zt, yt, xm), v: (zt, ym, xt), w: (zm, yt, xt)

        # Extract observations for all sensors at once using vectorized indexing
        obs_list = []
        for state_var in self.obs_states:
            # Get dimension names for this variable
            dims = self.dim_mapping[state_var]

            # Use xarray's isel with DataArray objects sharing a common dimension
            # This enables vectorized indexing: selects (obs_ids_x[i], obs_ids_y[i], obs_ids_z[i]) for all i
            # Result shape: (time, sensor) where sensor dimension has size num_sensors
            sensor_obs = state[state_var].isel(
                **{
                    self.dim_mapping[state_var]["z"]: xarray.DataArray(
                        self.obs_ids_z, dims="sensor"
                    ),
                    self.dim_mapping[state_var]["y"]: xarray.DataArray(
                        self.obs_ids_y, dims="sensor"
                    ),
                    self.dim_mapping[state_var]["x"]: xarray.DataArray(
                        self.obs_ids_x, dims="sensor"
                    ),
                }
            )
            # Flatten to handle time dimension: (time, sensor) -> (time * sensor,)
            # If time dimension is size 1, this gives shape (num_sensors,)
            obs_list.append(sensor_obs.values.ravel())

        # Concatenate all state variables: pattern is [all_sensors_var0, all_sensors_var1, ...]
        obs_values = np.concatenate(obs_list)

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
            obs_matrix[i, :] = self._observation_single(states.isel(ensemble=i))

        return obs_matrix

    def __call__(self, state: xarray.Dataset) -> np.ndarray:
        if "ensemble" in state.dims:
            return self._observation_ensemble(state)
        else:
            return self._observation_single(state)
