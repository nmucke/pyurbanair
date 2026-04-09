"""Observation operator for the data assimilation."""

import numpy as np
import xarray
from data_assimilation.interpolation import interpolate_dataarray_at_points
from pandas.core.indexing import Any


class ObservationOperator:
    """Observation operator for the data assimilation."""

    def __init__(
        self,
        obs_ids_x: list[int] | None = None,
        obs_ids_y: list[int] | None = None,
        obs_ids_z: list[int] | None = None,
        obs_states: list[str] | None = None,
        solver_name: str = "pylbm",
        obs_x: list[float] | None = None,
        obs_y: list[float] | None = None,
        obs_z: list[float] | None = None,
    ):
        """Initialize the observation operator."""
        if obs_states is None or len(obs_states) == 0:
            raise ValueError("obs_states must be provided and non-empty.")

        has_indices = (
            obs_ids_x is not None and obs_ids_y is not None and obs_ids_z is not None
        )
        has_coordinates = obs_x is not None and obs_y is not None and obs_z is not None

        if has_indices and has_coordinates:
            raise ValueError(
                "Provide either index-based observations (obs_ids_*) or coordinate-"
                "based observations (obs_*), not both."
            )
        if not has_indices and not has_coordinates:
            raise ValueError(
                "Provide either obs_ids_x/obs_ids_y/obs_ids_z or obs_x/obs_y/obs_z."
            )

        self.use_interpolation = has_coordinates
        if self.use_interpolation:
            self.obs_x = np.asarray(obs_x, dtype=float)
            self.obs_y = np.asarray(obs_y, dtype=float)
            self.obs_z = np.asarray(obs_z, dtype=float)
            num_sensors = self.obs_x.size
            if self.obs_y.size != num_sensors or self.obs_z.size != num_sensors:
                raise ValueError("obs_x, obs_y, and obs_z must have the same length.")
        else:
            self.obs_ids_x = np.asarray(obs_ids_x, dtype=int)
            self.obs_ids_y = np.asarray(obs_ids_y, dtype=int)
            self.obs_ids_z = np.asarray(obs_ids_z, dtype=int)
            num_sensors = self.obs_ids_x.size
            if self.obs_ids_y.size != num_sensors or self.obs_ids_z.size != num_sensors:
                raise ValueError(
                    "obs_ids_x, obs_ids_y, and obs_ids_z must have the same length."
                )

        self.obs_states = obs_states

        self.num_sensors = num_sensors
        self.num_obs = num_sensors * len(obs_states)

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

        if "time" in state.dims:
            state = state.isel(time=-1)

        # Extract observations for all sensors at once using vectorized indexing
        obs_list = []
        for state_var in self.obs_states:
            # Get dimension names for this variable
            dims = self.dim_mapping[state_var]

            if self.use_interpolation:
                sensor_obs = interpolate_dataarray_at_points(
                    state[state_var],
                    x_dim=dims["x"],
                    y_dim=dims["y"],
                    z_dim=dims["z"],
                    obs_x=self.obs_x,
                    obs_y=self.obs_y,
                    obs_z=self.obs_z,
                )
            else:
                # Use xarray's isel with DataArray objects sharing a common dimension.
                # This enables vectorized indexing: selects
                # (obs_ids_x[i], obs_ids_y[i], obs_ids_z[i]) for all i.
                # Result shape: (time, sensor) where sensor dimension has size num_sensors.
                sensor_obs = state[state_var].isel(
                    **{
                        dims["z"]: xarray.DataArray(self.obs_ids_z, dims="sensor"),
                        dims["y"]: xarray.DataArray(self.obs_ids_y, dims="sensor"),
                        dims["x"]: xarray.DataArray(self.obs_ids_x, dims="sensor"),
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


class TemporalObservationOperator:
    """Observation operator for the data assimilation with temporal averaging."""

    def __init__(
        self,
        observation_operator: ObservationOperator,
        mode: str = "mean",
        num_time_steps: int | None = None,
        interval_size: int | None = None,
        aggregation_mode: str = "mean",
    ):
        """
        Initialize the temporal observation operator.

        Args:
            observation_operator: ObservationOperator.
            mode: Mode to use for temporal aggregation.
                Valid modes are "mean", "median", "max", "min", "full",
                "intervals".
            num_time_steps: Number of time steps in the state. Optional; if
                not provided for "full" mode, it is detected from the first
                observed state.
            interval_size: Number of time steps per interval. Required when
                mode is "intervals".
            aggregation_mode: Aggregation function to apply within each
                interval. Must be one of "mean", "median", "max", "min".
                Only used when mode is "intervals".
        """
        self.observation_operator = observation_operator
        self.mode = mode

        self.mode_mapping = {
            "mean": lambda state: state.mean(dim="time"),
            "median": lambda state: state.median(dim="time"),
            "max": lambda state: state.max(dim="time"),
            "min": lambda state: state.min(dim="time"),
        }

        valid_modes = set(self.mode_mapping.keys()) | {"full", "intervals"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")

        if mode == "full":
            self._num_time_steps = num_time_steps  # None is OK; set lazily

        if mode == "intervals":
            if interval_size is None:
                raise ValueError(
                    "interval_size must be provided when mode is 'intervals'."
                )
            if aggregation_mode not in self.mode_mapping:
                raise ValueError(
                    f"Invalid aggregation_mode '{aggregation_mode}'. "
                    f"Must be one of {list(self.mode_mapping.keys())}."
                )
            self.interval_size = interval_size
            self.aggregation_mode = aggregation_mode
            self._num_intervals: int | None = None

    @property
    def num_obs(self) -> int | Any:
        """Number of observations produced by the operator."""
        if self.mode == "full":
            return self.observation_operator.num_obs * self._num_time_steps
        if self.mode == "intervals":
            if self._num_intervals is None:
                raise RuntimeError(
                    "num_obs is not available until the operator has been "
                    "called at least once (number of intervals is unknown)."
                )
            return self._num_intervals * self.observation_operator.num_obs
        return self.observation_operator.num_obs

    def _observation_single(self, state: xarray.Dataset) -> np.ndarray:
        """Apply observation operator to one state with temporal aggregation.

        Args:
            state: xarray Dataset with time dimension.

        Returns:
            Observation vector. Length is ``num_sensors * num_states`` for
            aggregation modes, or ``num_sensors * num_states * num_time_steps``
            for "full" mode.
        """
        if self.mode == "full":
            num_time_steps = state.sizes["time"]
            if self._num_time_steps is None:
                self._num_time_steps = num_time_steps
            obs_per_time = []
            for t in range(num_time_steps):
                state_t = state.isel(time=t)
                obs_per_time.append(
                    self.observation_operator._observation_single(state_t)
                )
            return np.concatenate(obs_per_time)

        if self.mode == "intervals":
            num_time_steps = state.sizes["time"]
            num_intervals = num_time_steps // self.interval_size
            if num_intervals == 0:
                raise ValueError(
                    f"interval_size ({self.interval_size}) exceeds the number "
                    f"of time steps ({num_time_steps})."
                )
            if self._num_intervals is None:
                self._num_intervals = num_intervals

            agg_fn = self.mode_mapping[self.aggregation_mode]
            obs_per_interval = []
            for i in range(num_intervals):
                interval_state = state.isel(
                    time=slice(i * self.interval_size, (i + 1) * self.interval_size)
                )
                aggregated = agg_fn(interval_state)
                obs_per_interval.append(
                    self.observation_operator._observation_single(aggregated)
                )
            return np.concatenate(obs_per_interval)

        # Aggregation path (mean, median, max, min)
        state_avg = self.mode_mapping[self.mode](state)
        obs_values = self.observation_operator(state_avg)

        return obs_values

    def _observation_ensemble(self, states: xarray.Dataset) -> np.ndarray:
        """Apply observation operator to each ensemble member.

        Args:
            states: xarray Dataset with ensemble and time dimensions.

        Returns:
            Matrix of shape (ensemble, num_obs).
        """
        ensemble_size = states.sizes["ensemble"]
        observations_list = []
        for i in range(ensemble_size):
            observations_list.append(self._observation_single(states.isel(ensemble=i)))
        return np.stack(observations_list, axis=0)

    def __call__(self, state: xarray.Dataset) -> np.ndarray:
        """Apply the observation operator to a state or ensemble of states.

        Args:
            state: xarray Dataset with time dimension, optionally with ensemble dimension.

        Returns:
            Observation vector or matrix.
        """
        if "ensemble" in state.dims:
            return self._observation_ensemble(state)
        else:
            return self._observation_single(state)
