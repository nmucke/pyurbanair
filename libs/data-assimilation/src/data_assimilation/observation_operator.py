"""Observation operator for the data assimilation."""

from typing import Any, cast

import numpy as np
import xarray
from data_assimilation.interpolation import interpolate_dataarray_at_points


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

        self.num_sensors = int(num_sensors)
        self.num_obs = self.num_sensors * len(obs_states)

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
        elif solver_name == "palm":
            # PALM staggers u on xu and v on yv. pypalm's postprocess
            # unifies the vertical axis: zu_3d -> z (wall layer dropped)
            # and w is interpolated from zw_3d onto z, so all three share
            # the single z dim.
            self.dim_mapping = {
                "u": {"z": "z", "y": "y", "x": "xu"},
                "v": {"z": "z", "y": "yv", "x": "x"},
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
            # Any time dimension was already selected away (isel(time=-1)
            # above), so this is a plain (sensor,) vector per variable.
            obs_list.append(sensor_obs.values.ravel())

        # Concatenate all state variables: pattern is [all_sensors_var0, all_sensors_var1, ...]
        obs_values = np.concatenate(obs_list)

        return obs_values

    def _observation_ensemble(self, states: xarray.Dataset) -> np.ndarray:
        """Apply observation operator to each ensemble member.

        Args:
            states: xarray Dataset with ensemble dimension.

        Returns:
            Matrix of shape (ensemble, num_obs) where
            num_obs = num_sensors * len(obs_states).
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
    """Observation operator returning time-resolved observations as an xarray.

    Wraps an :class:`ObservationOperator` and applies it to every frame of the
    state's ``time`` dimension, keeping the result labelled so downstream code
    (aggregation, persistence) can still see which frame each observation came
    from. Temporal aggregation is NOT done here -- it lives in
    :class:`AggregateObservations`, which the data assimilation classes apply to
    both the real and the predicted observations through one shared path.
    """

    def __init__(self, observation_operator: ObservationOperator):
        """
        Initialize the temporal observation operator.

        Args:
            observation_operator: ObservationOperator applied per time frame.
        """
        self.observation_operator = observation_operator

    def _observation_single(self, state: xarray.Dataset) -> xarray.DataArray:
        """Apply observation operator to one state, frame by frame.

        Args:
            state: xarray Dataset with a time dimension.

        Returns:
            DataArray with dims ("time", "obs"); "obs" has length
            ``num_sensors * len(obs_states)`` with the sensor index innermost.
        """
        if "time" not in state.dims:
            raise ValueError(
                "TemporalObservationOperator requires a state with a 'time' "
                "dimension."
            )

        num_time_steps = state.sizes["time"]
        obs_per_time = [
            self.observation_operator._observation_single(state.isel(time=t))
            for t in range(num_time_steps)
        ]
        obs_values = np.stack(obs_per_time, axis=0)  # (time, obs)

        if "time" in state.coords:
            time_values = np.asarray(state["time"].values, dtype=float)
        else:
            # No time coordinate to carry (rare; interval aggregation needs one,
            # and will complain loudly downstream). Fall back to frame indices.
            time_values = np.arange(num_time_steps, dtype=float)

        return xarray.DataArray(
            obs_values,
            dims=("time", "obs"),
            coords={"time": time_values},
        )

    def _observation_ensemble(self, states: xarray.Dataset) -> xarray.DataArray:
        """Apply observation operator to each ensemble member.

        Args:
            states: xarray Dataset with ensemble and time dimensions.

        Returns:
            DataArray with dims ("ensemble", "time", "obs").
        """
        members = [
            self._observation_single(states.isel(ensemble=i))
            for i in range(states.sizes["ensemble"])
        ]
        observations = xarray.concat(members, dim="ensemble")
        return observations.transpose("ensemble", "time", "obs")

    def __call__(self, state: xarray.Dataset) -> xarray.DataArray:
        """Apply the observation operator to a state or ensemble of states.

        Args:
            state: xarray Dataset with a time dimension, optionally with an
                ensemble dimension.

        Returns:
            DataArray with dims ("time", "obs"), or ("ensemble", "time", "obs")
            when the state carries an ensemble dimension. The "time" dimension
            carries the state's (seconds-valued) time coordinate.
        """
        if "ensemble" in state.dims:
            return self._observation_ensemble(state)
        else:
            return self._observation_single(state)


class AggregateObservations:
    """Aggregate time-resolved observations into fixed-length time intervals.

    Applied by the data assimilation classes to BOTH the real observations and
    the predicted observations ``H(x)``, so the two always live in the same
    space. Aggregating observations rather than states is identical for the
    (default) "mean" mode, since the observation operator is linear in the
    state.
    """

    def __init__(self, interval_seconds: float, mode: str = "mean"):
        """
        Initialize the observation aggregator.

        Args:
            interval_seconds: Length of each aggregation interval in seconds.
                Frames are binned by their ``time`` coordinate (in seconds):
                all frames whose time falls in
                ``[t0 + k*interval_seconds, t0 + (k+1)*interval_seconds)``
                are aggregated together.
            mode: Aggregation function applied within each interval. Must be
                one of "mean", "median", "max", "min".
        """
        if interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be positive, got {interval_seconds}."
            )

        self.mode_mapping = {
            "mean": lambda obs: obs.mean(dim="time"),
            "median": lambda obs: obs.median(dim="time"),
            "max": lambda obs: obs.max(dim="time"),
            "min": lambda obs: obs.min(dim="time"),
        }
        if mode not in self.mode_mapping:
            raise ValueError(
                f"Invalid mode '{mode}'. "
                f"Must be one of {list(self.mode_mapping.keys())}."
            )

        self.interval_seconds = float(interval_seconds)
        self.mode = mode
        self._num_intervals: int | None = None

    def __call__(self, observations: xarray.DataArray) -> xarray.DataArray:
        """Aggregate observations over absolute ``interval_seconds`` bins.

        Args:
            observations: DataArray with dims (..., "time", "obs") and a
                seconds-valued "time" coordinate.

        Returns:
            DataArray with the same dims, where "time" now holds one entry per
            interval with coordinate ``t0 + k * interval_seconds`` (the
            interval's start time).
        """
        if "time" not in observations.dims:
            raise ValueError(
                "AggregateObservations requires observations with a 'time' "
                "dimension."
            )
        if "time" not in observations.coords:
            raise ValueError(
                "AggregateObservations requires a 'time' coordinate (in "
                "seconds) on the observations."
            )
        time_values = np.asarray(observations["time"].values, dtype=float)
        if time_values.size == 0:
            raise ValueError("Observations have no time steps to aggregate.")

        # Bin each frame by its time coordinate (seconds): frame t belongs to
        # interval floor((t - t0) / interval_seconds). Frames are emitted at a
        # uniform cadence, so each populated bin holds the frames whose time
        # falls in [t0 + k*dt, t0 + (k+1)*dt).
        bin_indices = np.floor(
            (time_values - time_values[0]) / self.interval_seconds
        ).astype(int)

        # Bin against the ABSOLUTE interval range spanned by the window, not
        # just the populated bins. Iterating only over np.unique(bin_indices)
        # would make element k of the returned vector "the k-th populated
        # interval" rather than "absolute interval k" -- if the model output
        # ever skips an interval (irregular cadence, a short window, a missing
        # frame), predicted and real observations would silently misalign and
        # shift the whole innovation vector.
        num_intervals = (
            int(np.floor((time_values[-1] - time_values[0]) / self.interval_seconds))
            + 1
        )
        if self._num_intervals is None:
            self._num_intervals = num_intervals
        elif num_intervals != self._num_intervals:
            # The observation vector length (and hence C_D, sized from the first
            # window) is fixed by the first call's interval count. A later
            # window with a different number of absolute intervals -- a short
            # final window, a changed output cadence -- would silently return a
            # mismatched-length vector.
            raise ValueError(
                f"Interval count changed between calls: first call produced "
                f"{self._num_intervals} intervals, this one produced "
                f"{num_intervals}. The observation vector length must stay "
                "constant across windows (C_D is sized from the first "
                "window); check the output cadence and window length."
            )

        agg_fn = self.mode_mapping[self.mode]
        obs_per_interval = []
        for b in range(num_intervals):
            frame_ids = np.nonzero(bin_indices == b)[0]
            if frame_ids.size == 0:
                # A silent gap here would misalign every subsequent element of
                # the observation vector against absolute interval index, rather
                # than raising loudly.
                raise ValueError(
                    f"Absolute interval {b} (of {num_intervals}, "
                    f"interval_seconds={self.interval_seconds}) has no frames. "
                    "Output cadence is irregular or a frame is missing, so "
                    "absolute-interval alignment cannot be guaranteed; check "
                    "the model output cadence and window length."
                )
            obs_per_interval.append(agg_fn(observations.isel(time=frame_ids)))

        aggregated = xarray.concat(obs_per_interval, dim="time")
        # Label each interval by its start time so the output stays a valid
        # input to another aggregation / to persistence.
        aggregated = aggregated.assign_coords(
            time=time_values[0] + np.arange(num_intervals) * self.interval_seconds
        )
        return cast(xarray.DataArray, aggregated.transpose(*observations.dims))


def flatten_observations(obs: xarray.DataArray) -> np.ndarray:
    """Flatten labelled observations into the time-major array the DA uses.

    ``("time", "obs")`` becomes ``(T * num_obs,)`` and
    ``("ensemble", "time", "obs")`` becomes ``(N_e, T * num_obs)``. The layout
    is time-major -- ``[t0: var0 sensors..., var1 sensors..., t1: ...]`` -- so
    the sensor index stays the innermost axis and
    :func:`sensor_observation_coords` keeps mapping observation ``j`` to sensor
    ``j % num_sensors``.
    """
    if "ensemble" in obs.dims:
        values = np.asarray(obs.transpose("ensemble", "time", "obs").values)
        return values.reshape(values.shape[0], -1)
    values = np.asarray(obs.transpose("time", "obs").values)
    return values.reshape(-1)


def sensor_observation_coords(observation_operator: Any, n_d: int) -> np.ndarray:
    """Physical (x, y, z) coordinate of each of the ``n_d`` observations.

    Every observation is a sensor reading; its spatial location is the
    sensor's, independent of which state component (u/v/w) or time interval
    it belongs to.  In the flattened observation vector the sensor index is
    the innermost/fastest axis (see :class:`ObservationOperator` and
    :class:`TemporalObservationOperator`), so observation ``j`` sits at sensor
    ``j % num_sensors``.  Tiling the sensor coordinates therefore reproduces
    the per-observation coordinates regardless of the number of state
    components or temporal intervals.

    Requires coordinate-based observations (``obs_x``/``obs_y``/``obs_z``).
    Accepts either an :class:`ObservationOperator` or a temporal wrapper
    around one. Used by distance-based localization (smoothing and filtering).
    """
    base_op = cast(
        ObservationOperator,
        getattr(observation_operator, "observation_operator", observation_operator),
    )
    if not getattr(base_op, "use_interpolation", False):
        raise ValueError(
            "Distance-based localization requires coordinate-based "
            "observations (obs_x/obs_y/obs_z), not index-based ones."
        )
    sensor_xyz = np.stack(
        [base_op.obs_x, base_op.obs_y, base_op.obs_z], axis=1
    )  # (num_sensors, 3)
    num_sensors = base_op.num_sensors
    reps, remainder = divmod(int(n_d), int(num_sensors))
    if remainder != 0:
        raise ValueError(
            f"Observation count {n_d} is not a multiple of the sensor count "
            f"{num_sensors}; cannot map observations to sensor coordinates."
        )
    return np.tile(sensor_xyz, (reps, 1))  # (n_d, 3)
