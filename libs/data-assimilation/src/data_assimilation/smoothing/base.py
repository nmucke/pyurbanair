import pathlib
from abc import abstractmethod
from typing import Optional

import jax.numpy as jnp
import xarray
from data_assimilation.observation_operator import ObservationOperator

from pyurbanair.base_forward_model import BaseForwardModel


class BaseSmoothing:
    """Base class for smoothing."""

    def __init__(
        self,
        observation_operator: ObservationOperator,
        forward_model: BaseForwardModel,
    ) -> None:
        self.observation_operator = observation_operator
        self.forward_model = forward_model

    def _forecast_step(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> xarray.Dataset:
        """Forecast the state."""
        return self.forward_model.run_ensemble(state=state, params=params)

    def _observation_step(
        self,
        state: Optional[xarray.Dataset] = None,
        results_dir: Optional[pathlib.Path] = None,
    ) -> jnp.ndarray:
        """
        Observe the state.

        Args:
            state: The state to observe. If None, the state is loaded from the results directory.
            results_dir: The directory to load the states from.

        Returns:
            The observations as a jnp.ndarray of shape (num_observations, num_sensors).
        """
        if state is not None:
            return self.observation_operator(state)
        elif results_dir is not None:
            file_list = [f for f in results_dir.iterdir() if f.is_file()]
            observations_list: list[jnp.ndarray] = []
            for state_file in file_list:
                state = xarray.open_dataset(state_file)
                observations = self.observation_operator(state)
                observations_list.append(observations)
            return jnp.concatenate(observations_list, axis=0)

    @abstractmethod
    def _analysis(
        self,
        params: xarray.Dataset,
        observations: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
        return_params_history: bool = False,
        return_state_history: bool = False,
    ) -> xarray.Dataset | tuple[xarray.Dataset, xarray.Dataset]:
        """Analyze the state."""
        raise NotImplementedError

    def __call__(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        observations: Optional[jnp.ndarray] = None,
        return_params_history: bool = False,
        return_state_history: bool = False,
    ) -> xarray.Dataset | tuple[xarray.Dataset, xarray.Dataset]:
        """Smooth the state."""
        return self._analysis(
            state=state,
            params=params,
            observations=observations,
            return_params_history=return_params_history,
            return_state_history=return_state_history,
        )
