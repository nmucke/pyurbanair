import pathlib
from abc import abstractmethod
from typing import Optional

import jax.numpy as jnp
import xarray
from data_assimilation.io import get_sorted_state_files
from data_assimilation.observation_operator import ObservationOperator

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel


class BaseSmoothing:
    """Base class for smoothing."""

    def __init__(
        self,
        observation_operator: ObservationOperator,
        forward_model: BaseEnsembleForwardModel,
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
            The observations. The in-memory path returns whatever the operator
            returns for the given state -- ``(N_e, num_obs)`` for an ensemble
            Dataset. The disk path stacks per-member vectors into ``(N_e,
            num_obs)``.
        """
        if state is not None:
            return self.observation_operator(state)
        elif results_dir is not None:
            file_list = self._get_sorted_state_files(results_dir)
            if not file_list:
                raise FileNotFoundError(
                    f"No state_*.nc files found in results directory: {results_dir}"
                )
            observations_list: list[jnp.ndarray] = []
            for state_file in file_list:
                with xarray.open_dataset(state_file) as member_state:
                    observations_list.append(
                        self.observation_operator(member_state.load())
                    )

            return jnp.stack(observations_list, axis=0)
        raise ValueError("Either state or results_dir must be provided.")

    @staticmethod
    def _get_sorted_state_files(results_dir: pathlib.Path) -> list[pathlib.Path]:
        """Return state files sorted by ensemble index (shared io helper)."""
        return get_sorted_state_files(results_dir)

    @abstractmethod
    def _analysis(
        self,
        params: xarray.Dataset,
        observations: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
        return_params_history: bool = False,
        return_state_history: bool = False,
    ) -> xarray.Dataset | tuple[xarray.Dataset, xarray.Dataset]:
        """Perform the analysis."""
        raise NotImplementedError

    def __call__(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        observations: Optional[jnp.ndarray] = None,
        return_params_history: bool = False,
        return_state_history: bool = False,
    ) -> xarray.Dataset | tuple[xarray.Dataset, xarray.Dataset]:
        """Perform the analysis."""
        return self._analysis(
            state=state,
            params=params,
            observations=observations,
            return_params_history=return_params_history,
            return_state_history=return_state_history,
        )
