import pathlib
import re
from abc import abstractmethod
from typing import Optional

import jax.numpy as jnp
import xarray
from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

from data_assimilation.observation_operator import ObservationOperator


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
            The observations as a jnp.ndarray of shape (num_observations, num_sensors).
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
                state = xarray.open_dataset(state_file)
                observations_list.append(self.observation_operator(state))

            return jnp.stack(observations_list, axis=0)

    @staticmethod
    def _get_sorted_state_files(results_dir: pathlib.Path) -> list[pathlib.Path]:
        """Return state files sorted by ensemble index.

        Only files matching state_<int>.nc are considered to avoid stale or
        unrelated NetCDF files from previous runs polluting ensemble size.
        """
        state_file_regex = re.compile(r"state_(\d+)\.nc")
        state_files_with_idx: list[tuple[int, pathlib.Path]] = []

        for file_path in results_dir.iterdir():
            if not file_path.is_file():
                continue
            match = state_file_regex.fullmatch(file_path.name)
            if match is None:
                continue
            state_files_with_idx.append((int(match.group(1)), file_path))

        state_files_with_idx.sort(key=lambda item: item[0])
        return [file_path for _, file_path in state_files_with_idx]

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
