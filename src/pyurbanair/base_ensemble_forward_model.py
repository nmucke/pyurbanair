import os
import pathlib
from abc import abstractmethod
from typing import Any, Optional

import xarray
from tqdm import tqdm

from pyurbanair.base_forward_model import BaseForwardModel


class BaseEnsembleForwardModel:
    """
    Base class for ensemble forward models.

    Manages running multiple forward model instances as an ensemble,
    either sequentially or in parallel, with results saved in memory
    or on disk.

    Subclasses must:
    - Populate self.ensemble_forward_models in __init__
    - Implement _run_parallel for parallel execution
    - Optionally override sequential methods for custom behavior
    """

    def __init__(
        self,
        forward_model: BaseForwardModel,
        *args: Any,
        ensemble_size: int = 10,
        results_dir: Optional[pathlib.Path] = None,
        num_parallel_processes: int = 1,
        num_cpus_per_process: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the ensemble forward model.

        Args:
            forward_model: The forward model template.
            ensemble_size: Number of ensemble members.
            results_dir: Directory for saving results. If None, uses
                forward_model.results_dir.
            num_parallel_processes: Number of parallel processes.
            num_cpus_per_process: Number of CPUs per process.
        """
        self.forward_model = forward_model
        self.ensemble_size = ensemble_size
        self.num_parallel_processes = num_parallel_processes
        self.num_cpus_per_process = num_cpus_per_process
        self.parallel_execution = num_parallel_processes > 1

        # Determine results directory: explicit > forward model's
        effective_results_dir = (
            results_dir if results_dir is not None else forward_model.results_dir
        )
        self._set_save_mode(effective_results_dir)

        # Subclasses must populate this list with individual forward models
        self.ensemble_forward_models: list[BaseForwardModel] = []

    def _set_save_mode(self, results_dir: Optional[pathlib.Path]) -> None:
        """Set save mode based on results_dir."""
        self.results_dir = results_dir
        self.save_on_disk = results_dir is not None
        self.save_in_memory = results_dir is None
        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)

    def set_results_dir(self, results_dir: Optional[pathlib.Path]) -> None:
        """Change the results directory for the ensemble.

        This updates save mode flags and creates the directory if needed.
        Individual ensemble member forward models are updated when
        running sequentially on disk.
        """
        self._set_save_mode(results_dir)

    @staticmethod
    def _extract_member_state(
        state: Optional[xarray.Dataset | pathlib.Path],
        member_index: int,
        sim_name: str = "state",
    ) -> Optional[xarray.Dataset]:
        """Extract state for a single ensemble member.

        Handles xarray.Dataset (with/without ensemble dim) and
        pathlib.Path (directory of per-member files or single file).
        """
        if state is None:
            return None
        if isinstance(state, pathlib.Path):
            if state.is_dir():
                return xarray.open_dataset(
                    state / f"{sim_name}_{member_index}.nc", engine="netcdf4"
                ).load()
            ds = xarray.open_dataset(state, engine="netcdf4").load()
            if "ensemble" in ds.dims:
                return ds.isel(ensemble=member_index)
            return ds
        if "ensemble" in state.dims:
            return state.isel(ensemble=member_index)
        return state

    @staticmethod
    def _extract_member_params(
        params: Optional[xarray.Dataset],
        member_index: int,
    ) -> Optional[xarray.Dataset]:
        """Extract params for a single ensemble member."""
        if params is None:
            return None
        if "ensemble" in params.dims:
            return params.isel(ensemble=member_index)
        return params

    @abstractmethod
    def _pre_run_ensemble(
        self,
        params: Optional[xarray.Dataset] = None,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        sim_name: Optional[str] = "state",
    ) -> None:
        """Prepare the ensemble for a run.

        This can include applying inflow settings, handling warm starts, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def _post_run_ensemble(self, sim_name: str) -> xarray.Dataset | None:
        """Clean up after a run and collect results.

        This can include moving results to disk, cleaning output directories, etc.
        """
        raise NotImplementedError

    def _run_ensemble_sequentially_in_memory(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset:
        """Run the ensemble sequentially, returning results in memory.

        Override in subclasses for custom behavior.
        """
        states = []
        pbar = tqdm(
            enumerate(self.ensemble_forward_models),
            total=self.ensemble_size,
            desc="Running ensemble",
        )
        for i, model in pbar:
            state_i = self._extract_member_state(state, i, sim_name)  # type: ignore[arg-type]
            params_i = self._extract_member_params(params, i)
            result = model.run_single(
                state=state_i, params=params_i, sim_name=f"{sim_name}_{i}"
            )
            if result is not None:
                states.append(result)

        if not states:
            raise RuntimeError("No results returned from ensemble members")

        return xarray.concat(states, dim="ensemble", join="override")

    def _run_ensemble_sequentially_on_disk(
        self,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> None:
        """Run the ensemble sequentially, saving results to disk.

        Each member's results_dir is set to the ensemble's results_dir
        before running. Override in subclasses for custom behavior.
        """
        self._pre_run_ensemble(state=state, params=params, sim_name=sim_name)
        pbar = tqdm(
            enumerate(self.ensemble_forward_models),
            total=self.ensemble_size,
            desc="Running ensemble",
        )
        for i, model in pbar:
            model.set_results_dir(self.results_dir)
            state_i = self._extract_member_state(state, i, sim_name)  # type: ignore[arg-type]
            params_i = self._extract_member_params(params, i)
            model.run_single(state=state_i, params=params_i, sim_name=f"{sim_name}_{i}")
        self._post_run_ensemble(sim_name=sim_name)  # type: ignore[arg-type]

        return None

    def get_states(self) -> xarray.Dataset:
        """Get the state from disk."""
        states = []
        for i, model in enumerate(self.ensemble_forward_models):
            result_file = self.results_dir / f"state_{i}.nc"  # type: ignore[operator]
            states.append(xarray.open_dataset(result_file, engine="netcdf4").load())
        return xarray.concat(states, dim="ensemble", join="override")

    @abstractmethod
    def _apply_inflow_settings(self, params: xarray.Dataset) -> None:
        """Apply the inflow settings to the ensemble forward models."""
        raise NotImplementedError

    @abstractmethod
    def _load_results(self) -> xarray.Dataset:
        """Load the results from the output folders."""
        raise NotImplementedError

    @abstractmethod
    def _clean_output(self) -> None:
        """Clean the output folders."""
        raise NotImplementedError

    @abstractmethod
    def _run_parallel(
        self,
        params: Optional[xarray.Dataset] = None,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """Run the ensemble in parallel. Must be implemented by subclasses."""
        raise NotImplementedError

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """
        Run the forward model ensemble.

        Dispatches to parallel, sequential in-memory, or sequential on-disk
        based on configuration.

        Args:
            state: The state for the ensemble. Can be an xarray.Dataset with
                an ensemble dimension, a pathlib.Path to a directory of
                per-member files, or None.
            params: The parameters with an ensemble dimension.
            sim_name: The base simulation name. Each member will be saved
                as "{sim_name}_{i}.nc".

        Returns:
            The ensemble state if save_in_memory, otherwise None.
        """
        if self.parallel_execution:
            return self._run_parallel(
                state=state,
                params=params,
                sim_name=sim_name,
            )
        elif self.save_in_memory:
            return self._run_ensemble_sequentially_in_memory(
                state=state,
                params=params,
                sim_name=sim_name,
            )
        else:
            return self._run_ensemble_sequentially_on_disk(  # type: ignore[func-returns-value]
                state=state,
                params=params,
                sim_name=sim_name,
            )
