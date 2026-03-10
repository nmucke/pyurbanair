import os
import pathlib
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Optional

import xarray
from tqdm import tqdm

from pyurbanair.base_forward_model import BaseForwardModel


def create_dir(
    dir_path: pathlib.Path,
) -> pathlib.Path:
    """Create a temporary directory in the given directory."""
    os.makedirs(pathlib.Path(dir_path), exist_ok=True)
    return pathlib.Path(dir_path)


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
        temp_dir: Optional[pathlib.Path] = None,
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
            temp_dir: Temporary directory for ensemble experiments.
        """
        self.forward_model = forward_model
        self.ensemble_size = ensemble_size
        self.num_parallel_processes = num_parallel_processes
        self.num_cpus_per_process = num_cpus_per_process
        self.parallel_execution = num_parallel_processes > 1

        # Determine results directory: explicit > forward model's
        results_dir = (
            results_dir if results_dir is not None else forward_model.results_dir
        )
        self._set_save_mode(results_dir)

        if hasattr(forward_model, "rollout_step"):
            self.rollout = True
            self.rollout_step = 0
        else:
            self.rollout = False

        # Create ensemble experiment base directory
        if hasattr(forward_model, "dirs"):
            ensemble_temp_dir = forward_model.dirs.temp_dir
        else:
            ensemble_temp_dir = (
                temp_dir if temp_dir is not None else pathlib.Path(".temp")
            )
        self.ensemble_experiment_base_dir = create_dir(
            ensemble_temp_dir / "ensemble_experiments"
        )

        # Subclasses must populate this list with individual forward models
        self.ensemble_forward_models: list[BaseForwardModel] = []
        for ensemble_number in range(self.ensemble_size):
            self.ensemble_forward_models.append(
                self._create_new_forward_model(
                    forward_model=forward_model,
                    experiment_base_dir=self.ensemble_experiment_base_dir,
                    experiment_name=f"{ensemble_number:03d}",
                )
            )

    @abstractmethod
    def _create_new_forward_model(
        self,
        forward_model: BaseForwardModel,
        experiment_base_dir: pathlib.Path,
        experiment_name: str,
    ) -> BaseForwardModel:
        """Create a new forward model for the ensemble."""
        raise NotImplementedError

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

    def get_member_state(
        self,
        state: Optional[xarray.Dataset | pathlib.Path],
        member_index: int,
        sim_name: str = "state",
    ) -> Optional[xarray.Dataset]:
        """
        Get state for a single ensemble member.

        Args:
            state: The state for the ensemble. Can be an xarray.Dataset with
                an ensemble dimension, a pathlib.Path to a directory of
                per-member files, or None.
            member_index: The index of the ensemble member.
            sim_name: The base simulation name. Each member will be saved
                as "{sim_name}_{i}.nc".

        Returns:
            The state for the ensemble member.
        """
        if isinstance(state, xarray.Dataset):
            return state.isel(ensemble=member_index)

        if isinstance(state, pathlib.Path):
            return xarray.open_dataset(
                state / f"{sim_name}_{member_index}.nc", engine="netcdf4"
            ).load()

        if self.save_on_disk:
            file_name = self.results_dir / f"{sim_name}_{member_index}.nc"  # type: ignore[operator]
            if file_name.exists():
                return xarray.open_dataset(file_name, engine="netcdf4").load()

        return None

    def get_member_params(
        self,
        params: Optional[xarray.Dataset],
        member_index: int,
    ) -> Optional[xarray.Dataset]:
        """Get params for a single ensemble member."""
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
        """
        Prepare the ensemble for a run.

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
            result = model(
                state=self.get_member_state(state, i, sim_name),  # type: ignore[arg-type]
                params=self.get_member_params(params, i),
                sim_name=f"{sim_name}_{i}",
            )
            states.append(result)

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
        # self._pre_run_ensemble(state=state, params=params, sim_name=sim_name)
        pbar = tqdm(
            enumerate(self.ensemble_forward_models),
            total=self.ensemble_size,
            desc="Running ensemble",
        )
        for i, model in pbar:
            model(
                state=self.get_member_state(state, i, sim_name),  # type: ignore[arg-type]
                params=self.get_member_params(params, i),
                sim_name=f"{sim_name}_{i}",
            )

        return None

    def get_states(self) -> xarray.Dataset:
        """Get the state from disk."""
        states = []
        for i, _ in enumerate(self.ensemble_forward_models):
            result_file = self.results_dir / f"state_{i}.nc"  # type: ignore[operator]
            states.append(xarray.open_dataset(result_file, engine="netcdf4").load())
        return xarray.concat(states, dim="ensemble", join="override")

    def _clean_output(self) -> None:
        """Clean the output folders."""
        for model in self.ensemble_forward_models:
            model._clean_output()

    def _run_parallel(
        self,
        params: Optional[xarray.Dataset] = None,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """Run the ensemble in parallel."""
        with ProcessPoolExecutor(max_workers=self.num_parallel_processes) as executor:
            futures = [
                executor.submit(
                    model.__call__,
                    state=self.get_member_state(state, i, sim_name),  # type: ignore[arg-type]
                    params=self.get_member_params(params, i),
                    sim_name=f"{sim_name}_{i}",
                )
                for i, model in enumerate(self.ensemble_forward_models)
            ]

            states = {i: None for i in range(self.ensemble_size)}
            for i, future in enumerate(as_completed(futures)):
                states[i] = future.result()

        states = list(states.values())  # type: ignore[assignment]
        if self.save_on_disk:
            return None
        else:
            return xarray.concat(states, dim="ensemble", join="override")

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
