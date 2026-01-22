import os
import pathlib
from abc import abstractmethod
from typing import Any, Optional

import xarray
from tqdm import tqdm

from pyurbanair.base_forward_model import BaseForwardModel


class BaseEnsembleForwardModel:
    """
    Base class for forward models.

    All forward models must implement the __call__ method.

    The base class provides a way to save the results in memory or on disk.

    The base class also provides a way to run the forward model ensemble given
    an implementation of the __call__ method for a single state.

    All inputs and outputs are expected to be xarray.Dataset objects. If they
    are saved on disk, they are saved to the results directory as netcdf files.
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
        """Initialize the forward model."""
        self.forward_model = forward_model
        self.ensemble_size = ensemble_size
        self.results_dir = results_dir
        self.num_parallel_processes = num_parallel_processes
        self.num_cpus_per_process = num_cpus_per_process
        self.parallel_execution = num_parallel_processes > 1

        if results_dir is not None:
            self.apply_save_on_disk(results_dir)
        else:
            self.apply_save_in_memory()

    def apply_save_in_memory(self) -> None:
        """Apply the save in memory flag."""
        self.save_in_memory = True
        self.save_on_disk = False
        self.forward_model.save_in_memory = True
        self.forward_model.save_on_disk = False

    def apply_save_on_disk(self, results_dir: pathlib.Path) -> None:
        """Apply the save on disk flag."""
        self.save_in_memory = False
        self.save_on_disk = True
        self.forward_model.save_in_memory = False
        self.forward_model.save_on_disk = True
        self.forward_model.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)  # type: ignore[arg-type]

    @abstractmethod
    def _run_parallel(
        self,
        params: Optional[xarray.Dataset] = None,
        state: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> None:
        """Run the forward model ensemble in parallel."""
        raise NotImplementedError

    def _run_ensemble_sequentially_in_memory(
        self,
        state: xarray.Dataset,
        params: xarray.Dataset,
    ) -> xarray.Dataset:
        """Run the forward model ensemble sequentially in memory."""

        states = []
        for i in tqdm(
            range(self.ensemble_size), desc="Running ensemble", total=self.ensemble_size
        ):
            states.append(
                self.forward_model.__call__(
                    params=params.isel(ensemble=i) if params is not None else None,
                    state=state.isel(ensemble=i) if state is not None else None,
                )
            )
        return xarray.concat(states, dim="ensemble", join="override")

    def _run_ensemble_sequentially_on_disk(
        self,
        state: xarray.Dataset,
        params: xarray.Dataset,
        sim_name: str,
    ) -> xarray.Dataset:
        """Run the forward model ensemble sequentially on disk."""

        for i in tqdm(
            range(self.ensemble_size), desc="Running ensemble", total=self.ensemble_size
        ):
            _ = self.forward_model.__call__(
                params=params.isel(ensemble=i) if params is not None else None,
                state=state.isel(ensemble=i) if state is not None else None,
                sim_name=f"{sim_name}_{i}",
            )
        return None

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """
        Run the forward model ensemble.

        Args:
            state: The state of the forward model. If None, the state is initialized
                according to the speicific forward model implementation.
            params: The parameters of the forward model. If None, the parameters are
                initialized according to the speicific forward model implementation.
            sim_name: The name of the simulation. If None, the simulation name is "state".
                If the simulation is saved on disk, the state is saved to the results
                directory with the name "sim_name_i.nc" for each ensemble member.

        Returns:
            The state of the forward model ensemble if saved in memory, otherwise None.
            If saved on disk, the state is saved to the results directory.
        """

        if self.parallel_execution:
            return self._run_parallel(
                state=state,
                params=params,
                sim_name=sim_name,
            )
        else:
            if self.save_in_memory:
                return self._run_ensemble_sequentially_in_memory(
                    state=state,
                    params=params,
                )
            else:
                return self._run_ensemble_sequentially_on_disk(
                    state=state,
                    params=params,
                    sim_name=sim_name,  # type: ignore[arg-type]
                )
