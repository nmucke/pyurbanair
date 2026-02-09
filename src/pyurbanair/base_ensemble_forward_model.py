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
        num_parallel_processes: int = 1,
        num_cpus_per_process: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the forward model."""
        self.forward_model = forward_model
        self.ensemble_size = ensemble_size
        self.num_parallel_processes = num_parallel_processes
        self.num_cpus_per_process = num_cpus_per_process
        self.parallel_execution = num_parallel_processes > 1

        if forward_model.dirs.results_dir is not None:  # type: ignore[attr-defined]
            self.save_on_disk = True
            self.save_in_memory = False
        else:
            self.save_in_memory = True
            self.save_on_disk = False

    @abstractmethod
    def _run_parallel(
        self,
        params: Optional[xarray.Dataset] = None,
        state: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> None:
        """Run the forward model ensemble in parallel."""
        raise NotImplementedError

    @abstractmethod
    def _run_ensemble_sequentially_in_memory(
        self,
        state: xarray.Dataset,
        params: xarray.Dataset,
    ) -> xarray.Dataset:
        """Run the forward model ensemble sequentially in memory."""
        raise NotImplementedError

    @abstractmethod
    def _run_ensemble_sequentially_on_disk(
        self,
        state: xarray.Dataset,
        params: xarray.Dataset,
        sim_name: str,
    ) -> xarray.Dataset:
        """Run the forward model ensemble sequentially on disk."""
        raise NotImplementedError

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """
        Run the forward model ensemble.

        Args:
            state: The state of the forward model. If None, the state is initialized
                according to the speicific forward model implementation. If a pathlib.Path
                is provided, the state is loaded from the path.
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
