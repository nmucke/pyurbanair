import os
import pathlib
from abc import abstractmethod
from typing import Any, Optional

import xarray
from tqdm import tqdm


class BaseForwardModel:
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
        *args: Any,
        results_dir: Optional[pathlib.Path] = None,
        parallel_execution: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the forward model."""
        self.results_dir = results_dir
        self.parallel_execution = parallel_execution

        if results_dir is not None:
            self.apply_save_on_disk(results_dir)
        else:
            self.apply_save_in_memory()

    def apply_save_in_memory(self) -> None:
        """Apply the save in memory flag."""
        self.save_in_memory = True
        self.save_on_disk = False

    def apply_save_on_disk(self, results_dir: pathlib.Path) -> None:
        """Apply the save on disk flag."""
        self.save_in_memory = False
        self.save_on_disk = True
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    @abstractmethod
    def run_single(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """
        Run the forward model for a single state.

        Args:
            state: The state of the forward model. If None, the state is initialized
                according to the speicific forward model implementation.
            params: The parameters of the forward model. If None, the parameters are
                initialized according to the speicific forward model implementation.
            sim_name: The name of the simulation. If None, the simulation name is "state".

        Returns:
            The state of the forward model if saved in memory, otherwise None.
            If saved on disk, the state is saved to the results directory.
        """
        raise NotImplementedError

    def __call__(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """
        Run the forward model.

        Args:
            state: The state of the forward model. If None, the state is initialized
                according to the speicific forward model implementation.
            params: The parameters of the forward model. If None, the parameters are
                initialized according to the speicific forward model implementation.
            **kwargs: Additional keyword arguments.

        Returns:
            The state of the forward model if saved in memory, otherwise None.
            If saved on disk, the state is saved to the results directory.
        """
        state = self.run_single(state=state, params=params, sim_name=sim_name)

        if self.save_in_memory:
            return state
        else:
            # outfile = (
            #     self.results_dir / f"{sim_name}.nc"  # type: ignore[operator]
            #     if sim_name is not None
            #     else self.results_dir / "state.nc"  # type: ignore[operator]
            # )
            # state.to_netcdf(str(outfile))  # type: ignore[union-attr]
            return None
