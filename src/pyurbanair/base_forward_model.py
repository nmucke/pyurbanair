import os
import pathlib
from abc import abstractmethod
from typing import Any, Optional

import xarray


class BaseForwardModel:
    """
    Base class for forward models.

    All forward models must implement the run_single method.

    The base class manages save mode (in-memory vs on-disk) and provides
    a consistent interface for running simulations.

    All inputs and outputs are expected to be xarray.Dataset objects.
    """

    def __init__(
        self,
        *args: Any,
        results_dir: Optional[pathlib.Path] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the forward model."""
        self._set_save_mode(results_dir)

    def _set_save_mode(self, results_dir: Optional[pathlib.Path]) -> None:
        """Set save mode based on results_dir."""
        self.results_dir = results_dir
        self.save_on_disk = results_dir is not None
        self.save_in_memory = results_dir is None
        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)

    def set_results_dir(self, results_dir: Optional[pathlib.Path]) -> None:
        """Change the results directory and update save mode.

        Override in subclasses if additional directory state needs updating
        (e.g., a dirs dataclass).
        """
        self._set_save_mode(results_dir)

    def get_states(self) -> xarray.Dataset:
        """Get the states from the results directory."""
        return xarray.open_dataset(self.results_dir / "state.nc", engine="netcdf4").load()

    @abstractmethod
    def save_results(self, state: xarray.Dataset, sim_name: str = "state") -> None:
        """Save simulation results to disk.

        Args:
            state: The simulation state to save.
            sim_name: The name of the simulation.
        """
        raise NotImplementedError

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
            state: The state of the forward model. If None, the state is
                initialized according to the specific forward model
                implementation.
            params: The parameters of the forward model. If None, the
                parameters are initialized according to the specific forward
                model implementation.
            sim_name: The name of the simulation.

        Returns:
            The state of the forward model if save_in_memory, otherwise None.
            If save_on_disk, the state is saved to the results directory.
        """
        raise NotImplementedError

    def __call__(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """Run the forward model."""
        return self.run_single(state=state, params=params, sim_name=sim_name)
