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
        **kwargs: Any,
    ) -> None:
        """Initialize the forward model."""
        self.results_dir = results_dir

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
    ) -> xarray.Dataset | None:
        """Run the forward model for a single state."""
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
        state = self.run_single(state=state, params=params)

        if self.save_in_memory:
            return state
        else:
            outfile = (
                self.results_dir / f"{sim_name}.nc"  # type: ignore[operator]
                if sim_name is not None
                else self.results_dir / "state.nc"  # type: ignore[operator]
            )
            state.to_netcdf(str(outfile))  # type: ignore[union-attr]
            return None

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        ensemble_size: Optional[int] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """
        Run the forward model ensemble.

        Args:
            state: The state of the forward model. If None, the state is initialized
                according to the speicific forward model implementation.
            params: The parameters of the forward model. If None, the parameters are
                initialized according to the speicific forward model implementation.
            ensemble_size: The size of the ensemble. If None, the ensemble size is
                the number of parameters in the parameters dataset or the number of states in the state dataset.
            sim_name: The name of the simulation. If None, the simulation name is "state".
                If the simulation is saved on disk, the state is saved to the results
                directory with the name "sim_name_i.nc" for each ensemble member.

        Returns:
            The state of the forward model ensemble if saved in memory, otherwise None.
            If saved on disk, the state is saved to the results directory.
        """

        if ensemble_size is None:
            if params is not None:
                ensemble_size = len(params.ensemble)
            elif state is not None:
                ensemble_size = len(state.ensemble)
            else:
                raise ValueError(
                    "Ensemble size is not specified and cannot be inferred from the parameters or state."
                )
        elif ensemble_size is not None:
            ensemble_size = ensemble_size

        if self.save_in_memory:
            states = []
            for i in tqdm(
                range(ensemble_size), desc="Running ensemble", total=ensemble_size
            ):
                states.append(
                    self.__call__(
                        params=params.isel(ensemble=i) if params is not None else None,
                        state=state.isel(ensemble=i) if state is not None else None,
                    )
                )
            return xarray.concat(states, dim="ensemble", join="override")
        else:
            for i in tqdm(
                range(ensemble_size), desc="Running ensemble", total=ensemble_size
            ):
                _ = self.__call__(
                    params=params.isel(ensemble=i) if params is not None else None,
                    state=state.isel(ensemble=i) if state is not None else None,
                    sim_name=f"{sim_name}_{i}",
                )
            return None
