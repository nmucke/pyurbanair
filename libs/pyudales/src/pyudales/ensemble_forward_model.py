import logging
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import xarray
from pyudales.forward_model import ForwardModel
from pyudales.utils.dir_utils import create_dir
from pyudales.utils.forward_model_utils import create_new_forward_model

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _run_simulation(
    forward_model: ForwardModel,
    params: xarray.Dataset,
    state: xarray.Dataset,
    sim_name: str,
) -> xarray.Dataset | None:
    """Run a single simulation. Module-level function for multiprocessing."""

    return forward_model(
        params=params,
        state=state,
        sim_name=sim_name,
    )


class EnsembleForwardModel(BaseEnsembleForwardModel):
    """
    Forward model class.

    The forward model is a wrapper around the uDALES code.
    """

    def __init__(
        self,
        forward_model: ForwardModel,
        ensemble_size: int = 10,
        temp_dir: Optional[pathlib.Path] = None,
        results_dir: Optional[pathlib.Path] = None,
        num_parallel_processes: int = 1,
        num_cpus_per_process: int = 1,
    ) -> None:
        """
        Initialize the ForwardModel.

        Args:
            forward_model: The forward model to use.
            results_dir: The directory where the results will be saved.
            num_parallel_processes: The number of parallel processes to use.
            num_cpus_per_process: The number of CPUs per process to use.
        """
        super().__init__(
            forward_model=forward_model,
            ensemble_size=ensemble_size,
            results_dir=results_dir,
            num_parallel_processes=num_parallel_processes,
            num_cpus_per_process=num_cpus_per_process,
        )

        if self.parallel_execution:

            self.ensemble_experiment_base_dir = create_dir(
                self.forward_model.dirs.temp_dir / "ensemble_experiments"
            )

            self.ensemble_forward_models = []

            for ensemble_number in range(self.ensemble_size):
                self.ensemble_forward_models.append(
                    create_new_forward_model(
                        forward_model=self.forward_model,
                        experiment_base_dir=self.ensemble_experiment_base_dir,
                        experiment_name=f"{ensemble_number:03d}",
                    )
                )

    def _run_parallel(
        self,
        params: Optional[xarray.Dataset] = None,
        state: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset:

        # Run simulations in parallel
        with ProcessPoolExecutor(max_workers=self.num_parallel_processes) as executor:
            # Submit all tasks and store futures with their ensemble numbers
            future_to_ensemble = {
                executor.submit(
                    _run_simulation,
                    forward_model=self.ensemble_forward_models[ensemble_number],
                    params=(
                        params.isel(ensemble=ensemble_number)
                        if params is not None
                        else None
                    ),
                    state=(
                        state.isel(ensemble=ensemble_number)
                        if state is not None
                        else None
                    ),
                    sim_name=f"{sim_name}_{ensemble_number}",
                ): ensemble_number
                for ensemble_number in range(self.ensemble_size)
            }

            # Collect results, preserving order
            states = [None] * self.ensemble_size
            for future in as_completed(future_to_ensemble):
                ensemble_number = future_to_ensemble[future]
                result = future.result()
                states[ensemble_number] = result

        if self.save_on_disk:
            return None
        else:
            return xarray.concat(states, dim="ensemble", join="override")
