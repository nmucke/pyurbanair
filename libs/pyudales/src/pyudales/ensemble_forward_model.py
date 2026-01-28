import logging
import os
import pathlib
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import xarray
from pyudales.forward_model import (
    ForwardModel,
    apply_inflow_settings,
    clean_output_dir,
    merge_params,
)
from pyudales.rollout_forward_model import RolloutForwardModel
from pyudales.utils.dir_utils import DirectoryPaths, create_dir
from pyudales.utils.forward_model_utils import create_new_forward_model
from pyudales.utils.rollout_utils import collect_rollout_results
from pyudales.utils.warm_start_utils import (
    clean_output_except_warmstart_files,
    remove_old_warmstart_files,
    set_warm_start,
)

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _run_simulation(
    experiment_dir: pathlib.Path,
    udales_root_path: pathlib.Path,
) -> xarray.Dataset | None:
    """Run a single simulation. Module-level function for multiprocessing."""

    logger.info(f"Running simulation in {experiment_dir}...")
    command = str(pathlib.Path(udales_root_path).joinpath("tools", "local_execute.sh"))
    command = ["bash", command, str(experiment_dir)]  # type: ignore[assignment]
    subprocess.run(
        command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    return None


class EnsembleForwardModel(BaseEnsembleForwardModel):
    """
    Forward model class.

    The forward model is a wrapper around the uDALES code.
    """

    def __init__(
        self,
        forward_model: ForwardModel | RolloutForwardModel,
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

        if isinstance(forward_model, RolloutForwardModel):
            self.rollout = True
            self.rollout_step = 0
        else:
            self.rollout = False

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

    def _apply_inflow_settings(self, params: xarray.Dataset) -> None:
        """Apply the inflow settings to the ensemble forward models."""
        for i, model in enumerate(self.ensemble_forward_models):
            _params = merge_params(
                model.params, params.isel(ensemble=i) if params is not None else None
            )
            apply_inflow_settings(_params, model.dirs)
            model.params = _params

    def _move_results_to_disk(self, sim_name: str) -> None:
        """Move the results to disk."""
        for i, model in enumerate(self.ensemble_forward_models):
            result_file = model.dirs.output_dir.joinpath(
                model.dirs.experiment_name, f"fielddump.{model.dirs.experiment_name}.nc"
            )
            shutil.move(
                str(result_file), str(model.results_dir / f"{sim_name}_{i}.nc")
            )

    def _move_and_collect_rollout_results_to_disk(
        self, sim_name: str, rollout_step: int
    ) -> None:
        """Move the rollout results to disk."""
        for i, model in enumerate(self.ensemble_forward_models):
            result_file = model.dirs.output_dir.joinpath(
                model.dirs.experiment_name, f"fielddump.{model.dirs.experiment_name}.nc"
            )
            shutil.move(
                str(result_file),
                str(model.results_dir / f"{sim_name}_{i}_rollout_{rollout_step}.nc"),
            )

            collect_rollout_results(
                sim_name=f"{sim_name}_{i}",
                rollout_step=rollout_step,
                dirs=model.dirs,
            )

    def _load_results(self) -> xarray.Dataset:
        """Load the results from the output folders."""
        states = []
        for i, model in enumerate(self.ensemble_forward_models):
            result_file = model.dirs.output_dir.joinpath(
                model.dirs.experiment_name, f"fielddump.{model.dirs.experiment_name}.nc"
            )
            states.append(xarray.open_dataset(result_file, engine="netcdf4").load())
        return xarray.concat(states, dim="ensemble", join="override")

    def _clean_output(self) -> None:
        """Clean the output folders."""
        for model in self.ensemble_forward_models:
            clean_output_dir(model.dirs)

    def _set_warm_rollout_start(self) -> None:
        """Set the warm start for the rollout forward models."""
        for model in self.ensemble_forward_models:
            set_warm_start(model.dirs)

    def _clean_rollout_output(self) -> None:
        """Clean the output folders for the first step of the rollout forward models."""
        for model in self.ensemble_forward_models:
            clean_output_except_warmstart_files(model.dirs)

    def _clean_old_rollout_warmstart_files(self) -> None:
        """Clean the warmstart files for the rollout forward models."""
        for model in self.ensemble_forward_models:
            remove_old_warmstart_files(model.dirs)

    def _launch_simulations(self) -> None:
        """Launch the simulations in parallel."""

        with ProcessPoolExecutor(max_workers=self.num_parallel_processes) as executor:
            futures = [
                executor.submit(
                    _run_simulation,
                    experiment_dir=model.dirs.experiment_dir,
                    udales_root_path=model.dirs.udales_root_path,
                )
                for model in self.ensemble_forward_models
            ]

            for future in as_completed(futures):
                future.result()

    def get_states(self) -> xarray.Dataset:
        """Get the state from disk."""
        states = []
        for i, model in enumerate(self.ensemble_forward_models):
            result_file = model.results_dir / f"state_{i}.nc"
            states.append(xarray.open_dataset(result_file, engine="netcdf4").load())
        return xarray.concat(states, dim="ensemble", join="override")

    def _run_parallel(
        self,
        params: Optional[xarray.Dataset] = None,
        state: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset:
        """Run the forward model ensemble in parallel."""

        self._apply_inflow_settings(params)

        if self.rollout and self.rollout_step > 0:
            self._set_warm_rollout_start()

        self._launch_simulations()

        # Load the results
        if self.save_on_disk:
            if self.rollout:
                self._move_and_collect_rollout_results_to_disk(
                    sim_name=sim_name,  # type: ignore[arg-type]
                    rollout_step=self.rollout_step,
                )
            else:
                self._move_results_to_disk(sim_name)  # type: ignore[arg-type]
            states = None
        else:
            states = self._load_results()

        # Clean the output for the next step
        if self.rollout:
            self._clean_rollout_output()
            if self.rollout_step > 0:
                self._clean_old_rollout_warmstart_files()
            self.rollout_step += 1
        else:
            self._clean_output()

        return states
