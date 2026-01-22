import copy
import logging
import pathlib
import pdb
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import xarray
from pyudales.forward_model import ForwardModel, create_dir

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MATLAB_BIN = pathlib.Path("/Applications/MATLAB_R2025b.app/bin/matlab")

DEFAULT_TEMP_DIR = lambda cwd: pathlib.Path(f"{cwd}/.temp/experiments")


def _run_simulation(
    temp_dir: pathlib.Path,
    udales_root_path: pathlib.Path,
) -> None:
    """Run a single simulation. Module-level function for multiprocessing."""
    command = [
        "bash",
        str(udales_root_path.joinpath("tools", "local_execute.sh")),
        str(temp_dir),
    ]
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

        self.cwd = self.forward_model.cwd  # type: ignore[attr-defined]

        if self.parallel_execution:
            if temp_dir is None:
                self.base_temp_dir: pathlib.Path = create_dir(
                    pathlib.Path(f"{self.cwd}/.temp/ensemble_experiments")
                )
                self.base_output_dir: pathlib.Path = create_dir(
                    pathlib.Path(f"{self.cwd}/.temp/ensemble_outputs")
                )
            else:
                self.base_temp_dir: pathlib.Path = create_dir(  # type: ignore[no-redef]
                    pathlib.Path(f"{temp_dir}/.temp/ensemble_experiments")
                )
                self.base_output_dir: pathlib.Path = create_dir(  # type: ignore[no-redef]
                    pathlib.Path(f"{temp_dir}/.temp/ensemble_outputs")
                )

            self.ensemble_experiment_names = [f"{i:03d}" for i in range(ensemble_size)]
            for experiment_name in self.ensemble_experiment_names:
                create_dir(self.base_temp_dir / experiment_name)

    def _prepare_forward_models(self) -> Tuple[List[pathlib.Path], List[ForwardModel]]:
        """Prepare the forward models."""
        temp_dirs = []
        forward_models = []
        for ensemble_number in range(self.ensemble_size):
            temp_dirs.append(
                self.base_temp_dir / self.ensemble_experiment_names[ensemble_number]
            )
            _forward_model = copy.deepcopy(self.forward_model)
            _forward_model.change_dirs(  # type: ignore[attr-defined]
                temp_dir=temp_dirs[ensemble_number],
                output_dir=self.base_output_dir,
                experiment_name=self.ensemble_experiment_names[ensemble_number],
            )
            forward_models.append(_forward_model)
        return temp_dirs, forward_models  # type: ignore[return-value]

    def _apply_ensemble_inflow_settings(
        self, forward_models: List[ForwardModel], params: xarray.Dataset
    ) -> None:
        for ensemble_number, forward_model in enumerate(forward_models):
            _params = (
                params.isel(ensemble=ensemble_number) if params is not None else None
            )
            _inflow = forward_model.inflow_angle
            _velocity_magnitude = forward_model.velocity_magnitude
            _pressure_gradient_magnitude = forward_model.pressure_gradient_magnitude
            if "inflow_angle" in params:
                _inflow = _params.inflow_angle.item()
            if "velocity_magnitude" in params:
                _velocity_magnitude = _params.velocity_magnitude.item()
            if "pressure_gradient_magnitude" in params:
                _pressure_gradient_magnitude = (
                    _params.pressure_gradient_magnitude.item()
                )
            forward_model._apply_inflow_settings(
                inflow_angle=_inflow,
                velocity_magnitude=_velocity_magnitude,
                pressure_gradient_magnitude=_pressure_gradient_magnitude,
            )

    def _run_parallel(
        self,
        params: Optional[xarray.Dataset] = None,
        state: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset:

        temp_dirs, _forward_models = self._prepare_forward_models()

        self._apply_ensemble_inflow_settings(
            forward_models=_forward_models,
            params=params,
        )

        # Run simulations in parallel
        with ProcessPoolExecutor(max_workers=self.num_parallel_processes) as executor:
            futures = [
                executor.submit(
                    _run_simulation,
                    temp_dir=temp_dirs[ensemble_number],
                    udales_root_path=self.forward_model.udales_root_path,  # type: ignore[attr-defined]
                )
                for ensemble_number in range(self.ensemble_size)
            ]
            # Wait for all simulations to complete
            for future in as_completed(futures):
                future.result()  # Raise any exceptions that occurred

        states = []
        if self.save_on_disk:
            for i, forward_model in enumerate(_forward_models):
                src_path = forward_model.work_dir.joinpath(
                    forward_model.experiment_name,
                    f"fielddump.{forward_model.experiment_name}.nc",
                )
                shutil.move(str(src_path), self.results_dir / f"{sim_name}_{i}.nc")  # type: ignore[operator]

                forward_model._clean_work_dir()
            return None
        else:
            for forward_model in _forward_models:
                states.append(
                    xarray.open_dataset(
                        self.base_output_dir.joinpath(
                            forward_model.experiment_name,
                            f"fielddump.{forward_model.experiment_name}.nc",
                        ),
                        engine="netcdf4",
                    )
                )
                forward_model._clean_work_dir()

            return xarray.concat(states, dim="ensemble", join="override")
