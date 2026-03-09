import logging
import os
import pathlib
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, cast

import xarray

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

from .forward_model import ForwardModel
from .rollout_forward_model import RolloutForwardModel
from .utils import apply_inflow_settings
from .utils.dir_utils import create_dir
from .utils.forward_model_utils import create_new_forward_model
from .utils.rollout_utils import collect_rollout_results
from .utils.state_utils import scale_velocity_to_physical
from .utils.warm_start_utils import (
    clean_output_files,
    identify_latest_restart_iteration,
    remove_old_restart_files,
    write_restart_file_from_xarray,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _run_simulation(
    experiment_dir: pathlib.Path,
    executable_path: pathlib.Path,
    pixi_env_path: pathlib.Path,
) -> None:
    """
    Run a single LBM simulation. Module-level function for multiprocessing.

    Args:
        experiment_dir: Directory containing the experiment files (infile.in, etc.)
        executable_path: Path to the boltzmann executable
        pixi_env_path: Path to the pixi/conda environment (for HOME env var)
    """
    logger.info(f"Running simulation in {experiment_dir}...")

    original_cwd = pathlib.Path.cwd()
    os.chdir(experiment_dir)

    # Set up environment
    env = os.environ.copy()
    env["HOME"] = str(pixi_env_path)
    if "PIXI_ENVIRONMENT" not in env:
        env["PIXI_ENVIRONMENT"] = str(pixi_env_path)

    # Run the executable
    shell_cmd = str(executable_path)
    subprocess.run(
        shell_cmd,
        shell=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    # Return to original directory
    os.chdir(original_cwd)


class EnsembleForwardModel(BaseEnsembleForwardModel):
    """
    Ensemble forward model class for LBM.

    The ensemble forward model manages multiple ForwardModel instances
    and runs them in parallel or sequentially.
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
        Initialize the EnsembleForwardModel.

        Args:
            forward_model: The forward model to use as a template.
            ensemble_size: Number of ensemble members.
            temp_dir: Temporary directory for ensemble experiments.
            results_dir: Directory where results will be saved.
            num_parallel_processes: Number of parallel processes to use.
            num_cpus_per_process: Number of CPUs per process (not used for LBM).
        """
        super().__init__(
            forward_model=forward_model,
            ensemble_size=ensemble_size,
            results_dir=results_dir,
            num_parallel_processes=num_parallel_processes,
            num_cpus_per_process=num_cpus_per_process,
        )

        self.rollout = isinstance(forward_model, RolloutForwardModel)
        self.rollout_step = 0

        # Create ensemble experiment base directory
        ensemble_temp_dir = (
            temp_dir if temp_dir is not None else forward_model.dirs.temp_dir
        )
        self.ensemble_experiment_base_dir = create_dir(
            ensemble_temp_dir / "ensemble_experiments"
        )

        # Create ensemble forward models
        self.ensemble_forward_models = []
        for ensemble_number in range(self.ensemble_size):
            self.ensemble_forward_models.append(
                create_new_forward_model(
                    forward_model=forward_model,
                    experiment_base_dir=self.ensemble_experiment_base_dir,
                    experiment_name=f"{ensemble_number:03d}",
                )
            )

    def _typed_models(self) -> list[ForwardModel | RolloutForwardModel]:
        """Typed view over ensemble members for static type checking."""
        return cast(
            list[ForwardModel | RolloutForwardModel], self.ensemble_forward_models
        )

    def _apply_inflow_settings(self, params: Optional[xarray.Dataset]) -> None:
        """Apply the inflow settings to the ensemble forward models."""
        if params is None:
            return
        for i, model in enumerate(self._typed_models()):
            params_i = params.isel(ensemble=i)
            apply_inflow_settings(params=params_i, dirs=model.dirs)

    def _get_output_files(
        self, model: ForwardModel | RolloutForwardModel
    ) -> List[pathlib.Path]:
        """
        Get all output files from the model.

        LBM outputs files like: out_0000_F{timestep:06d}.nc
        If output_frequency < num_timesteps, there will be multiple files.
        Otherwise, there will be a single file at the final timestep.

        Args:
            model: The ForwardModel instance.

        Returns:
            List of paths to output files.
        """
        return model._get_output_files_for_current_run()

    def _load_results(self) -> xarray.Dataset:
        """Load the results from the output folders."""
        states = []
        for i, model in enumerate(self._typed_models()):
            output_files = self._get_output_files(model)

            if not output_files:
                logger.warning(
                    f"No output files found for ensemble member {i} in {model.dirs.output_dir}"
                )
                continue

            # Load and concatenate all output files (similar to ForwardModel.run_single)
            if len(output_files) > 1:
                # Multiple timesteps: concatenate along time dimension
                state_parts = []
                for output_file in output_files:
                    state_parts.append(
                        xarray.open_dataset(output_file, engine="netcdf4").load()
                    )

                state = xarray.concat(state_parts, dim="time", join="override")
            else:
                # Single timestep: load and add time dimension
                state = xarray.open_dataset(output_files[0], engine="netcdf4").load()
                state = state.expand_dims("time", axis=0)

            state = state.assign(x=model.x_grid, y=model.y_grid, z=model.z_grid)
            state = scale_velocity_to_physical(state, dirs=model.dirs)
            states.append(state)

        if not states:
            raise FileNotFoundError("No output files found for any ensemble member")

        return xarray.concat(states, dim="ensemble", join="override")

    def _move_results_to_disk(self, sim_name: str) -> None:
        """Move the results to disk."""
        for i, model in enumerate(self._typed_models()):
            output_files = self._get_output_files(model)

            if not output_files:
                logger.warning(
                    f"No output files found for ensemble member {i} in {model.dirs.output_dir}"
                )
                continue

            if self.results_dir is None:
                logger.warning(
                    f"results_dir is None for ensemble member {i}, skipping move"
                )
                continue

            # Load and concatenate all output files, then save as single file
            if len(output_files) > 1:
                # Multiple timesteps: concatenate along time dimension
                state_parts = []
                for output_file in output_files:
                    state_parts.append(
                        xarray.open_dataset(output_file, engine="netcdf4").load()
                    )
                state = xarray.concat(state_parts, dim="time", join="override")
            else:
                # Single timestep: load and add time dimension
                state = xarray.open_dataset(output_files[0], engine="netcdf4").load()
                state = state.expand_dims("time", axis=0)
            state = state.assign(x=model.x_grid, y=model.y_grid, z=model.z_grid)
            state = scale_velocity_to_physical(state, dirs=model.dirs)
            # Save concatenated state to results directory
            dest_file = self.results_dir / f"{sim_name}_{i}.nc"
            state.to_netcdf(str(dest_file))

            # Remove original output files after moving
            for output_file in output_files:
                output_file.unlink()

    def _move_and_collect_rollout_results_to_disk(self, sim_name: str) -> None:
        """Move and aggregate rollout results per ensemble member."""
        for i, model in enumerate(self._typed_models()):
            output_files = self._get_output_files(model)

            if not output_files:
                logger.warning(
                    f"No output files found for ensemble member {i} in {model.dirs.output_dir}"
                )
                continue

            if self.results_dir is None:
                logger.warning(
                    f"results_dir is None for ensemble member {i}, skipping move"
                )
                continue

            if len(output_files) > 1:
                state_parts = []
                for output_file in output_files:
                    state_parts.append(
                        xarray.open_dataset(output_file, engine="netcdf4").load()
                    )
                state = xarray.concat(state_parts, dim="time", join="override")
                state = state.assign(x=model.x_grid, y=model.y_grid, z=model.z_grid)
            else:
                state = xarray.open_dataset(output_files[0], engine="netcdf4").load()
                state = state.expand_dims("time", axis=0)
                state = state.assign(x=model.x_grid, y=model.y_grid, z=model.z_grid)
            state = scale_velocity_to_physical(state, dirs=model.dirs)

            member_sim_name = f"{sim_name}_{i}"
            rollout_file = (
                self.results_dir
                / f"{member_sim_name}_rollout_{self.rollout_step + 1}.nc"
            )
            state.to_netcdf(str(rollout_file))
            collect_rollout_results(
                sim_name=member_sim_name,
                rollout_step=self.rollout_step + 1,
                results_dir=self.results_dir,
            )

            for output_file in output_files:
                output_file.unlink()

    def _configure_rollout_for_parallel(self) -> None:
        """Set nt0/nt1 for parallel rollout runs per ensemble member."""
        for model in self._typed_models():
            if self.rollout_step == 0:
                model._set_infile_value("nt0", 0)
                model._set_infile_value("nt1", model.num_timesteps)
                continue

            restart_iteration = identify_latest_restart_iteration(model.dirs)
            if restart_iteration is None:
                raise FileNotFoundError(
                    f"No restart files found in {model.dirs.experiment_dir / 'restart'} "
                    f"for ensemble member {model.dirs.experiment_name} warmstart rollout."
                )
            model._set_infile_value("nt0", restart_iteration)
            model._set_infile_value("nt1", restart_iteration + model.num_timesteps)

    def _clean_old_rollout_restarts(self) -> None:
        """Keep only newest restart generation for each ensemble member."""
        for model in self._typed_models():
            remove_old_restart_files(model.dirs)

    def _pre_run_ensemble(
        self,
        params: Optional[xarray.Dataset] = None,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        sim_name: Optional[str] = "state",
    ) -> None:
        """
        Pre-run setup for the ensemble.

        Args:
            params: Parameters for each ensemble member.
            state: State for each ensemble member (not used for LBM currently).
            sim_name: Simulation name (not used in pre-run).
        """
        self._apply_inflow_settings(params)
        resolved_sim_name = sim_name if sim_name is not None else "state"

        if state is not None:
            for i, model in enumerate(self._typed_models()):
                state_i = self._extract_member_state(
                    state=state,
                    member_index=i,
                    sim_name=resolved_sim_name,
                )
                restart_iteration = write_restart_file_from_xarray(
                    state=state_i,
                    dirs=model.dirs,
                )
                model._set_infile_value("nt0", restart_iteration)
                model._set_infile_value("nt1", restart_iteration + model.num_timesteps)
        elif self.rollout:
            self._configure_rollout_for_parallel()

    def _post_run_ensemble(self, sim_name: str) -> xarray.Dataset | None:
        """
        Post-run processing for the ensemble.

        Args:
            sim_name: Simulation name for saving results.

        Returns:
            Dataset with ensemble results if save_in_memory, else None.
        """
        # Load or move results
        if self.save_on_disk:
            if self.rollout:
                self._move_and_collect_rollout_results_to_disk(sim_name)
            else:
                self._move_results_to_disk(sim_name)
            states = None
        else:
            states = self._load_results()

        # Clean output for next step
        self._clean_output()

        return states

    def _clean_output(self) -> None:
        """Clean the output folders."""
        for model in self._typed_models():
            clean_output_files(model.dirs)
        if self.rollout:
            self._clean_old_rollout_restarts()
            self.rollout_step += 1

    def _run_parallel(
        self,
        params: Optional[xarray.Dataset] = None,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """Run the forward model ensemble in parallel."""
        self._pre_run_ensemble(params, state, sim_name)

        with ProcessPoolExecutor(max_workers=self.num_parallel_processes) as executor:
            futures = [
                executor.submit(
                    _run_simulation,
                    experiment_dir=model.dirs.experiment_dir,
                    executable_path=model.dirs.executable_path,
                    pixi_env_path=model.dirs.pixi_env_path,
                )
                for model in self._typed_models()
            ]

            for future in as_completed(futures):
                future.result()

        resolved_sim_name = sim_name if sim_name is not None else "state"
        return self._post_run_ensemble(resolved_sim_name)
