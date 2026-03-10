import logging
import os
import pathlib
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, cast

import xarray
from pyudales import LOCAL_EXECUTE_SCRIPT
from pyudales.forward_model import (
    ForwardModel,
    _augment_runtime_library_paths,
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
    identify_warmstart_file,
    remove_old_warmstart_files,
    set_warm_start,
    update_warmstart_file_from_xarray,
)

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _run_simulation(
    experiment_dir: pathlib.Path,
) -> xarray.Dataset | None:
    """Run a single simulation. Module-level function for multiprocessing."""

    logger.info(f"Running simulation in {experiment_dir}...")
    command = ["bash", str(LOCAL_EXECUTE_SCRIPT), str(experiment_dir)]
    env = os.environ.copy()
    _augment_runtime_library_paths(env)
    subprocess.run(
        command,
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
            temp_dir=temp_dir,
        )

    def _create_new_forward_model(
        self,
        forward_model: ForwardModel | RolloutForwardModel,
        experiment_base_dir: pathlib.Path,
        experiment_name: str,
    ) -> ForwardModel | RolloutForwardModel:
        """Create a new forward model for the ensemble."""
        return create_new_forward_model(
            forward_model, experiment_base_dir, experiment_name
        )

    def _typed_models(self) -> list[ForwardModel | RolloutForwardModel]:
        """Typed view over ensemble members for static type checking."""
        return cast(
            list[ForwardModel | RolloutForwardModel], self.ensemble_forward_models
        )

    def _apply_inflow_settings(
        self,
        params: Optional[xarray.Dataset],
        model: ForwardModel | RolloutForwardModel,
    ) -> None:
        """Apply the inflow settings to the ensemble forward models."""
        _params = merge_params(model.params, params)
        if _params is None:
            return
        apply_inflow_settings(_params, model.dirs)
        model.params = _params

    def _move_results_to_disk(self, sim_name: str) -> None:
        """Move the results to disk."""
        if self.results_dir is None:
            raise ValueError("Cannot move results because results_dir is not set.")
        for i, model in enumerate(self._typed_models()):
            result_file = self._get_output_file(model)
            shutil.move(str(result_file), str(self.results_dir / f"{sim_name}_{i}.nc"))

    def _move_and_collect_rollout_results_to_disk(
        self, sim_name: str, rollout_step: int
    ) -> None:
        """Move the rollout results to disk."""
        if self.results_dir is None:
            raise ValueError(
                "Cannot move rollout results because results_dir is not set."
            )
        for i, model in enumerate(self._typed_models()):
            result_file = self._get_output_file(model)

            shutil.move(
                str(result_file),
                str(self.results_dir / f"{sim_name}_{i}_rollout_{rollout_step}.nc"),
            )
            collect_rollout_results(
                sim_name=f"{sim_name}_{i}",
                rollout_step=rollout_step,
                results_dir=self.results_dir,
            )

    def _get_output_file(
        self, model: ForwardModel | RolloutForwardModel
    ) -> pathlib.Path:
        """Get the output file from the model."""
        # Check for merged file first (multi-processor case after gather_outputs.sh)
        output_file = model.dirs.output_dir.joinpath(
            model.dirs.experiment_name, f"fielddump.{model.dirs.experiment_name}.nc"
        )
        # If merged file doesn't exist, check for single-processor file
        # (gather_outputs.sh doesn't merge when there's only one processor)
        if not output_file.exists():
            single_proc_file = model.dirs.output_dir.joinpath(
                model.dirs.experiment_name,
                f"fielddump.000.000.{model.dirs.experiment_name}.nc",
            )
            if single_proc_file.exists():
                output_file = single_proc_file

        return output_file

    def _load_results(self) -> xarray.Dataset:
        """Load the results from the output folders."""
        states = []
        for i, model in enumerate(self._typed_models()):
            output_file = self._get_output_file(model)

            states.append(xarray.open_dataset(output_file, engine="netcdf4").load())
        return xarray.concat(states, dim="ensemble", join="override")

    def _set_warm_rollout_start(self) -> None:
        """Set the warm start for the rollout forward models."""
        for model in self._typed_models():
            set_warm_start(model.dirs)

    def _clean_rollout_output(self) -> None:
        """Clean the output folders for the first step of the rollout forward models."""
        for model in self._typed_models():
            clean_output_except_warmstart_files(model.dirs)

    def _clean_old_rollout_warmstart_files(self) -> None:
        """Clean the warmstart files for the rollout forward models."""
        for model in self._typed_models():
            remove_old_warmstart_files(model.dirs)

    def _pre_run_ensemble(
        self,
        params: Optional[xarray.Dataset] = None,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        sim_name: Optional[str] = "state",
    ) -> None:
        """Pre-run the ensemble."""
        self._apply_inflow_settings(params)
        resolved_sim_name = sim_name if sim_name is not None else "state"

        if state is not None:
            for i, model in enumerate(self._typed_models()):
                state_i = self._extract_member_state(state, i, resolved_sim_name)
                if state_i is None:
                    continue

                warmstart_file = identify_warmstart_file(model.dirs)
                update_warmstart_file_from_xarray(
                    state_i, model.dirs, warmstart_file=warmstart_file
                )
                set_warm_start(model.dirs)

        if self.rollout and self.rollout_step > 0:
            self._set_warm_rollout_start()

    def _post_run_ensemble(self, sim_name: str) -> xarray.Dataset | None:
        """Post-run the ensemble."""

        # Load the results
        if self.save_on_disk:
            import pdb

            pdb.set_trace()
            if self.rollout:
                self._move_and_collect_rollout_results_to_disk(
                    sim_name=sim_name,
                    rollout_step=self.rollout_step,
                )
            else:
                self._move_results_to_disk(sim_name)
            states = None
        else:
            states = self._load_results()

        # Clean the output for the next step
        self._clean_output()

        return states

    def _clean_output(self) -> None:
        """Clean the output folders."""
        if self.rollout:
            self._clean_rollout_output()
            if self.rollout_step > 0:
                self._clean_old_rollout_warmstart_files()
            self.rollout_step += 1
        else:
            for model in self._typed_models():
                clean_output_dir(model.dirs)

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
                )
                for model in self._typed_models()
            ]

            for future in as_completed(futures):
                future.result()

        states = self._post_run_ensemble(sim_name if sim_name is not None else "state")

        return states
