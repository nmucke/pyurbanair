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
            forward_model=forward_model,  # type: ignore[arg-type]
            ensemble_size=ensemble_size,
            results_dir=results_dir,
            num_parallel_processes=num_parallel_processes,
            num_cpus_per_process=num_cpus_per_process,
            temp_dir=temp_dir,
        )

    def _create_new_forward_model(  # type: ignore[override]
        self,
        forward_model: ForwardModel | RolloutForwardModel,  # type: ignore[override]
        experiment_base_dir: pathlib.Path,
        experiment_name: str,
    ) -> ForwardModel | RolloutForwardModel:
        """Create a new forward model for the ensemble."""

        if isinstance(forward_model, ForwardModel):
            return create_new_forward_model(
                forward_model,
                experiment_base_dir,
                experiment_name,
            )
        if isinstance(forward_model, RolloutForwardModel):
            return RolloutForwardModel(
                forward_model=create_new_forward_model(
                    forward_model.forward_model,  # type: ignore[arg-type]
                    experiment_base_dir,
                    experiment_name,
                ),
            )
