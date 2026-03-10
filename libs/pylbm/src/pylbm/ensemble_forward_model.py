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
from .utils.forward_model_utils import create_new_forward_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
