"""pypalm ensemble wrapper.

Mirrors libs/pylbm/src/pylbm/ensemble_forward_model.py. Only the dispatch to
``create_new_forward_model`` differs; the rest of the ensemble orchestration
lives in ``pyurbanair.base_ensemble_forward_model.BaseEnsembleForwardModel``.
"""

import logging
import pathlib
from typing import Optional

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

from .forward_model import ForwardModel
from .utils.forward_model_utils import create_new_forward_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EnsembleForwardModel(BaseEnsembleForwardModel):
    """Ensemble forward model class for PALM."""

    def __init__(
        self,
        forward_model: ForwardModel,
        ensemble_size: int = 10,
        temp_dir: Optional[pathlib.Path] = None,
        results_dir: Optional[pathlib.Path] = None,
        num_parallel_processes: int = 1,
        num_cpus_per_process: int = 1,
        failure: Optional[dict] = None,
    ) -> None:
        super().__init__(
            forward_model=forward_model,
            ensemble_size=ensemble_size,
            results_dir=results_dir,
            num_parallel_processes=num_parallel_processes,
            num_cpus_per_process=num_cpus_per_process,
            temp_dir=temp_dir,
            failure=failure,
        )

    def _create_new_forward_model(  # type: ignore[override]
        self,
        forward_model: ForwardModel,  # type: ignore[override]
        experiment_base_dir: pathlib.Path,
        experiment_name: str,
    ) -> ForwardModel:
        return create_new_forward_model(
            forward_model,
            experiment_base_dir,
            experiment_name,
        )
