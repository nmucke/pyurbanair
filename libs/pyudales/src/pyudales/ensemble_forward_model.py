import logging
import pathlib
from typing import Optional

import xarray

from pyudales.forward_model import ForwardModel
from pyudales.utils.forward_model_utils import create_new_forward_model
from pyudales.utils.warm_start_utils import copy_carry

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
        forward_model: ForwardModel,
        ensemble_size: int = 10,
        temp_dir: Optional[pathlib.Path] = None,
        results_dir: Optional[pathlib.Path] = None,
        num_parallel_processes: int = 1,
        num_cpus_per_process: int = 1,
        failure: Optional[dict] = None,
    ) -> None:
        """
        Initialize the ForwardModel.

        Args:
            forward_model: The forward model to use.
            results_dir: The directory where the results will be saved.
            num_parallel_processes: The number of parallel processes to use.
            num_cpus_per_process: The number of CPUs per process to use.
            failure: Failure-handling policy mapping (see
                ``BaseEnsembleForwardModel``).
        """
        super().__init__(
            forward_model=forward_model,  # type: ignore[arg-type]
            ensemble_size=ensemble_size,
            results_dir=results_dir,
            num_parallel_processes=num_parallel_processes,
            num_cpus_per_process=num_cpus_per_process,
            temp_dir=temp_dir,
            failure=failure,
        )

    def _create_new_forward_model(  # type: ignore[override]
        self,
        forward_model: ForwardModel,
        experiment_base_dir: pathlib.Path,
        experiment_name: str,
    ) -> ForwardModel:
        """Create a new forward model for the ensemble."""
        return create_new_forward_model(
            forward_model,
            experiment_base_dir,
            experiment_name,
        )

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """Run the ensemble, then realign warmstart carries after failures.

        Each member persists its own end-of-run warmstart carry to disk inside
        ``run_single``. When a member fails it is resampled from a donor whose
        *state* then seeds the failed member's next window, so the failed
        member must also inherit the donor's carry (its subgrid fields) for the
        warm start to be consistent. The base records the failure-to-donor map
        in ``_last_failure_substitutions``; we apply the matching carry copies
        here, in the parent process after the (possibly parallel) run.
        """
        result = super().run_ensemble(state=state, params=params, sim_name=sim_name)
        for failed, donor in self._last_failure_substitutions.items():
            copy_carry(
                self.ensemble_forward_models[donor].dirs,
                self.ensemble_forward_models[failed].dirs,
            )
        return result
