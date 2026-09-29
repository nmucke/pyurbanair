import json
import logging
import pathlib
from typing import Any, Optional, cast

import xarray
from pyudales.forward_model import ForwardModel
from pyudales.utils.forward_model_utils import create_new_forward_model
from pyudales.utils.inlet_turbulence_utils import copy_elapsed_time, read_elapsed_time
from pyudales.utils.warm_start_utils import copy_carry

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel
from pyurbanair.base_forward_model import BaseForwardModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _attach_member_discrepancy(
    result: xarray.Dataset, metadata: list[dict[str, Any] | None]
) -> None:
    """Keep member-specific provenance through xarray's attribute override."""
    if any(value is not None for value in metadata):
        result.attrs.pop("model_discrepancy", None)
        result.attrs["model_discrepancy_by_member"] = json.dumps(
            metadata, sort_keys=True
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
        forward_model.prepare_solver()
        super().__init__(
            forward_model=forward_model,
            ensemble_size=ensemble_size,
            results_dir=results_dir,
            num_parallel_processes=num_parallel_processes,
            num_cpus_per_process=num_cpus_per_process,
            temp_dir=temp_dir,
            failure=failure,
        )

    def _create_new_forward_model(
        self,
        forward_model: BaseForwardModel,
        experiment_base_dir: pathlib.Path,
        experiment_name: str,
    ) -> ForwardModel:
        """Create a new forward model for the ensemble."""
        return create_new_forward_model(
            cast(ForwardModel, forward_model),
            experiment_base_dir,
            experiment_name,
        )

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """Run the ensemble, then realign per-member disk state after failures.

        Each member persists its own end-of-run warmstart carry — and its
        inlet-turbulence clock — to disk inside ``run_single``. When a member
        fails it is resampled from a donor whose *state* then seeds the failed
        member's next window, so the failed member must inherit both:

        * the donor's **carry** (its subgrid fields), or the warm start is
          inconsistent with the state it was given;
        * the donor's **clock**, or its synthetic inlet turbulence jumps at the
          substitution and stays permanently offset from the flow it is now
          carrying (the failed member's own clock did not advance, because it
          simulated nothing).

        The base records the failure-to-donor map in
        ``_last_failure_substitutions``; we apply the matching copies here, in
        the parent process after the (possibly parallel) run.

        Refreshing ``_elapsed_time`` on every member — not just the substituted
        ones — is what keeps the parent's in-memory copies in step with what the
        forkserver workers wrote, since those mutations never came back.
        """
        result: xarray.Dataset | None = super().run_ensemble(
            state=state, params=params, sim_name=sim_name
        )
        for failed, donor in self._last_failure_substitutions.items():
            donor_model = cast(ForwardModel, self.ensemble_forward_models[donor])
            failed_model = cast(ForwardModel, self.ensemble_forward_models[failed])
            copy_carry(donor_model.dirs, failed_model.dirs)
            if not copy_elapsed_time(donor_model.dirs, failed_model.dirs):
                logger.info(
                    "Donor %s had no inlet-turbulence clock to pass to member "
                    "%s; its turbulence history restarts from its own clock. "
                    "Expected when inlet_turbulence is off.",
                    donor_model.dirs.experiment_name,
                    failed_model.dirs.experiment_name,
                )
        for base_model in self.ensemble_forward_models:
            model = cast(ForwardModel, base_model)
            model._elapsed_time = read_elapsed_time(model.dirs, model._elapsed_time)
        if result is not None:
            metadata: list[dict[str, Any] | None] = []
            for index in range(self.ensemble_size):
                donor = self._last_failure_substitutions.get(index, index)
                member = cast(ForwardModel, self.ensemble_forward_models[donor])
                if member.model_discrepancy.get("enabled", False):
                    metadata.append(
                        json.loads(
                            (
                                member.dirs.experiment_dir / "model_discrepancy.json"
                            ).read_text()
                        )
                    )
                else:
                    metadata.append(None)
            _attach_member_discrepancy(result, metadata)
        return result

    def get_states(self) -> xarray.Dataset:
        """Load each saved member's provenance before concatenating its state."""
        states = []
        metadata: list[dict[str, Any] | None] = []
        for index, model in enumerate(self.ensemble_forward_models):
            state = model.get_states(sim_name=f"state_{index}")
            states.append(state)
            raw = state.attrs.get("model_discrepancy")
            metadata.append(json.loads(raw) if raw is not None else None)
        result = cast(
            xarray.Dataset, xarray.concat(states, dim="ensemble", join="override")
        )
        _attach_member_discrepancy(result, metadata)
        return result
