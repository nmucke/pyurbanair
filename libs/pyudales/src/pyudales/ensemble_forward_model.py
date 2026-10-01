import copy
import json
import logging
import math
import pathlib
from typing import Any, Optional, cast

import xarray
from pyudales.forward_model import ForwardModel
from pyudales.utils.forward_model_utils import create_new_forward_model
from pyudales.utils.inlet_turbulence_utils import (
    copy_elapsed_time,
    derive_seed,
    elapsed_time_path,
    read_elapsed_time,
    write_elapsed_time,
)
from pyudales.utils.namoptions_utils import NamoptionsFile
from pyudales.utils.warm_start_utils import CARRY_DIRNAME, clear_carry, copy_carry
from pyudales.utils.window_checkpoint import validate_carry

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


def _native_layout(model: ForwardModel) -> tuple[int, int, int, int, int]:
    nam = NamoptionsFile(
        model.dirs.experiment_dir / f"namoptions.{model.dirs.experiment_name}"
    )
    try:
        layout = tuple(
            int(nam.get_value(section, key) or default)
            for section, key, default in (
                ("DOMAIN", "itot", 0),
                ("DOMAIN", "jtot", 0),
                ("DOMAIN", "ktot", 0),
                ("RUN", "nprocx", 1),
                ("RUN", "nprocy", 1),
            )
        )
    except ValueError as exc:
        raise ValueError("SGS hybrid handoff found invalid native grid/ranks.") from exc
    if min(layout) <= 0:
        raise ValueError("SGS hybrid handoff found invalid native grid/ranks.")
    return layout  # type: ignore[return-value]


def _handoff_clock(model: ForwardModel, has_carry: bool) -> float:
    path = elapsed_time_path(model.dirs)
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            if payload["experiment_name"] != model.dirs.experiment_name:
                raise ValueError("member identity mismatch")
            clock = float(payload["elapsed_time"])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "SGS hybrid handoff found an invalid physical clock."
            ) from exc
    else:
        clock = float(model._elapsed_time)
        if has_carry or clock != 0:
            raise ValueError("SGS hybrid handoff requires a persisted physical clock.")
    if not math.isfinite(clock) or clock < 0:
        raise ValueError("SGS hybrid handoff found an invalid physical clock.")
    if not has_carry and clock > 0:
        raise ValueError(
            "SGS hybrid handoff is missing native carry for an advanced clock."
        )
    return clock


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

    @property
    def forecast_window_replay_enabled(self) -> bool:
        return any(
            model.forecast_window_replay_enabled
            for model in self.ensemble_forward_models
        )

    def synchronize_forecast_state_from(self, source: BaseEnsembleForwardModel) -> None:
        """Give this smoother stack the filter stack's accepted native state.

        The stacks have different solver horizons, so only member carry, clock
        and inlet realization cross this boundary. In particular, destination
        namoptions, paths and runtime remain owned by this stack.
        """
        if not self.forecast_window_replay_enabled:
            return
        if not isinstance(source, EnsembleForwardModel):
            raise ValueError("SGS hybrid handoff requires a uDALES source ensemble.")
        if source is self or not source.forecast_window_replay_enabled:
            raise ValueError("SGS hybrid handoff requires a distinct enabled source.")
        if self._failure_policy != "raise" or source._failure_policy != "raise":
            raise ValueError(
                "SGS hybrid handoff requires failure.policy=raise in both stacks."
            )
        if (
            getattr(self, "_forecast_window_ensemble_state", None) is not None
            or getattr(source, "_forecast_window_ensemble_state", None) is not None
        ):
            raise RuntimeError("Cannot hand off during an active forecast window.")
        if self.ensemble_size != source.ensemble_size or len(
            self.ensemble_forward_models
        ) != len(source.ensemble_forward_models):
            raise ValueError("SGS hybrid handoff requires matching ensemble sizes.")

        handoffs: list[tuple[ForwardModel, ForwardModel, bool, float, dict]] = []
        for index in range(self.ensemble_size):
            destination = cast(ForwardModel, self.ensemble_forward_models[index])
            origin = cast(ForwardModel, source.ensemble_forward_models[index])
            if (
                not origin.forecast_window_replay_enabled
                or not destination.forecast_window_replay_enabled
            ):
                raise ValueError("SGS hybrid handoff requires every member enabled.")
            if (
                getattr(origin, "_forecast_window_original", None) is not None
                or getattr(destination, "_forecast_window_original", None) is not None
            ):
                raise RuntimeError("Cannot hand off during an active forecast window.")
            if origin.dirs.experiment_name != destination.dirs.experiment_name:
                raise ValueError(
                    "SGS hybrid handoff requires aligned member identities."
                )
            if origin.dirs.experiment_dir == destination.dirs.experiment_dir:
                raise ValueError("SGS hybrid stacks must use separate experiment dirs.")
            if origin.model_discrepancy != destination.model_discrepancy:
                raise ValueError(
                    "SGS hybrid handoff requires matching discrepancy settings."
                )
            if _native_layout(origin) != _native_layout(destination):
                raise ValueError(
                    "SGS hybrid handoff requires matching grid and MPI ranks."
                )

            has_carry = (origin.dirs.experiment_dir / CARRY_DIRNAME).exists()
            validate_carry(origin, required=has_carry)
            clock = _handoff_clock(origin, has_carry)
            inlet = copy.deepcopy(origin.inlet_turbulence)
            if inlet.get("seed") is None:
                inlet["seed"] = derive_seed(origin.dirs.experiment_name)
            handoffs.append((origin, destination, has_carry, clock, inlet))

        # All members have passed structural and native-file validation before
        # any destination is changed. The source remains authoritative.
        for origin, destination, has_carry, clock, inlet in handoffs:
            if has_carry:
                if not copy_carry(origin.dirs, destination.dirs):
                    raise ValueError("SGS hybrid handoff could not copy native carry.")
                validate_carry(destination, required=True)
            else:
                clear_carry(destination.dirs)
            write_elapsed_time(destination.dirs, clock)
            destination._elapsed_time = clock
            destination.inlet_turbulence = inlet

    def begin_forecast_window(self) -> None:
        if not self.forecast_window_replay_enabled:
            return
        if getattr(self, "_forecast_window_ensemble_state", None) is not None:
            raise RuntimeError("A discrepancy forecast window is already active")
        begun = []
        try:
            for model in self.ensemble_forward_models:
                model.begin_forecast_window()
                begun.append(model)
        except BaseException:
            for model in begun:
                model.end_forecast_window(commit=False)
            raise
        rng = getattr(self, "_failure_rng", None)
        self._forecast_window_ensemble_state: dict[str, Any] | None = {
            "rng": copy.deepcopy(rng.bit_generator.state) if rng is not None else None,
            "substitutions": dict(getattr(self, "_last_failure_substitutions", {})),
        }

    def restore_forecast_window(self) -> None:
        for model in self.ensemble_forward_models:
            model.restore_forecast_window()

    def end_forecast_window(self, commit: bool) -> None:
        if not self.forecast_window_replay_enabled:
            return
        # Prepare every member before releasing a single rollback checkpoint.
        for base_model in self.ensemble_forward_models:
            cast(ForwardModel, base_model)._prepare_end_forecast_window(commit)
        transaction = getattr(self, "_forecast_window_ensemble_state", None)
        if not commit and transaction is not None:
            if transaction["rng"] is not None:
                self._failure_rng.bit_generator.state = transaction["rng"]
            self._last_failure_substitutions = transaction["substitutions"]
        self._forecast_window_ensemble_state = None
        for base_model in self.ensemble_forward_models:
            cast(ForwardModel, base_model)._release_forecast_window()

    def get_member_state(
        self,
        state: Optional[xarray.Dataset | pathlib.Path],
        member_index: int,
        sim_name: str = "state",
    ) -> Optional[xarray.Dataset]:
        # During replay, None explicitly requests the pinned cold start. Old
        # saved forecast outputs must not silently become initial conditions.
        model = self.ensemble_forward_models[member_index]
        if state is None and getattr(model, "_forecast_window_start", None) is not None:
            return None
        return super().get_member_state(state, member_index, sim_name)

    def apply_failure_substitutions_to_params(
        self, params: xarray.Dataset
    ) -> xarray.Dataset:
        if not self.forecast_window_replay_enabled:
            return cast(
                xarray.Dataset, super().apply_failure_substitutions_to_params(params)
            )
        # The accepted donor trajectory used exactly these coefficients. Jitter
        # would mislabel that trajectory and its hidden native checkpoint.
        return cast(xarray.Dataset, self.apply_failure_substitutions_to_state(params))

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
            checkpoint = getattr(donor_model, "_forecast_window_start", None)
            if checkpoint is not None:
                failed_model._forecast_window_start = checkpoint
                failed_model.params = copy.deepcopy(donor_model.params)
                failed_model._discrepancy_defaults = copy.deepcopy(
                    donor_model._discrepancy_defaults
                )
                # Preserve the donor forcing realization after committing too.
                failed_model.inlet_turbulence = copy.deepcopy(
                    donor_model.inlet_turbulence
                )
                if failed_model.inlet_turbulence.get("seed") is None:
                    failed_model.inlet_turbulence["seed"] = derive_seed(
                        checkpoint.experiment_name
                    )
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
