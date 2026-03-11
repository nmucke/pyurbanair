import pathlib
from typing import Any, Optional

import xarray
from pylbm.forward_model import ForwardModel

from pyurbanair.base_rollout_forward_model import BaseRolloutForwardModel

from .utils.rollout_utils import collect_rollout_results
from .utils.warm_start_utils import (
    clean_output_files,
    identify_latest_restart_iteration,
    remove_old_restart_files,
    write_restart_file_from_xarray,
)


class RolloutForwardModel(BaseRolloutForwardModel):
    """
    Rollout forward model for LBM using restart files for warmstart.

    Warmstart behavior in LBM is controlled by `nt0` in `infile.in`:
    - `nt0 == 0`: cold start
    - `nt0 > 0`: reads restart files at iteration `nt0` from `restart/`
    """

    def __init__(
        self,
        *args: Any,
        forward_model: ForwardModel,
        **kwargs: Any,
    ) -> None:
        """Initialize the rollout forward model."""
        super().__init__(*args, forward_model=forward_model, **kwargs)
        self.dirs = self.forward_model.dirs

    def _pre_run_rollout_step(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> None:
        """Prepare the state for the rollout step."""
        if state is not None:
            latest_restart = identify_latest_restart_iteration(self.dirs)
            restart_iteration = write_restart_file_from_xarray(
                state=state,
                dirs=self.dirs,
                restart_iteration=latest_restart,
            )
            self.forward_model._nt0_override = restart_iteration
        else:
            self._configure_for_rollout_step()

    def _configure_for_rollout_step(self) -> None:
        """Configure infile restart/timestep settings for current rollout step."""
        if self.rollout_step == 0:
            # Cold start: _set_scaling_factors defaults to nt0=0.
            return

        restart_iteration = identify_latest_restart_iteration(self.dirs)
        if restart_iteration is None:
            raise FileNotFoundError(
                f"No restart files found in {self.dirs.experiment_dir / 'restart'} "
                "for warmstart rollout."
            )
        self.forward_model._nt0_override = restart_iteration

    def _get_restart_iteration_from_state(
        self,
        state: xarray.Dataset | pathlib.Path,
    ) -> Optional[int]:
        """Infer restart iteration from xarray state time coordinate if available."""
        if isinstance(state, pathlib.Path):
            state_ds = xarray.open_dataset(state, engine="netcdf4").load()
        else:
            state_ds = state

        if "time" not in state_ds.coords and "time" not in state_ds.dims:
            return None

        time_values = state_ds["time"].values
        if getattr(time_values, "size", 0) == 0:
            return None

        try:
            return int(round(float(time_values[-1])))
        except Exception:
            return None

    def _post_run_rollout_step(
        self,
        state: xarray.Dataset,
        sim_name: Optional[str] = "state",
        rollout_step: Optional[int] = 0,
    ) -> None:
        """Post-run the rollout step."""
        # When save_on_disk, the ensemble's _post_run_ensemble needs the output
        # files for _move_and_collect_rollout_results_to_disk; it will clean them.
        if not self.forward_model.save_on_disk:
            clean_output_files(self.dirs)
        remove_old_restart_files(self.dirs)

        # if (
        #     self.forward_model.save_on_disk
        #     and self.forward_model.results_dir is not None
        #     and sim_name is not None
        # ):
        #     collect_rollout_results(
        #         sim_name=sim_name,
        #         rollout_step=rollout_step,
        #         results_dir=self.forward_model.results_dir,
        #     )
