from typing import Any, Optional

import xarray

from .forward_model import ForwardModel
from .utils.rollout_utils import collect_rollout_results
from .utils.warm_start_utils import (
    clean_output_files,
    identify_latest_restart_iteration,
    remove_old_restart_files,
)


class RolloutForwardModel(ForwardModel):
    """
    Rollout forward model for LBM using restart files for warmstart.

    Warmstart behavior in LBM is controlled by `nt0` in `infile.in`:
    - `nt0 == 0`: cold start
    - `nt0 > 0`: reads restart files at iteration `nt0` from `restart/`
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rollout_step = 0

        # Ensure first rollout step starts from cold-start settings.
        self._set_infile_value("nt0", 0)
        self._set_infile_value("nt1", self.num_timesteps)

    def _configure_for_rollout_step(self) -> None:
        """Configure infile restart/timestep settings for current rollout step."""
        if self.rollout_step == 0:
            self._set_infile_value("nt0", 0)
            self._set_infile_value("nt1", self.num_timesteps)
            return

        restart_iteration = identify_latest_restart_iteration(self.dirs)
        if restart_iteration is None:
            raise FileNotFoundError(
                f"No restart files found in {self.dirs.experiment_dir / 'restart'} "
                "for warmstart rollout."
            )

        self._set_infile_value("nt0", restart_iteration)
        self._set_infile_value("nt1", restart_iteration + self.num_timesteps)

    def run_single(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """
        Run one rollout step.

        State injection is not yet supported for LBM rollout because restart
        files are Fortran unformatted binaries with model-specific layout.
        """
        if state is not None:
            raise NotImplementedError(
                "State initialization is not supported for pylbm rollout yet. "
                "Run a cold step first, then continue with warmstart."
            )

        self._configure_for_rollout_step()
        self.rollout_step += 1

        step_sim_name = (
            f"{sim_name}_rollout_{self.rollout_step}"
            if sim_name is not None
            else f"state_rollout_{self.rollout_step}"
        )
        result_state = super().run_single(
            state=None,
            params=params,
            sim_name=step_sim_name,
        )

        clean_output_files(self.dirs)
        remove_old_restart_files(self.dirs)

        if self.save_on_disk and self.results_dir is not None and sim_name is not None:
            collect_rollout_results(
                sim_name=sim_name,
                rollout_step=self.rollout_step,
                results_dir=self.results_dir,
            )

        return result_state
