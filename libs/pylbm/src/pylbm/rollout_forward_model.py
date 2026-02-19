import pathlib
from typing import Any, Optional

import xarray

from .forward_model import ForwardModel
from .utils.rollout_utils import collect_rollout_results
from .utils.warm_start_utils import (
    clean_output_files,
    identify_latest_restart_iteration,
    remove_old_restart_files,
    write_restart_file_from_xarray,
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

    def run_single(
        self,
        state: Optional[xarray.Dataset | pathlib.Path] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """
        Run one rollout step.

        If `state` is provided, it is converted to an equilibrium restart file
        and used as warmstart for this rollout step.
        """
        if state is not None:

            latest_restart = identify_latest_restart_iteration(self.dirs)
            restart_iteration = write_restart_file_from_xarray(
                state=state,
                dirs=self.dirs,
                restart_iteration=latest_restart,
            )

            self._set_infile_value("nt0", restart_iteration)
            self._set_infile_value("nt1", restart_iteration + self.num_timesteps)
        else:
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
