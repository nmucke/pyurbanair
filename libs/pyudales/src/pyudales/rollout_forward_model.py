import pdb
from typing import Any, Optional

import xarray
from pyudales.forward_model import ForwardModel
from pyudales.utils.rollout_utils import collect_rollout_results
from pyudales.utils.warm_start_utils import (
    clean_output_except_warmstart_files,
    identify_warmstart_file,
    remove_old_warmstart_files,
    set_trestart,
    set_warm_start,
    update_warmstart_file_from_xarray,
)


class RolloutForwardModel(ForwardModel):
    """Rollout forward model.

    This forward model is used to run a sequence of forward model steps.
    It is used to run a sequence of forward model steps with warm start.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the rollout forward model."""
        super().__init__(*args, **kwargs)

        self.rollout_step = 0

        self.clean_output = False

        set_trestart(self.dirs)

    def run_single(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """Run the rollout forward model"""
        if state is not None:
            # An explicit state is given. Update an existing warmstart file with new flow values.
            # This uses an existing warmstart file as template (created by uDALES from a previous run)
            # to avoid issues with 2DECOMP&FFT memory layout.
            warmstart_file = identify_warmstart_file(self.dirs)
            update_warmstart_file_from_xarray(
                state, self.dirs, warmstart_file=warmstart_file
            )
            set_warm_start(self.dirs)
        elif self.rollout_step > 0:
            # No state given, but it's not the first step, so warm-start from previous.
            set_warm_start(self.dirs)

        # For rollout_step == 0 and state is None, it will be a cold start (no warm start settings).

        self.rollout_step += 1

        # Run the simulation
        result_state = super().run_single(
            state=None,  # state is always handled via files for this model now.
            params=params,
            sim_name=f"{sim_name}_rollout_{self.rollout_step}",
        )

        # Post-processing
        clean_output_except_warmstart_files(self.dirs)

        if self.rollout_step > 1:
            remove_old_warmstart_files(self.dirs)

        if self.save_on_disk:
            if self.dirs.results_dir is None:
                raise ValueError(
                    "Cannot collect rollout results because results_dir is not set."
                )
            collect_rollout_results(
                sim_name=sim_name,  # type: ignore[arg-type]
                rollout_step=self.rollout_step,
                results_dir=self.dirs.results_dir,
            )

        return result_state
