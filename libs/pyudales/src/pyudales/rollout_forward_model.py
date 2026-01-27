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
        if self.rollout_step == 0:
            self.rollout_step += 1
            state = super().run_single(
                state=state,
                params=params,
                sim_name=f"{sim_name}_rollout_{self.rollout_step}",
            )
            clean_output_except_warmstart_files(self.dirs)

        else:
            self.rollout_step += 1
            set_warm_start(self.dirs)
            state = super().run_single(
                state=state,
                params=params,
                sim_name=f"{sim_name}_rollout_{self.rollout_step}",
            )
            clean_output_except_warmstart_files(self.dirs)
            remove_old_warmstart_files(self.dirs)

        if self.save_on_disk:
            collect_rollout_results(
                sim_name=sim_name,
                rollout_step=self.rollout_step,
                dirs=self.dirs,
            )

        return state


