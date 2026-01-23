import pathlib
from typing import Any, Optional

import xarray
from pyudales.forward_model import ForwardModel
from pyudales.utils.warm_start_utils import (
    clean_output_except_warmstart_files,
    identify_warmstart_file,
    remove_old_warmstart_files,
    set_trestart,
    set_warm_start,
)

DEFAULT_WARMSTART_DIR = pathlib.Path("warmstart")


class RolloutForwardModel(ForwardModel):
    """
    Rollout forward model.
    """

    def __init__(
        self,
        *args: Any,
        # warmstart_dir: Optional[pathlib.Path] = DEFAULT_WARMSTART_DIR,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.first_step = True

        self.clean_output = False

        set_trestart(self.dirs)

    def run_single(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> xarray.Dataset | None:
        """
        Run the rollout forward model.
        """
        if self.first_step:
            self.first_step = False
            state = super().run_single(state=state, params=params)
            self.warm_start_file_name = identify_warmstart_file(self.dirs)
            clean_output_except_warmstart_files(self.dirs)
            return state
        else:
            set_warm_start(self.dirs)
            state = super().run_single(state=state, params=params)
            clean_output_except_warmstart_files(self.dirs)
            remove_old_warmstart_files(self.dirs)
            return state
