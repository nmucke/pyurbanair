import logging
import pathlib
import re
import shutil
from typing import Any, Optional

import xarray
from pyudales.forward_model import ForwardModel
from pyudales.utils.namoptions_utils import NamoptionsFile
from pyudales.utils.warm_start_utils import (
    clean_output_except_warmstart_files,
    identify_warmstart_file,
    remove_old_warmstart_files,
    set_trestart,
    set_warm_start,
    update_warmstart_file_from_xarray,
)

from pyurbanair.base_rollout_forward_model import BaseRolloutForwardModel

logger = logging.getLogger(__name__)


class RolloutForwardModel(BaseRolloutForwardModel):
    """Rollout forward model.

    This forward model is used to run a sequence of forward model steps
    with warm start. During initialization, a very short cold-start run
    is performed to generate a warmstart file template with the correct
    2DECOMP&FFT memory layout. Subsequent rollout steps copy this template
    and fill in flow values from the provided xarray state.
    """

    def __init__(
        self,
        *args: Any,
        forward_model: ForwardModel,
        **kwargs: Any,
    ) -> None:
        """Initialize the rollout forward model.

        Runs a very short cold-start simulation to generate a warmstart
        file template. This template is stored in the experiment directory
        and reused for all subsequent rollout steps.

        Args:
            forward_model: The uDALES ForwardModel instance. Preprocessing
                must have been run before creating this rollout model
                (i.e., forward_model.run_preprocessing() should have been
                called already).
        """
        super().__init__(*args, forward_model=forward_model, **kwargs)

        self.dirs = self.forward_model.dirs  # type: ignore[attr-defined]
        self._warmstart_template_dir = self.dirs.experiment_dir / "warmstart_template"
        self._generate_warmstart_template()

    def _generate_warmstart_template(self) -> None:
        """Run a very short cold-start simulation to generate warmstart template files.

        Temporarily overrides the namoptions runtime to a single timestep,
        runs the forward model, copies the resulting warmstart files to a
        template directory, then restores the original namoptions settings.
        """
        namoptions_path = (
            self.dirs.experiment_dir / f"namoptions.{self.dirs.experiment_name}"
        )
        namoptions = NamoptionsFile(namoptions_path)

        # Save original values to restore later
        original_runtime = namoptions.get_value("RUN", "runtime")
        original_trestart = namoptions.get_value("RUN", "trestart")
        original_lwarmstart = namoptions.get_value("RUN", "lwarmstart")

        # Set a very short runtime (single timestep) and enable restart writing
        dtmax_str = namoptions.get_value("RUN", "dtmax")
        short_runtime = float(dtmax_str) if dtmax_str else 1.0
        namoptions.set_value("RUN", "runtime", short_runtime)
        namoptions.set_value("RUN", "trestart", short_runtime)
        namoptions.set_value("RUN", "lwarmstart", ".false.")
        namoptions.write()

        self.forward_model.run_single(
            state=None,
            params=None,
            sim_name="warmstart_template",
        )

        warmstart_filename = identify_warmstart_file(self.dirs)

        shutil.move(
            self.dirs.output_dir / self.dirs.experiment_name / warmstart_filename,
            self.dirs.experiment_dir / warmstart_filename,
        )
        self.warmstart_template_file: pathlib.Path = (
            self.dirs.experiment_dir / warmstart_filename
        )

        namoptions.set_value("RUN", "runtime", original_runtime or short_runtime)
        namoptions.write()

        self.forward_model._clean_output()

    def _pre_run_rollout_step(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> None:
        """Prepare the state for the rollout step."""
        set_trestart(self.dirs)

        if state is not None:
            # Copy the warmstart template and update with flow values
            update_warmstart_file_from_xarray(
                state, self.dirs, warmstart_file=self.warmstart_template_file
            )
            shutil.copy(
                self.warmstart_template_file,
                self.dirs.output_dir
                / self.dirs.experiment_name
                / self.warmstart_template_file.name,
            )
            set_warm_start(self.dirs)

    def _post_run_rollout_step(
        self,
        state: xarray.Dataset,
        sim_name: Optional[str] = "state",
        rollout_step: Optional[int] = 0,
    ) -> None:
        """Post-run the rollout step."""
        clean_output_except_warmstart_files(self.dirs)
        remove_old_warmstart_files(self.dirs)
