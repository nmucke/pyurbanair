import os
import pathlib
from abc import abstractmethod
from typing import Any, Optional

import xarray

from .base_forward_model import BaseForwardModel


class BaseRolloutForwardModel:
    """
    Base class for rollout forward models.

    All rollout forward models must implement the run_single method.

    The base class manages save mode (in-memory vs on-disk) and provides
    a consistent interface for running simulations.

    All inputs and outputs are expected to be xarray.Dataset objects.
    """

    def __init__(
        self,
        *args: Any,
        forward_model: BaseForwardModel,
        spinup_first_step_only: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the rollout forward model.

        Args:
            forward_model: The underlying forward model to run at each step.
            spinup_first_step_only: When True (default), automatically call
                ``forward_model.disable_spinup()`` after the first rollout
                step so that only the cold-start run pays the spinup cost.
        """
        self.forward_model = forward_model
        self.rollout_step = 0
        self.spinup_first_step_only = spinup_first_step_only

    @property
    def results_dir(self) -> Optional[pathlib.Path]:
        """Delegate to the underlying forward model's results directory."""
        return self.forward_model.results_dir

    def set_results_dir(self, results_dir: Optional[pathlib.Path]) -> None:
        """Delegate to the underlying forward model."""
        self.forward_model.set_results_dir(results_dir)

    @abstractmethod
    def _pre_run_rollout_step(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> None:
        """Prepare the state for the rollout step."""
        raise NotImplementedError

    @abstractmethod
    def _post_run_rollout_step(
        self,
        state: xarray.Dataset,
        sim_name: Optional[str] = "state",
        rollout_step: Optional[int] = 0,
    ) -> None:
        """Post-run the rollout step."""
        raise NotImplementedError

    def get_states(
        self,
        sim_name: Optional[str] = "state",
        rollout_step: Optional[int] = None,
    ) -> xarray.Dataset:
        """Get the states from the results directory."""
        if rollout_step is None:
            rollout_step = self.rollout_step - 1

        sim_name = "state" if sim_name is None else sim_name
        return self.forward_model.get_states(
            sim_name=f"{sim_name}_rollout_{rollout_step}",
        )

    def __call__(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """Run the forward model."""

        sim_name = f"{sim_name}_rollout_{self.rollout_step}"
        self._pre_run_rollout_step(
            state=state,
            params=params,
            sim_name=sim_name,
        )
        state = self.forward_model(
            state=None,
            params=params,
            sim_name=sim_name,
        )

        self._post_run_rollout_step(
            state=state,
            sim_name=sim_name,
            rollout_step=self.rollout_step,
        )
        self.rollout_step += 1

        if (
            self.spinup_first_step_only
            and self.rollout_step == 1
            and hasattr(self.forward_model, "spinup_time")
            and self.forward_model.spinup_time > 0
        ):
            self.forward_model.disable_spinup()

        return state
