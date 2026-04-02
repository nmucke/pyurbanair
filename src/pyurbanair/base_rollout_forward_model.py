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
        **kwargs: Any,
    ) -> None:
        """Initialize the rollout forward model."""
        self.forward_model = forward_model
        self.rollout_step = 0

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
    ) -> xarray.Dataset:
        """Get the states from the results directory."""
        sim_name = "state" if sim_name is None else sim_name
        return self.forward_model.get_states(sim_name=sim_name)

    def __call__(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = "state",
    ) -> xarray.Dataset | None:
        """Run the forward model."""

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

        return state
