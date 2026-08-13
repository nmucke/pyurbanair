import pathlib
from abc import abstractmethod
from typing import Any, Optional, Union

import jax.numpy as jnp
import numpy as np
import xarray
from data_assimilation.io import get_sorted_state_files
from data_assimilation.observation_operator import (
    AggregateObservations,
    ObservationOperator,
    flatten_observations,
)

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

#: What ``_analysis``/``__call__`` accept as observations: the time-resolved
#: DataArray the (temporal) observation operator returns, or an already
#: flattened vector.
Observations = Union[xarray.DataArray, np.ndarray, jnp.ndarray]


class BaseSmoothing:
    """Base class for smoothing."""

    #: Optional interval aggregation applied to BOTH the real observations and
    #: the predicted ones. Also a class attribute so instances built without
    #: ``__init__`` (as some unit tests do via ``__new__``) still resolve it.
    aggregate_observations: Optional[AggregateObservations] = None

    def __init__(
        self,
        observation_operator: ObservationOperator,
        forward_model: BaseEnsembleForwardModel,
        aggregate_observations: Optional[AggregateObservations] = None,
    ) -> None:
        self.observation_operator = observation_operator
        self.forward_model = forward_model
        self.aggregate_observations = aggregate_observations

    def _get_observations(self, observations: Observations) -> jnp.ndarray:
        """Bring observations into the flat space the Kalman update works in.

        The one path both the real observations and the predicted ones
        ``H(x)`` go through, so the two always live in the same space: optional
        interval aggregation, then the time-major flattening.

        A plain array is passed through untouched -- pre-flattened input is the
        caller's responsibility (low-level unit tests and a non-temporal
        observation operator both rely on this).
        """
        if not isinstance(observations, xarray.DataArray):
            return jnp.asarray(observations)
        if self.aggregate_observations is not None:
            observations = self.aggregate_observations(observations)
        return jnp.asarray(flatten_observations(observations))

    def _forecast_step(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> xarray.Dataset:
        """Forecast the state."""
        return self.forward_model.run_ensemble(state=state, params=params)

    def _observation_step(
        self,
        state: Optional[xarray.Dataset] = None,
        results_dir: Optional[pathlib.Path] = None,
    ) -> jnp.ndarray:
        """
        Observe the state.

        Args:
            state: The state to observe. If None, the state is loaded from the results directory.
            results_dir: The directory to load the states from.

        Returns:
            The predicted observations, ``(N_e, N_d)`` for an ensemble state,
            in the same (aggregated, flattened) space as the real observations
            -- both go through :meth:`_get_observations`.
        """
        if state is not None:
            return self._get_observations(self.observation_operator(state))
        elif results_dir is not None:
            file_list = self._get_sorted_state_files(results_dir)
            if not file_list:
                raise FileNotFoundError(
                    f"No state_*.nc files found in results directory: {results_dir}"
                )
            # ``Any``: the element type depends on the operator (a labelled
            # DataArray from the temporal wrapper, a plain vector from the bare
            # operator), which the ``ObservationOperator`` annotation does not
            # capture.
            observations_list: list[Any] = []
            for state_file in file_list:
                with xarray.open_dataset(state_file) as member_state:
                    observations_list.append(
                        self.observation_operator(member_state.load())
                    )

            # A temporal operator returns labelled per-member observations:
            # stack them into one ("ensemble", "time", "obs") DataArray so the
            # aggregation/flattening runs once, exactly as on the in-memory
            # path. A bare (non-temporal) operator returns plain per-member
            # vectors, which stack directly.
            #
            # ``join="override"`` (as in ``_get_states``): members are the same
            # window at the same output cadence, but a backend can write time
            # coordinates that differ in the last bits. The default outer join
            # would then silently take the UNION of those axes and pad the
            # observations with NaN; override keeps member 0's axis and raises
            # if a member really has a different number of frames.
            if isinstance(observations_list[0], xarray.DataArray):
                return self._get_observations(
                    xarray.concat(observations_list, dim="ensemble", join="override")
                )
            return jnp.stack([jnp.asarray(o) for o in observations_list], axis=0)
        raise ValueError("Either state or results_dir must be provided.")

    @staticmethod
    def _get_sorted_state_files(results_dir: pathlib.Path) -> list[pathlib.Path]:
        """Return state files sorted by ensemble index (shared io helper)."""
        return get_sorted_state_files(results_dir)

    @abstractmethod
    def _analysis(
        self,
        params: xarray.Dataset,
        observations: Observations,
        state: Optional[xarray.Dataset] = None,
        return_params_history: bool = False,
        return_state_history: bool = False,
    ) -> xarray.Dataset | tuple[xarray.Dataset, xarray.Dataset]:
        """Perform the analysis."""
        raise NotImplementedError

    def __call__(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        observations: Optional[Observations] = None,
        return_params_history: bool = False,
        return_state_history: bool = False,
    ) -> xarray.Dataset | tuple[xarray.Dataset, xarray.Dataset]:
        """Perform the analysis.

        ``observations`` is the time-resolved DataArray the observation
        operator returns (aggregated and flattened inside ``_analysis``) or an
        already flattened vector.
        """
        return self._analysis(
            state=state,
            params=params,
            observations=observations,
            return_params_history=return_params_history,
            return_state_history=return_state_history,
        )
