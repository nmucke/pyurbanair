"""Ensemble Smoother with Multiple Data Assimilation (ESMDA) in JAX.

Public API re-exported here so consumers can import from the package root
(``from data_assimilation import ParameterESMDA``) and refactors of the file
layout do not break callers. See ``docs/data_assimilation.md``.
"""

from data_assimilation.localization.base import BaseLocalization
from data_assimilation.localization.correlation import CorrelationLocalization
from data_assimilation.localization.distance import DistanceLocalization
from data_assimilation.observation_operator import (
    ObservationOperator,
    TemporalObservationOperator,
)
from data_assimilation.reduction import OnlineStateReduction
from data_assimilation.smoothing.base import BaseSmoothing
from data_assimilation.smoothing.esmda import (
    ParameterESMDA,
    StateAndParameterESMDA,
    StateAndTimeVaryingParameterESMDA,
    TimeVaryingParameterESMDA,
)

__all__ = [
    "BaseLocalization",
    "BaseSmoothing",
    "CorrelationLocalization",
    "DistanceLocalization",
    "ObservationOperator",
    "OnlineStateReduction",
    "ParameterESMDA",
    "StateAndParameterESMDA",
    "StateAndTimeVaryingParameterESMDA",
    "TemporalObservationOperator",
    "TimeVaryingParameterESMDA",
]
