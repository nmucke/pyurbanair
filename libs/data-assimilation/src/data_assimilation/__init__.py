"""Ensemble data assimilation in JAX: ESMDA smoothing and EnKF filtering.

Public API re-exported here so consumers can import from the package root
(``from data_assimilation import ParameterESMDA``) and refactors of the file
layout do not break callers. See ``docs/data_assimilation.md``.
"""

from data_assimilation.filter_smoothing import (
    FilterSmoothingESMDA,
    FilterSmoothingResult,
    IterationDiagnostics,
    TemporalLocalization,
)
from data_assimilation.filtering import (
    AnalysisScheme,
    BaseFilter,
    CycleDiagnostics,
    EnsembleKalmanFilter,
    ETKFAnalysis,
    FilterResult,
    IdentityEvolution,
    LETKFAnalysis,
    ObservationTSVD,
    ParameterEvolution,
    RandomWalkEvolution,
    StochasticEnKFAnalysis,
)
from data_assimilation.inflation import (
    RTPP,
    RTPS,
    InflationScheme,
    MultiplicativeInflation,
)
from data_assimilation.localization.base import BaseLocalization
from data_assimilation.localization.correlation import CorrelationLocalization
from data_assimilation.localization.distance import DistanceLocalization
from data_assimilation.observation_operator import (
    ObservationOperator,
    TemporalObservationOperator,
)
from data_assimilation.reduction import OnlineStateReduction, StreamingStateReduction
from data_assimilation.smoothing.base import BaseSmoothing
from data_assimilation.smoothing.esmda import (
    ParameterESMDA,
    StateAndParameterESMDA,
    StateAndTimeVaryingParameterESMDA,
    StateESMDA,
    TimeVaryingParameterESMDA,
)

__all__ = [
    "AnalysisScheme",
    "BaseFilter",
    "BaseLocalization",
    "BaseSmoothing",
    "CorrelationLocalization",
    "CycleDiagnostics",
    "DistanceLocalization",
    "ETKFAnalysis",
    "EnsembleKalmanFilter",
    "FilterResult",
    "FilterSmoothingESMDA",
    "FilterSmoothingResult",
    "IdentityEvolution",
    "IterationDiagnostics",
    "InflationScheme",
    "LETKFAnalysis",
    "MultiplicativeInflation",
    "ObservationOperator",
    "ObservationTSVD",
    "OnlineStateReduction",
    "StreamingStateReduction",
    "ParameterESMDA",
    "ParameterEvolution",
    "RandomWalkEvolution",
    "RTPP",
    "RTPS",
    "StateESMDA",
    "StateAndParameterESMDA",
    "StateAndTimeVaryingParameterESMDA",
    "StochasticEnKFAnalysis",
    "TemporalLocalization",
    "TemporalObservationOperator",
    "TimeVaryingParameterESMDA",
]
