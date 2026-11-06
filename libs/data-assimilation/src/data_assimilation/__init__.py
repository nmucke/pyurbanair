"""Ensemble data assimilation in JAX: ESMDA smoothing and EnKF filtering.

Public API re-exported here so consumers can import from the package root
(``from data_assimilation import ParameterESMDA``) and refactors of the file
layout do not break callers. See ``docs/data_assimilation.md``.
"""

from data_assimilation.adaptive_observation_error import AdaptiveObservationError
from data_assimilation.filter_smoothing import (
    FilterSmoothing,
    FilterSmoothingResult,
    TemperingPolicy,
    resolve_tempering_policy,
)
from data_assimilation.filtering import (
    AnalysisScheme,
    BaseFilter,
    CycleDiagnostics,
    EnsembleKalmanFilter,
    ETKFAnalysis,
    FilterResult,
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
    AggregateObservations,
    ObservationOperator,
    TemporalObservationOperator,
    flatten_observations,
)
from data_assimilation.observation_statistics import (
    ObservationStatistics,
    ResolvedStatisticsError,
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
from data_assimilation.smoothing.field_analysis import (
    FieldAnalysisInfo,
    field_analysis,
    field_analysis_netcdf,
    global_field_weights,
)

__all__ = [
    "AdaptiveObservationError",
    "AggregateObservations",
    "AnalysisScheme",
    "BaseFilter",
    "BaseLocalization",
    "BaseSmoothing",
    "CorrelationLocalization",
    "CycleDiagnostics",
    "DistanceLocalization",
    "ETKFAnalysis",
    "EnsembleKalmanFilter",
    "FieldAnalysisInfo",
    "FilterResult",
    "FilterSmoothing",
    "FilterSmoothingResult",
    "InflationScheme",
    "LETKFAnalysis",
    "MultiplicativeInflation",
    "ObservationOperator",
    "ObservationStatistics",
    "ObservationTSVD",
    "OnlineStateReduction",
    "StreamingStateReduction",
    "ParameterESMDA",
    "ParameterEvolution",
    "RandomWalkEvolution",
    "ResolvedStatisticsError",
    "RTPP",
    "RTPS",
    "StateESMDA",
    "StateAndParameterESMDA",
    "StateAndTimeVaryingParameterESMDA",
    "StochasticEnKFAnalysis",
    "TemperingPolicy",
    "TemporalObservationOperator",
    "TimeVaryingParameterESMDA",
    "field_analysis",
    "field_analysis_netcdf",
    "flatten_observations",
    "global_field_weights",
    "resolve_tempering_policy",
]
