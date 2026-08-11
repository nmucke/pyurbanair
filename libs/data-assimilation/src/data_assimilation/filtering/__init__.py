"""Sequential (filtering) data assimilation: EnKF and extensions.

The analysis math in :mod:`data_assimilation.filtering.analysis` is shared
with the ESMDA smoothers (``smoothing/``); the cycle loop lives in
:mod:`data_assimilation.filtering.base`.

Re-exported here: the schemes a config instantiates (``ETKFAnalysis``,
``LETKFAnalysis``, ``ObservationTSVD``). Deliberately *not*, because nothing
outside their defining module imports them:

* the ETKF/LETKF kernel functions (``ensemble_transform``,
  ``apply_ensemble_transform``, ``whiten_observations``) — shared internals of
  one module;
* the transform and diagnostic types (``ObservationTransform``,
  ``LocalTransformDiagnostics``) — ``BaseFilter`` reads what a scheme published
  by attribute name and never imports their types, which is what keeps it
  independent of the ensemble-transform module;
* the ``LocalizationPolicy`` literal annotating
  ``AnalysisScheme.localization_policy``.

All of them stay importable from the module that defines them
(:mod:`data_assimilation.filtering.etkf`, :mod:`data_assimilation.filtering.\
analysis`) without becoming package API.
"""

from data_assimilation.filtering.analysis import (
    AnalysisScheme,
    StochasticEnKFAnalysis,
    stochastic_enkf_update,
)
from data_assimilation.filtering.base import (
    BaseFilter,
    CycleDiagnostics,
    EnsembleKalmanFilter,
    FilterResult,
)
from data_assimilation.filtering.etkf import (
    ETKFAnalysis,
    LETKFAnalysis,
    ObservationTSVD,
)
from data_assimilation.filtering.parameter_evolution import (
    IdentityEvolution,
    ParameterEvolution,
    RandomWalkEvolution,
)

__all__ = [
    "AnalysisScheme",
    "BaseFilter",
    "CycleDiagnostics",
    "ETKFAnalysis",
    "EnsembleKalmanFilter",
    "FilterResult",
    "IdentityEvolution",
    "LETKFAnalysis",
    "ObservationTSVD",
    "ParameterEvolution",
    "RandomWalkEvolution",
    "StochasticEnKFAnalysis",
    "stochastic_enkf_update",
]
