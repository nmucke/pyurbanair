"""Sequential (filtering) data assimilation: EnKF and extensions.

The analysis math in :mod:`data_assimilation.filtering.analysis` is shared
with the ESMDA smoothers (``smoothing/``); the cycle loop lives in
:mod:`data_assimilation.filtering.base`.
"""

from data_assimilation.filtering.analysis import (
    AnalysisScheme,
    StochasticEnKFAnalysis,
    stochastic_enkf_update,
)

__all__ = [
    "AnalysisScheme",
    "StochasticEnKFAnalysis",
    "stochastic_enkf_update",
]
