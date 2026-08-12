"""Filter smoothing: windowed parameter-trajectory ESMDA with EnKF filtering.

The hybrid of the two other flavors — sequential filtering for the
high-dimensional state, an outer ESMDA loop for the low-dimensional parameter
trajectory over a window of cycles. See
:mod:`data_assimilation.filter_smoothing.base` for the algorithm and
``docs/filter_smoothing/parameter_trajectory_esmda_state_filter.pdf`` for the
derivation.

Re-exported here: the classes a config instantiates
(``FilterSmoothingESMDA``, ``TemporalLocalization``), the moving-window
orchestrator a run script calls (``run_moving_window``) and the result types a
caller reads. Deliberately *not* the inner filter subclass
``_TrajectoryStateFilter``: it is an implementation detail of the outer loop,
which builds it itself.
"""

from data_assimilation.filter_smoothing.base import (
    FilterSmoothingESMDA,
    FilterSmoothingResult,
    IterationDiagnostics,
)
from data_assimilation.filter_smoothing.moving_window import (
    MovingWindowResult,
    WindowDiagnostics,
    run_moving_window,
)
from data_assimilation.filter_smoothing.temporal_localization import (
    TemporalLocalization,
)

__all__ = [
    "FilterSmoothingESMDA",
    "FilterSmoothingResult",
    "IterationDiagnostics",
    "MovingWindowResult",
    "TemporalLocalization",
    "WindowDiagnostics",
    "run_moving_window",
]
