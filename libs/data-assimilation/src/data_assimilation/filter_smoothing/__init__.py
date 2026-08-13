"""Filter-smoothing hybrid: ESMDA parameter estimation + a sequential filter.

Per assimilation window the ESMDA smoother estimates the parameters over the
whole window (static or as a knot trajectory) and a sequential filter then
produces the window's posterior state, cycle by cycle, with those parameters.
The estimator lives in :mod:`data_assimilation.filter_smoothing.base`, which
also exports the pure trajectory helpers the segment geometry is built from —
they are useful (and testable) on their own, and the run script reads segment
boundaries with them.

See ``docs/data_assimilation.md`` and the class docstring of
:class:`FilterSmoothing` for the algorithm.
"""

from data_assimilation.filter_smoothing.base import (
    FilterSmoothing,
    FilterSmoothingResult,
    knot_times,
    params_for_segment,
    segment_bounds,
    trajectory_values_at,
)

__all__ = [
    "FilterSmoothing",
    "FilterSmoothingResult",
    "knot_times",
    "params_for_segment",
    "segment_bounds",
    "trajectory_values_at",
]
