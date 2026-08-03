"""Reductions of pre-extracted sensor and probe series to window statistics.

Consumes ``(ensemble, time, sensor)`` arrays a script has already pulled out of
the state files. Extraction itself stays in ``scripts/esmda/_esmda_common.py``:
it needs ``data_assimilation``'s observation operator (jax) and the run-dir
layout, both forbidden here.

Populated in WP0.2 (move), extended in WP1.3.
"""

# mypy: ignore-errors
# Moved in WP0.2 from ``scripts/esmda/_esmda_common.py``, which carries a
# file-level mypy waiver; kept here rather than annotated during a pure
# refactor.

from __future__ import annotations

import numpy as np


def sensor_magnitude(components):
    """Velocity magnitude |U| from a ``(component, ...)`` sensor series."""
    return np.sqrt((components**2).sum("component"))
