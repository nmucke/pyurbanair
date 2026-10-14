"""Turbulence statistics for comparing rollouts (scripts/surrogate_comparison/).

Fields are ``(T, C, nz, ny, nx)`` arrays; ``fluid`` is the ``(nz, ny, nx)``
boolean mask. Fluctuations are taken about each cell's time mean over the
frames given, as the papers do for their statistics.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def fluctuations(fields: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    """Departures from each cell's time mean, zero in solid cells."""
    result: np.ndarray = (fields - fields.mean(axis=0, keepdims=True)) * fluid
    return result


def profiles(
    fields: np.ndarray,
    fluid: np.ndarray,
    u: int = 0,
    w: int = 2,
    velocity: Sequence[int] = (0, 1, 2),
) -> dict[str, np.ndarray]:
    """Horizontally averaged ``(nz,)`` profiles over fluid cells: mean
    streamwise velocity, resolved TKE (over the ``velocity`` channels) and
    Reynolds shear stress ``-<u'w'>``."""
    prime = fluctuations(fields, fluid)
    count = np.maximum(fluid.sum(axis=(-2, -1)), 1)

    def average(x: np.ndarray) -> np.ndarray:
        result: np.ndarray = (x * fluid).sum(axis=(-2, -1)).mean(axis=0) / count
        return result

    return {
        "mean_u": average(fields[:, u]),
        "tke": average(0.5 * (prime[:, list(velocity)] ** 2).sum(axis=1)),
        "uw": -average(prime[:, u] * prime[:, w]),
    }
