"""Fluid-masked errors and turbulence statistics for comparing rollouts.

Fields are ``(T, C, nz, ny, nx)`` arrays; ``fluid`` is the ``(nz, ny, nx)``
boolean mask. Fluctuations are taken about each cell's time mean over the
frames given, as the papers do for their statistics.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def persistence(truth: np.ndarray, start: int) -> np.ndarray:
    """The rollout that repeats frame ``start`` (the truth before it)."""
    pred = truth.copy()
    pred[start + 1 :] = truth[start]
    return pred


def masked_errors(
    pred: np.ndarray, truth: np.ndarray, fluid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame RMSE and MAE over the fluid cells and all channels."""
    error = (pred - truth)[:, :, fluid]  # (T, C, n_fluid)
    return np.sqrt((error**2).mean(axis=(1, 2))), np.abs(error).mean(axis=(1, 2))


def spatial_std(fields: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    """``(T, C)`` standard deviation over the fluid cells: a smoothed or
    laminarising rollout loses it, an overenergetic one gains it."""
    result: np.ndarray = fields[:, :, fluid].std(axis=-1)
    return result


def fluctuations(fields: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    """Departures from each cell's time mean, zero in solid cells."""
    result: np.ndarray = (fields - fields.mean(axis=0, keepdims=True)) * fluid
    return result


def lateral_spectrum(
    field: np.ndarray, fluid: np.ndarray, dy: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided energy spectrum along y of ``(T, ny, nx)`` fluctuations at one
    height, averaged over x and time. Returns wavenumbers (rad per unit of
    ``dy``) and the spectrum, which sums to the mean squared fluctuation."""
    ny = field.shape[-2]
    coeffs = np.fft.rfft(field * fluid, axis=-2) / ny
    energy = (np.abs(coeffs) ** 2).mean(axis=(0, -1))
    energy[1 : (ny + 1) // 2] *= 2  # fold the negative wavenumbers
    k = 2 * np.pi * np.fft.rfftfreq(ny, d=dy)
    return k, energy


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
