"""Diagnostic skill metrics for time-varying parameter assimilation.

Helpers operate on numpy arrays of shape ``(ensemble, time)`` and a truth
array of shape ``(time,)``. They are intentionally pure-numpy (no JAX) so
they can be applied to ``xarray.Dataset`` outputs after an ESMDA run.
"""

from __future__ import annotations

import numpy as np


def per_knot_error(ens: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-knot magnitude of (ensemble mean - truth)."""
    return np.abs(ens.mean(axis=0) - truth)


def per_knot_spread(ens: np.ndarray) -> np.ndarray:
    """Per-knot ensemble standard deviation (ddof=1)."""
    if ens.shape[0] < 2:
        return np.zeros(ens.shape[1])
    return ens.std(axis=0, ddof=1)


def per_knot_crps(ens: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-knot sample CRPS using the energy-form estimator.

    ``CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|`` where ``X, X'`` are
    independent draws from ``F``. With a finite ensemble of size ``N`` the
    pairwise term is computed as the mean of ``|x_i - x_j|`` over all
    ``(i, j)`` pairs.
    """
    n = ens.shape[0]
    term1 = np.mean(np.abs(ens - truth[None, :]), axis=0)
    if n < 2:
        return term1
    diffs = np.abs(ens[:, None, :] - ens[None, :, :])
    term2 = 0.5 * diffs.mean(axis=(0, 1))
    return term1 - term2


def per_knot_in_band(
    ens: np.ndarray, truth: np.ndarray, alpha: float = 0.9
) -> np.ndarray:
    """Boolean per-knot indicator: truth in central ``alpha`` ensemble band."""
    lo = np.quantile(ens, 0.5 - alpha / 2.0, axis=0)
    hi = np.quantile(ens, 0.5 + alpha / 2.0, axis=0)
    return (truth >= lo) & (truth <= hi)


def summary_scalars(
    ens: np.ndarray, truth: np.ndarray, alpha: float = 0.9
) -> dict[str, float]:
    """Time-averaged skill scalars for one parameter at one ESMDA step."""
    err = per_knot_error(ens, truth)
    spr = per_knot_spread(ens)
    crps = per_knot_crps(ens, truth)
    band = per_knot_in_band(ens, truth, alpha=alpha)
    return {
        "time_avg_error": float(np.sqrt(np.mean(err**2))),
        "time_avg_spread": float(np.mean(spr)),
        "mean_crps": float(np.mean(crps)),
        "coverage": float(np.mean(band)),
    }
