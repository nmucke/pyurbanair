"""Diagnostic skill metrics for time-varying parameter assimilation.

Helpers operate on numpy arrays of shape ``(ensemble, time)`` and a truth
array of shape ``(time,)``. They are intentionally pure-numpy (no JAX) so
they can be applied to ``xarray.Dataset`` outputs after an ESMDA run.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class EnsembleUniqueness(TypedDict):
    """Typed result returned by :func:`ensemble_uniqueness`."""

    n_members: int
    n_unique: int
    min_pairwise: float | None
    median_pairwise: float | None


def per_knot_error(ens: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-knot magnitude of (ensemble mean - truth)."""
    return np.abs(ens.mean(axis=0) - truth)


def per_knot_spread(ens: np.ndarray) -> np.ndarray:
    """Per-knot ensemble standard deviation (ddof=1)."""
    if ens.shape[0] < 2:
        return np.zeros(ens.shape[1])
    return ens.std(axis=0, ddof=1)


def per_knot_crps(ens: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-knot fair sample CRPS using the energy-form estimator.

    ``CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|`` where ``X, X'`` are
    independent draws from ``F``. With a finite ensemble of size ``N`` the
    pairwise term excludes the zero diagonal and divides by ``N(N - 1)``.
    This is the fair estimator: unlike the all-pairs ``N**2`` form, it does
    not reward under-dispersion at finite ensemble size.
    """
    n = ens.shape[0]
    term1 = np.mean(np.abs(ens - truth[None, :]), axis=0)
    if n < 2:
        return term1
    # For sorted scalar samples x_(i),
    #   sum_{i,j} |x_i - x_j| = 2 sum_i (2i - n + 1) x_(i).
    # This is algebraically identical to forming the full pairwise-difference
    # tensor, but avoids its O(N**2) memory cost.
    sorted_ens = np.sort(ens, axis=0)
    weights = (2 * np.arange(n) - n + 1).reshape((n,) + (1,) * (sorted_ens.ndim - 1))
    term2 = np.sum(weights * sorted_ens, axis=0) / (n * (n - 1))
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
        "time_avg_spread": float(np.sqrt(np.mean(spr**2))),
        "mean_crps": float(np.mean(crps)),
        "coverage": float(np.mean(band)),
    }


def ensemble_uniqueness(members: np.ndarray) -> EnsembleUniqueness:
    """Summarize exact duplicate rows and pairwise distances in an ensemble.

    Args:
        members: Two-dimensional ``(n_members, n_features)`` array containing
            flattened parameter vectors. ``n_unique`` uses exact row matching;
            legitimately close-but-distinct members remain distinct.

    Returns:
        Member and exact-unique counts plus the minimum and median off-diagonal
        pairwise L2 distances. Distances are ``None`` for fewer than two
        members.
    """
    values = np.asarray(members)
    if values.ndim != 2:
        raise ValueError(
            "members must have shape (n_members, n_features); "
            f"got array with shape {values.shape}"
        )

    n_members = int(values.shape[0])
    n_unique = int(np.unique(values, axis=0).shape[0])
    if n_members < 2:
        return {
            "n_members": n_members,
            "n_unique": n_unique,
            "min_pairwise": None,
            "median_pairwise": None,
        }

    distances = np.linalg.norm(
        values[:, None, :].astype(float) - values[None, :, :].astype(float),
        axis=-1,
    )
    pairwise = distances[np.triu_indices(n_members, k=1)]
    return {
        "n_members": n_members,
        "n_unique": n_unique,
        "min_pairwise": float(np.min(pairwise)),
        "median_pairwise": float(np.median(pairwise)),
    }
