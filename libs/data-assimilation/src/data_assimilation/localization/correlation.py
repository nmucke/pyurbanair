"""Adaptive correlation-based localization.

Implements the correlation-truncation localization of Vossepoel et al.
(2025, MWR-D-24-0269.1, sections 3a, 3c, 3d).  For each augmented-state row
``l`` and predicted measurement ``j`` we estimate the ensemble correlation
``rho(l, j)`` and:

* exclude the observation when ``|rho| < rho_t`` (the correlation truncation
  threshold);
* otherwise inflate its error variance based on the "correlation distance"
  ``d_c = 1 - |rho|`` so that observations near the truncation distance are
  tapered, yielding smoother updates (the paper attributes the method's
  success to this inflation).

Unlike distance-based localization, this strategy needs no spatial
coordinates for the state rows — only the ensemble anomalies — which makes it
applicable to abstract parameter rows as well as gridded state rows.
"""

import math
from typing import Optional

import jax.numpy as jnp
from data_assimilation.localization.base import BaseLocalization


class CorrelationLocalization(BaseLocalization):
    """Adaptive correlation-based localization with error-variance tapering.

    Args:
        truncation_correlation: Correlation truncation threshold ``rho_t``.
            Observations with ``|rho| < rho_t`` are excluded.  If ``None``,
            the theoretical first guess ``3 / sqrt(N_e)`` is used per ensemble
            size (Eq. 6); the paper found tuned values in ``[0.3, 0.4]`` to
            work best.
        tapering_beta: ``beta in (0, 1)``; the fraction of the truncation
            distance within which observations are *not* tapered (Eq. 9).
            Defaults to ``0.5`` as in the paper.
        max_inflation: Maximum inflation ``E_max`` reached at the truncation
            distance (Eq. 10).  As in the paper and reference code, ``E_max``
            multiplies the observation-error perturbation (std), so the error
            *variance* there is scaled by ``E_max ** 2``.  Defaults to ``8.0``
            — the value the paper found best for correlation-based
            localization.
        block_grouping: Local-analysis granularity.  ``False`` (default)
            updates each augmented row on its own (per-row local analysis).
            ``True`` requests the paper's "grid block" analysis (sec. 3b):
            co-located rows — the ``u``/``v``/``w`` state at one cell, or all
            time knots of one parameter — are updated *jointly* with a single
            observation selection and transition matrix.  The smoother builds
            the block ids from the augmented-state layout; this flag only
            selects which behaviour it asks for.
    """

    def __init__(
        self,
        truncation_correlation: Optional[float] = None,
        tapering_beta: float = 0.5,
        max_inflation: float = 8.0,
        block_grouping: bool = False,
    ) -> None:
        if truncation_correlation is not None and not (
            0.0 < truncation_correlation < 1.0
        ):
            raise ValueError("truncation_correlation must lie in (0, 1).")
        if not (0.0 < tapering_beta < 1.0):
            raise ValueError("tapering_beta must lie in (0, 1).")
        if max_inflation < 1.0:
            raise ValueError("max_inflation must be >= 1.")

        self.truncation_correlation = truncation_correlation
        self.tapering_beta = tapering_beta
        self.max_inflation = max_inflation
        self.block_grouping = block_grouping

    def _truncation_correlation(self, N_e: int) -> float:
        """Resolve the truncation threshold, defaulting to ``3 / sqrt(N_e)``."""
        if self.truncation_correlation is not None:
            return self.truncation_correlation
        return min(3.0 / math.sqrt(N_e), 0.99)

    def inflation_factors(
        self,
        aug_dev: jnp.ndarray,
        pred_obs_dev: jnp.ndarray,
    ) -> jnp.ndarray:
        N_aug, N_e = aug_dev.shape
        N_d = pred_obs_dev.shape[0]

        rho_t = self._truncation_correlation(N_e)
        d_t = 1.0 - rho_t  # truncation distance, Eq. (8)

        # Ensemble correlation between each augmented row and each predicted
        # observation.  Guard zero-variance rows/observations (constant across
        # the ensemble) by treating their correlation as zero -> excluded.
        # Use a consistent (N_e - 1) normalization for both the covariance and
        # the standard deviations so the ratio is the exact sample correlation
        # (the reference uses a consistent 1/N in numerator and denominators).
        aug_std = jnp.std(aug_dev, axis=1, ddof=1)  # (N_aug,)
        obs_std = jnp.std(pred_obs_dev, axis=1, ddof=1)  # (N_d,)
        cov = jnp.dot(aug_dev, pred_obs_dev.T) / (N_e - 1)  # (N_aug, N_d)
        denom = jnp.outer(aug_std, obs_std)  # (N_aug, N_d)
        rho = jnp.where(denom > 0.0, cov / jnp.where(denom > 0.0, denom, 1.0), 0.0)
        rho = jnp.clip(rho, -1.0, 1.0)

        d_c = 1.0 - jnp.abs(rho)  # correlation distance, Eq. (7)

        # Error-variance inflation, Eqs. (9)-(10). The tapering turns on once
        # d_c exceeds beta * d_t and reaches max_inflation at d_c = d_t.
        b = (1.0 - self.tapering_beta) * d_t / jnp.sqrt(jnp.log(self.max_inflation))
        taper_active = d_c > (self.tapering_beta * d_t)
        # Guard b == 0 (max_inflation == 1 -> no tapering) to avoid div-by-0.
        safe_b = jnp.where(b > 0.0, b, 1.0)
        taper = jnp.where(
            taper_active,
            jnp.exp(((d_c - self.tapering_beta * d_t) / safe_b) ** 2),
            1.0,
        )
        inflation = jnp.where(b > 0.0, taper, 1.0)

        # Exclude observations below the correlation threshold (|rho| < rho_t,
        # i.e. d_c > d_t) by assigning infinite inflation.
        return jnp.where(jnp.abs(rho) >= rho_t, inflation, jnp.inf)
