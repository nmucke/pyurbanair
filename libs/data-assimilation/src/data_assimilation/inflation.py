"""Covariance (spread) inflation schemes for ensemble analyses.

A sequential filter re-integrates every analysis, so under-dispersion
compounds cycle after cycle until the filter ignores the observations
(filter divergence). Inflation counteracts that by rescaling ensemble
anomalies; it lives at the package root because the smoother can use the
same schemes.

Two hooks, both optional (the base class is a no-op for each):

* ``inflate_prior(dev)`` — applied to the forecast anomalies *before* the
  analysis (multiplicative inflation);
* ``inflate_posterior(dev_prior, dev_post)`` — applied to the analysis
  anomalies *after* the update, with the prior anomalies available for
  relaxation methods (RTPS/RTPP).

All anomaly arrays have shape ``(N_rows, N_e)`` with the ensemble mean
already removed; schemes must return the same shape and must not shift the
mean (rescale/blend rows only).
"""

import jax.numpy as jnp


class InflationScheme:
    """Base inflation scheme: both hooks default to the identity."""

    def inflate_prior(self, dev: jnp.ndarray) -> jnp.ndarray:
        """Rescale forecast (prior) anomalies before the analysis."""
        return dev

    def inflate_posterior(
        self, dev_prior: jnp.ndarray, dev_post: jnp.ndarray
    ) -> jnp.ndarray:
        """Rescale analysis (posterior) anomalies after the update."""
        return dev_post


class MultiplicativeInflation(InflationScheme):
    """Fixed multiplicative prior inflation: anomalies scaled by ``factor``."""

    def __init__(self, factor: float) -> None:
        if factor <= 0.0:
            raise ValueError(
                f"Multiplicative inflation factor must be positive, got {factor}."
            )
        self.factor = factor

    def inflate_prior(self, dev: jnp.ndarray) -> jnp.ndarray:
        return self.factor * dev


class RTPS(InflationScheme):
    """Relaxation to prior spread (Whitaker & Hamill 2012).

    Rescales each posterior anomaly row so its ensemble spread relaxes a
    fraction ``alpha`` of the way back toward the prior spread:
    ``sigma_a <- sigma_a + alpha * (sigma_f - sigma_a)``. ``alpha = 0`` is a
    no-op; ``alpha = 1`` restores the prior spread exactly. Rows with zero
    posterior spread are left unchanged (nothing to rescale).
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"RTPS alpha must be in [0, 1], got {alpha}.")
        self.alpha = alpha

    def inflate_posterior(
        self, dev_prior: jnp.ndarray, dev_post: jnp.ndarray
    ) -> jnp.ndarray:
        sigma_prior = jnp.std(dev_prior, axis=1, ddof=1)
        sigma_post = jnp.std(dev_post, axis=1, ddof=1)
        factor = jnp.where(
            sigma_post > 0.0,
            1.0
            + self.alpha
            * (sigma_prior - sigma_post)
            / jnp.where(sigma_post > 0.0, sigma_post, 1.0),
            1.0,
        )
        return dev_post * factor[:, None]


class RTPP(InflationScheme):
    """Relaxation to prior perturbations (Zhang et al. 2004).

    Blends the posterior anomalies with the prior anomalies member-wise:
    ``dev_a <- alpha * dev_f + (1 - alpha) * dev_a``. ``alpha = 0`` is a
    no-op; ``alpha = 1`` keeps the prior anomalies entirely (only the mean is
    updated by the analysis).
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"RTPP alpha must be in [0, 1], got {alpha}.")
        self.alpha = alpha

    def inflate_posterior(
        self, dev_prior: jnp.ndarray, dev_post: jnp.ndarray
    ) -> jnp.ndarray:
        return self.alpha * dev_prior + (1.0 - self.alpha) * dev_post
