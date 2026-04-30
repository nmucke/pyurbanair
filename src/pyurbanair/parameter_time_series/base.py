"""Base class and shared utilities for parameter time-series models.

A :class:`ParameterTimeSeries` ties together the *prior* sampling and
*between-window extrapolation* for a single generative model of
time-varying parameters.  Subclasses implement one specific method.
"""

from __future__ import annotations

import abc
from typing import Optional

import jax
import jax.numpy as jnp
import xarray


class ParameterTimeSeries(abc.ABC):
    """ABC for time-varying parameter prior + extrapolation.

    Args:
        external_priors: Mapping ``name -> {"mean", "std", optional "min",
            optional "max"}`` defining the external prior (the paper's
            ``x_ext`` and ``Σ_ext``) and optional value clips.
        ensemble_size: Number of ensemble members.
    """

    def __init__(
        self,
        external_priors: dict[str, dict[str, float]],
        ensemble_size: int,
    ) -> None:
        self.external_priors = external_priors
        self.param_names = list(external_priors.keys())
        self.ensemble_size = ensemble_size

    @abc.abstractmethod
    def sample_prior(
        self,
        time_coords: jnp.ndarray,
        rng_key: jax.Array,
    ) -> xarray.Dataset:
        """Draw the initial-window prior ensemble."""

    @abc.abstractmethod
    def extrapolate(
        self,
        posterior: xarray.Dataset,
        prediction_times: jnp.ndarray,
        rng_key: jax.Array,
    ) -> xarray.Dataset:
        """Build the next window's prior given the previous window's posterior.

        ``prediction_times`` is the time grid for the next window.  Methods
        that fit-and-roll-forward from the posterior expect
        ``prediction_times[0] == posterior.time[-1]`` so the first predicted
        value coincides with each member's end-of-window value.  Methods
        that synthesize a fresh trajectory (e.g. AR(2) relaxation) treat
        ``prediction_times`` as the local time axis for the new window.
        """

    # ------------------------------------------------------------------
    # Shared helpers for subclasses
    # ------------------------------------------------------------------

    def _ext(self, name: str) -> dict[str, float]:
        return self.external_priors[name]

    def _apply_clips(self, name: str, values: jnp.ndarray) -> jnp.ndarray:
        spec = self.external_priors[name]
        lo = spec.get("min")
        hi = spec.get("max")
        if lo is not None:
            values = jnp.maximum(values, lo)
        if hi is not None:
            values = jnp.minimum(values, hi)
        return values

    def _build_dataset(
        self,
        arrays: dict[str, jnp.ndarray],
        time_coords: jnp.ndarray,
        passthrough: Optional[dict[str, xarray.DataArray]] = None,
    ) -> xarray.Dataset:
        data_vars: dict = {
            name: (("time", "ensemble"), self._apply_clips(name, arr))
            for name, arr in arrays.items()
        }
        if passthrough:
            data_vars.update(passthrough)
        return xarray.Dataset(
            data_vars=data_vars,
            coords={
                "time": jnp.asarray(time_coords),
                "ensemble": jnp.arange(self.ensemble_size),
            },
        )


# ---------------------------------------------------------------------------
# RBF / GP helpers (shared across GP-based methods)
# ---------------------------------------------------------------------------


def rbf_kernel(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    correlation_length: float,
) -> jnp.ndarray:
    dt = x1[:, None] - x2[None, :]
    return jnp.exp(-0.5 * (dt / jnp.maximum(correlation_length, 1e-6)) ** 2)


def gp_predict(
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_pred: jnp.ndarray,
    correlation_length: float,
    amplitude: float,
    noise_var: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Posterior mean and std at ``x_pred`` for a 1-D GP regression."""
    amp_sq = amplitude**2
    K_tt = amp_sq * rbf_kernel(x_train, x_train, correlation_length)
    K_tt = K_tt + jnp.diag(noise_var) + 1e-8 * jnp.eye(K_tt.shape[0])

    K_pt = amp_sq * rbf_kernel(x_pred, x_train, correlation_length)
    K_pp = amp_sq * rbf_kernel(x_pred, x_pred, correlation_length)

    L = jnp.linalg.cholesky(K_tt)
    alpha = jax.scipy.linalg.cho_solve((L, True), y_train)
    V = jax.scipy.linalg.solve_triangular(L, K_pt.T, lower=True)

    mean = K_pt @ alpha
    var = jnp.diag(K_pp) - jnp.sum(V**2, axis=0)
    std = jnp.sqrt(jnp.maximum(var, 0.0))
    return mean, std


def sample_gp_ensemble(
    rng_key: jax.Array,
    time_coords: jnp.ndarray,
    mean: float,
    std: float,
    ensemble_size: int,
    correlation_length: float,
) -> jnp.ndarray:
    """Draw a smooth ensemble from an RBF GP prior with given marginals."""
    n_t = time_coords.shape[0]
    K = rbf_kernel(time_coords, time_coords, correlation_length)
    K = K + 1e-6 * jnp.eye(n_t)
    L = jnp.linalg.cholesky(K)
    z = jax.random.normal(rng_key, (n_t, ensemble_size))
    return mean + std * (L @ z)
