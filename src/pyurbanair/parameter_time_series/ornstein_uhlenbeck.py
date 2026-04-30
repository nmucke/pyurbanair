"""Ornstein-Uhlenbeck parameter time-series model.

Prior:
    Stationary OU SDE draws around the configured mean with
    mean-reversion ``θ = 1 / l_corr`` and stationary marginal variance
    ``std²``.  Each ensemble member uses independent Brownian
    increments.

Extrapolation:
    OU SDE anchored to the external prior:

        dX_t = θ (x_ext - X_t) dt + σ dW_t,

    with ``θ = 1 / l_corr`` (matching how ``α`` depends on ``l_corr``
    in ``ar2_relaxation``) and ``σ = σ_ext √(2θ)`` so the marginal
    standard deviation of ``X_t`` approaches ``σ_ext`` as ``t → ∞``.
    Each member starts from its end-of-window posterior value for
    continuity; the ensemble relaxes toward the external prior with
    independent Brownian increments per member.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import xarray

from .base import ParameterTimeSeries


class OrnsteinUhlenbeckModel(ParameterTimeSeries):
    """Stationary OU prior + per-member OU-with-drift extrapolation."""

    def __init__(
        self,
        external_priors: dict[str, dict[str, float]],
        ensemble_size: int,
        correlation_length: float,
        phi_max: float = 0.999,
    ) -> None:
        super().__init__(external_priors, ensemble_size)
        self.correlation_length = correlation_length
        self.phi_max = phi_max

    def sample_prior(
        self,
        time_coords: jnp.ndarray,
        rng_key: jax.Array,
    ) -> xarray.Dataset:
        time_coords = jnp.asarray(time_coords)
        n_t = time_coords.shape[0]
        dt = jnp.median(jnp.diff(time_coords))
        # Exact discretization of dx = -θ x dt + σ dW with θ = 1/l_corr,
        # stationary variance std²: φ = exp(-θ dt), innovation variance
        # = std² (1 - φ²).
        phi = jnp.exp(-dt / jnp.maximum(self.correlation_length, 1e-6))
        phi = jnp.clip(phi, -self.phi_max, self.phi_max)
        innovation_std = jnp.sqrt(jnp.maximum(1.0 - phi**2, 0.0))

        keys = jax.random.split(rng_key, len(self.param_names))
        arrays: dict[str, jnp.ndarray] = {}
        for key, name in zip(keys, self.param_names):
            spec = self._ext(name)
            mean = spec["mean"]
            std = spec["std"]

            key_init, key_eps = jax.random.split(key)
            x0 = jax.random.normal(key_init, (self.ensemble_size,)) * std
            eps = jax.random.normal(key_eps, (n_t - 1, self.ensemble_size))

            def step(x_prev, eps_k):
                x_new = phi * x_prev + std * innovation_std * eps_k
                return x_new, x_new

            _, x_rest = jax.lax.scan(step, x0, eps)
            anomaly = jnp.concatenate([x0[None, :], x_rest], axis=0)
            arrays[name] = mean + anomaly

        return self._build_dataset(arrays, time_coords)

    def extrapolate(
        self,
        posterior: xarray.Dataset,
        prediction_times: jnp.ndarray,
        rng_key: jax.Array,
    ) -> xarray.Dataset:
        prediction_times = jnp.asarray(prediction_times)
        n_pred = prediction_times.shape[0]
        dt = jnp.median(jnp.diff(prediction_times))

        # Exact discretization of dX = θ(x_ext - X) dt + σ dW with
        # θ = 1/l_corr and σ = σ_ext √(2θ): φ = exp(-θ dt), conditional
        # variance σ_ext² (1 - φ²) per step.
        phi = jnp.exp(-dt / jnp.maximum(self.correlation_length, 1e-6))
        phi = jnp.clip(phi, -self.phi_max, self.phi_max)
        innovation_std = jnp.sqrt(jnp.maximum(1.0 - phi**2, 0.0))

        var_names = [
            n
            for n in self.param_names
            if n in posterior.data_vars and "time" in posterior[n].dims
        ]
        var_keys = jax.random.split(rng_key, max(len(var_names), 1))

        arrays: dict[str, jnp.ndarray] = {}
        passthrough = {
            n: posterior[n]
            for n in posterior.data_vars
            if "time" not in posterior[n].dims
        }

        for name, key in zip(var_names, var_keys):
            spec = self._ext(name)
            x_ext = spec["mean"]
            std = spec["std"]

            da = posterior[name].transpose("time", "ensemble")
            y_train = jnp.asarray(da.values)
            n_e = y_train.shape[1]

            x0 = y_train[-1, :]
            eps = jax.random.normal(key, (n_pred - 1, n_e))

            def step(x_prev_state, eps_k):
                x_new = (
                    x_ext
                    + phi * (x_prev_state - x_ext)
                    + std * innovation_std * eps_k
                )
                return x_new, x_new

            _, x_rest = jax.lax.scan(step, x0, eps)
            arrays[name] = jnp.concatenate([x0[None, :], x_rest], axis=0)

        return self._build_dataset(arrays, prediction_times, passthrough=passthrough)
