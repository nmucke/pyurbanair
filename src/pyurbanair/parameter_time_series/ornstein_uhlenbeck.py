"""Ornstein-Uhlenbeck parameter time-series model.

Prior:
    Stationary OU SDE draws around the configured mean with
    mean-reversion ``θ = 1 / l_corr`` and stationary marginal variance
    ``std²``.  Each ensemble member uses independent Brownian
    increments.

Extrapolation:
    Per-member OU-with-drift fit ``x_{k+1} = c_e + φ_e x_k + σ ε`` by
    OLS on the posterior trajectory, with ensemble-pooled diffusion
    ``σ``.  Rolled forward stochastically (Euler-Maruyama) with
    independent Brownian increments per member, so the ensemble spread
    grows across the extrapolation window instead of collapsing.
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
            da = posterior[name].transpose("time", "ensemble")
            y_train = jnp.asarray(da.values)
            n_e = y_train.shape[1]

            x_prev = y_train[:-1, :]
            x_next = y_train[1:, :]

            mean_prev = x_prev.mean(axis=0, keepdims=True)
            mean_next = x_next.mean(axis=0, keepdims=True)
            cov = jnp.sum((x_prev - mean_prev) * (x_next - mean_next), axis=0)
            var_prev = jnp.sum((x_prev - mean_prev) ** 2, axis=0)
            phi = jnp.where(
                var_prev > 1e-12, cov / jnp.maximum(var_prev, 1e-12), 0.0
            )
            phi = jnp.clip(phi, -self.phi_max, self.phi_max)
            c = mean_next[0] - phi * mean_prev[0]

            residuals = x_next - (c[None, :] + phi[None, :] * x_prev)
            sigma = jnp.sqrt(jnp.mean(residuals**2))

            eps = jax.random.normal(key, (n_pred - 1, n_e))
            x0 = y_train[-1, :]

            def step(x_prev_state, eps_k):
                x_new = c + phi * x_prev_state + sigma * eps_k
                return x_new, x_new

            _, x_rest = jax.lax.scan(step, x0, eps)
            arrays[name] = jnp.concatenate([x0[None, :], x_rest], axis=0)

        return self._build_dataset(arrays, prediction_times, passthrough=passthrough)
