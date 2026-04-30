"""First-order autoregressive parameter time-series model.

Prior:
    Stationary AR(1) trajectories around the configured mean with
    persistence ``phi = exp(-Δt / l_corr)`` (so ``correlation_length``
    parametrizes a smooth Markov process with the requested marginal
    standard deviation).

Extrapolation:
    Per-member AR(1) coefficient fitted by ordinary least squares on
    the posterior trajectory and rolled forward deterministically from
    the last training value.  ``phi_max`` clips the fit for stability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import xarray

from .base import ParameterTimeSeries


class AR1Model(ParameterTimeSeries):
    """Stationary AR(1) prior + per-member AR(1) extrapolation."""

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
        # Use the median spacing — handles non-uniform grids gracefully.
        dt = jnp.median(jnp.diff(time_coords))
        phi = jnp.exp(-dt / jnp.maximum(self.correlation_length, 1e-6))
        phi = jnp.clip(phi, -self.phi_max, self.phi_max)
        innovation_std = jnp.sqrt(jnp.maximum(1.0 - phi**2, 0.0))

        keys = jax.random.split(rng_key, len(self.param_names))
        arrays: dict[str, jnp.ndarray] = {}
        for key, name in zip(keys, self.param_names):
            spec = self._ext(name)
            mean = spec["mean"]
            std = spec["std"]

            # Stationary AR(1): x_0 ~ N(mean, std²); x_{k+1} - mean =
            # phi (x_k - mean) + std * sqrt(1 - phi²) * eps.
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
        del rng_key  # deterministic rollout
        prediction_times = jnp.asarray(prediction_times)
        n_pred = prediction_times.shape[0]

        arrays: dict[str, jnp.ndarray] = {}
        passthrough = {
            n: posterior[n]
            for n in posterior.data_vars
            if "time" not in posterior[n].dims
        }

        for name in self.param_names:
            if name not in posterior.data_vars:
                continue
            da = posterior[name].transpose("time", "ensemble")
            y_train = jnp.asarray(da.values)

            mu = y_train.mean(axis=0)
            yc = y_train - mu[None, :]
            num = jnp.sum(yc[1:] * yc[:-1], axis=0)
            den = jnp.sum(yc[:-1] ** 2, axis=0)
            phi = jnp.where(den > 1e-12, num / jnp.maximum(den, 1e-12), 0.0)
            phi = jnp.clip(phi, -self.phi_max, self.phi_max)

            last = y_train[-1]
            k = jnp.arange(n_pred)
            powers = phi[None, :] ** k[:, None]
            arrays[name] = mu[None, :] + powers * (last - mu)[None, :]

        return self._build_dataset(arrays, prediction_times, passthrough=passthrough)
