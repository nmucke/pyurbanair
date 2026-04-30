"""GP-based time-varying parameter model with linear trend extrapolation.

Prior:
    Smooth ensemble drawn from an RBF Gaussian process with marginal
    distribution ``N(mean, std)`` per parameter and configured
    correlation length.

Extrapolation:
    Per-member least-squares linear trend fitted to the posterior
    trajectory plus a zero-mean GP residual with heteroscedastic noise
    from cross-ensemble spread.  The trend continues into the next
    window (optionally exponentially damped) so the prediction does not
    collapse to a constant.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import xarray

from .base import ParameterTimeSeries, gp_predict, sample_gp_ensemble


def _linear_trend_at(
    t: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    t_end: float,
    slope_damping_time: Optional[float],
) -> jnp.ndarray:
    if slope_damping_time is None or slope_damping_time <= 0:
        return a + b * t
    tau = slope_damping_time
    delta = jnp.maximum(t - t_end, 0.0)
    return a + b * t_end + b * tau * (1.0 - jnp.exp(-delta / tau))


class GPLinearTrendModel(ParameterTimeSeries):
    """Smooth GP prior + linear-trend-with-GP-residual extrapolation."""

    def __init__(
        self,
        external_priors: dict[str, dict[str, float]],
        ensemble_size: int,
        correlation_length: float,
        slope_damping_time: Optional[float] = None,
        continuity_jitter: float = 1e-6,
    ) -> None:
        super().__init__(external_priors, ensemble_size)
        self.correlation_length = correlation_length
        self.slope_damping_time = slope_damping_time
        self.continuity_jitter = continuity_jitter

    def sample_prior(
        self,
        time_coords: jnp.ndarray,
        rng_key: jax.Array,
    ) -> xarray.Dataset:
        time_coords = jnp.asarray(time_coords)
        keys = jax.random.split(rng_key, len(self.param_names))
        arrays: dict[str, jnp.ndarray] = {}
        for key, name in zip(keys, self.param_names):
            spec = self._ext(name)
            arrays[name] = sample_gp_ensemble(
                key,
                time_coords,
                mean=spec["mean"],
                std=spec["std"],
                ensemble_size=self.ensemble_size,
                correlation_length=self.correlation_length,
            )
        return self._build_dataset(arrays, time_coords)

    def extrapolate(
        self,
        posterior: xarray.Dataset,
        prediction_times: jnp.ndarray,
        rng_key: jax.Array,
    ) -> xarray.Dataset:
        del rng_key  # deterministic
        time_coords = jnp.asarray(posterior.coords["time"].values)
        prediction_times = jnp.asarray(prediction_times)
        t_end = float(time_coords[-1])

        t_mean = time_coords.mean()
        t_centered = time_coords - t_mean
        denom = jnp.sum(t_centered**2)

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
            y_train = jnp.asarray(da.values)  # (N_t, N_e)

            y_means = y_train.mean(axis=0, keepdims=True)
            b = jnp.sum(t_centered[:, None] * (y_train - y_means), axis=0) / denom
            a = y_means[0] - b * t_mean

            trend_train = a[None, :] + b[None, :] * time_coords[:, None]
            residual = y_train - trend_train

            sigma_t = jnp.asarray(da.std(dim="ensemble").values)
            noise_var = jnp.maximum(sigma_t**2, self.continuity_jitter)
            noise_var = noise_var.at[-1].set(self.continuity_jitter)

            amplitude = jnp.maximum(jnp.std(residual), 1e-6)

            def _predict(r):
                return gp_predict(
                    time_coords,
                    r,
                    prediction_times,
                    self.correlation_length,
                    float(amplitude),
                    noise_var,
                )

            res_means, _ = jax.vmap(_predict, in_axes=1, out_axes=1)(residual)

            trend_pred = jax.vmap(
                lambda ae, be: _linear_trend_at(
                    prediction_times, ae, be, t_end, self.slope_damping_time
                ),
                in_axes=(0, 0),
                out_axes=1,
            )(a, b)

            arrays[name] = trend_pred + res_means

        return self._build_dataset(arrays, prediction_times, passthrough=passthrough)
