"""Parameter samplers for training-data generation.

Each sampler exposes a `sample_prior(time_coords, rng_key)` method,
returning an `xarray.Dataset` with `(time, ensemble)` arrays for every
parameter.
The training-data script swaps them in via Hydra `_target_` blocks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import xarray

from pyurbanair.dynamic_parameters.ar2_relaxation import AR2RelaxationModel
from pyurbanair.static_parameters.distributions import Normal


class UniformParameterSampler:
    """Draw a fresh uniform `[min, max]` time series per simulation.

    For each (time, ensemble) cell, sample independently from a uniform
    distribution. Every simulation member gets its own time-varying
    trajectory of every parameter — the forward model linearly
    interpolates between the sampled values, so `time_coords` length
    controls smoothness (fewer points → smoother, more points → noisier).
    """

    def __init__(
        self,
        bounds: dict[str, dict[str, float]],
        ensemble_size: int = 1,
    ) -> None:
        if not bounds:
            raise ValueError("UniformParameterSampler: `bounds` must be non-empty.")
        for name, spec in bounds.items():
            if "min" not in spec or "max" not in spec:
                raise ValueError(
                    f"UniformParameterSampler: bounds[{name!r}] needs `min` and `max`."
                )
            if float(spec["min"]) > float(spec["max"]):
                raise ValueError(
                    f"UniformParameterSampler: bounds[{name!r}] has min > max."
                )
        self.bounds = bounds
        self.ensemble_size = ensemble_size
        self.param_names = list(bounds.keys())

    def sample_prior(
        self,
        time_coords: jnp.ndarray,
        rng_key: jax.Array,
    ) -> xarray.Dataset:
        time_coords = jnp.asarray(time_coords)
        n_t = int(time_coords.shape[0])

        data_vars: dict = {}
        for name in self.param_names:
            spec = self.bounds[name]
            rng_key, subkey = jax.random.split(rng_key)
            arr = jax.random.uniform(
                subkey,
                shape=(n_t, self.ensemble_size),
                minval=float(spec["min"]),
                maxval=float(spec["max"]),
            )
            data_vars[name] = (("time", "ensemble"), arr)

        return xarray.Dataset(
            data_vars=data_vars,
            coords={
                "time": np.asarray(time_coords),
                "ensemble": np.arange(self.ensemble_size),
            },
        )


def _resolve_external_hyperparam(
    spec: float | int | dict,
    *,
    ensemble_size: int,
    rng_key: jax.Array,
    label: str,
) -> tuple[jax.Array, jnp.ndarray]:
    """Resolve one external hyperparameter to a per-member `(ensemble,)` array.

    `spec` is either a scalar (fixed for every simulation) or a
    `{min, max}` dict (drawn uniformly per simulation). Returns the
    updated `rng_key` plus the resolved values.
    """
    if isinstance(spec, (int, float)):
        return rng_key, jnp.full((ensemble_size,), float(spec))
    if isinstance(spec, dict):
        if "min" not in spec or "max" not in spec:
            raise ValueError(
                f"{label}: dict spec must contain both `min` and `max` keys."
            )
        lo = float(spec["min"])
        hi = float(spec["max"])
        if lo > hi:
            raise ValueError(f"{label}: min ({lo}) > max ({hi}).")
        rng_key, subkey = jax.random.split(rng_key)
        return rng_key, jax.random.uniform(
            subkey, shape=(ensemble_size,), minval=lo, maxval=hi
        )
    raise TypeError(
        f"{label}: must be a number (fixed) or a dict with `min`/`max` "
        f"(uniform); got {type(spec).__name__}."
    )


class UniformExternalAR2Sampler:
    """Per-sim external hyperparameters + AR(2)-relaxation time series.

    Config layout:

      external:
        <param_name>:
          mean: <scalar OR {min, max}>
          std:  <scalar OR {min, max}>
      time_series:
        correlation_length: <float>

    For each simulation member and each parameter the sampler:
      1. Draws `mean_e` from the param's `mean` spec — fixed (scalar) or
         `Uniform(min, max)` (dict). Same for `std_e`.
      2. Integrates a critically-damped AR(2) anomaly `z(t)` with the
         configured correlation length (unit-variance, smooth).
      3. Returns `x(t, e) = mean_e + std_e · z(t, e)`.

    Each hyperparameter (e.g. `inflow_mean`, `inflow_std`,
    `velocity_mean`, `velocity_std`) is independently selectable from
    the config as either fixed or uniformly sampled.
    """

    def __init__(
        self,
        external: dict[str, dict[str, object]],
        time_series: dict[str, object],
        ensemble_size: int = 1,
    ) -> None:
        if not external:
            raise ValueError("UniformExternalAR2Sampler: `external` must be non-empty.")
        for name, spec in external.items():
            for key in ("mean", "std"):
                if key not in spec:
                    raise ValueError(
                        f"UniformExternalAR2Sampler: external[{name!r}] needs `{key}`."
                    )

        if "correlation_length" not in time_series:
            raise ValueError(
                "UniformExternalAR2Sampler: time_series must define `correlation_length`."
            )

        self.external = external
        self.time_series = time_series
        self.ensemble_size = ensemble_size
        self.param_names = list(external.keys())
        self.correlation_length = float(time_series["correlation_length"])

    def sample_prior(
        self,
        time_coords: jnp.ndarray,
        rng_key: jax.Array,
    ) -> xarray.Dataset:
        time_coords = jnp.asarray(time_coords)

        # Per-member z(t) for each parameter. The AR(2) model needs the
        # window's `time_coords` at construction, so build it here. Zero-mean,
        # unit-std `Normal` envelopes make `sample` return the bare anomaly
        # `z(t)` (x_ext(t) + Σ_ext(t)·z with x_ext=0, Σ_ext=1). The threaded
        # `rng_key` seeds the model's internal PRNG for reproducibility.
        rng_key, ar2_key = jax.random.split(rng_key)
        seed = int(jax.random.randint(ar2_key, (), 0, 2_000_000_000))
        ar2 = AR2RelaxationModel(
            external_parameters={
                n: Normal(mean=0.0, std=1.0) for n in self.param_names
            },
            time_coords=time_coords,
            correlation_length=self.correlation_length,
            seed=seed,
        )
        z_ds = ar2.sample(self.ensemble_size)

        data_vars: dict = {}
        for name in self.param_names:
            spec = self.external[name]
            rng_key, mean_e = _resolve_external_hyperparam(
                spec["mean"],
                ensemble_size=self.ensemble_size,
                rng_key=rng_key,
                label=f"external[{name!r}].mean",
            )
            rng_key, std_e = _resolve_external_hyperparam(
                spec["std"],
                ensemble_size=self.ensemble_size,
                rng_key=rng_key,
                label=f"external[{name!r}].std",
            )
            z_traj = jnp.asarray(z_ds[name].transpose("time", "ensemble").values)
            arr = mean_e[None, :] + std_e[None, :] * z_traj

            # AR(2) anomalies are unbounded; clip the final trajectory so
            # solver-unsafe values (e.g. uDALES SIGILL at sub-physical
            # ubulk) never reach the forward model. Bounds come from an
            # explicit `clip: {min?, max?}` block or, if absent, fall
            # back to the uniform mean's `{min, max}` (the user's stated
            # parameter range).
            clip_spec = spec.get("clip")
            if clip_spec is None and isinstance(spec["mean"], dict):
                clip_spec = spec["mean"]
            if isinstance(clip_spec, dict):
                if "min" in clip_spec:
                    arr = jnp.maximum(arr, float(clip_spec["min"]))
                if "max" in clip_spec:
                    arr = jnp.minimum(arr, float(clip_spec["max"]))

            data_vars[name] = (("time", "ensemble"), arr)

        return xarray.Dataset(
            data_vars=data_vars,
            coords={
                "time": np.asarray(time_coords),
                "ensemble": np.arange(self.ensemble_size),
            },
        )
