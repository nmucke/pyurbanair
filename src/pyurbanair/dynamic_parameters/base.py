"""Base class and shared utilities for parameter time-series models.

A :class:`ParameterTimeSeries` ties together the *prior* sampling and
*between-window extrapolation* for a single generative model of
time-varying parameters.  Subclasses implement one specific method.

The class mirrors the static :class:`pyurbanair.static_parameters.ParameterSampler`
interface: every constructor argument (the external parameters, the time grid,
the RNG seed, and any method-specific hyperparameters) is passed at
construction time, and :meth:`sample` takes a single ``ensemble_size``
argument.  This lets a model be built declaratively with
``hydra.utils.instantiate(cfg.<group>)`` and then drawn from with
``model.sample(ensemble_size)``.

The external prior for each parameter is given as a
:class:`pyurbanair.static_parameters.Distribution` (the same objects the static
sampler uses): a :class:`~pyurbanair.static_parameters.Normal` supplies the
relaxation target ``x_ext`` (its ``mean``), the external spread ``Σ_ext`` (its
``std``), and optional ``min`` / ``max`` clips; a
:class:`~pyurbanair.static_parameters.Constant` pins ``x_ext`` with zero spread.
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
        external_parameters: Mapping ``name -> Distribution`` defining the
            external prior (the paper's ``x_ext`` and ``Σ_ext``) for each
            parameter. A ``Normal`` contributes its ``mean`` / ``std`` (each
            may be a scalar or a per-window control-point sequence) and optional
            ``min`` / ``max`` clips; a ``Constant`` contributes its ``value`` as
            ``x_ext`` with zero spread.
        time_coords: The time grid for the initial-window prior draw.
        seed: Seed for the model's internal PRNG.
    """

    def __init__(
        self,
        external_parameters: dict[str, object],
        time_coords: jnp.ndarray,
        seed: int = 0,
    ) -> None:
        self.external_parameters = external_parameters
        self.param_names = list(external_parameters.keys())
        self.time_coords = jnp.asarray(time_coords)
        self.rng_key = jax.random.PRNGKey(seed)

    @abc.abstractmethod
    def sample(self, ensemble_size: int) -> xarray.Dataset:
        """Draw the initial-window prior ensemble of ``ensemble_size`` members."""

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

    def _moments(self, name: str) -> tuple[object, object, Optional[float], Optional[float]]:
        """Extract ``(mean, std, min, max)`` from an external-prior Distribution.

        A ``Normal`` exposes all four attributes directly; a ``Constant`` is
        treated as a fixed location (``value``) with zero spread and no clips.
        ``mean`` / ``std`` are returned unconverted so a control-point sequence
        survives for :meth:`_resolve_profile`.
        """
        dist = self.external_parameters[name]
        if hasattr(dist, "mean"):  # Normal-style location/scale prior
            return dist.mean, dist.std, dist.min, dist.max
        if hasattr(dist, "value"):  # Constant: fixed x_ext, no spread
            return dist.value, 0.0, None, None
        raise TypeError(
            f"external parameter {name!r} ({dist!r}) is not usable as a "
            "dynamic external prior: it needs a mean/std (Normal) or a "
            "constant value (Constant)."
        )

    def _resolve_profile(
        self,
        value: object,
        time_coords: jnp.ndarray,
    ) -> jnp.ndarray:
        """Resolve an external-prior value to a per-time profile ``(n_t,)``.

        A scalar broadcasts to a constant profile (the historical behavior);
        a sequence is treated as control points spaced evenly over the window
        and linearly interpolated onto ``time_coords`` — letting the external
        mean/std *vary over the window* (``x_ext(t)`` / ``Σ_ext(t)``). The
        models build a unit-variance anomaly and apply this envelope, so
        ``mean(t) + std(t)·z(t)`` stays mathematically sound even though the
        underlying AR process is stationary.
        """
        arr = jnp.asarray(value, dtype=float)
        time_coords = jnp.asarray(time_coords)
        if arr.ndim == 0:
            return jnp.broadcast_to(arr, (time_coords.shape[0],))
        ctrl_t = jnp.linspace(time_coords[0], time_coords[-1], arr.shape[0])
        return jnp.interp(time_coords, ctrl_t, arr)

    def _ext_profile(
        self,
        name: str,
        time_coords: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(mean(t), std(t))`` profiles for an external parameter."""
        mean, std, _, _ = self._moments(name)
        return (
            self._resolve_profile(mean, time_coords),
            self._resolve_profile(std, time_coords),
        )

    def _ext_scalar(self, name: str, key: str) -> float:
        """Representative scalar for a (possibly time-varying) external value.

        Between-window extrapolation models a single relaxation target /
        stationary spread, so a time profile is reduced to its mean.
        """
        mean, std, _, _ = self._moments(name)
        value = mean if key == "mean" else std
        arr = jnp.asarray(value, dtype=float)
        return float(arr.mean()) if arr.ndim else float(arr)

    def _apply_clips(self, name: str, values: jnp.ndarray) -> jnp.ndarray:
        _, _, lo, hi = self._moments(name)
        if lo is not None:
            values = jnp.maximum(values, lo)
        if hi is not None:
            values = jnp.minimum(values, hi)
        return values

    def _build_dataset(
        self,
        arrays: dict[str, jnp.ndarray],
        time_coords: jnp.ndarray,
        ensemble_size: int,
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
                "ensemble": jnp.arange(ensemble_size),
            },
        )
