"""Per-parameter value specifications for :class:`ParameterSampler`.

Each parameter in a :class:`~pyurbanair.static_parameters.ParameterSampler` is a
:class:`Distribution`: either a random prior (:class:`Normal`,
:class:`LogNormal`, :class:`Uniform`) or a fixed :class:`Constant`.  They share a single
``sample(rng_key, ensemble_size)`` method returning a 1-D array of
``ensemble_size`` values, so the sampler treats random priors and constants
uniformly. The same objects double as the per-parameter external prior for the
time-varying :class:`pyurbanair.dynamic_parameters.ParameterTimeSeries` models.

All three are built declaratively with ``hydra.utils.instantiate`` from a
``_target_`` block, e.g.::

    inflow_angle:
      _target_: pyurbanair.static_parameters.Normal
      mean: 0.0
      std: 10.0
"""

from __future__ import annotations

import abc
from typing import Optional

import jax
import jax.numpy as jnp


class Distribution(abc.ABC):
    """A per-parameter value generator over an ensemble."""

    @abc.abstractmethod
    def sample(self, rng_key: jax.Array, ensemble_size: int) -> jnp.ndarray:
        """Return a ``(ensemble_size,)`` array of parameter values."""


class Normal(Distribution):
    """Gaussian prior with optional ``min`` / ``max`` clamps."""

    def __init__(
        self,
        mean: float,
        std: float,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> None:
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max

    def sample(self, rng_key: jax.Array, ensemble_size: int) -> jnp.ndarray:
        values = jax.random.normal(rng_key, (ensemble_size,)) * self.std + self.mean
        if self.min is not None:
            values = jnp.maximum(values, self.min)
        if self.max is not None:
            values = jnp.minimum(values, self.max)
        return values


class LogNormal(Distribution):
    """Log-normal prior for a strictly positive parameter, estimated in log space.

    ``log(value) ~ Normal(log(median), log_std)``: ``median`` is in the
    parameter's units, ``log_std`` is dimensionless (0.7 is about a factor 2 per
    standard deviation). Optional ``min`` / ``max`` truncate the draws to
    ``(min, max)``. The data assimilation updates ``log(value)`` for every
    parameter with this prior (``log_space_parameter_names``), so the forward
    models always receive a positive value; with both ``min`` and ``max`` it
    updates a bounded transform instead, so every update also stays inside
    ``(min, max)``.
    """

    def __init__(
        self,
        median: float,
        log_std: float,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> None:
        if not median > 0.0:
            raise ValueError(f"LogNormal median must be > 0, got {median}.")
        if not log_std >= 0.0:
            raise ValueError(f"LogNormal log_std must be >= 0, got {log_std}.")
        if min is not None and not 0.0 < min < median:
            raise ValueError(f"LogNormal needs 0 < min < median, got min {min}.")
        if max is not None and not median < max:
            raise ValueError(f"LogNormal needs median < max, got max {max}.")
        self.median = median
        self.log_std = log_std
        self.min = min
        self.max = max

    @property
    def bounds(self) -> Optional[tuple[float, float]]:
        """``(min, max)`` when both are set: the DA keeps updates inside them."""
        if self.min is None or self.max is None:
            return None
        return (float(self.min), float(self.max))

    def sample(self, rng_key: jax.Array, ensemble_size: int) -> jnp.ndarray:
        if (self.min is None and self.max is None) or self.log_std == 0.0:
            log_values = jax.random.normal(rng_key, (ensemble_size,)) * self.log_std
        else:
            # Truncated in log space, in units of log_std around log(median).
            lower, upper = (
                -jnp.inf if bound is None else jnp.log(bound / self.median)
                for bound in (self.min, self.max)
            )
            log_values = self.log_std * jax.random.truncated_normal(
                rng_key, lower / self.log_std, upper / self.log_std, (ensemble_size,)
            )
        return jnp.exp(log_values + jnp.log(self.median))


def log_space_parameter_names(
    sampler: object,
) -> dict[str, Optional[tuple[float, float]]]:
    """A sampler's static parameters whose prior is :class:`LogNormal`.

    Reads a :class:`~pyurbanair.static_parameters.ParameterSampler`'s
    ``parameters`` or a time-varying model's ``static_parameters`` and maps
    each name to its prior's ``bounds``: the data assimilation analyses these
    as ``log(value)``, or bounded to ``(min, max)`` when both are set.
    """
    priors = {
        **getattr(sampler, "parameters", {}),
        **(getattr(sampler, "static_parameters", None) or {}),
    }
    return {
        name: dist.bounds
        for name, dist in priors.items()
        if isinstance(dist, LogNormal)
    }


class Uniform(Distribution):
    """Uniform prior on ``[low, high)``."""

    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high

    def sample(self, rng_key: jax.Array, ensemble_size: int) -> jnp.ndarray:
        return jax.random.uniform(
            rng_key, (ensemble_size,), minval=self.low, maxval=self.high
        )


class Constant(Distribution):
    """A fixed value broadcast across the whole ensemble."""

    def __init__(self, value: float) -> None:
        self.value = value

    def sample(self, rng_key: jax.Array, ensemble_size: int) -> jnp.ndarray:
        return jnp.full((ensemble_size,), self.value)
