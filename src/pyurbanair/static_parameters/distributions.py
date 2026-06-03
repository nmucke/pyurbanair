"""Per-parameter value specifications for :class:`ParameterSampler`.

Each parameter in a :class:`~pyurbanair.static_parameters.ParameterSampler` is a
:class:`Distribution`: either a random prior (:class:`Normal`,
:class:`Uniform`) or a fixed :class:`Constant`.  They share a single
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
