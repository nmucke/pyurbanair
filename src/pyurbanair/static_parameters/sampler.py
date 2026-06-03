"""Static (non-time-varying) parameter sampler.

:class:`ParameterSampler` mirrors the
:class:`pyurbanair.dynamic_parameters.ParameterTimeSeries` interface — all
configuration is passed at construction time and :meth:`sample` takes a
single ``ensemble_size`` — but it produces *scalar* per-member parameters (an
``ensemble`` dim only, no ``time`` dim) rather than time series.

A run draws parameters with the same two lines whether it wants random priors
or fixed truth values::

    params_sampler = hydra.utils.instantiate(cfg.params)
    params = params_sampler.sample(ensemble_size)

Each parameter is a :class:`~pyurbanair.static_parameters.distributions.Distribution`
(a random prior or a :class:`Constant`), so the same class covers both the
"sample an ensemble from a prior" and "use these fixed values" cases.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import xarray

from .distributions import Distribution


class ParameterSampler:
    """Draw an ensemble of scalar parameters from per-parameter distributions.

    Args:
        parameters: Mapping ``name -> Distribution``. Each distribution may be
            a random prior (e.g. :class:`Normal`) or a :class:`Constant`.
        seed: Seed for the sampler's internal PRNG.
    """

    def __init__(
        self,
        parameters: dict[str, Distribution],
        seed: int = 0,
    ) -> None:
        self.parameters = parameters
        self.param_names = list(parameters.keys())
        self.rng_key = jax.random.PRNGKey(seed)

    def sample(self, ensemble_size: int) -> xarray.Dataset:
        """Draw an ``ensemble_size``-member ensemble of scalar parameters."""
        key = self.rng_key
        data_vars: dict = {}
        for name, dist in self.parameters.items():
            key, subkey = jax.random.split(key)
            data_vars[name] = ("ensemble", dist.sample(subkey, ensemble_size))
        return xarray.Dataset(
            data_vars=data_vars,
            coords={"ensemble": jnp.arange(ensemble_size)},
        )
