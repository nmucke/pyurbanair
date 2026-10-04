"""Parameter evolution models for sequential filtering.

In a smoother, parameters are static unknowns of the window. In a filter, a
parameter that receives no forecast noise collapses to (near-)zero spread
after a few cycles and stops learning — so parameter-bearing filter modes
need explicit spread maintenance between cycles. The strategies here are
the parameters' "forecast model": the filter applies one to the estimated
parameters immediately before every forecast that follows an analysis, so the
saved posterior parameters stay pure analyses.

The interface takes the whole parameter ``xarray.Dataset`` (not a bare
array) so future models can be structured — e.g. evolving time-varying
inflow parameters with the prior's own AR model.
"""

from abc import ABC, abstractmethod
from typing import Mapping

import jax
import jax.numpy as jnp
import xarray


class ParameterEvolution(ABC):
    """Evolves the parameter ensemble between filter cycles."""

    @abstractmethod
    def evolve(self, params: xarray.Dataset, rng_key: jax.Array) -> xarray.Dataset:
        """Return the parameters to use for the next cycle's forecast.

        Args:
            params: Analyzed parameter Dataset (scalar ``(ensemble,)``
                variables in Phase 1).
            rng_key: PRNG key for the evolution noise (consumed as-is; the
                caller splits).

        Returns:
            Evolved parameter Dataset with the same variables/coords.
        """
        raise NotImplementedError


class RandomWalkEvolution(ParameterEvolution):
    """Additive Gaussian random walk: ``theta_{k+1} = theta_k + xi``.

    The standard augmented-state approach to parameter estimation in a
    filter. ``std`` maps ``{param_name: std}``, each in its parameter's own
    units; names absent from the mapping are left unchanged (no noise).
    """

    def __init__(self, std: Mapping[str, float]) -> None:
        if not isinstance(std, Mapping):
            raise ValueError(
                f"RandomWalkEvolution needs a per-parameter std mapping "
                f"{{param_name: std}}, got {std!r}: parameters have different "
                "units, so one scalar std cannot fit them all."
            )
        bad = {k: v for k, v in std.items() if v < 0.0}
        if bad:
            raise ValueError(f"Random-walk stds must be >= 0, got {bad}.")
        self.std = std

    def _std_for(self, name: str) -> float:
        return float(self.std.get(name, 0.0))

    def evolve(self, params: xarray.Dataset, rng_key: jax.Array) -> xarray.Dataset:
        data_vars: dict = {}
        # Iterate in sorted order so the key -> noise mapping is independent
        # of the Dataset's insertion order.
        names = sorted(params.data_vars)
        subkeys = jax.random.split(rng_key, len(names))
        for name, subkey in zip(names, subkeys):
            std = self._std_for(name)
            values = jnp.asarray(params[name].values)
            if std > 0.0:
                values = values + std * jax.random.normal(subkey, values.shape)
            data_vars[name] = (params[name].dims, values)
        return xarray.Dataset(data_vars=data_vars, coords=params.coords)
