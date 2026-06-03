"""Static parameter sampling.

A :class:`ParameterSampler` draws scalar per-member parameters from a mapping
of per-parameter :class:`Distribution` objects (random priors or constants).
It shares the ``sample(ensemble_size)`` interface with the time-varying
:class:`pyurbanair.dynamic_parameters.ParameterTimeSeries` models, so a run
samples parameters the same way for static and time-varying cases:

    params_sampler = hydra.utils.instantiate(cfg.params)
    params = params_sampler.sample(ensemble_size)
"""

from .distributions import Constant, Distribution, Normal, Uniform
from .sampler import ParameterSampler

__all__ = [
    "Constant",
    "Distribution",
    "Normal",
    "ParameterSampler",
    "Uniform",
]
