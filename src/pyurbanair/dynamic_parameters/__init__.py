"""Time-varying parameter prior + extrapolation models.

:class:`AR2RelaxationModel` is a critically damped stochastic AR(2) prior with
relaxation between windows. :class:`HarmonicParameterModel` instead produces
deterministic sine/cosine forcing profiles for controlled forward comparisons.

Like :class:`pyurbanair.static_parameters.ParameterSampler`, every constructor
argument is passed at build time and :meth:`sample` takes only
``ensemble_size``, so a model is built declaratively with
``hydra.utils.instantiate(cfg.<group>)`` and drawn with
``model.sample(ensemble_size)``.
"""

from .ar2_relaxation import AR2RelaxationModel, build_knot_times
from .base import ParameterTimeSeries
from .harmonic import HarmonicParameterModel

__all__ = [
    "AR2RelaxationModel",
    "HarmonicParameterModel",
    "ParameterTimeSeries",
    "build_knot_times",
]
