"""Time-varying parameter prior + extrapolation models.

The single supported method, :class:`AR2RelaxationModel`, is a critically
damped AR(2) prior with relaxation toward the external prior between
windows.  It exposes :meth:`sample` (cold-start draw for the initial
assimilation window) and :meth:`extrapolate` (next window's prior given the
previous posterior).

Like :class:`pyurbanair.static_parameters.ParameterSampler`, every constructor
argument is passed at build time and :meth:`sample` takes only
``ensemble_size``, so a model is built declaratively with
``hydra.utils.instantiate(cfg.<group>)`` and drawn with
``model.sample(ensemble_size)``.
"""

from .ar2_relaxation import AR2RelaxationModel
from .base import ParameterTimeSeries

__all__ = [
    "AR2RelaxationModel",
    "ParameterTimeSeries",
]
