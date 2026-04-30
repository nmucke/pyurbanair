"""Time-varying parameter prior + extrapolation models.

Each method is a class that exposes :meth:`sample_prior` (cold-start
draw for the initial assimilation window) and :meth:`extrapolate` (next
window's prior given the previous posterior), so the rollout loop can
treat them interchangeably.
"""

from .ar1 import AR1Model
from .ar2_relaxation import AR2RelaxationModel
from .base import ParameterTimeSeries
from .gp_linear_trend import GPLinearTrendModel
from .ornstein_uhlenbeck import OrnsteinUhlenbeckModel

_REGISTRY: dict[str, type[ParameterTimeSeries]] = {
    "gp_linear_trend": GPLinearTrendModel,
    "ar1": AR1Model,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckModel,
    "ar2_relaxation": AR2RelaxationModel,
}


def build_parameter_time_series(
    method: str,
    external_priors: dict[str, dict[str, float]],
    ensemble_size: int,
    method_kwargs: dict | None = None,
) -> ParameterTimeSeries:
    """Construct a :class:`ParameterTimeSeries` for ``method``.

    Args:
        method: One of the keys in :data:`_REGISTRY`.
        external_priors: Per-parameter ``{"mean", "std", optional "min",
            optional "max"}``.
        ensemble_size: Number of ensemble members.
        method_kwargs: Extra method-specific hyperparameters forwarded
            to the class constructor.
    """
    try:
        cls = _REGISTRY[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown parameter time-series method: {method!r}. "
            f"Expected one of {sorted(_REGISTRY)}."
        ) from exc
    return cls(
        external_priors=external_priors,
        ensemble_size=ensemble_size,
        **(method_kwargs or {}),
    )


__all__ = [
    "AR1Model",
    "AR2RelaxationModel",
    "GPLinearTrendModel",
    "OrnsteinUhlenbeckModel",
    "ParameterTimeSeries",
    "build_parameter_time_series",
]
