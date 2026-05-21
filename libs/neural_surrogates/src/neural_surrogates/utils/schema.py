"""The source-solver contract carried by every checkpoint.

A neural-surrogate checkpoint records the *source solver contract* —
``source_solver_name`` + ``param_schema`` + ``state_var_names`` — separately
from the pyurbanair backend name ``neural_surrogate``
(``docs/neural_surrogate_plan.md`` §0, §7). Inference uses this schema to
decide which params are required and which state variables are emitted, so a
uDALES-trained checkpoint receives ``pressure_gradient_magnitude`` while a
pylbm-trained one does not — *without* keying off ``model.name``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Parameters that live on a circular domain and must be sin/cos encoded so the
# 359°->1° wrap doesn't sweep through the whole range (§1.5).
DEFAULT_ANGULAR_PARAMS: tuple[str, ...] = ("inflow_angle",)


@dataclass(frozen=True)
class ParamSchema:
    """Ordered conditioning-parameter contract for a checkpoint.

    Args:
        names: Ordered parameter variable names the checkpoint consumes
            (e.g. ``("inflow_angle", "velocity_magnitude")`` for pylbm, plus
            ``"pressure_gradient_magnitude"`` for uDALES).
        angular: Subset of ``names`` that are angles (degrees), sin/cos
            encoded into two conditioning channels each.
    """

    names: tuple[str, ...]
    angular: tuple[str, ...] = DEFAULT_ANGULAR_PARAMS

    def __post_init__(self) -> None:
        unknown = set(self.angular) - set(self.names)
        if unknown:
            raise ValueError(
                f"angular params {sorted(unknown)} are not in names {self.names}."
            )

    def is_angular(self, name: str) -> bool:
        return name in self.angular

    @property
    def conditioning_dim(self) -> int:
        """Width ``P`` of the per-step conditioning vector after encoding."""
        return sum(2 if self.is_angular(n) else 1 for n in self.names)

    def to_dict(self) -> dict:
        return {"names": list(self.names), "angular": list(self.angular)}

    @classmethod
    def from_dict(cls, data: dict) -> "ParamSchema":
        return cls(
            names=tuple(data["names"]),
            angular=tuple(data.get("angular", DEFAULT_ANGULAR_PARAMS)),
        )


@dataclass(frozen=True)
class ContractSchema:
    """Full source-solver contract baked into a checkpoint (``schema.json``)."""

    source_solver_name: str
    param_schema: ParamSchema
    state_var_names: tuple[str, ...]
    dtype: str = "float32"

    def to_dict(self) -> dict:
        return {
            "source_solver_name": self.source_solver_name,
            "param_schema": self.param_schema.to_dict(),
            "state_var_names": list(self.state_var_names),
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContractSchema":
        return cls(
            source_solver_name=data["source_solver_name"],
            param_schema=ParamSchema.from_dict(data["param_schema"]),
            state_var_names=tuple(data["state_var_names"]),
            dtype=data.get("dtype", "float32"),
        )


# Canonical per-solver contracts used by the data generator and tests. The
# checkpoint records the resolved schema; this is the source of truth when
# generating a corpus from a given solver.
_SOLVER_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "pylbm": ("inflow_angle", "velocity_magnitude"),
    "palm": ("inflow_angle", "velocity_magnitude"),
    "udales": (
        "inflow_angle",
        "velocity_magnitude",
        "pressure_gradient_magnitude",
    ),
}


def param_schema_for_solver(solver_name: str) -> ParamSchema:
    """Return the canonical ``ParamSchema`` for a source solver name.

    ``pressure_gradient_magnitude`` is uDALES-only (``docs/codebase_guide.md``
    §4); every solver carries ``inflow_angle`` + ``velocity_magnitude``.
    """
    if solver_name not in _SOLVER_PARAM_NAMES:
        raise ValueError(
            f"No canonical param schema for solver {solver_name!r}. "
            f"Known: {sorted(_SOLVER_PARAM_NAMES)}."
        )
    return ParamSchema(names=_SOLVER_PARAM_NAMES[solver_name])


def default_state_var_names(include_pressure: bool = False) -> tuple[str, ...]:
    """Default state variables ``(u, v, w[, pres])`` (``docs/codebase_guide.md`` §4)."""
    return ("u", "v", "w", "pres") if include_pressure else ("u", "v", "w")
