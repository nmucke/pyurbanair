"""Per-variable, mask-aware standardization (``docs/neural_surrogate_plan.md`` §6.3).

Fit per-variable mean/std over the **training split only**, store in the
checkpoint manifest (``normalization.json``), and apply at inference. Never
recompute on assimilation data. Solid cells are excluded from the statistics
(mask-aware), since they carry no flow information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

# Channel axis in the [..., C, Z, Y, X] layout.
_CHANNEL_AXIS = -4


@dataclass(frozen=True)
class Normalization:
    """Per-channel mean/std for the state variables."""

    var_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (len(self.var_names) == len(self.mean) == len(self.std)):
            raise ValueError("var_names, mean, std must have equal length.")

    def _broadcast(self, values: tuple[float, ...]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return arr.reshape(arr.shape[0], 1, 1, 1)

    def apply(self, fields: np.ndarray) -> np.ndarray:
        """Standardize ``[..., C, Z, Y, X]`` fields."""
        mean = self._broadcast(self.mean)
        std = self._broadcast(self.std)
        return (np.asarray(fields) - mean) / std

    def invert(self, fields: np.ndarray) -> np.ndarray:
        """Undo :meth:`apply` (denormalize predictions back to physical units)."""
        mean = self._broadcast(self.mean)
        std = self._broadcast(self.std)
        return np.asarray(fields) * std + mean

    def to_dict(self) -> dict:
        return {
            "var_names": list(self.var_names),
            "mean": list(self.mean),
            "std": list(self.std),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Normalization":
        return cls(
            var_names=tuple(data["var_names"]),
            mean=tuple(float(v) for v in data["mean"]),
            std=tuple(float(v) for v in data["std"]),
        )


def fit_normalization(
    trajectories: Iterable[np.ndarray],
    var_names: Sequence[str],
    *,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> Normalization:
    """Fit per-channel mean/std from an iterable of ``[T, C, Z, Y, X]`` arrays.

    Uses a streaming (sum / sum-of-squares) accumulation so the full corpus
    never has to be materialized. If ``mask`` (``[Z, Y, X]``, 1 = solid) is
    given, solid cells are excluded from the statistics (mask-aware).

    Pass only the **training-split** trajectories (§6.3).
    """
    n_c = len(var_names)
    count = np.zeros(n_c, dtype=np.float64)
    total = np.zeros(n_c, dtype=np.float64)
    total_sq = np.zeros(n_c, dtype=np.float64)

    fluid = None if mask is None else (np.asarray(mask) <= 0.5)

    for traj in trajectories:
        traj = np.asarray(traj, dtype=np.float64)  # [T, C, Z, Y, X]
        if traj.shape[1] != n_c:
            raise ValueError(
                f"Trajectory has {traj.shape[1]} channels, expected {n_c}."
            )
        for c in range(n_c):
            vals = traj[:, c]  # [T, Z, Y, X]
            if fluid is not None:
                vals = vals[:, fluid]
            vals = vals.reshape(-1)
            count[c] += vals.size
            total[c] += vals.sum()
            total_sq[c] += np.square(vals).sum()

    if np.any(count == 0):
        raise ValueError("No samples to fit normalization (empty corpus/split).")

    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 0.0)
    std = np.sqrt(var) + eps
    return Normalization(
        var_names=tuple(var_names),
        mean=tuple(float(m) for m in mean),
        std=tuple(float(s) for s in std),
    )
