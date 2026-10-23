"""Online adaptive observation error from innovation statistics (Desroziers).

The configured observation error (``ObservationErrorSpec``) usually holds what
is known a priori -- instrument and realisation noise. Against real data, or a
model other than the truth's, the innovations also carry a structural error the
model cannot fit, and an update that ignores it is overconfident. This module
estimates that error online, from innovations only (no truth, no second model):
Desroziers et al. (2005) show ``E[d_a d_f] = R`` for a consistent analysis,
with ``d_f = y - mean prior prediction`` and ``d_a = y - mean posterior
prediction``.

Before each window ``k > 0`` the estimate pools ``r_i = d_a,i d_f,i / R_conf,i``
of the previous windows per group of observations (one group per sensor height
or per configured height band) into an inflation ``lambda_g**2 = mean r_i`` of
the configured variance ``R_conf``; window ``k`` then assimilates with
``lambda_g**2 R_conf``. Re-estimating from windows that used the previous
estimate is the usual fixed-point iteration of the Desroziers diagnostic (Li et
al. 2009). The diagnostic is only unbiased for an analysis that used the right
``R``, which window 0 (assimilated with ``initial_inflation``) did not: window 1
uses window 0's estimate, later windows pool from window 1 on.

The inflation multiplies the whole configured variance of a group, instrument
part included: it assumes the unknown error scales with ``R_conf`` within a
group. That is cleanest when the groups match or are finer than the
``representation_std`` height bands of the observation error and the
aggregation intervals have equal length (``R_conf`` then constant per group).

For few observations per group:

* ``min_inflation`` floors it (1 keeps the configured noise as a floor: an
  innovation sample can then never make the update more confident than the
  noise the configuration already knows of);
* ``max_inflation`` caps it (an unlucky sample cannot switch the observations
  off);
* ``memory`` keeps only the last windows (a sliding estimate).

Non-finite ratios (a diverged member's prediction) are dropped; a group with no
ratio to pool keeps its previous inflation (``initial_inflation`` at first).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def observation_heights(resolved: Any, observation_operator: Any) -> np.ndarray:
    """The sensor height of every entry of ``resolved.covariance_diag``.

    ``resolved`` is a ``ResolvedObservationError``: its vector is time-major
    with ``sensor_indices`` labelling one frame.
    """
    base = getattr(observation_operator, "observation_operator", observation_operator)
    heights = getattr(base, "obs_z", None)
    if heights is None:
        raise ValueError("Adaptive observation error needs sensor heights (obs_z).")
    frame = np.asarray(heights, dtype=float)[list(resolved.sensor_indices)]
    return np.tile(frame, len(resolved.times))


class AdaptiveObservationError:
    """Per-group Desroziers inflation of the configured observation variance.

    Args:
        height_bands: ``[{min_z, max_z}, ...]`` (half-open ``[min_z, max_z)``,
            not overlapping) pooling the observations per band; ``None`` pools
            per distinct sensor height. Every observation must fall in a band.
        memory: Number of most recent windows to pool; ``None`` pools all.
        initial_inflation: Std inflation of window 0.
        min_inflation, max_inflation: Bounds on the std inflation.
    """

    def __init__(
        self,
        height_bands: Sequence[Mapping[str, float]] | None = None,
        memory: int | None = None,
        initial_inflation: float = 1.0,
        min_inflation: float = 1.0,
        max_inflation: float = 10.0,
    ) -> None:
        if memory is not None and int(memory) < 1:
            raise ValueError("memory must be a positive window count or None.")
        if not 0 < min_inflation <= max_inflation or not np.isfinite(max_inflation):
            raise ValueError("Need 0 < min_inflation <= max_inflation < inf.")
        if not min_inflation <= initial_inflation <= max_inflation:
            raise ValueError("initial_inflation must lie in the inflation bounds.")
        self.bands = (
            None
            if height_bands is None
            else [(float(b["min_z"]), float(b["max_z"])) for b in height_bands]
        )
        ordered = sorted(self.bands or [])
        if any(lo >= hi for lo, hi in ordered) or any(
            following[0] < upper for (_, upper), following in zip(ordered, ordered[1:])
        ):
            raise ValueError(
                f"Adaptive height bands need min_z < max_z and must not overlap: "
                f"{self.bands}."
            )
        self.memory = None if memory is None else int(memory)
        self.initial_inflation = float(initial_inflation)
        self.min_inflation = float(min_inflation)
        self.max_inflation = float(max_inflation)
        # One entry per assimilated window: (group name, d_a d_f / R_conf) per
        # finite observation; names, not ids, so a changed sensor set cannot
        # shift them.
        self._records: list[tuple[np.ndarray, np.ndarray]] = []
        self.history: list[dict[str, Any]] = []

    def groups(self, heights: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Group id per observation and the group names."""
        heights = np.asarray(heights, dtype=float)
        if self.bands is None:
            levels, ids = np.unique(heights, return_inverse=True)
            return ids, [f"z={z:g}" for z in levels]
        ids = np.full(heights.shape, -1)
        for g, (lower, upper) in enumerate(self.bands):
            ids[(heights >= lower) & (heights < upper)] = g
        if np.any(ids < 0):
            raise ValueError(
                f"Sensor heights {sorted(set(heights[ids < 0]))} lie in no "
                "adaptive height band."
            )
        return ids, [f"z=[{lo:g},{hi:g})" for lo, hi in self.bands]

    def inflation(self, names: Sequence[str]) -> np.ndarray:
        """The std inflation per group for the next window."""
        # Window 0 used initial_inflation, not the estimate: leave it out once
        # a later window is there.
        records = self._records[1:] if len(self._records) > 1 else self._records
        if self.memory is not None:
            records = records[-self.memory :]
        previous = (
            dict(zip(self.history[-1]["groups"], self.history[-1]["inflation"]))
            if self.history
            else {}
        )
        factor = np.array([previous.get(n, self.initial_inflation) for n in names])
        if records:
            labels = np.concatenate([r[0] for r in records])
            ratios = np.concatenate([r[1] for r in records])
            for g, name in enumerate(names):
                pooled = ratios[labels == name]
                if pooled.size:
                    factor[g] = np.sqrt(max(float(pooled.mean()), 0.0))
        return np.asarray(np.clip(factor, self.min_inflation, self.max_inflation))

    def variance(
        self, configured_variance: np.ndarray, heights: np.ndarray
    ) -> np.ndarray:
        """The variance to assimilate the next window with.

        Also appends the window's entry to ``history`` (groups, inflation and
        the std used, mean over each group; ``None`` for a group without
        observations) and logs it.
        """
        configured_variance = np.asarray(configured_variance, dtype=float)
        ids, names = self.groups(heights)
        factor = self.inflation(names)
        variance = configured_variance * factor[ids] ** 2
        window = len(self._records)
        std_used = [
            float(np.sqrt(np.mean(variance[ids == g]))) if np.any(ids == g) else None
            for g in range(len(names))
        ]
        self.history.append(
            {
                "window": window,
                "groups": names,
                "inflation": [float(f) for f in factor],
                "std_used": std_used,
            }
        )
        logger.info(
            "Adaptive observation error, window %d: %s",
            window,
            ", ".join(
                f"{n} x{f:.2f} (std {s if s is None else f'{s:.3f}'})"
                for n, f, s in zip(names, factor, std_used)
            ),
        )
        return variance

    def record(
        self,
        observations: np.ndarray,
        forecast: np.ndarray,
        analysis: np.ndarray,
        configured_variance: np.ndarray,
        heights: np.ndarray,
    ) -> None:
        """Add one assimilated window's innovations to the statistics.

        ``forecast`` / ``analysis`` are the prior and posterior predicted
        observations, ``(N_d, N_e)``; ``configured_variance`` is the window's
        configured (not inflated) variance. Non-finite ratios are dropped.
        """
        y = np.asarray(observations, dtype=float).ravel()
        d_f = y - np.asarray(forecast, dtype=float).mean(axis=1)
        d_a = y - np.asarray(analysis, dtype=float).mean(axis=1)
        ratios = d_a * d_f / np.asarray(configured_variance, dtype=float)
        ids, names = self.groups(heights)
        finite = np.isfinite(ratios)
        if not finite.all():
            logger.warning(
                "Adaptive observation error, window %d: dropped %d of %d "
                "non-finite innovation ratios.",
                len(self._records),
                int((~finite).sum()),
                ratios.size,
            )
        ids, ratios = ids[finite], ratios[finite]
        self._records.append((np.asarray(names)[ids], ratios))
        # The window's own estimate, before pooling and bounds; None for a
        # group without a positive mean ratio.
        means = [
            ratios[ids == g].mean() if np.any(ids == g) else 0.0
            for g in range(len(names))
        ]
        estimate = [float(np.sqrt(m)) if m > 0 else None for m in means]
        if self.history and self.history[-1]["window"] == len(self._records) - 1:
            self.history[-1]["window_estimate"] = estimate
        logger.info(
            "Desroziers inflation estimate of window %d: %s",
            len(self._records) - 1,
            ", ".join(
                f"{n} x{e if e is None else f'{e:.2f}'}"
                for n, e in zip(names, estimate)
            ),
        )
