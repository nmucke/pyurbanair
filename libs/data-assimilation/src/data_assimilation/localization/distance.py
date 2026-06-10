"""Distance-based localization.

Implements the standard physical-distance localization (Vossepoel et al. 2025,
MWR-D-24-0269.1, sec. 3 and 3d) within the same local-analysis machinery as the
correlation strategy.  For each state grid point ``l`` and sensor ``j`` we use
the **physical Euclidean distance** ``dist(l, j) = ||x_l - x_j||`` between the
grid point and the sensor location — *not* any ensemble correlation:

* observations farther than ``localization_radius`` are excluded;
* the rest are tapered by an error-variance inflation that grows to
  ``max_inflation`` at the radius (the same taper the paper applies to the
  correlation distance — "It is straightforward to use the same tapering
  strategy in standard distance-based localization", sec. 3d).

Because it needs grid-point and sensor coordinates, this strategy only applies
to the state rows of a state-bearing smoother (``state_and_parameter`` /
``state_and_dynamic``); parameter rows have no spatial location and always get
the global update.
"""

from typing import Optional

import jax.numpy as jnp
from data_assimilation.localization.base import BaseLocalization, taper_inflation


class DistanceLocalization(BaseLocalization):
    """Physical-distance localization with error-variance tapering.

    Args:
        localization_radius: Truncation distance (the "localization radius") in
            the domain's length units.  Observations whose sensor lies farther
            than this from a state grid point are excluded from that grid
            point's update.  Domain-dependent — tune to the flow's correlation
            length scale.
        tapering_beta: ``beta in (0, 1)``; the fraction of the radius within
            which observations are *not* tapered (Eq. 9).  Defaults to ``0.5``.
        max_inflation: Maximum error-variance inflation ``E_max`` reached at the
            radius (Eq. 10).  ``E_max`` multiplies the observation-error
            perturbation (std), so the error *variance* there is scaled by
            ``E_max ** 2``.  Defaults to ``4.0`` — the value the paper found
            best for distance-based localization.
        block_grouping: Local-analysis granularity.  ``False`` (default) updates
            each augmented row on its own.  ``True`` requests the paper's "grid
            block" analysis (sec. 3b): co-located state rows (``u``/``v``/``w``
            at one cell) are updated *jointly* with a single observation
            selection and transition.  The smoother builds the block ids.
        horizontal_only: When ``True``, use only the horizontal ``(x, y)``
            separation, ignoring the vertical distance — useful when the
            relevant correlation scale is horizontal (mesoscale-like).  Defaults
            to ``False`` (full 3-D Euclidean distance).
    """

    requires_coordinates: bool = True

    def __init__(
        self,
        localization_radius: float,
        tapering_beta: float = 0.5,
        max_inflation: float = 4.0,
        block_grouping: bool = False,
        horizontal_only: bool = False,
    ) -> None:
        if localization_radius <= 0.0:
            raise ValueError("localization_radius must be > 0.")
        if not (0.0 < tapering_beta < 1.0):
            raise ValueError("tapering_beta must lie in (0, 1).")
        if max_inflation < 1.0:
            raise ValueError("max_inflation must be >= 1.")

        self.localization_radius = float(localization_radius)
        self.tapering_beta = tapering_beta
        self.max_inflation = max_inflation
        self.block_grouping = block_grouping
        self.horizontal_only = horizontal_only

    def inflation_factors(
        self,
        aug_dev: jnp.ndarray,
        pred_obs_dev: jnp.ndarray,
        row_coords: Optional[jnp.ndarray] = None,
        obs_coords: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # ``aug_dev`` / ``pred_obs_dev`` are unused: distance-based localization
        # depends only on the physical coordinates.
        if row_coords is None or obs_coords is None:
            raise ValueError(
                "DistanceLocalization requires row_coords and obs_coords. "
                "It only applies to a state-bearing smoother "
                "(esmda/smoother=state_and_parameter or state_and_dynamic) with "
                "coordinate-based observations."
            )

        n_axes = 2 if self.horizontal_only else 3
        row = row_coords[:, :n_axes]  # (N_aug, n_axes)
        obs = obs_coords[:, :n_axes]  # (N_d, n_axes)

        # Pairwise Euclidean distance via ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b,
        # avoiding the (N_aug, N_d, 3) broadcast intermediate.
        row_sq = jnp.sum(row**2, axis=1)[:, None]  # (N_aug, 1)
        obs_sq = jnp.sum(obs**2, axis=1)[None, :]  # (1, N_d)
        cross = row @ obs.T  # (N_aug, N_d)
        dist_sq = jnp.maximum(row_sq + obs_sq - 2.0 * cross, 0.0)
        distance = jnp.sqrt(dist_sq)  # (N_aug, N_d)

        return taper_inflation(
            distance,
            self.localization_radius,
            self.tapering_beta,
            self.max_inflation,
        )
