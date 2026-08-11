"""Temporal localization of the windowed parameter-trajectory update.

The outer ESMDA control vector of :class:`~data_assimilation.filter_smoothing.\
base.FilterSmoothingESMDA` is the *whole* parameter trajectory ``Theta``, and
its observation vector is the *whole* window's stacked forecast observations.
With a finite ensemble, sampling error therefore produces spurious correlations
between a knot at one time and observations far away in time. This strategy is
the paper's Eq. (17): taper the parameter-observation covariance by
``rho(|t_j - t_k|)``, reusing the shared Vossepoel taper
(:func:`~data_assimilation.localization.base.taper_inflation`) with time in
place of physical distance.

Deliberately **symmetric in time**: a later observation may still update an
earlier knot — that temporal credit assignment is the defining feature of the
method (paper §4.3), not something localization should suppress. The taper
exists to remove implausible long-range sampling correlations, not to enforce
filtering causality.
"""

from typing import Optional

import jax.numpy as jnp
from data_assimilation.localization.base import BaseLocalization, taper_inflation


class TemporalLocalization(BaseLocalization):
    """Time-distance localization with error-variance tapering.

    Slots into the same local-analysis machinery as the spatial strategies: the
    only difference is what ``distance`` means. Rows and observations carry
    their *time* in the first component of the ``(N, 3)`` coordinate arrays
    (the remaining components are unused), and

    ``distance(row j, observation k) = |t_j - t_k|``.

    :class:`~data_assimilation.filter_smoothing.base.FilterSmoothingESMDA`
    builds those coordinates — knot ``j`` at the start of segment ``j`` and
    observation batch ``k`` at the end of segment ``k`` — so this class stays
    geometry-agnostic and is unit-agnostic with it: ``temporal_radius`` is
    expressed in whatever unit the coordinates use (cycles, as the outer loop
    builds them).

    Args:
        temporal_radius: Truncation distance in time. Observations more than
            this far from a knot are excluded from that knot's update. Tune to
            the characteristic time over which a parameter change still
            influences the observed flow.
        tapering_beta: ``beta in (0, 1)``; the fraction of the radius within
            which observations are *not* tapered (Eq. 9). Defaults to ``0.5``.
        max_inflation: Maximum error-variance inflation ``E_max`` reached at
            the radius (Eq. 10). ``E_max`` multiplies the observation-error
            perturbation (std), so the error *variance* there is scaled by
            ``E_max ** 2``. Defaults to ``4.0``, the distance-localization
            value of the paper this taper comes from.
        block_grouping: Local-analysis granularity. ``False`` (default)
            updates each knot row on its own. ``True`` requests the "grid
            block" analysis with the blocks
            :meth:`~data_assimilation.augmentation.ParamAugmentation.group_ids`
            defines — all knots of one parameter. Note what that means here:
            a block shares ONE observation selection (the per-observation
            minimum inflation over its rows), so grouping a parameter's knots
            erases the temporal taper *within* that parameter and leaves only
            the between-parameter selection. Useful when the trajectory should
            move as a whole; not the default.
    """

    #: Time is supplied through ``row_coords`` / ``obs_coords`` (component 0).
    requires_coordinates: bool = True
    #: Every row this strategy sees IS a parameter row — the outer update has
    #: no state block at all.
    localizes_parameters: bool = True

    def __init__(
        self,
        temporal_radius: float,
        tapering_beta: float = 0.5,
        max_inflation: float = 4.0,
        block_grouping: bool = False,
    ) -> None:
        if temporal_radius <= 0.0:
            raise ValueError("temporal_radius must be > 0.")
        if not (0.0 < tapering_beta < 1.0):
            raise ValueError("tapering_beta must lie in (0, 1).")
        if max_inflation < 1.0:
            raise ValueError("max_inflation must be >= 1.")

        self.temporal_radius = float(temporal_radius)
        self.tapering_beta = tapering_beta
        self.max_inflation = max_inflation
        self.block_grouping = block_grouping

    def inflation_factors(
        self,
        aug_dev: jnp.ndarray,
        pred_obs_dev: jnp.ndarray,
        row_coords: Optional[jnp.ndarray] = None,
        obs_coords: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # ``aug_dev`` / ``pred_obs_dev`` are unused: temporal localization
        # depends only on when each row and each observation live.
        if row_coords is None or obs_coords is None:
            raise ValueError(
                "TemporalLocalization requires row_coords and obs_coords "
                "carrying the row/observation times in component 0. They are "
                "built by FilterSmoothingESMDA; this strategy is the outer "
                "trajectory update's localization, not the inner state "
                "filter's (use a distance/correlation strategy there)."
            )

        distance = jnp.abs(row_coords[:, 0][:, None] - obs_coords[:, 0][None, :])

        return taper_inflation(
            distance,
            self.temporal_radius,
            self.tapering_beta,
            self.max_inflation,
        )
