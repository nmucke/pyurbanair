"""Localization for ensemble-smoother analysis updates.

Localization reduces the effect of spurious ensemble correlations in the
Kalman update by restricting which observations influence each row of the
augmented state.  Following Vossepoel et al. (2025, MWR-D-24-0269.1), we
implement it as a *local analysis*: each row ``l`` of the augmented state is
updated with an individual transition matrix computed from only the
observations relevant to that row.

The different localization strategies (distance-based, correlation-based,
...) differ *only* in how they decide which observations are relevant and how
strongly to taper the ones near the cutoff.  That policy is expressed here as
an observation-error *inflation matrix* ``E_inf`` of shape ``(N_aug, N_d)``:

* ``E_inf[l, j] == 1``    -> observation ``j`` fully influences row ``l``;
* ``1 < E_inf[l, j] < inf`` -> observation ``j`` is tapered for row ``l``
  (its perturbation/std is multiplied by ``E_inf``, i.e. its error variance
  is inflated by ``E_inf ** 2``, reducing its impact);
* ``E_inf[l, j] == inf``  -> observation ``j`` is excluded from row ``l``.

Subclasses implement :meth:`inflation_factors`.  The shared local-analysis
math lives in :meth:`localized_update`.

Mathematical notes
------------------

*The taper (Vossepoel Eqs. 9-10), derived once.*  With un-tapered fraction
``beta``, truncation distance ``T`` and peak inflation ``E_max``, the width is
``b = (1 - beta) T / sqrt(log E_max)`` and, for ``d > beta T``, the inflation is
``E_inf(d) = exp(((d - beta T) / b)^2)``.  At ``d = beta T`` the exponent is 0 so
``E_inf = 1`` (taper switches on continuously); at ``d = T`` the exponent is
``((1 - beta) T / b)^2 = log E_max`` so ``E_inf = E_max`` exactly; beyond ``T`` the
observation is excluded (``inf``).  The ``E_max = 1`` edge case gives ``log = 0``
so ``b -> inf`` and ``E_inf`` collapses to 1 -- a hard cutoff with no taper.  This
single formula drives both strategies; only ``d`` and ``T`` differ (correlation:
``d = 1 - |rho|``, ``T = 1 - rho_t``; distance: ``d = ||x_l - x_j||``,
``T = radius``).  See :func:`taper_inflation`.

*Localization and the ES-MDA likelihood budget.*  Per-row observation-error
inflation means different augmented rows assimilate *different tempered
likelihoods*.  The ES-MDA identity ``sum_k 1/alpha_k = 1`` (one Bayesian
conditioning) holds per row only in the linear-Gaussian limit AND when the
inflation pattern is the same at every step.  Correlation localization re-selects
observations every step from the current (shrinking) ensemble, so the effective
per-row data weight over the schedule is not strictly controlled.  This is
inherent to the paper's method, not a bug here; treat localized posteriors as
approximate and prefer innovation-consistency diagnostics over exact-MDA
assumptions.  (Also note ``rho_t = 3/sqrt(N_e)`` is fixed across steps while the
ensemble spread shrinks; per-step adaptive thresholds are an open choice.)
"""

from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp


def _group_inflation(inflation: jnp.ndarray, group_ids: jnp.ndarray) -> jnp.ndarray:
    """Share the per-observation inflation across rows in the same block.

    Implements the paper's "grid block" local analysis (Vossepoel et al. 2025,
    sec. 3b): co-located augmented rows are updated jointly with a *single*
    observation selection and transition matrix.  For each block and each
    observation we take the minimum inflation over the block's member rows,
    which reproduces the reference EnKF_MS semantics
    (``m_assimilation_old.F90``):

    * an observation is active for the block if it is active for *any* member
      (``|corrA| > rho_t .or. |corrO| > rho_t``) — the min of a finite and an
      ``inf`` is finite;
    * the taper is driven by the *strongest* correlation in the block
      (``dist = 1 - max(|corr|)``) — the largest ``|rho|`` gives the smallest
      ``d_c`` and hence the smallest inflation, i.e. the min.

    Args:
        inflation: Per-row inflation, shape ``(N_aug, N_d)``.
        group_ids: Block id per augmented row, shape ``(N_aug,)``.  Rows
            sharing an id are updated jointly.

    Returns:
        Inflation of shape ``(N_aug, N_d)`` in which every row has been
        replaced by its block's shared inflation vector.
    """
    num_segments = int(group_ids.max()) + 1
    block_min = jax.ops.segment_min(
        inflation, group_ids, num_segments=num_segments
    )  # (num_segments, N_d)
    return block_min[group_ids]


def taper_inflation(
    distance: jnp.ndarray,
    truncation: float,
    tapering_beta: float,
    max_inflation: float,
) -> jnp.ndarray:
    """Error-variance inflation as a function of a (generic) distance.

    Implements the tapering of Vossepoel et al. (2025), Eqs. (9)-(10), written
    against an abstract ``distance`` and ``truncation`` distance so the *same*
    taper drives both localization strategies:

    * correlation-based: ``distance = 1 - |rho|``, ``truncation = 1 - rho_t``;
    * physical-distance-based: ``distance = ||grid point - sensor||``,
      ``truncation = localization_radius``.

    Observations with ``distance <= truncation`` are kept; their error variance
    is inflated by a factor that is ``1`` for ``distance <= beta*truncation`` and
    grows to ``max_inflation`` at ``distance == truncation``.  Observations with
    ``distance > truncation`` are excluded (``jnp.inf``).

    Args:
        distance: Distance per ``(row, observation)`` pair, any shape.
        truncation: Truncation distance (scalar).
        tapering_beta: ``beta in (0, 1)``; fraction of ``truncation`` left
            un-tapered.
        max_inflation: ``E_max >= 1`` reached at ``distance == truncation``.

    Returns:
        Inflation array, same shape as ``distance``: ``1.0`` (full weight),
        ``> 1`` (tapered) or ``jnp.inf`` (excluded).
    """
    # b such that the inflation equals max_inflation at distance == truncation
    # (Eq. 10). For max_inflation == 1 (log == 0) b -> inf, giving no taper.
    b = (1.0 - tapering_beta) * truncation / jnp.sqrt(jnp.log(max_inflation))
    taper_active = distance > (tapering_beta * truncation)
    safe_b = jnp.where(b > 0.0, b, 1.0)  # guard div-by-zero when b == 0
    taper = jnp.where(
        taper_active,
        jnp.exp(((distance - tapering_beta * truncation) / safe_b) ** 2),
        1.0,
    )
    inflation = jnp.where(b > 0.0, taper, 1.0)
    # Exclude observations beyond the truncation distance.
    return jnp.where(distance <= truncation, inflation, jnp.inf)


class BaseLocalization(ABC):
    """Base class for ESMDA localization strategies."""

    #: Whether :meth:`inflation_factors` needs the physical coordinates of the
    #: augmented rows and observations (``row_coords`` / ``obs_coords``).  The
    #: correlation strategy works from ensemble anomalies alone (``False``); the
    #: distance strategy needs grid/sensor coordinates (``True``), which only the
    #: state-bearing smoothers can supply.
    requires_coordinates: bool = False

    #: Whether the strategy can localize non-spatial parameter rows. Adaptive
    #: correlation localization can, because it uses only ensemble
    #: correlations. Physical-distance localization cannot assign a meaningful
    #: location to a scalar/global parameter, so those rows must retain the
    #: global update in a joint analysis.
    localizes_parameters: bool = True

    #: Whether the strategy requests the paper's "grid block" joint local
    #: analysis (sec. 3b): co-located augmented rows are updated with a single
    #: observation selection and transition.  Subclasses set it from their
    #: constructor; declared here so the smoother can read it without a
    #: ``getattr`` guard.
    block_grouping: bool = False

    @abstractmethod
    def inflation_factors(
        self,
        aug_dev: jnp.ndarray,
        pred_obs_dev: jnp.ndarray,
        row_coords: Optional[jnp.ndarray] = None,
        obs_coords: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Observation-error inflation factors for the local analysis.

        Args:
            aug_dev: Augmented-state anomalies, shape ``(N_aug, N_e)``
                (ensemble mean already subtracted).
            pred_obs_dev: Predicted-observation anomalies, shape
                ``(N_d, N_e)`` (ensemble mean already subtracted).
            row_coords: Optional physical coordinates of each augmented row,
                shape ``(N_aug, 3)``.  Supplied by state-bearing smoothers when
                :attr:`requires_coordinates` is ``True`` (parameter rows are
                padded and then masked out via ``localize_mask``).
            obs_coords: Optional physical coordinates of each observation
                (sensor location), shape ``(N_d, 3)``.

        Returns:
            Array of shape ``(N_aug, N_d)``.  Entry ``[l, j]`` is the factor
            ``E_inf`` that multiplies observation ``j``'s error perturbation
            (std) when updating augmented-state row ``l`` — equivalently its
            error variance is inflated by ``E_inf ** 2``.  ``1.0`` keeps the
            observation, larger values taper it, and ``jnp.inf`` excludes it.
        """
        raise NotImplementedError

    def localized_update(
        self,
        augmented: jnp.ndarray,
        aug_dev: jnp.ndarray,
        pred_obs: jnp.ndarray,
        pred_obs_dev: jnp.ndarray,
        obs: jnp.ndarray,
        C_D: jnp.ndarray,
        C_D_sqrt: jnp.ndarray,
        alpha: float,
        rng_key: jax.Array,
        group_ids: Optional[jnp.ndarray] = None,
        localize_mask: Optional[jnp.ndarray] = None,
        row_coords: Optional[jnp.ndarray] = None,
        obs_coords: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Apply the localized (row- or block-wise) ESMDA Kalman update.

        This mirrors the global update in
        ``_BaseESMDA._compute_kalman_update`` but computes an individual
        transition for each augmented-state row, scaling the (diagonal)
        observation-error covariance ``C_D`` by this localization's
        per-row inflation factors.  Rows whose observations are all excluded
        receive the identity update (no change), matching the paper's
        "transition matrix is just the identity" case.

        When ``group_ids`` is given, co-located rows sharing a block id are
        updated *jointly* with a single observation selection and transition
        matrix (the paper's "grid block" local analysis, sec. 3b) — e.g. the
        ``u``/``v``/``w`` state at one cell, or all time knots of one
        parameter.  When ``group_ids`` is ``None`` each row is its own block
        (per-row local analysis).

        Args:
            augmented: Augmented state, shape ``(N_aug, N_e)``.
            aug_dev: Augmented-state anomalies, shape ``(N_aug, N_e)``.
            pred_obs: Predicted observations, shape ``(N_d, N_e)``.
            pred_obs_dev: Predicted-observation anomalies, shape
                ``(N_d, N_e)``.
            obs: True observations, shape ``(N_d,)``.
            C_D: Diagonal observation-error covariance, shape
                ``(N_d, N_d)``.
            C_D_sqrt: Element-wise square root of ``C_D``, shape
                ``(N_d, N_d)``.
            alpha: ESMDA inflation coefficient for this step.
            rng_key: PRNG key for the observation perturbations.
            group_ids: Optional block id per augmented row, shape
                ``(N_aug,)``.  Rows sharing an id are updated jointly with a
                single selection/transition.  ``None`` -> per-row analysis.
            localize_mask: Optional boolean array, shape ``(N_aug,)``.  Rows
                where ``False`` receive the exact global (unlocalized) Kalman
                update; rows where ``True`` (or when ``None``) use this
                strategy's inflation factors. Used when a strategy does not
                apply to every augmented block (for example distance
                localization on joint parameter rows).
            row_coords: Optional augmented-row coordinates, shape ``(N_aug, 3)``,
                forwarded to :meth:`inflation_factors` (distance strategy).
            obs_coords: Optional observation coordinates, shape ``(N_d, 3)``,
                forwarded to :meth:`inflation_factors` (distance strategy).

        Returns:
            Updated augmented array, shape ``(N_aug, N_e)``.
        """
        N_aug, N_e = augmented.shape
        N_d = obs.shape[0]

        # Cross- and auto-covariances (the auto-covariance C_DD does not
        # depend on the per-row inflation; only the alpha * C_D term does).
        C_MD = jnp.dot(aug_dev, pred_obs_dev.T) / (N_e - 1)
        C_DD = jnp.dot(pred_obs_dev, pred_obs_dev.T) / (N_e - 1)

        Z = jax.random.normal(rng_key, (N_d, N_e))
        # Center the perturbations so their ensemble mean is exactly zero,
        # removing the O(1/sqrt(N_e)) sampling bias in the analysis mean (Evensen
        # 2004); matches the global path in ``_compute_kalman_update``.
        Z = Z - jnp.mean(Z, axis=1, keepdims=True)
        # Base (un-inflated) measurement-error perturbation E.  Following
        # Vossepoel et al. (2025) and the reference EnKF_MS implementation
        # (``m_assimilation_old.F90``: ``subE(m,:) = subE(m,:) * Einfl``), the
        # per-row tapering multiplies the perturbation matrix E — a *standard-
        # deviation* quantity — by the inflation factor E_inf.  Consequently
        # the measurement-error *variance* is scaled by ``E_inf ** 2``.
        base_perturbation = jnp.sqrt(alpha) * (C_D_sqrt @ Z)  # (N_d, N_e)

        C_D_diag = jnp.diag(C_D)  # (N_d,)
        inflation = self.inflation_factors(
            aug_dev, pred_obs_dev, row_coords=row_coords, obs_coords=obs_coords
        )  # (N_aug, N_d)

        # Joint "grid block" update: rows in the same block share one selection
        # and transition. Masked/global rows are temporarily set to infinity so
        # they cannot pull a localized block's minimum inflation down to 1 when
        # a caller accidentally gives both kinds of row the same group id.
        if group_ids is not None:
            grouping_inflation = inflation
            if localize_mask is not None:
                grouping_inflation = jnp.where(
                    localize_mask[:, None], inflation, jnp.inf
                )
            inflation = _group_inflation(grouping_inflation, group_ids)

        # Force masked-out rows to all-ones inflation so they receive the exact
        # global update. An all-ones row keeps every observation with no taper,
        # reducing its solve to the corresponding row of the global Kalman
        # update with the SAME observation-perturbation realization.
        if localize_mask is not None:
            inflation = jnp.where(localize_mask[:, None], inflation, 1.0)

        def update_row(
            aug_row: jnp.ndarray,
            c_md_row: jnp.ndarray,
            inflation_row: jnp.ndarray,
        ) -> jnp.ndarray:
            # An observation is "active" for this row when its inflation is
            # finite; an infinite inflation factor means "excluded".
            active = jnp.isfinite(inflation_row)  # (N_d,) bool

            # E_inf for active observations; placeholder 1.0 for excluded ones
            # (they are decoupled below, so the value is moot).  E_inf scales
            # the perturbation E (std), so the error variance C_D — and hence
            # the EE^T term in the analysis denominator — scales by E_inf ** 2.
            e_inf = jnp.where(active, inflation_row, 1.0)  # (N_d,)
            C_D_row = C_D_diag * e_inf**2  # (N_d,) variance scaled by E_inf**2

            perturbed_obs = obs[:, None] + e_inf[:, None] * base_perturbation
            # Excluded observations carry no innovation.
            innovation = jnp.where(
                active[:, None], perturbed_obs - pred_obs, 0.0
            )  # (N_d, N_e)

            C_DD_alpha = C_DD + alpha * jnp.diag(C_D_row)  # (N_d, N_d)
            # Decouple excluded observations: zero their rows/columns and put
            # 1 on their diagonal. The resulting linear system is block
            # structured, so the solve yields the exact active-only solution
            # (x = 0 for excluded obs) — equivalent to extracting the active
            # submatrix as in Vossepoel et al. (2025), but shape-stable for
            # ``jax.vmap``.
            keep = active[:, None] * active[None, :]
            C_DD_alpha = C_DD_alpha * keep + jnp.diag(jnp.where(active, 0.0, 1.0))

            x = jnp.linalg.solve(C_DD_alpha, innovation)  # (N_d, N_e)
            # Excluded observations must not enter this row's update.
            c_md_row = jnp.where(active, c_md_row, 0.0)
            return aug_row + c_md_row @ x  # (N_e,)

        return jax.vmap(update_row)(augmented, C_MD, inflation)
