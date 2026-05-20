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
  (its error variance is inflated, reducing its impact);
* ``E_inf[l, j] == inf``  -> observation ``j`` is excluded from row ``l``.

Subclasses implement :meth:`inflation_factors`.  The shared local-analysis
math lives in :meth:`localized_update`.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


class BaseLocalization(ABC):
    """Base class for ESMDA localization strategies."""

    @abstractmethod
    def inflation_factors(
        self,
        aug_dev: jnp.ndarray,
        pred_obs_dev: jnp.ndarray,
    ) -> jnp.ndarray:
        """Observation-error inflation factors for the local analysis.

        Args:
            aug_dev: Augmented-state anomalies, shape ``(N_aug, N_e)``
                (ensemble mean already subtracted).
            pred_obs_dev: Predicted-observation anomalies, shape
                ``(N_d, N_e)`` (ensemble mean already subtracted).

        Returns:
            Array of shape ``(N_aug, N_d)``.  Entry ``[l, j]`` is the factor
            by which observation ``j``'s error variance is inflated when
            updating augmented-state row ``l``.  ``1.0`` keeps the
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
    ) -> jnp.ndarray:
        """Apply the localized (row-by-row) ESMDA Kalman update.

        This mirrors the global update in
        ``_BaseESMDA._compute_kalman_update`` but computes an individual
        transition for each augmented-state row, scaling the (diagonal)
        observation-error covariance ``C_D`` by this localization's
        per-row inflation factors.  Rows whose observations are all excluded
        receive the identity update (no change), matching the paper's
        "transition matrix is just the identity" case.

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
        # Base perturbation (un-inflated). Per-row inflation scales the
        # observation errors, so the perturbed observations for row l use
        # sqrt(E_inf[l]) on the perturbation as well as on C_D (the paper
        # multiplies each row of the perturbation matrix E by E_inf).
        base_perturbation = jnp.sqrt(alpha) * (C_D_sqrt @ Z)  # (N_d, N_e)

        C_D_diag = jnp.diag(C_D)  # (N_d,)
        inflation = self.inflation_factors(aug_dev, pred_obs_dev)  # (N_aug, N_d)

        def update_row(
            aug_row: jnp.ndarray,
            c_md_row: jnp.ndarray,
            inflation_row: jnp.ndarray,
        ) -> jnp.ndarray:
            # An observation is "active" for this row when its inflation is
            # finite; an infinite inflation factor means "excluded".
            active = jnp.isfinite(inflation_row)  # (N_d,) bool

            # Finite tapering for active observations; placeholder 1.0 for
            # excluded ones (they are decoupled below, so the value is moot).
            tapered = jnp.where(active, inflation_row, 1.0)  # (N_d,)
            sqrt_inf = jnp.sqrt(tapered)  # (N_d,)
            C_D_row = C_D_diag * tapered  # (N_d,)

            perturbed_obs = obs[:, None] + sqrt_inf[:, None] * base_perturbation
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
