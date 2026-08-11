"""Analysis (update) schemes shared by the filter cycle loop and ESMDA.

The stochastic (perturbed-observation) EnKF analysis with tempering weight
``alpha`` *is* the ESMDA per-step update; with ``alpha = 1`` it is the classic
stochastic EnKF analysis. The single implementation lives here as the pure
function :func:`stochastic_enkf_update` — explicit ``rng_key``, no hidden
state — and both packages call it:

* ``smoothing/esmda.py`` passes its tempered ``alpha`` (and its own key
  splitting), so the smoother behavior is unchanged by the extraction;
* ``filtering/base.py`` calls it through :class:`StochasticEnKFAnalysis` with
  the filter's full-weight ``alpha = 1``.

New update flavors for the filter (ETKF, particle-style updates, ...) are new
:class:`AnalysisScheme` implementations — the cycle loop in ``BaseFilter``
does not change.
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from data_assimilation.localization.base import BaseLocalization

LocalizationPolicy = Literal["optional", "forbidden", "required"]

#: Runtime spelling of :data:`LocalizationPolicy`, so a typo in a scheme's
#: declaration fails loudly in ``BaseFilter.__init__`` instead of quietly
#: matching none of the checks (type annotations are not enforced at runtime).
LOCALIZATION_POLICIES: tuple[str, ...] = ("optional", "forbidden", "required")


def validate_variances(C_D_diag: jnp.ndarray) -> jnp.ndarray:
    """Validate a 1-D observation-error variance vector.

    The honest contract for the diagonal-``C_D`` assumption baked into the
    perturbation draw and the localized update: a flat vector of strictly
    positive variances. A zero/negative variance -- a config typo or a
    "perfect" synthetic sensor -- makes the analysis system singular in the
    directions outside the ensemble span, and JAX's solve returns NaN without
    raising.
    """
    C_D_diag = jnp.asarray(C_D_diag)
    if C_D_diag.ndim != 1:
        raise ValueError(
            f"C_D_diag must be a 1-D variance vector, got shape {C_D_diag.shape}. "
            "Pass the diagonal of the observation-error covariance (sigma**2 "
            "per observation)."
        )
    if not bool(jnp.all(C_D_diag > 0.0)):
        raise ValueError(
            "Observation-error variances must be strictly positive; a zero or "
            "negative variance makes the analysis system singular "
            "(NaN-poisoning the ensemble). Check obs_error_std."
        )
    return C_D_diag


def stochastic_enkf_update(
    augmented: jnp.ndarray,
    pred_obs: jnp.ndarray,
    obs: jnp.ndarray,
    C_D_diag: jnp.ndarray,
    rng_key: jax.Array,
    alpha: float = 1.0,
    localization: Optional[BaseLocalization] = None,
    group_ids: Optional[jnp.ndarray] = None,
    localize_mask: Optional[jnp.ndarray] = None,
    row_coords: Optional[jnp.ndarray] = None,
    obs_coords: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """The stochastic (perturbed-observation) EnKF / ESMDA analysis.

    Pure function: the caller supplies (and is responsible for splitting) the
    PRNG key. Observation perturbations are centered (their ensemble mean is
    subtracted), removing the O(1/sqrt(N_e)) sampling bias in the analysis
    mean (Evensen 2004).

    Args:
        augmented: Array of shape (N_aug, N_e) — the augmented state
            (parameters only, state only, or state + parameters).
        pred_obs: Array of shape (N_d, N_e) — predicted observations.
        obs: Array of shape (N_d,) — true observations.
        C_D_diag: Diagonal of the observation-error covariance as a 1-D
            variance vector, shape (N_d,).
        rng_key: PRNG key for the observation perturbations (consumed as-is).
        alpha: Tempering weight of this update. ``1.0`` is the full-weight
            filter analysis; the ESMDA smoother passes its per-step
            coefficient.
        localization: Optional localization strategy. When set, each augmented
            row is updated via a local analysis driven by the strategy's
            inflation factors instead of the global update.
        group_ids: Optional block id per augmented row, shape (N_aug,),
            used only by a localized update with block grouping enabled.
            Rows sharing an id are updated jointly. ``None`` -> per-row.
        localize_mask: Optional boolean array, shape (N_aug,), forwarded to
            a localized update. Rows where ``False`` get the exact global
            update; ``True`` rows get the localized update. Ignored on the
            global (``localization is None``) path.
        row_coords: Optional augmented-row coordinates, shape (N_aug, 3),
            and ``obs_coords``: observation coordinates, shape (N_d, 3),
            forwarded to a distance-based localized update. Ignored on the
            global path and by coordinate-free strategies.

    Returns:
        Updated augmented array of the same shape.
    """
    N_e = augmented.shape[1]
    N_d = obs.shape[0]
    if C_D_diag.ndim != 1 or C_D_diag.shape[0] != N_d:
        raise ValueError(
            f"C_D_diag must be a 1-D variance vector of length N_d={N_d}, got "
            f"shape {C_D_diag.shape}."
        )

    aug_mean = jnp.mean(augmented, axis=1, keepdims=True)
    pred_obs_mean = jnp.mean(pred_obs, axis=1, keepdims=True)

    aug_dev = augmented - aug_mean
    pred_obs_dev = pred_obs - pred_obs_mean

    # Localized (local-analysis) update: each augmented row is updated
    # with only the observations the localization strategy deems relevant.
    if localization is not None:
        return localization.localized_update(
            augmented=augmented,
            aug_dev=aug_dev,
            pred_obs=pred_obs,
            pred_obs_dev=pred_obs_dev,
            obs=obs,
            C_D=jnp.diag(C_D_diag),
            C_D_sqrt=jnp.diag(jnp.sqrt(C_D_diag)),
            alpha=alpha,
            rng_key=rng_key,
            group_ids=group_ids,
            localize_mask=localize_mask,
            row_coords=row_coords,
            obs_coords=obs_coords,
        )

    C_MD = jnp.dot(aug_dev, pred_obs_dev.T) / (N_e - 1)
    C_DD = jnp.dot(pred_obs_dev, pred_obs_dev.T) / (N_e - 1)

    Z = jax.random.normal(rng_key, (N_d, N_e))
    # Center the observation perturbations: the raw draw has a sample mean of
    # order 1/sqrt(N_e), which biases the analysis mean. Removing the row
    # mean makes the perturbations exactly zero-mean across the ensemble --
    # the standard, essentially free stochastic-EnKF correction (Evensen
    # 2004), most visible at the N_e ~ 50-100 used here.
    Z = Z - jnp.mean(Z, axis=1, keepdims=True)
    perturbed_obs = obs[:, None] + jnp.sqrt(alpha) * (jnp.sqrt(C_D_diag)[:, None] * Z)

    innovation = perturbed_obs - pred_obs
    C_DD_alpha = C_DD + alpha * jnp.diag(C_D_diag)

    # ``C_DD + alpha * C_D`` is symmetric positive definite by construction
    # (positive observation-error variances; see the C_D validation at the
    # call sites), so solve it with a Cholesky factorization: ~2x cheaper
    # than generic LU and better-conditioned. A non-SPD system (a collapsed
    # ensemble or a degenerate C_D slipping through) yields NaNs in the
    # factor, which the finiteness check below turns into a loud failure
    # rather than a silently NaN-poisoned ensemble.
    x = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(C_DD_alpha), innovation)
    if not bool(jnp.all(jnp.isfinite(x))):
        raise FloatingPointError(
            "Kalman solve produced non-finite values: the system "
            "C_DD + alpha * C_D is singular or ill-conditioned. Check for "
            "collapsed ensemble spread or a degenerate observation-error "
            "covariance."
        )

    return augmented + C_MD @ x


class AnalysisScheme(ABC):
    """The per-cycle analysis (update) math of a sequential filter.

    A pure function of arrays: the :class:`~data_assimilation.filtering.base.\
BaseFilter` cycle loop owns time management, augmentation, inflation and
    parameter evolution; the scheme only maps the forecast ensemble to the
    analysis ensemble. New filter flavors (ETKF/LETKF, particle-style updates)
    implement this interface — the cycle loop does not change.

    ``localization_policy`` declares what the scheme can *do* with a
    localization strategy, and ``BaseFilter`` validates it at construction so an
    invalid combination fails before the first forecast instead of inside cycle
    0. ``"optional"`` (the default, inherited by the stochastic analysis)
    accepts either; ``"forbidden"`` is a global-only scheme such as the ETKF,
    whose localized counterpart is an explicit LETKF class so a config name
    cannot promise localization the analysis ignores; ``"required"`` is that
    LETKF, which has no meaning without a strategy.
    """

    localization_policy: LocalizationPolicy = "optional"

    @abstractmethod
    def __call__(
        self,
        augmented: jnp.ndarray,
        pred_obs: jnp.ndarray,
        obs: jnp.ndarray,
        C_D_diag: jnp.ndarray,
        rng_key: jax.Array,
        localization: Optional[BaseLocalization] = None,
        group_ids: Optional[jnp.ndarray] = None,
        localize_mask: Optional[jnp.ndarray] = None,
        row_coords: Optional[jnp.ndarray] = None,
        obs_coords: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Return the analysis ensemble for one full-weight observation batch.

        Arguments as in :func:`stochastic_enkf_update` (with ``alpha`` fixed
        to the filter's full weight of 1).
        """
        raise NotImplementedError


class StochasticEnKFAnalysis(AnalysisScheme):
    """The stochastic (perturbed-observation) EnKF analysis, full weight."""

    def __call__(
        self,
        augmented: jnp.ndarray,
        pred_obs: jnp.ndarray,
        obs: jnp.ndarray,
        C_D_diag: jnp.ndarray,
        rng_key: jax.Array,
        localization: Optional[BaseLocalization] = None,
        group_ids: Optional[jnp.ndarray] = None,
        localize_mask: Optional[jnp.ndarray] = None,
        row_coords: Optional[jnp.ndarray] = None,
        obs_coords: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return stochastic_enkf_update(
            augmented=augmented,
            pred_obs=pred_obs,
            obs=obs,
            C_D_diag=C_D_diag,
            rng_key=rng_key,
            alpha=1.0,
            localization=localization,
            group_ids=group_ids,
            localize_mask=localize_mask,
            row_coords=row_coords,
            obs_coords=obs_coords,
        )
