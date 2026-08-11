"""Deterministic ensemble-transform analyses (ETKF) and the shared TSVD kernel.

The stochastic analysis in :mod:`data_assimilation.filtering.analysis` draws
perturbed observations; the ensemble-transform family instead computes one
**ensemble-space weight matrix** and right-multiplies the forecast anomalies
with it, so the posterior sample covariance is the Kalman covariance exactly
(given the sample forecast moments) rather than in expectation over the
perturbation draw.

Four things in this module are load-bearing contracts rather than
implementation choices:

1. **Pure right-multiplication.** The analysis is
   ``mean + X @ (wbar 1^T + W_a)`` with ``X`` the raw forecast anomalies and
   the weights depending *only* on ``(pred_obs, obs, C_D_diag)``. Nothing may
   depend on the analyzed rows themselves — not their count, not their spread.
   ``BaseFilter._record_reduction_diagnostics`` exploits exactly this: it calls
   the analysis a second time on the tiny ``(k, N_e)`` coordinate array of the
   reduction fit and subtracts, which measures the increment the truncation
   discards *only* if both calls share one weight matrix. It is also what makes
   PR 1's reduced state representation work on this path for free: encoding the
   state block is a left multiplication, and left and right multiplication
   commute.
2. **The symmetric square root.** ``W_a = sqrt(N_e-1) C^{-1/2}`` is formed with
   the symmetric (minimum-distance, mean-preserving) root, not a Cholesky or
   one-sided root. Any root ``W_a Q`` with orthogonal ``Q`` gives the same
   posterior covariance, so the choice looks free — it is not. RTPP
   (:class:`~data_assimilation.inflation.RTPP`) blends posterior anomalies with
   the *prior* anomalies member by member. A rotated root scrambles member
   identity, so RTPP would blend a member against an unrelated prior
   perturbation. The symmetric root is additionally the unique root with
   ``W_a 1 = 1``, which is what preserves the posterior mean (see
   :func:`ensemble_transform`).
3. **No inflation, no re-centering.** ``BaseFilter`` owns prior inflation
   (already applied to both ``augmented`` and ``pred_obs`` when the scheme is
   called) and posterior inflation. The scheme adds neither.
4. **Factored, fixed-shape weights.** The transform is stored as
   ``(V, scale, wbar)`` and applied as
   ``X + (X V) diag(scale - 1) V^T``; the dense ``N_e x N_e`` matrix is a
   convenience, never a required intermediate. Spectral truncation is a
   *retention mask over a fixed rank* ``min(N_d, N_e)``, not a reshape. Both
   matter for the LETKF that will share this kernel: at the canonical shape
   (``N_e = 50``, ``N_d = 12``, so ``r = 12``) one transform costs
   ``N_e*r + N_e + r = 662`` floats factored against ``N_e**2 = 2500`` dense,
   and a shape-changing truncation cannot be batched across blocks whose active
   observation count differs. The ratio is what makes a per-*row* table
   thinkable at all: ``N_s = 230k`` rows would be 2.3 GB dense against 610 MB
   factored in float32 (:class:`LETKFAnalysis` then stores neither — see there).

The observation-space TSVD lives here too, as one helper shared by the global
ETKF and (later) every LETKF block: localize -> form ``R_eff`` -> whiten ``Y``
-> TSVD -> ensemble transform. It removes weak *linear combinations* of the
whitened observation anomalies; it never modifies the physical observation
error variances.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from data_assimilation.filtering.analysis import (
    AnalysisScheme,
    LocalizationPolicy,
    validate_variances,
)
from data_assimilation.localization.base import (
    BaseLocalization,
    active_observations,
    resolve_row_inflation,
)
from data_assimilation.reduction import _RANK_TOLERANCE_FLOOR, _validate_fraction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObservationTSVD:
    """Spectral truncation of the whitened predicted-observation anomalies.

    Nested on the analysis object (``ETKFAnalysis(tsvd=ObservationTSVD(...))``)
    rather than exposed as a separate filter class, because it regularizes the
    transform the analysis already computes. Disabled by default: with the
    shipped ``N_d ~ 12`` there is nothing to regularize, and truncation can only
    remove update directions.

    Args:
        enabled: Apply the scientific truncation (``energy_fraction`` and
            ``max_rank``). When ``False`` and no ``numerical_tolerance`` is
            given, the kernel retains **every** thin-SVD direction — see
            :func:`ensemble_transform` for why keeping round-off directions is
            exactly harmless here (unlike ESMDA's whitened encode, which
            divides by the singular values). Because ``max_rank`` *is* part of
            the scientific truncation, setting it while disabled is rejected
            rather than silently ignored.
        energy_fraction: Smallest retained rank whose squared singular values
            hold this fraction of the whitened spectrum. ``1.0`` needs no
            special case: the suffix criterion of :func:`_retention_mask` then
            counts every strictly nonzero singular value and the numerical cap
            reduces it to "retain every numerically nonzero direction".
        max_rank: Optional hard cap on the retained rank. Requires
            ``enabled=True``.
        numerical_tolerance: Optional relative singular-value floor
            (``sigma_i > numerical_tolerance * sigma_max``). It redefines
            "numerically zero" rather than making a scientific choice, so it is
            applied *in addition* to the scientific cut when that is enabled and
            **on its own when it is not** — a run can retune the round-off
            decision without turning the energy criterion on. ``None`` uses the
            dtype- and shape-aware floor shared with the state reduction
            (:func:`data_assimilation.reduction._numerical_rank`).
    """

    enabled: bool = False
    energy_fraction: float = 0.99
    max_rank: Optional[int] = None
    numerical_tolerance: Optional[float] = None

    def __post_init__(self) -> None:
        # ``_validate_fraction`` is the same (0, 1] check the state reduction
        # applies, kept identical so one energy_fraction means one thing.
        object.__setattr__(
            self, "energy_fraction", _validate_fraction(self.energy_fraction)
        )
        if self.max_rank is not None:
            if self.max_rank < 1:
                raise ValueError(f"max_rank must be >= 1, got {self.max_rank}.")
            if not self.enabled:
                # A cap that cannot take effect is a configuration error, not a
                # default: ``max_rank`` is part of the scientific truncation
                # ``enabled`` gates, so honouring it here would make a block
                # spelled "enabled: false" truncate anyway, and ignoring it
                # would be a silent no-op. ``numerical_tolerance`` is different
                # — it redefines "numerically zero" and does apply when
                # disabled; see :func:`_retention_mask`.
                raise ValueError(
                    "max_rank is part of the scientific truncation and needs "
                    f"enabled=True, got max_rank={self.max_rank} with "
                    "enabled=False. Enable the TSVD, or drop the cap (use "
                    "numerical_tolerance for a floor that applies either way)."
                )
        if self.numerical_tolerance is not None:
            tolerance = float(self.numerical_tolerance)
            if not math.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
                raise ValueError(
                    "numerical_tolerance must be a finite relative floor in "
                    f"[0, 1), got {self.numerical_tolerance}."
                )
            object.__setattr__(self, "numerical_tolerance", tolerance)


@dataclass(frozen=True)
class ObservationTransform:
    """One deterministic analysis in factored ensemble-space form.

    The posterior is ``mean + X @ (wbar 1^T + W_a)`` with

    * ``mean_weights`` — ``wbar``, shape ``(N_e,)``, moving the ensemble mean;
    * ``modes`` — the right singular vectors ``V`` of the whitened observation
      anomalies, shape ``(N_e, r)`` with the fixed ``r = min(N_d, N_e)``;
    * ``anomaly_scales`` — shape ``(r,)``, the symmetric root's eigenvalue on
      each mode. Discarded modes carry exactly ``1.0``, i.e. the identity, so
      truncation never changes an array shape.

    so that ``W_a = I + V diag(anomaly_scales - 1) V^T``. ``W_a`` is symmetric
    positive definite with ``W_a @ 1 = 1``; :meth:`anomaly_weights` assembles it
    when a caller genuinely wants the dense matrix (one global analysis), while
    :func:`apply_ensemble_transform` never does.

    ``retained_rank`` is the number of retained directions. With the TSVD
    disabled it is ``r`` and therefore says nothing about the ensemble;
    ``available_rank`` — the numerically nonzero count, bounded by
    ``min(N_d, N_e - 1)`` because observation anomalies sum to zero across the
    ensemble — is the meaningful diagnostic. Both, plus ``retained_energy`` (the
    retained share of the *whole* spectrum's energy) and ``discarded_spectrum``
    (empty when nothing was truncated), are always populated by
    :func:`ensemble_transform`; the optional declarations only keep the
    dataclass constructible without them.
    """

    mean_weights: jnp.ndarray
    modes: jnp.ndarray
    anomaly_scales: jnp.ndarray
    retained_rank: int
    discarded_spectrum: jnp.ndarray
    available_rank: Optional[int] = None
    retained_energy: Optional[float] = None

    @property
    def num_members(self) -> int:
        return int(self.mean_weights.shape[0])

    def anomaly_weights(self) -> jnp.ndarray:
        """Assemble the dense ``(N_e, N_e)`` symmetric transform ``W_a``.

        Only for callers that want the matrix itself (a single global analysis,
        or a test asserting symmetry/mean preservation). Writing it as a
        correction to the identity keeps an untruncated or zero-singular-value
        transform exactly ``I`` instead of the round-off of ``V V^T``.
        """
        return jnp.eye(self.num_members, dtype=self.modes.dtype) + self.modes @ (
            (self.anomaly_scales - 1.0)[:, None] * self.modes.T
        )


def whiten_observations(
    pred_obs: jnp.ndarray,
    obs: jnp.ndarray,
    C_D_diag: jnp.ndarray,
    error_inflation: Optional[jnp.ndarray] = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Whitened observation anomalies ``Y_w`` and innovation ``d_w``.

    ``Y_w = R_eff^{-1/2} (pred_obs - mean(pred_obs))`` and
    ``d_w = R_eff^{-1/2} (obs - mean(pred_obs))``, the only two quantities the
    ensemble transform depends on. Keeping the whitening boundary *outside*
    :func:`ensemble_transform` is what lets a localized analysis reuse the
    kernel verbatim: a LETKF block passes its own per-observation inflation
    (``R_eff = E_inf**2 * R``) here and the kernel never learns about
    localization.

    An excluded observation (``E_inf = +inf``) becomes a **zero row** of
    ``Y_w`` and a zero entry of ``d_w``, contributing exactly nothing to
    ``Y_w^T Y_w`` or ``Y_w^T d_w``. It is written with an explicit ``where`` and
    a safe placeholder rather than relying on ``1/inf``, so no ``inf * 0`` is
    ever formed (that would be NaN, not zero).

    Args:
        pred_obs: Predicted observations, shape ``(N_d, N_e)``.
        obs: Observations, shape ``(N_d,)``.
        C_D_diag: Physical observation-error variances, shape ``(N_d,)``,
            strictly positive.
        error_inflation: Optional per-observation error inflation ``E_inf``,
            shape ``(N_d,)``, as returned by
            :meth:`~data_assimilation.localization.base.BaseLocalization.\
inflation_factors`. ``None`` means no inflation (all ones).

    Returns:
        ``(whitened_anomalies, whitened_innovation)`` of shapes ``(N_d, N_e)``
        and ``(N_d,)``.
    """
    pred_obs = jnp.asarray(pred_obs)
    obs = jnp.asarray(obs)
    if pred_obs.ndim != 2:
        raise ValueError(f"pred_obs must be (N_d, N_e), got shape {pred_obs.shape}.")
    N_d = pred_obs.shape[0]
    if obs.ndim != 1 or obs.shape[0] != N_d:
        raise ValueError(
            f"obs must be a 1-D vector of length N_d={N_d}, got shape {obs.shape}."
        )
    C_D_diag = validate_variances(C_D_diag)
    if C_D_diag.shape[0] != N_d:
        raise ValueError(
            f"C_D_diag must be a 1-D variance vector of length N_d={N_d}, got "
            f"shape {C_D_diag.shape}."
        )
    return _whiten(pred_obs, obs, C_D_diag, error_inflation)


def _whiten(
    pred_obs: jnp.ndarray,
    obs: jnp.ndarray,
    C_D_diag: jnp.ndarray,
    error_inflation: Optional[jnp.ndarray] = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """:func:`whiten_observations` without the host-side validation.

    Split out because :func:`validate_variances` ends in
    ``bool(jnp.all(...))``, which is a host synchronization: harmless once per
    cycle, fatal inside the LETKF block loop, where ``lax.map`` lifts the
    closed-over ``C_D_diag`` into a scan constant and the ``bool`` then raises
    ``TracerBoolConversionError``. The validation is not skipped, only hoisted:
    :meth:`LETKFAnalysis.__call__` runs it once before the loop.
    """
    N_d = pred_obs.shape[0]
    weights = 1.0 / jnp.sqrt(C_D_diag)
    if error_inflation is not None:
        inflation = jnp.asarray(error_inflation)
        if inflation.shape != (N_d,):
            raise ValueError(
                f"error_inflation must have shape ({N_d},), got {inflation.shape}."
            )
        active = active_observations(inflation)
        safe = jnp.where(active, inflation, 1.0)  # placeholder: never inf * 0
        weights = jnp.where(active, weights / safe, 0.0)

    pred_obs_mean = jnp.mean(pred_obs, axis=1, keepdims=True)
    whitened_anomalies = weights[:, None] * (pred_obs - pred_obs_mean)
    whitened_innovation = weights * (obs - pred_obs_mean.ravel())
    return whitened_anomalies, whitened_innovation


def _retention_mask(
    singular_values: jnp.ndarray,
    matrix_shape: tuple[int, ...],
    tsvd: ObservationTSVD,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The rank policy: ``(retain, retained_rank, available_rank)``.

    **One criterion, two readouts.** The global ETKF and every LETKF block call
    this same function; the ETKF then reads ``int(retained_rank)`` for its
    diagnostics while the block loop consumes ``retain`` under a trace. There is
    deliberately no host-side twin: an earlier revision had one (a float64
    ``np.cumsum`` in :func:`data_assimilation.reduction._select_rank` next to a
    dtype-native ``jnp.cumsum`` here) and the two genuinely disagreed — at
    ``energy_fraction=1.0`` on 36 percent of a 600-problem random sweep, by up
    to 4 modes, moving the posterior by 5x the float32 test tolerance. A
    float32 cumulative sum saturates at ``1.0`` after ~7 significant digits, so
    a prefix comparison against ``1 - 1e-12`` stops counting while a float64 one
    keeps going.

    Everything below is therefore expressed so that dtype does not decide the
    rank. The cut is on the **suffix** energy — *retain the smallest prefix
    whose complement holds at most ``1 - energy_fraction`` of the spectrum*.
    Mathematically identical to the prefix form, numerically far better behaved
    on a decaying spectrum, because the discarded tail is summed from its
    smallest entry up instead of being recovered as a difference of two nearly
    equal large numbers. The spectrum is normalized by ``sigma_max`` first, so
    no overflow and no dependence on the physical units of ``R``.

    ``energy_fraction = 1.0`` needs no special case in this form, which is why
    there is none: the tolerated tail is then exactly ``0``, so the suffix
    comparison counts every *strictly nonzero* singular value, and the
    ``minimum(rank, available)`` below reduces that to the numerically nonzero
    count. "Retain every numerically nonzero direction" is therefore what the
    general path already computes — it is a statement about the round-off
    floor, and the floor is where it is imposed. (A *prefix* cumulative sum
    could not express it: a float32 prefix sum saturates at ``1.0`` after ~7
    significant digits and stops counting. That earlier form needed an
    ``energy_fraction >= 1.0`` short-circuit; this one does not, and carrying
    one anyway was dead code — removing it changed the retained rank in 0 of
    108,324 swept cases.)

    Every criterion (energy, ``max_rank``, the round-off floor, the optional
    relative tolerance) selects a *prefix* of the descending spectrum, so the
    rank and the mask ``arange(r) < rank`` are two readouts of one number. That
    is what lets truncation be a fixed-shape mask downstream.

    The round-off floor is ``_numerical_rank(conservative=False)``'s: nothing in
    the transform divides by a singular value. ``wbar`` weights direction ``i``
    by ``s_i / ((N_e-1) + s_i**2)`` and ``W_a`` by
    ``sqrt((N_e-1) / ((N_e-1) + s_i**2))``; both are *damped* as ``s_i -> 0``
    (they tend to 0 and 1, i.e. to "this direction is not assimilated"), never a
    ``1/s`` amplification, so admitting a round-off direction costs essentially
    nothing. The ``conservative=True`` cut, in contrast, scales with
    ``max(N_d, N_e)`` and would start discarding genuine observation directions
    — real information, never recovered — as soon as a dense sensor network
    makes ``N_d`` large. Between the two failure modes, the damped one is safe.

    Args:
        singular_values: Descending thin-SVD spectrum of ``Y_w``, shape
            ``(r,)`` with ``r = min(N_d, N_e)``.
        matrix_shape: Shape of ``Y_w``, for the round-off floor.
        tsvd: Truncation settings.

    Returns:
        ``(retain, retained_rank, available_rank)``: a ``(r,)`` boolean prefix
        mask and two integer scalars, all traced arrays.
    """
    count = singular_values.shape[0]
    if count == 0:
        # No observations at all: nothing to retain, and no ``[0]`` to read.
        zero = jnp.asarray(0)
        return jnp.zeros((0,), dtype=bool), zero, zero
    largest = singular_values[0]
    degenerate = ~jnp.isfinite(largest) | (largest <= 0.0)
    safe_largest = jnp.where(degenerate, 1.0, largest)

    # ``_numerical_rank(conservative=False)``: the float32 round-off floor of a
    # thin SVD, floored at _RANK_TOLERANCE_FLOOR * eps so a small ensemble
    # cannot tighten the cut below the measured noise level.
    dimension = max(min(matrix_shape), _RANK_TOLERANCE_FLOOR)
    eps = float(np.finfo(singular_values.dtype).eps)
    available = jnp.where(
        degenerate, 0, jnp.sum(singular_values > largest * eps * dimension)
    )

    # The optional relative floor is *not* part of the scientific truncation:
    # it redefines "numerically zero", so it applies whether or not ``enabled``
    # is set and is folded into ``available`` before the branch below reads it.
    if tsvd.numerical_tolerance is not None:
        available = jnp.minimum(
            available,
            jnp.sum(singular_values > tsvd.numerical_tolerance * safe_largest),
        )

    if not tsvd.enabled:
        if tsvd.numerical_tolerance is None:
            # No scientific cut and no explicit floor: retain every thin-SVD
            # direction, round-off ones included (see
            # :func:`ensemble_transform` for why that is harmless here). That is
            # the untruncated ETKF bit for bit, and ``available`` is reported as
            # a diagnostic only.
            return jnp.ones((count,), dtype=bool), jnp.asarray(count), available
        # A floor *was* asked for. ``available`` already folds in both it and
        # the round-off floor, so this is "retain every numerically nonzero
        # direction" with the caller's definition of nonzero — never a silently
        # ignored knob.
        rank = jnp.where(degenerate, 0, available)
        return jnp.arange(count) < rank, rank, available

    # Suffix energies of the sigma_max-normalized spectrum: ``tail[i]`` is the
    # energy discarded by keeping ``i + 1`` modes. Keep growing the prefix while
    # its complement still exceeds the tolerated share. Strictly ``>``: on a
    # tied spectrum the prefix that leaves *exactly* the tolerated share is
    # already sufficient, which is the same boundary
    # :func:`data_assimilation.reduction._select_rank` takes.
    energy = (singular_values / safe_largest) ** 2
    tolerated = (1.0 - tsvd.energy_fraction) * jnp.sum(energy)
    tail = jnp.cumsum(energy[::-1])[::-1][1:]
    rank = 1 + jnp.sum(tail > tolerated)
    if tsvd.max_rank is not None:
        rank = jnp.minimum(rank, tsvd.max_rank)
    rank = jnp.where(degenerate, 0, jnp.minimum(rank, available))
    return jnp.arange(count) < rank, rank, available


def _retained_energy(singular_values: jnp.ndarray, rank: int) -> float:
    """Share of the whitened spectrum's energy held by the first ``rank`` modes.

    A *readout* of the rank :func:`_retention_mask` already chose, never a
    second criterion — which is why it may safely be computed in float64 (host
    side, once per cycle) while the decision itself is dtype-independent.
    Measured against the whole spectrum, so energy lost to the numerical cut
    shows up as a value below one instead of being normalized away.
    """
    energy = np.square(np.asarray(singular_values, dtype=np.float64))
    total = float(energy.sum())
    if total <= 0.0 or rank <= 0:
        return 0.0
    return float(energy[:rank].sum() / total)


def _transform_factors(
    left: jnp.ndarray,
    singular_values: jnp.ndarray,
    right_t: jnp.ndarray,
    innovation: jnp.ndarray,
    retain: jnp.ndarray,
    num_members: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``(mean_weights, modes, anomaly_scales)`` from a thin SVD and a mask.

    The arithmetic core of :func:`ensemble_transform`, split out **only** so the
    batched LETKF block loop can reuse it under a ``vmap``/``lax.map`` trace.
    :func:`ensemble_transform` itself is not traceable — it materializes the
    rank as a Python ``int`` for its diagnostics and raises a Python error on
    non-finite weights — so a localized analysis that called it per block would
    either fork this formula or run one dispatch per block. Sharing this
    function instead means there is exactly one place where the ETKF/LETKF
    weights are defined, just as :func:`_retention_mask` is the one place where
    ``retain`` is decided.

    Every input is a traced-friendly array and ``retain`` is a fixed-shape
    boolean prefix mask, so a truncated direction contributes ``0`` to ``wbar``
    and the identity (``scale = 1``) to ``W_a`` without any reshaping. See
    :func:`ensemble_transform` for the derivation of both expressions.
    """
    damping = (num_members - 1) + singular_values**2
    mean_coefficients = jnp.where(
        retain, singular_values / damping * (left.T @ innovation), 0.0
    )
    modes = right_t.T  # V, shape (N_e, min(N_d, N_e))
    anomaly_scales = jnp.where(retain, jnp.sqrt((num_members - 1) / damping), 1.0)
    return modes @ mean_coefficients, modes, anomaly_scales


def ensemble_transform(
    whitened_anomalies: jnp.ndarray,
    whitened_innovation: jnp.ndarray,
    tsvd: Optional[ObservationTSVD] = None,
) -> ObservationTransform:
    """The shared ETKF/LETKF kernel: transform weights from ``(Y_w, d_w)``.

    A LETKF block calls this with its own localized whitening (see
    :func:`whiten_observations`); the global ETKF calls it once with everything.

    With ``N = N_e`` and the thin SVD ``Y_w = U S V^T`` (Hunt et al. 2007, in
    SVD form)::

        C     = (N-1) I + Y_w^T Y_w
        W_a   = sqrt(N-1) C^{-1/2}
              = I + V diag(sqrt((N-1)/((N-1)+s^2)) - 1) V^T
        wbar  = C^{-1} Y_w^T d_w
              = V diag(s / ((N-1)+s^2)) U^T d_w

    Both are written as a *correction to the identity* on the retained
    directions, which makes the two structural facts explicit:

    * every direction outside the retained ``V`` columns contributes exactly the
      identity to ``W_a`` and exactly zero to ``wbar``. Truncation therefore
      means "do not assimilate this combination of observations", and leaves
      ``R`` untouched;
    * ``Y_w`` has centered rows, so ``Y_w 1 = 0``, hence ``V^T 1 = 0`` and
      ``W_a 1 = 1``. The posterior mean is exactly ``mean + X wbar``
      *including under truncation* — mean preservation is structural, not a
      tolerance.

    Because the weights are damped at ``s -> 0`` (see :func:`_retention_mask`),
    retaining a round-off direction is numerically harmless. Disabling the TSVD
    therefore keeps every thin-SVD direction rather than truncating at the
    numerical rank: that is the untruncated ETKF bit-for-bit. (An explicit
    ``numerical_tolerance`` still applies when disabled — it defines what
    "numerically zero" means rather than making a scientific choice.)

    The rank comes from :func:`_retention_mask`, the *same* call the LETKF block
    loop makes; only the readout differs (a Python ``int`` here for the
    diagnostics, a traced mask there). This function is consequently not usable
    under a trace — it also ends in a ``bool(...)`` finiteness check — which is
    exactly why the LETKF calls the mask and :func:`_transform_factors`
    directly instead.

    Args:
        whitened_anomalies: ``Y_w``, shape ``(N_d, N_e)``.
        whitened_innovation: ``d_w``, shape ``(N_d,)``.
        tsvd: Optional truncation settings; ``None`` means disabled.

    Returns:
        An :class:`ObservationTransform`.
    """
    tsvd = ObservationTSVD() if tsvd is None else tsvd
    anomalies = jnp.asarray(whitened_anomalies)
    innovation = jnp.asarray(whitened_innovation)
    if anomalies.ndim != 2:
        raise ValueError(
            f"whitened_anomalies must be (N_d, N_e), got shape {anomalies.shape}."
        )
    N_d, N_e = anomalies.shape
    if N_e < 2:
        raise ValueError(
            f"The ensemble transform needs at least 2 members, got N_e={N_e}."
        )
    if innovation.shape != (N_d,):
        raise ValueError(
            f"whitened_innovation must have shape ({N_d},), got {innovation.shape}."
        )

    left, singular_values, right_t = jnp.linalg.svd(anomalies, full_matrices=False)

    # Truncation as a fixed-shape prefix mask: discarded modes get weight 0 in
    # ``wbar`` and scale 1 (the identity) in ``W_a``, which is exactly what
    # dropping their columns would do, without changing any array's shape.
    retain, retained_rank, available = _retention_mask(
        singular_values, anomalies.shape, tsvd
    )
    rank = int(retained_rank)
    available_rank = int(available)
    retained_energy = _retained_energy(singular_values, rank)
    mean_weights, modes, anomaly_scales = _transform_factors(
        left, singular_values, right_t, innovation, retain, N_e
    )

    # Loud failure, matching the stochastic update's contract: a collapsed
    # ensemble or a degenerate C_D must not NaN-poison the analyzed ensemble.
    if not bool(
        jnp.all(jnp.isfinite(mean_weights)) & jnp.all(jnp.isfinite(anomaly_scales))
    ):
        raise FloatingPointError(
            "Ensemble transform produced non-finite weights: the whitened "
            "observation anomalies or the innovation are not finite. Check for "
            "a collapsed/diverged ensemble or a degenerate observation-error "
            "covariance."
        )

    return ObservationTransform(
        mean_weights=mean_weights,
        modes=modes,
        anomaly_scales=anomaly_scales,
        retained_rank=rank,
        discarded_spectrum=singular_values[rank:],
        available_rank=available_rank,
        retained_energy=retained_energy,
    )


def apply_ensemble_transform(
    rows: jnp.ndarray, transform: ObservationTransform
) -> jnp.ndarray:
    """Apply one transform to any set of rows: ``mean + X @ (wbar 1^T + W_a)``.

    Applied in factored form — ``X + (X V) diag(scale - 1) V^T`` — so the dense
    ``N_e x N_e`` matrix is never materialized. Centering explicitly (rather
    than folding ``11^T/N_e`` into one weight matrix) is a float32 accuracy
    choice: physical fields have means far larger than their ensemble
    anomalies, and a single ``rows @ T`` would compute each posterior entry as a
    near-cancelling sum of large numbers.
    """
    rows = jnp.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != transform.num_members:
        raise ValueError(
            f"rows must have shape (N_rows, N_e={transform.num_members}), got "
            f"{rows.shape}."
        )
    mean = jnp.mean(rows, axis=1, keepdims=True)
    anomalies = rows - mean
    projected = anomalies @ transform.modes
    transformed = (
        anomalies + (projected * (transform.anomaly_scales - 1.0)) @ transform.modes.T
    )
    return mean + transformed + (anomalies @ transform.mean_weights)[:, None]


class ETKFAnalysis(AnalysisScheme):
    """Global deterministic ensemble transform Kalman filter analysis.

    Deterministic and unlocalized: one :func:`ensemble_transform` per cycle is
    applied to every augmented row (state, parameters and the appended
    predicted-observation diagnostic rows alike), so the scheme supports
    ``state``/``parameter``/``joint`` modes and PR 1's reduced state
    representation without knowing about any of them.

    ``localization_policy = "forbidden"``: a localized deterministic analysis is
    the separate LETKF class, so a config name can never claim localization the
    analysis silently ignores. :class:`~data_assimilation.filtering.base.\
BaseFilter` enforces this at construction.

    ``rng_key`` is accepted and **ignored** — the transform is deterministic.
    Dropping the argument would break the :class:`AnalysisScheme` contract, and
    raising on it would break the cycle loop, which splits a key every cycle
    regardless of scheme.

    Args:
        tsvd: Optional observation-space spectral truncation (disabled by
            default). See :class:`ObservationTSVD`.
    """

    localization_policy: LocalizationPolicy = "forbidden"

    def __init__(self, tsvd: Optional[ObservationTSVD] = None) -> None:
        self.tsvd = ObservationTSVD() if tsvd is None else tsvd
        # Rank diagnostics of the most recent call. The filter calls the
        # analysis a second time per cycle for the reduction diagnostic, with
        # the same observations and therefore the same transform, so this is
        # the current cycle's value either way.
        self.last_transform: Optional[ObservationTransform] = None

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
        del rng_key, group_ids, localize_mask, row_coords, obs_coords
        if localization is not None:
            # Defense in depth: BaseFilter rejects this combination at
            # construction, but a direct caller must not get a silently
            # unlocalized "localized" analysis.
            raise ValueError(
                f"{type(self).__name__} cannot localize: it applies one global "
                "ensemble transform. Use the LETKF analysis for a localized "
                "deterministic update, or drop the localization strategy."
            )
        augmented = jnp.asarray(augmented)
        pred_obs = jnp.asarray(pred_obs)
        if augmented.ndim != 2 or augmented.shape[1] != pred_obs.shape[1]:
            raise ValueError(
                "augmented must be (N_aug, N_e) with the same N_e as pred_obs, "
                f"got {augmented.shape} and {pred_obs.shape}."
            )

        transform = ensemble_transform(
            *whiten_observations(pred_obs, obs, C_D_diag), tsvd=self.tsvd
        )
        self.last_transform = transform
        if self.tsvd.enabled:
            logger.info(
                "ETKF observation TSVD: rank %d/%s retained (energy %s, largest "
                "discarded singular value %s)",
                transform.retained_rank,
                transform.available_rank,
                (
                    None
                    if transform.retained_energy is None
                    else f"{transform.retained_energy:.4f}"
                ),
                (
                    None
                    if transform.discarded_spectrum.size == 0
                    else f"{float(transform.discarded_spectrum[0]):.3e}"
                ),
            )
        return apply_ensemble_transform(augmented, transform)


# ---------------------------------------------------------------------------
# LETKF: the same kernel, once per distinct local observation selection
# ---------------------------------------------------------------------------

#: Upper bound on how many blocks/rows are vectorized in one chunk. Paired with
#: :data:`_CHUNK_ELEMENT_BUDGET` so the bound is a *memory* budget rather than a
#: count; see :func:`_resolve_chunk_size` for the per-item accounting.
#: 16e6 elements is ~64 MB in float32 — negligible next to an ensemble of CFD
#: states, and small enough to stay in cache-friendly territory. The count cap
#: keeps a tiny-``N_d`` case (the shipped ``N_d = 12``, where the budget alone
#: would allow ~8k blocks per chunk) from building a large gather anyway.
_CHUNK_ELEMENT_BUDGET = 16_000_000
_MAX_CHUNK = 4096


def _resolve_chunk_size(
    num_members: int, num_observations: int, override: Optional[int] = None
) -> int:
    """Chunk size for the block/row loops, as a fixed memory budget.

    The plan asks for a chunk size that is an implementation detail unless one
    fixed value cannot cover the supported grids. A fixed *count* cannot: the
    per-block footprint spans two orders of magnitude across the backends and
    sensor networks. A fixed *element budget* can — but only if it counts the
    arrays that actually dominate. Per item of the block loop
    (``LETKFAnalysis._block_transforms.one_block``), with
    ``r = min(N_d, N_e)``:

    * ``whitened_anomalies``, ``N_d * N_e``;
    * the thin SVD's ``left``, ``N_d * r``, and ``right_t``, ``r * N_e``;
    * the returned ``modes``, ``N_e * r``.

    so ``per_item = N_d * (N_e + r) + 2 * N_e * r``. Both leading terms are
    linear in ``N_d``, which an earlier ``N_e * r`` accounting omitted entirely:
    at ``N_d = 400, N_e = 50`` it under-counted by 16x (41 MB assumed against
    655 MB actual for a 4096-block chunk), and at ``N_d = 2000`` by 80x. The row
    loop (:meth:`LETKFAnalysis._apply_blocks`) gathers ``modes[index]`` per row,
    ``N_e * r``, which the same expression already bounds.

    ``override`` exists for tests and benchmark sweeps, not for configs.

    Args:
        num_members: ``N_e``.
        num_observations: ``N_d`` — the *full* observation dimension. Local
            exclusion is a zero row, not a smaller array, so an inactive
            observation still costs memory.
        override: Explicit chunk size, bypassing the budget.
    """
    if override is not None:
        if int(override) < 1:
            raise ValueError(f"block_chunk_size must be >= 1, got {override}.")
        return int(override)
    members = max(int(num_members), 1)
    observations = max(int(num_observations), 1)
    rank = min(observations, members)
    per_item = max(observations * (members + rank) + 2 * members * rank, 1)
    return int(max(1, min(_MAX_CHUNK, _CHUNK_ELEMENT_BUDGET // per_item)))


@dataclass(frozen=True)
class LocalTransformDiagnostics:
    """What the last :class:`LETKFAnalysis` call actually computed.

    Recorded because the cost and the meaning of a localized deterministic
    analysis are both invisible from its output: two configurations that differ
    by a factor of twenty in work produce identically shaped ensembles. The
    counts are also what the LETKF resource gate in
    ``docs/plans/filtering_state_reduction_and_transforms.md`` asks to be
    reported per cycle, and the four per-block spectral arrays are that plan's
    step-5 TSVD diagnostics — available rank, retained rank, retained energy and
    discarded spectrum — which :class:`ObservationTransform` reports for the one
    global transform and which are needed *per block* exactly where LETKF runs.

    **Per-block arrays are distributions, not scalars.** They are left as arrays
    here, in block order, and summarized to scalars by
    ``BaseFilter._record_transform_diagnostics`` — the same division of labour
    the rank arrays already use, because the per-cycle record must not carry one
    entry per state row (``N_s ~ 2e5`` in production) while an interactive
    caller still wants the full distribution. Every array is materialized as
    **NumPy** rather than left on the device: these are diagnostics nobody
    differentiates through, the call already synchronizes on its finiteness
    check, and ``write_yaml``'s ``_to_native`` converts NumPy scalars but not
    JAX ones, so handing back device arrays makes an easy persistence bug.

    Attributes:
        num_blocks: Distinct local observation selections found this cycle,
            after grouping, masking and deduplication.
        num_active_blocks: How many of those had at least one active
            observation, i.e. how many transforms were actually computed. The
            remainder are provably identity and are skipped.
        num_updated_rows: Augmented rows that belong to an active block. Every
            other row is returned bit-identical to its input.
        row_block: ``(N_aug,)`` block index per augmented row — the value that
            says *which* rows share a transform. Diagnostic rows (and, under
            distance localization, parameter rows) share the single all-ones
            block that is the global ETKF.
        active_observation_counts: ``(num_blocks,)`` number of observations with
            finite inflation per block.
        retained_ranks: ``(num_active_blocks,)`` retained rank per active block,
            ordered by block index over the blocks with a nonzero active count
            (i.e. aligned with
            ``np.flatnonzero(active_observation_counts > 0)``).
        available_ranks: ``(num_active_blocks,)`` numerically nonzero rank per
            active block, in the same order, bounded by
            ``min(N_d_active, N_e - 1)``.
        retained_energies: ``(num_active_blocks,)`` share of that block's
            whitened observation-anomaly energy held by its retained modes,
            measured against the block's *whole* spectrum so energy lost to the
            numerical cut shows as a value below one. ``1.0`` when the block
            retained everything (the untruncated default), ``0.0`` for a block
            whose spectrum is entirely zero — the same convention as the global
            :func:`_retained_energy`.
        discarded_spectrum_maxima: ``(num_active_blocks,)`` largest singular
            value the truncation actually dropped in that block. ``0.0`` means
            "nothing was discarded", matching the global
            ``transform_discarded_spectrum_max``; there is no ``None`` here
            because an inactive block is *absent* from these arrays rather than
            present with a placeholder.
        chunk_size: Blocks/rows vectorized per chunk (see
            :func:`_resolve_chunk_size`).
    """

    num_blocks: int
    num_active_blocks: int
    num_updated_rows: int
    row_block: np.ndarray
    active_observation_counts: np.ndarray
    retained_ranks: np.ndarray
    available_ranks: np.ndarray
    retained_energies: np.ndarray
    discarded_spectrum_maxima: np.ndarray
    chunk_size: int


class LETKFAnalysis(AnalysisScheme):
    """Localized deterministic ensemble transform (LETKF) analysis.

    One :func:`ensemble_transform` per *distinct local observation selection*,
    applied to the rows that share it. The localization strategy supplies the
    selection through
    :meth:`~data_assimilation.localization.base.BaseLocalization.\
inflation_factors`, exactly as it does for the stochastic localized update, and
    the effective observation-error covariance of a block is
    ``R_eff = C_D * E_inf**2`` — the same convention, so switching
    ``filtering/analysis`` between the stochastic scheme and this one changes
    the estimator and nothing about what "localization radius" means. Only
    ``R`` is inflated; the ensemble term is never touched.

    Three implementation choices are load-bearing at production size
    (``N_s ~ 230k``, ``N_e = 50``, ``N_d = 12``):

    1. **Deduplicate on the inflation vector, not on the block id.** The
       transform is a function of ``(pred_obs, obs, C_D, E_inf_row)`` alone, so
       any two rows with the same effective inflation share it. This subsumes
       ``group_ids`` grouping (co-located rows share an inflation row by
       construction) and does considerably more: at the shipped
       ``localization_radius`` roughly 95 percent of the domain sees no
       observation at all, and *every* such row collapses into one block, as do
       all globally updated rows. Grouping alone would deduplicate nothing on
       the staggered uDALES grid, where each variable already has its own
       ``group_ids`` range.
    2. **Skip the blocks with no active observation.** Their transform is
       provably the identity, so their rows are returned *bit-identical* rather
       than recomputed — which a ``mean + (row - mean)`` round trip would not
       be in float32. Nothing in this package is jitted, so the partition is a
       plain host-side NumPy mask.
    3. **Never assemble an ``N_e x N_e`` transform.** Weights are kept in the
       factored ``(V, scale, wbar)`` form of :class:`ObservationTransform`:
       ``N_e*r + N_e + r = 662`` floats per block at the shape above against
       ``N_e**2 = 2500`` dense, applied in ``2*N_e*r + N_e`` multiply-adds per
       row against ``N_e**2`` — and without the ``N_e**2 * r`` it would take to
       assemble each block's matrix in the first place. What bounds the table
       is choices 1 and 2 rather than any of those ratios: it holds one triple
       per *distinct active* block, not one per row. (For scale: the shipped
       case has ``N_s = 230,400`` rows and, at ``localization_radius = 7.5``,
       roughly 5 percent of blocks with an active observation — both recorded
       in ``docs/plans/filtering_state_reduction_and_transforms.md`` §6,
       corrections #1 and #4. How far the distinct-inflation dedup collapses
       the block count on a real run is one of the quantities
       ``docs/temp/filtering_ensemble_transform_benchmark.md`` exists to
       measure; it has not been measured yet.) Bounded chunking is a safety
       bound on top of that, not the mechanism.

    ``localization_policy = "required"``: without a strategy this is just a
    slower global ETKF, so :class:`~data_assimilation.filtering.base.BaseFilter`
    rejects the combination at construction. Combining LETKF with PR 1's global
    state reduction is rejected structurally by the same constructor, which
    already refuses ``state_reduction`` together with any localization; the
    reduction's global basis has no spatial support and cannot be localized.

    ``rng_key`` is accepted and **ignored** — the transform is deterministic.

    Args:
        tsvd: Optional observation-space spectral truncation, applied *per
            block* and therefore after that block's observation selection and
            whitening (the plan's ``localize -> R_eff -> whiten -> TSVD ->
            transform`` order). Disabled by default.
        block_chunk_size: Blocks/rows per vectorized chunk. ``None`` (the
            default, and what configs should use) derives it from a fixed
            memory budget; an explicit value exists for tests and benchmarks.
    """

    localization_policy: LocalizationPolicy = "required"

    def __init__(
        self,
        tsvd: Optional[ObservationTSVD] = None,
        block_chunk_size: Optional[int] = None,
    ) -> None:
        self.tsvd = ObservationTSVD() if tsvd is None else tsvd
        if block_chunk_size is not None and int(block_chunk_size) < 1:
            raise ValueError(f"block_chunk_size must be >= 1, got {block_chunk_size}.")
        self.block_chunk_size = (
            None if block_chunk_size is None else int(block_chunk_size)
        )
        #: Diagnostics of the most recent call; see
        #: :class:`LocalTransformDiagnostics`.
        self.last_diagnostics: Optional[LocalTransformDiagnostics] = None

    def _block_transforms(
        self,
        block_inflation: jnp.ndarray,
        pred_obs: jnp.ndarray,
        obs: jnp.ndarray,
        C_D_diag: jnp.ndarray,
        chunk_size: int,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        """Factored transforms for every active block, in bounded chunks.

        Returns stacked ``(mean_weights, modes, anomaly_scales, retained_rank,
        available_rank, retained_energy, discarded_spectrum_max)``. The inner
        function returns the raw factor triple rather than an
        :class:`ObservationTransform`, because that dataclass is not a
        registered pytree and registering it would put a global,
        rarely-exercised conversion on every ETKF call for the benefit of this
        one loop.

        The last four are the plan's step-5 TSVD diagnostics, computed *inside*
        the trace as reductions over the same fixed-rank spectrum and retention
        mask the transform already uses. They are deliberately not recovered
        afterwards from a second spectrum: the whitened anomalies of a block are
        never materialized outside this loop, so recomputing them per block
        would cost a second SVD each, while these two extra reductions over a
        ``min(N_d, N_e)``-length vector are free next to that SVD.
        """
        num_members = int(pred_obs.shape[1])
        tsvd = self.tsvd

        def one_block(
            inflation_row: jnp.ndarray,
        ) -> tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ]:
            # localize -> R_eff -> whiten -> TSVD -> transform. Excluded
            # observations arrive as ``inf`` and leave as exactly zero rows of
            # ``Y_w``/``d_w``; the shape never depends on how many are active,
            # which is what makes the block loop vectorizable at all.
            whitened_anomalies, whitened_innovation = _whiten(
                pred_obs, obs, C_D_diag, error_inflation=inflation_row
            )
            left, singular_values, right_t = jnp.linalg.svd(
                whitened_anomalies, full_matrices=False
            )
            retain, rank, available = _retention_mask(
                singular_values, whitened_anomalies.shape, tsvd
            )
            mean_weights, modes, anomaly_scales = _transform_factors(
                left, singular_values, right_t, whitened_innovation, retain, num_members
            )
            # Retained energy against the block's whole spectrum, and the
            # largest singular value the truncation actually dropped. Both are
            # traceable reductions with no host synchronization: no ``bool()``,
            # no ``int()``, no shape that depends on the rank. The zero-spectrum
            # guard mirrors :func:`_retained_energy`, so a collapsed block
            # reports 0.0 instead of NaN and the two readouts of one criterion
            # stay two readouts.
            #
            # Normalized by sigma_max first, exactly as :func:`_retention_mask`
            # does. The ratio is invariant under it, but the float32
            # intermediate then cannot overflow to ``inf`` (and the ratio to
            # NaN) on a spectrum whose squares do not fit — which is the only
            # way this float32 readout could report something the host-side
            # float64 :func:`_retained_energy` would not.
            scale = jnp.where(singular_values[0] > 0.0, singular_values[0], 1.0)
            energy = (singular_values / scale) ** 2
            total = jnp.sum(energy)
            retained_energy = jnp.sum(jnp.where(retain, energy, 0.0)) / jnp.where(
                total > 0.0, total, 1.0
            )
            discarded_max = jnp.max(jnp.where(retain, 0.0, singular_values))
            return (
                mean_weights,
                modes,
                anomaly_scales,
                rank,
                available,
                retained_energy,
                discarded_max,
            )

        # ``batch_size`` makes lax.map a vmap-inside-scan, so peak memory is
        # bounded by the chunk rather than by the block count. jax is pinned to
        # >=0.7, where the argument is built in.
        stacked: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
        ] = jax.lax.map(one_block, block_inflation, batch_size=chunk_size)
        return stacked

    def _apply_blocks(
        self,
        rows: jnp.ndarray,
        block_index: jnp.ndarray,
        mean_weights: jnp.ndarray,
        modes: jnp.ndarray,
        anomaly_scales: jnp.ndarray,
        chunk_size: int,
    ) -> jnp.ndarray:
        """Apply each row's own block transform, in bounded chunks.

        Literally :func:`apply_ensemble_transform` per row, with the transform
        gathered from the block table — including the explicit centering, which
        is the float32 accuracy choice documented there and matters more here,
        not less: a localized posterior differs from its prior in the sixth
        significant digit for most rows.
        """

        def one_row(item: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
            row, index = item
            mean = jnp.mean(row)
            anomalies = row - mean
            block_modes = modes[index]  # (N_e, r)
            projected = anomalies @ block_modes
            transformed = (
                anomalies + (projected * (anomaly_scales[index] - 1.0)) @ block_modes.T
            )
            return mean + transformed + anomalies @ mean_weights[index]

        # Annotated because ``lax.map`` is typed as returning ``Any`` (its
        # output structure follows ``f``), which mypy would otherwise report as
        # an ``Any`` escaping this ``-> jnp.ndarray`` signature.
        posterior: jnp.ndarray = jax.lax.map(
            one_row, (rows, block_index), batch_size=chunk_size
        )
        return posterior

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
        del rng_key
        if localization is None:
            # Defense in depth: BaseFilter rejects this at construction, but a
            # direct caller must not silently get a global ETKF from a class
            # whose name promises localization.
            raise ValueError(
                f"{type(self).__name__} requires a localization strategy: "
                "without one it is exactly the global ETKF, so use "
                "ETKFAnalysis instead of a localized class that localizes "
                "nothing."
            )
        augmented = jnp.asarray(augmented)
        pred_obs = jnp.asarray(pred_obs)
        obs = jnp.asarray(obs)
        if augmented.ndim != 2 or pred_obs.ndim != 2:
            raise ValueError(
                "augmented must be (N_aug, N_e) and pred_obs (N_d, N_e), got "
                f"{augmented.shape} and {pred_obs.shape}."
            )
        N_aug, N_e = augmented.shape
        N_d = pred_obs.shape[0]
        if pred_obs.shape[1] != N_e:
            raise ValueError(
                "augmented and pred_obs must share the ensemble dimension, got "
                f"{augmented.shape} and {pred_obs.shape}."
            )
        if obs.ndim != 1 or obs.shape[0] != N_d:
            # Hoisted here with the other shape checks: inside the traced block
            # loop a wrong-length ``obs`` surfaces as a broadcasting TypeError
            # from the middle of ``_whiten`` instead of this message.
            raise ValueError(
                f"obs must be a 1-D vector of length N_d={N_d}, got shape "
                f"{obs.shape}."
            )
        C_D_diag = validate_variances(C_D_diag)
        if C_D_diag.shape[0] != N_d:
            raise ValueError(
                f"C_D_diag must be a 1-D variance vector of length N_d={N_d}, "
                f"got shape {C_D_diag.shape}."
            )

        aug_dev = augmented - jnp.mean(augmented, axis=1, keepdims=True)
        pred_obs_dev = pred_obs - jnp.mean(pred_obs, axis=1, keepdims=True)
        inflation = localization.inflation_factors(
            aug_dev, pred_obs_dev, row_coords=row_coords, obs_coords=obs_coords
        )
        if inflation.shape != (N_aug, N_d):
            raise ValueError(
                f"{type(localization).__name__}.inflation_factors must return "
                f"shape ({N_aug}, {N_d}), got {tuple(inflation.shape)}."
            )
        # Grouping then masking, in that order, from the shared helper the
        # localized stochastic update also calls: masked rows are excluded from
        # the block ``segment_min`` and only then forced to all ones. All-ones
        # *is* the global ETKF selection, which is why LETKF needs no separate
        # global code path — the appended predicted-observation diagnostic rows
        # (always masked) and, under distance localization, the parameter rows
        # all collapse into one block whose transform is the global one.
        row_inflation = resolve_row_inflation(inflation, group_ids, localize_mask)

        # --- host-side block partition -------------------------------------
        # Canonicalize first: the analysis treats any non-finite or non-positive
        # inflation as "excluded" (:func:`active_observations`), so two rows
        # that differ only in *how* they spell exclusion are the same block.
        # Mapping them all to ``inf`` makes the deduplication exactly as coarse
        # as the analysis.
        table = np.asarray(row_inflation)
        active = np.isfinite(table) & (table > 0.0)
        canonical = np.where(active, table, np.inf)
        unique_inflation, row_block = np.unique(canonical, axis=0, return_inverse=True)
        # NumPy 2.0 briefly returned an inverse shaped like the input.
        row_block = np.asarray(row_block).reshape(-1)
        active_counts = np.isfinite(unique_inflation).sum(axis=1)
        active_blocks = np.flatnonzero(active_counts > 0)

        chunk_size = _resolve_chunk_size(N_e, N_d, self.block_chunk_size)

        if active_blocks.size == 0:
            # No row sees any observation: the analysis is the identity and the
            # ensemble must come back untouched, not round-tripped.
            #
            # Skipping the transform must not also skip the failure contract.
            # Every other path in the package raises on non-finite observation
            # inputs (they reach the weights and trip the check below), so this
            # shortcut checks them explicitly rather than being the one place
            # where a diverged forecast returns quietly. ``augmented`` is not
            # checked, here or anywhere: the transform never looks at it.
            if not bool(jnp.all(jnp.isfinite(pred_obs)) & jnp.all(jnp.isfinite(obs))):
                raise FloatingPointError(
                    "Localized ensemble transform received non-finite predicted "
                    "observations or observations. No block had an active "
                    "observation, so the analysis would have been the identity; "
                    "failing instead, because a diverged forecast must not be "
                    "reported as a successful no-op analysis."
                )
            self.last_diagnostics = LocalTransformDiagnostics(
                num_blocks=int(unique_inflation.shape[0]),
                num_active_blocks=0,
                num_updated_rows=0,
                row_block=row_block,
                active_observation_counts=active_counts,
                retained_ranks=np.zeros((0,), dtype=int),
                available_ranks=np.zeros((0,), dtype=int),
                retained_energies=np.zeros((0,), dtype=float),
                discarded_spectrum_maxima=np.zeros((0,), dtype=float),
                chunk_size=chunk_size,
            )
            return augmented

        (
            mean_weights,
            modes,
            anomaly_scales,
            ranks,
            available,
            retained_energies,
            discarded_maxima,
        ) = self._block_transforms(
            jnp.asarray(unique_inflation[active_blocks]),
            pred_obs,
            obs,
            C_D_diag,
            chunk_size,
        )
        # One finiteness check over every block, matching the global path's
        # contract: a collapsed ensemble must fail loudly instead of
        # NaN-poisoning the analysis. Done here rather than inside the loop
        # because the loop is traced.
        if not bool(
            jnp.all(jnp.isfinite(mean_weights)) & jnp.all(jnp.isfinite(anomaly_scales))
        ):
            raise FloatingPointError(
                "Localized ensemble transform produced non-finite weights: some "
                "block's whitened observation anomalies or innovation are not "
                "finite. Check for a collapsed/diverged ensemble or a "
                "degenerate observation-error covariance."
            )

        # Rows of an inactive block keep their exact input values.
        position = np.full(unique_inflation.shape[0], -1, dtype=np.int64)
        position[active_blocks] = np.arange(active_blocks.size)
        row_position = position[row_block]
        updated_rows = np.flatnonzero(row_position >= 0)

        # One device->host transfer of four ``(num_active_blocks,)`` vectors.
        # The call has already synchronized on the finiteness check above, and
        # every consumer of these arrays (the summary in ``BaseFilter``, the log
        # line below, tests) is host-side — see the class docstring for why they
        # must not leave as JAX arrays.
        block_ranks = np.asarray(ranks)
        block_available = np.asarray(available)
        block_energies = np.asarray(retained_energies, dtype=float)
        block_discarded = np.asarray(discarded_maxima, dtype=float)

        self.last_diagnostics = LocalTransformDiagnostics(
            num_blocks=int(unique_inflation.shape[0]),
            num_active_blocks=int(active_blocks.size),
            num_updated_rows=int(updated_rows.size),
            row_block=row_block,
            active_observation_counts=active_counts,
            retained_ranks=block_ranks,
            available_ranks=block_available,
            retained_energies=block_energies,
            discarded_spectrum_maxima=block_discarded,
            chunk_size=chunk_size,
        )
        if self.tsvd.enabled:
            logger.info(
                "LETKF observation TSVD: %d active of %d blocks, retained rank "
                "min/median/max %d/%d/%d of available max %d; retained energy "
                "min %.4f, largest discarded singular value %.3e",
                int(active_blocks.size),
                int(unique_inflation.shape[0]),
                int(block_ranks.min()),
                int(np.median(block_ranks)),
                int(block_ranks.max()),
                int(block_available.max()),
                float(block_energies.min()),
                float(block_discarded.max()),
            )

        if updated_rows.size == N_aug:
            # Every row is updated (a strategy that excludes nothing, or a
            # small problem): skip the gather/scatter round trip entirely.
            return self._apply_blocks(
                augmented,
                jnp.asarray(row_position),
                mean_weights,
                modes,
                anomaly_scales,
                chunk_size,
            )
        row_index = jnp.asarray(updated_rows)
        posterior = self._apply_blocks(
            augmented[row_index],
            jnp.asarray(row_position[updated_rows]),
            mean_weights,
            modes,
            anomaly_scales,
            chunk_size,
        )
        return augmented.at[row_index].set(posterior)
