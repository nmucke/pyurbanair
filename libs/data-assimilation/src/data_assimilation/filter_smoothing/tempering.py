"""Beta tempering of the hybrid: how one observation's influence is split.

The filter-smoothing hybrid assimilates every raw observation TWICE — once in
the ESMDA phase (the window's concatenated vector, over ``num_steps`` tempered
updates) and once in the filter phase (frame by frame). This module resolves,
from two scalars, how strongly each phase is allowed to use it. It is pure: no
arrays of the run, no collaborators, nothing mutated — the run script resolves
a policy once (before any forecast) and hands the same frozen object to both
constructors and to :class:`~data_assimilation.filter_smoothing.FilterSmoothing`,
which checks that the collaborators were built with matching weights.

Two knobs, both multipliers of a PHYSICAL covariance ``R`` that is never
changed itself:

* **filter beta** — every filter analysis uses ``beta R``
  (``BaseFilter.effective_C_D_diag``), i.e. the likelihood ``L^(1/beta)``.
  ``filter_weight = 1/beta``.
* **smoother likelihood weight** ``w`` — every ESMDA update uses
  ``alpha_effective = alpha_base / w`` on the NORMALIZED base schedule
  (``sum 1/alpha_base = 1``), so the whole MDA loop conditions on ``L^w``.

Two named allocations:

``filter_only`` (default)
    ``w = 1``: ESMDA keeps its full, normalized schedule and only the filter is
    tempered. For a repeatedly-used likelihood factor the nominal combined
    exponent is ``1 + 1/beta`` — no finite beta makes it one full likelihood.
    Conservative damping, not likelihood accounting; ``beta = 1`` is exactly
    the legacy hybrid.
``shared_budget`` (opt-in research mode)
    ``w = (beta - 1)/beta``, so ``sum 1/alpha_effective + 1/beta = 1``: each
    reused observation's unit budget is split between the phases (beta 2 →
    half each, beta 4 → three quarters to ESMDA). Requires ``beta > 1`` (at
    ``beta = 1`` the parameter stage would get a zero budget) and the SAME raw
    observation product in both phases; the latter is checked by the hybrid,
    not here. This is a NOMINAL allocation: parameter-only ESMDA followed by a
    state filter is not a factorization of the joint posterior in general.

``w`` is computed as ``(beta - 1)/beta`` rather than ``1 - 1/beta``: the two
are equal in exact arithmetic, but the latter cancels catastrophically near
``beta = 1``.
"""

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, get_args

import jax
import numpy as np

# Beta is validated by the FILTER's own rule, so the policy and the collaborator
# it configures can never disagree about what a valid beta is.
from data_assimilation.filtering import validate_beta
from numpy.typing import ArrayLike, DTypeLike

LikelihoodAllocation = Literal["filter_only", "shared_budget"]

#: Every accepted ``likelihood_allocation`` value, in documentation order.
LIKELIHOOD_ALLOCATIONS: tuple[str, ...] = get_args(LikelihoodAllocation)


def _smoother_weight(beta: float, allocation: str) -> float:
    """The ESMDA likelihood weight an allocation assigns, in float64."""
    if allocation == "filter_only":
        return 1.0
    # (beta - 1)/beta, not 1 - 1/beta: see the module docstring.
    return (beta - 1.0) / beta


def _default_dtype() -> np.dtype:
    """The float dtype the JAX analyses actually run in (float32 unless x64)."""
    return np.dtype(jax.dtypes.canonicalize_dtype(np.float64))


def _finite_in(value: float, dtype: np.dtype) -> bool:
    """Whether ``value`` survives the cast to ``dtype`` as a finite number."""
    with np.errstate(over="ignore", invalid="ignore"):
        return bool(np.isfinite(np.asarray(value, dtype=dtype)))


@dataclass(frozen=True)
class TemperingPolicy:
    """A resolved, immutable split of the observation influence between phases.

    Build it with :func:`resolve_tempering_policy`; direct construction is
    validated too (``__post_init__``), so a policy whose ``smoother_weight``
    disagrees with its ``beta``/``likelihood_allocation`` cannot exist.

    Attributes:
        beta: Filter covariance multiplier, ``R_filter = beta R``.
        likelihood_allocation: ``"filter_only"`` or ``"shared_budget"``.
        smoother_weight: ESMDA likelihood weight ``w``; every ESMDA update
            uses ``alpha_base / w``.
    """

    beta: float
    likelihood_allocation: LikelihoodAllocation
    smoother_weight: float

    def __post_init__(self) -> None:
        beta = validate_beta(self.beta)
        allocation = _validate_allocation(self.likelihood_allocation)
        if allocation == "shared_budget" and not beta > 1.0:
            raise ValueError(
                "likelihood_allocation='shared_budget' requires beta > 1, got "
                f"beta={beta}. At beta = 1 the filter takes the whole budget "
                "and ESMDA's share (beta - 1)/beta is zero, i.e. the parameter "
                "stage would assimilate nothing. Use a beta > 1, or "
                "likelihood_allocation='filter_only' (the legacy hybrid)."
            )
        expected = _smoother_weight(beta, allocation)
        if self.smoother_weight != expected:
            raise ValueError(
                f"smoother_weight={self.smoother_weight!r} does not match "
                f"{allocation!r} at beta={beta} (expected {expected!r}). The "
                "weight is derived, never configured; build the policy with "
                "resolve_tempering_policy."
            )
        if not expected > 0.0:
            raise ValueError(
                f"beta={beta} is so close to 1 that the shared-budget ESMDA "
                "weight (beta - 1)/beta underflows to 0."
            )
        # Normalize numpy scalars / ints to plain floats for stable equality
        # and YAML serialization (frozen, hence object.__setattr__).
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "smoother_weight", float(expected))

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def filter_weight(self) -> float:
        """The filter phase's likelihood exponent, ``1/beta``."""
        return 1.0 / self.beta

    @property
    def nominal_combined_exponent(self) -> float:
        """Total nominal exponent on one reused observation's likelihood.

        ``1 + 1/beta`` under ``filter_only`` (ESMDA's full unit plus the
        filter's share) and exactly ``1`` under ``shared_budget``. NOMINAL:
        it counts likelihood factors, and says nothing about whether the
        hybrid's sequence of updates reproduces the joint posterior.
        """
        if self.likelihood_allocation == "filter_only":
            return 1.0 + self.filter_weight
        return 1.0

    @property
    def is_legacy(self) -> bool:
        """``filter_only`` at ``beta = 1``: bitwise the pre-tempering hybrid."""
        return self.likelihood_allocation == "filter_only" and self.beta == 1.0

    def effective_alpha(self, base_alpha: float) -> float:
        """ESMDA's per-update coefficient: ``alpha_base / smoother_weight``."""
        return float(base_alpha) / self.smoother_weight

    # ------------------------------------------------------------------
    # Numerics
    # ------------------------------------------------------------------

    def check_numerics(
        self,
        *,
        base_alpha: Optional[float] = None,
        variances: Optional[ArrayLike] = None,
        dtype: Optional[DTypeLike] = None,
    ) -> None:
        """Reject a policy that overflows/underflows in the computation dtype.

        Finite configuration values are not enough: the analyses run in JAX's
        default float (float32 unless x64 is enabled), where a large beta or a
        tiny shared weight can still turn ``beta R`` or ``alpha_effective R``
        into ``inf`` — which the solvers would silently propagate as NaN — or
        round the weight itself to zero.

        Args:
            base_alpha: ESMDA's base (normalized-schedule) alpha; checks
                ``alpha_effective`` and ``alpha_effective * max(R)``.
            variances: Physical observation-error variances (any shape; only
                the extremes matter); checks ``beta * max(R)`` and, with
                ``base_alpha``, ``alpha_effective * max(R)``.
            dtype: Computation dtype; default JAX's canonical float.
        """
        target = np.dtype(dtype) if dtype is not None else _default_dtype()
        if not _finite_in(self.beta, target):
            raise ValueError(f"beta={self.beta} overflows {target}.")
        if not float(np.asarray(self.smoother_weight, dtype=target)) > 0.0:
            raise ValueError(
                f"The ESMDA likelihood weight {self.smoother_weight!r} "
                f"(beta={self.beta}, {self.likelihood_allocation}) rounds to "
                f"zero in {target}."
            )
        if not float(np.asarray(self.filter_weight, dtype=target)) > 0.0:
            raise ValueError(
                f"The filter weight 1/beta = {self.filter_weight!r} rounds to "
                f"zero in {target}; beta={self.beta} is too large."
            )
        alpha_eff: Optional[float] = None
        if base_alpha is not None:
            alpha = float(base_alpha)
            if not (math.isfinite(alpha) and alpha > 0.0):
                raise ValueError(f"base_alpha must be finite and > 0, got {alpha}.")
            alpha_eff = self.effective_alpha(alpha)
            if not (math.isfinite(alpha_eff) and _finite_in(alpha_eff, target)):
                raise ValueError(
                    f"The effective ESMDA alpha {alpha} / {self.smoother_weight} "
                    f"= {alpha_eff} overflows {target} (beta={self.beta})."
                )
        if variances is None:
            return
        values = np.asarray(variances, dtype=float)
        if values.size == 0:
            return
        largest = float(np.max(values))
        scaled = [("filter", "beta", self.beta)]
        if alpha_eff is not None:
            scaled.append(("ESMDA", "alpha_effective", alpha_eff))
        for phase, name, factor in scaled:
            product = factor * largest
            if not (math.isfinite(product) and _finite_in(product, target)):
                raise ValueError(
                    f"The {phase} phase's effective observation covariance "
                    f"{name} * R = {factor:g} * {largest:g} overflows {target}; "
                    "the analysis would divide by inf and NaN-poison the "
                    "ensemble. Lower beta or rescale the observations."
                )

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def metadata(
        self,
        *,
        base_alpha: Optional[float] = None,
        num_steps: Optional[int] = None,
    ) -> dict[str, Any]:
        """Plain-Python record of the policy, for run summaries.

        With ``base_alpha`` the scalar base/effective alphas are added; with
        ``num_steps`` too, the per-step schedules and ESMDA's realized share
        ``sum 1/alpha_effective`` (``== smoother_weight`` for the normalized
        scalar schedule).
        """
        record: dict[str, Any] = {
            "beta": float(self.beta),
            "likelihood_allocation": str(self.likelihood_allocation),
            "filter_weight": float(self.filter_weight),
            "smoother_weight": float(self.smoother_weight),
            "nominal_combined_exponent": float(self.nominal_combined_exponent),
        }
        if base_alpha is not None:
            alpha = float(base_alpha)
            alpha_eff = self.effective_alpha(alpha)
            record["base_alpha"] = alpha
            record["effective_alpha"] = float(alpha_eff)
            if num_steps is not None:
                steps = int(num_steps)
                record["base_alphas"] = [alpha] * steps
                record["effective_alphas"] = [float(alpha_eff)] * steps
                record["smoother_likelihood_share"] = float(steps / alpha_eff)
        return record


def _validate_allocation(likelihood_allocation: Any) -> LikelihoodAllocation:
    if (
        not isinstance(likelihood_allocation, str)
        or likelihood_allocation not in LIKELIHOOD_ALLOCATIONS
    ):
        raise ValueError(
            f"likelihood_allocation={likelihood_allocation!r} is not one of "
            f"{list(LIKELIHOOD_ALLOCATIONS)}."
        )
    return likelihood_allocation  # type: ignore[return-value]


def resolve_tempering_policy(
    beta: Any = 1.0,
    likelihood_allocation: Any = "filter_only",
    *,
    base_alpha: Optional[float] = None,
    variances: Optional[ArrayLike] = None,
    dtype: Optional[DTypeLike] = None,
) -> TemperingPolicy:
    """Resolve ``(beta, likelihood_allocation)`` into a validated policy.

    The single source of both phase weights: resolve ONCE, before building
    either collaborator, and construct the filter with ``beta=policy.beta`` and
    the smoother with ``likelihood_weight=policy.smoother_weight``.

    Rules:

    * both allocations: ``beta`` must be a finite real ``>= 1`` (NaN, inf,
      ``bool`` and strings rejected);
    * ``filter_only``: ``smoother_weight = 1``;
    * ``shared_budget``: ``beta > 1`` strictly, ``smoother_weight =
      (beta - 1)/beta``;
    * unknown allocations are rejected;
    * :meth:`TemperingPolicy.check_numerics` always runs in ``dtype`` (JAX's
      canonical float by default), including the effective-alpha/covariance
      overflow checks when ``base_alpha``/``variances`` are given.

    The observation-product requirements of ``shared_budget`` (no smoother
    aggregation, identical raw frames/operator/covariance in both phases) need
    the collaborators and the data, so :class:`FilterSmoothing` checks them.
    """
    value = validate_beta(beta)
    allocation = _validate_allocation(likelihood_allocation)
    policy = TemperingPolicy(
        beta=value,
        likelihood_allocation=allocation,
        smoother_weight=(
            _smoother_weight(value, allocation)
            if allocation == "filter_only" or value > 1.0
            # Let __post_init__ raise its beta > 1 message.
            else 0.0
        ),
    )
    policy.check_numerics(base_alpha=base_alpha, variances=variances, dtype=dtype)
    return policy
