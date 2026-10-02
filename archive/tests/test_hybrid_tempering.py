"""Beta tempering of the filter-smoothing hybrid (docs/plans/hybrid_beta_tempering.md).

Three groups, none touching a CFD solver:

* the pure policy resolver (``data_assimilation.filter_smoothing.tempering``):
  the plan's worked numbers, the budget identity, and every rejection;
* ANALYTICAL oracles for what the two allocations mean for a linear-Gaussian
  problem. The hybrid's phases are composed as ideal (infinite-ensemble)
  perturbed-observation updates with the RESOLVED coefficients — ESMDA
  ``num_steps`` updates with ``alpha_effective R`` on the parameter rows, then
  one filter update with ``beta R`` — and compared against once-conditioned
  Gaussian posteriors. These are statements about the allocation, not about any
  library kernel;
* the same x = theta problem through the REAL ``FilterSmoothing`` with a
  large ensemble (stochastic ESMDA + deterministic ETKF): exact tolerances for
  the transform, statistical ones for the MDA.
"""

import dataclasses
import math
from fractions import Fraction
from typing import Any, Optional, Sequence

import numpy as np
import pytest
from data_assimilation.filter_smoothing import (
    LIKELIHOOD_ALLOCATIONS,
    TemperingPolicy,
    resolve_tempering_policy,
)

# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def test_allocations_are_the_two_named_policies() -> None:
    assert LIKELIHOOD_ALLOCATIONS == ("filter_only", "shared_budget")


def test_default_policy_is_the_legacy_hybrid() -> None:
    policy = resolve_tempering_policy()
    assert policy.beta == 1.0
    assert policy.likelihood_allocation == "filter_only"
    assert policy.smoother_weight == 1.0
    assert policy.filter_weight == 1.0
    assert policy.is_legacy
    # Legacy double counting of every reused observation, stated explicitly.
    assert policy.nominal_combined_exponent == 2.0
    # A unit weight leaves the base alpha untouched, bit for bit.
    assert policy.effective_alpha(4.0) == 4.0


@pytest.mark.parametrize("beta", [1.0, 1.5, 2.0, 8.0, 1e6])  # type: ignore[misc]
def test_filter_only_keeps_full_esmda_weight(beta: float) -> None:
    policy = resolve_tempering_policy(beta, "filter_only")
    assert policy.smoother_weight == 1.0
    assert policy.filter_weight == 1.0 / beta
    assert policy.nominal_combined_exponent == 1.0 + 1.0 / beta
    assert policy.is_legacy == (beta == 1.0)


def test_worked_example_four_steps_beta_two() -> None:
    """Plan: four steps, beta 2 -> base alpha 4, effective 8, half each."""
    policy = resolve_tempering_policy(2.0, "shared_budget")
    assert policy.smoother_weight == 0.5
    assert policy.filter_weight == 0.5
    assert policy.effective_alpha(4.0) == 8.0
    assert not policy.is_legacy
    meta = policy.metadata(base_alpha=4.0, num_steps=4)
    assert meta["base_alphas"] == [4.0] * 4
    assert meta["effective_alphas"] == [8.0] * 4
    assert meta["smoother_likelihood_share"] == 0.5


def test_worked_example_beta_four() -> None:
    """Plan: beta 4 assigns three quarters to ESMDA and one to filtering."""
    policy = resolve_tempering_policy(4, "shared_budget")
    assert policy.beta == 4.0 and isinstance(policy.beta, float)
    assert policy.smoother_weight == 0.75
    assert policy.filter_weight == 0.25


@pytest.mark.parametrize("num_steps", [1, 2, 4, 7])  # type: ignore[misc]
@pytest.mark.parametrize("beta", [1.0 + 1e-9, 1.25, 2.0, 3.0, 4.0, 8.0, 1e3])  # type: ignore[misc]
def test_shared_budget_sums_to_one(num_steps: int, beta: float) -> None:
    """``sum 1/alpha_effective + 1/beta = 1`` on the normalized base schedule."""
    policy = resolve_tempering_policy(beta, "shared_budget")
    base_alpha = float(num_steps)  # the default, normalized schedule
    alpha_eff = policy.effective_alpha(base_alpha)
    total = num_steps / alpha_eff + 1.0 / beta
    assert total == pytest.approx(1.0, rel=1e-12)
    assert policy.nominal_combined_exponent == 1.0


def test_shared_weight_is_accurate_near_one() -> None:
    """``(beta - 1)/beta`` keeps full relative accuracy where ``1 - 1/beta``
    loses it to cancellation."""
    beta = 1.0 + 2.0**-40
    policy = resolve_tempering_policy(beta, "shared_budget")
    exact = (Fraction(beta) - 1) / Fraction(beta)
    rel_err = abs(Fraction(policy.smoother_weight) - exact) / exact
    assert rel_err < Fraction(1, 10**15)


@pytest.mark.parametrize("allocation", ["filter_only", "shared_budget"])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "beta",
    [math.nan, math.inf, -math.inf, 0.0, 0.5, 0.999999, -2.0, True, False, "2", None],
)
def test_invalid_beta_rejected_in_both_policies(allocation: str, beta: object) -> None:
    with pytest.raises(ValueError, match="beta"):
        resolve_tempering_policy(beta, allocation)


def test_shared_budget_rejects_beta_one() -> None:
    with pytest.raises(ValueError, match="zero"):
        resolve_tempering_policy(1.0, "shared_budget")


@pytest.mark.parametrize(  # type: ignore[misc]
    "allocation", ["shared", "FILTER_ONLY", "", None, 1, "filter-only"]
)
def test_unknown_allocation_rejected(allocation: object) -> None:
    with pytest.raises(ValueError, match="likelihood_allocation"):
        resolve_tempering_policy(2.0, allocation)


def test_policy_is_frozen_and_unforgeable() -> None:
    policy = resolve_tempering_policy(2.0, "shared_budget")
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.beta = 3.0  # type: ignore[misc]
    # A hand-built policy whose weight disagrees with its beta cannot exist.
    with pytest.raises(ValueError, match="derived"):
        TemperingPolicy(
            beta=2.0, likelihood_allocation="shared_budget", smoother_weight=0.4
        )
    with pytest.raises(ValueError, match="derived"):
        TemperingPolicy(
            beta=2.0, likelihood_allocation="filter_only", smoother_weight=0.5
        )
    # ... while a consistent one is equal to the resolved one.
    assert TemperingPolicy(2.0, "shared_budget", 0.5) == policy


# --- computation-dtype numerics ------------------------------------------------


def test_beta_overflowing_the_dtype_is_rejected() -> None:
    # Finite as a Python float, infinite in float32.
    with pytest.raises(ValueError, match="overflows"):
        resolve_tempering_policy(1e39, "filter_only", dtype=np.float32)
    resolve_tempering_policy(1e39, "filter_only", dtype=np.float64)


def test_effective_covariance_overflow_is_rejected() -> None:
    with pytest.raises(ValueError, match="filter phase"):
        resolve_tempering_policy(
            1e30, "filter_only", variances=[1.0, 1e10], dtype=np.float32
        )
    resolve_tempering_policy(1e30, "filter_only", variances=[1e10], dtype=np.float64)


def test_effective_alpha_overflow_is_rejected() -> None:
    # Near-one beta: w ~ 2.2e-16, alpha_effective ~ 1.8e16 -- fine alone, but
    # times a large (finite) variance it leaves float32.
    beta = 1.0 + 2.0**-52
    policy = resolve_tempering_policy(beta, "shared_budget", base_alpha=4.0)
    assert policy.effective_alpha(4.0) > 1e16
    with pytest.raises(ValueError, match="ESMDA phase"):
        resolve_tempering_policy(
            beta,
            "shared_budget",
            base_alpha=4.0,
            variances=np.full(3, 1e23),
            dtype=np.float32,
        )


def test_weight_rounding_to_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="rounds to zero"):
        resolve_tempering_policy(1.0 + 2.0**-52, "shared_budget", dtype=np.float16)


def test_metadata_records_both_phases() -> None:
    meta = resolve_tempering_policy(4.0, "shared_budget").metadata(
        base_alpha=3.0, num_steps=3
    )
    assert meta == {
        "beta": 4.0,
        "likelihood_allocation": "shared_budget",
        "filter_weight": 0.25,
        "smoother_weight": 0.75,
        "nominal_combined_exponent": 1.0,
        "base_alpha": 3.0,
        "effective_alpha": 4.0,
        "base_alphas": [3.0] * 3,
        "effective_alphas": [4.0] * 3,
        "smoother_likelihood_share": 0.75,
    }
    legacy = resolve_tempering_policy().metadata()
    assert legacy["nominal_combined_exponent"] == 2.0
    assert "base_alpha" not in legacy


# ---------------------------------------------------------------------------
# Analytical oracles (ideal linear-Gaussian updates)
# ---------------------------------------------------------------------------


def _perturbed_obs_update(
    mean: np.ndarray,
    cov: np.ndarray,
    h: np.ndarray,
    y: float,
    r: float,
    rows: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Infinite-ensemble moments of one perturbed-observation EnKF update.

    ``z_a = z + K (y + eps - h z)``, ``eps ~ N(0, r)``, with the gain
    ``K = C_zy (C_yy + r)^(-1)`` — the covariance form of
    ``stochastic_enkf_update`` with ``alpha * C_D = r``. Only ``rows`` are
    updated (the others' gain rows are zeroed), which is how a parameter-only
    ESMDA or a state-only filter acts on a larger augmented vector. The
    covariance is propagated exactly (Joseph form + the perturbation term), so
    with every row updated it reduces to Gaussian conditioning on ``y`` with
    variance ``r``.
    """
    c_zy = cov @ h
    gain = c_zy / (h @ c_zy + r)
    if rows is not None:
        mask = np.zeros_like(gain)
        mask[list(rows)] = 1.0
        gain = gain * mask
    a = np.eye(len(mean)) - np.outer(gain, h)
    new_mean = mean + gain * (y - h @ mean)
    new_cov = a @ cov @ a.T + r * np.outer(gain, gain)
    return new_mean, new_cov


def _condition(
    mean: np.ndarray, cov: np.ndarray, h: np.ndarray, y: float, r: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Gaussian conditioning on ONE use of ``y = h z + e``, ``e ~ N(0, r)``."""
    s = h @ cov @ h + r
    gain = cov @ h / s
    return mean + gain * (y - h @ mean), cov - np.outer(gain, h @ cov)


# Scalar x = theta: the state IS the parameter (identity forward model).
_PRIOR_MEAN = 0.7
_P = 1.3
_R = 0.4
_Y = 1.9


def _hybrid_x_equals_theta(
    policy: TemperingPolicy,
    num_steps: int,
    filter_rows: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """ESMDA on theta, then the forecast x := theta, then one filter update.

    Returns the moments of ``(x, theta)`` after the filter phase;
    ``filter_rows=(0, 1)`` is ``mode="joint"``, ``(0,)`` ``mode="state"``.
    """
    alpha_eff = policy.effective_alpha(float(num_steps))  # normalized base
    theta_mean = np.array([_PRIOR_MEAN])
    theta_cov = np.array([[_P]])
    for _ in range(num_steps):
        theta_mean, theta_cov = _perturbed_obs_update(
            theta_mean, theta_cov, np.array([1.0]), _Y, alpha_eff * _R
        )
    # Forecast: x = theta exactly, so (x, theta) is the rank-1 lift.
    lift = np.array([[1.0], [1.0]])
    mean = lift @ theta_mean
    cov = lift @ theta_cov @ lift.T
    return _perturbed_obs_update(
        mean, cov, np.array([1.0, 0.0]), _Y, policy.beta * _R, rows=filter_rows
    )


@pytest.mark.parametrize("num_steps", [1, 4])  # type: ignore[misc]
@pytest.mark.parametrize("beta", [1.25, 2.0, 4.0, 8.0])  # type: ignore[misc]
def test_shared_budget_recovers_once_conditioned_posterior(
    beta: float, num_steps: int
) -> None:
    """x = theta, joint filter: ``V^-1 = P^-1 + R^-1`` and the posterior mean
    of one full conditioning, with the joint covariance still rank 1."""
    policy = resolve_tempering_policy(beta, "shared_budget")
    mean, cov = _hybrid_x_equals_theta(policy, num_steps, filter_rows=(0, 1))

    v = 1.0 / (1.0 / _P + 1.0 / _R)
    m = v * (_PRIOR_MEAN / _P + _Y / _R)
    np.testing.assert_allclose(cov, np.full((2, 2), v), rtol=1e-12)
    np.testing.assert_allclose(mean, [m, m], rtol=1e-12)


@pytest.mark.parametrize("num_steps", [1, 4])  # type: ignore[misc]
@pytest.mark.parametrize("beta", [1.0, 2.0, 4.0, 8.0])  # type: ignore[misc]
def test_filter_only_overcounts_by_one_over_beta(beta: float, num_steps: int) -> None:
    """``V^-1 = P^-1 + (1 + 1/beta) R^-1``: no finite beta removes the excess."""
    policy = resolve_tempering_policy(beta, "filter_only")
    mean, cov = _hybrid_x_equals_theta(policy, num_steps, filter_rows=(0, 1))

    exponent = policy.nominal_combined_exponent
    assert exponent == 1.0 + 1.0 / beta
    v = 1.0 / (1.0 / _P + exponent / _R)
    m = v * (_PRIOR_MEAN / _P + exponent * _Y / _R)
    np.testing.assert_allclose(cov, np.full((2, 2), v), rtol=1e-12)
    np.testing.assert_allclose(mean, [m, m], rtol=1e-12)
    v_once = 1.0 / (1.0 / _P + 1.0 / _R)
    assert cov[0, 0] < v_once  # over-confident relative to one conditioning


@pytest.mark.parametrize("beta", [1.25, 2.0, 4.0, 8.0])  # type: ignore[misc]
def test_state_only_correct_state_variance_is_not_a_correct_joint(beta: float) -> None:
    """mode="state" on x = theta: Var x is exactly the once-conditioned V, yet
    theta keeps ESMDA's partial posterior — the joint covariance is wrong.

    Quantified in closed form: with ``P_w`` ESMDA's theta variance,
    ``Var theta - V = P_w^2 / (P_w + beta R) > 0``, and the (x, theta)
    covariance becomes full rank (correlation ``sqrt(V / P_w) < 1``) where the
    true posterior is rank 1 (``x = theta``). Cov(x, theta) happens to equal V.
    """
    policy = resolve_tempering_policy(beta, "shared_budget")
    _, cov = _hybrid_x_equals_theta(policy, num_steps=4, filter_rows=(0,))

    v = 1.0 / (1.0 / _P + 1.0 / _R)
    p_w = 1.0 / (1.0 / _P + policy.smoother_weight / _R)
    assert cov[0, 0] == pytest.approx(v, rel=1e-12)  # the state marginal is right
    assert cov[0, 1] == pytest.approx(v, rel=1e-12)
    assert cov[1, 1] == pytest.approx(p_w, rel=1e-12)  # theta is not
    excess = cov[1, 1] - v
    assert excess == pytest.approx(p_w**2 / (p_w + beta * _R), rel=1e-10)
    assert excess > 0.05 * v
    correlation = cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1])
    assert correlation == pytest.approx(math.sqrt(v / p_w), rel=1e-12)
    assert correlation < 1.0 - 1e-3
    assert np.linalg.matrix_rank(cov, tol=1e-9) == 2


def _hybrid_pinned_state_noise(
    policy: TemperingPolicy,
    num_steps: int,
    q: float,
) -> np.ndarray:
    """General joint Gaussian: x = theta + xi, xi ~ N(0, q) independent of theta.

    ``xi`` is the part of the forecast the parameter-only ESMDA cannot update
    (its pinned initial condition): it is fixed per member across the MDA
    iterations, so ``theta`` picks up a correlation with it. The filter then
    updates ``(x, theta)`` jointly with ``beta R``. Returns Cov(x, theta).
    """
    alpha_eff = policy.effective_alpha(float(num_steps))
    mean = np.zeros(2)  # (theta, xi); only covariances matter here
    cov = np.diag([_P, q])
    h = np.array([1.0, 1.0])
    for _ in range(num_steps):
        mean, cov = _perturbed_obs_update(mean, cov, h, _Y, alpha_eff * _R, rows=(0,))
    to_x_theta = np.array([[1.0, 1.0], [1.0, 0.0]])
    mean, cov = mean @ to_x_theta.T, to_x_theta @ cov @ to_x_theta.T
    _, cov = _perturbed_obs_update(
        mean, cov, np.array([1.0, 0.0]), _Y, policy.beta * _R
    )
    return cov


def _true_joint_covariance(q: float) -> np.ndarray:
    prior = np.array([[_P + q, _P], [_P, _P]])  # Cov(x, theta)
    _, post = _condition(np.zeros(2), prior, np.array([1.0, 0.0]), _Y, _R)
    return post


def test_pinned_state_noise_discrepancy_is_quantified_not_removed() -> None:
    """Beyond x = theta the shared budget is a NOMINAL allocation only.

    With unestimated forecast noise the hybrid's (x, theta) covariance differs
    from the once-conditioned joint posterior for every beta tried; the table
    below records by how much. Nothing here asserts a trend in beta — the plan
    is explicit that beta does not repair the state–parameter dependence.
    """
    q = 0.5
    truth = _true_joint_covariance(q)
    discrepancies = {}
    for beta in (1.5, 2.0, 4.0, 8.0):
        cov = _hybrid_pinned_state_noise(
            resolve_tempering_policy(beta, "shared_budget"), num_steps=4, q=q
        )
        discrepancies[beta] = float(np.max(np.abs(cov - truth)))
    # Material (percent-level of the posterior variances), for every beta.
    assert all(d > 0.01 for d in discrepancies.values()), discrepancies
    # Filter-only is off too (it over-counts on top of the structural error).
    filter_only = _hybrid_pinned_state_noise(
        resolve_tempering_policy(2.0, "filter_only"), num_steps=4, q=q
    )
    assert float(np.max(np.abs(filter_only - truth))) > 0.01


def test_pinned_state_noise_oracle_reduces_to_x_equals_theta() -> None:
    """Consistency of the two oracles: q -> 0 recovers the exact result."""
    policy = resolve_tempering_policy(2.0, "shared_budget")
    cov = _hybrid_pinned_state_noise(policy, num_steps=4, q=1e-14)
    np.testing.assert_allclose(cov, _true_joint_covariance(1e-14), atol=1e-10)


# ---------------------------------------------------------------------------
# Finite-ensemble versions through the real library kernels
# ---------------------------------------------------------------------------
#
# The same x = theta problem, now driven through ``FilterSmoothing`` itself: a
# ``ParameterESMDA`` built with ``likelihood_weight=policy.smoother_weight``
# (stochastic, so its moments carry sampling error) followed by an
# ``EnsembleKalmanFilter`` with the deterministic ETKF analysis and
# ``beta=policy.beta``. The ETKF phase is an exact transform of whatever
# ensemble ESMDA hands it, so it is checked to float32 precision against the
# scalar Kalman formula applied to that ensemble's own moments; only the ESMDA
# phase (and hence the end-to-end result) gets statistical tolerances.

_N_E_LARGE = 2000


class _IdentityEnsembleModel:
    """``u_t = a`` for every output frame: the state IS the parameter."""

    save_on_disk = False
    results_dir = None

    def __init__(self, num_frames: int = 1) -> None:
        self.num_frames = num_frames
        self.calls = 0

    def run_ensemble(self, state: Any = None, params: Any = None) -> Any:
        import xarray

        self.calls += 1
        a = np.asarray(params["a"].values, dtype=float)  # (N_e,)
        values = np.repeat(a[:, None, None], self.num_frames, axis=1)
        return xarray.Dataset(
            {"u": (("ensemble", "time", "x"), values)},
            coords={
                "ensemble": np.arange(a.size),
                "time": np.arange(1, self.num_frames + 1, dtype=float),
                "x": [0],
            },
        )

    def apply_failure_substitutions_to_params(self, params: Any) -> Any:
        return params

    def apply_failure_substitutions_to_state(self, state: Any) -> Any:
        return state


class _IdentityObsOp:
    """One sensor reading ``u`` directly, labelled like the temporal operators."""

    def __call__(self, state: Any) -> Any:
        import xarray

        u = np.asarray(state["u"].values, dtype=float)  # (..., time, 1)
        dims = (
            ("ensemble", "time", "obs") if "ensemble" in state.dims else ("time", "obs")
        )
        return xarray.DataArray(
            u,
            dims=dims,
            coords={"time": np.asarray(state["time"].values, dtype=float), "obs": [0]},
        )


def _kernel_hybrid(
    policy: TemperingPolicy, mode: str, num_steps: int = 4, seed: int = 0
) -> tuple[Any, np.ndarray]:
    """Run one window of the real hybrid on x = theta; return (result, prior)."""
    import jax
    import jax.numpy as jnp
    import xarray
    from data_assimilation import (
        EnsembleKalmanFilter,
        ETKFAnalysis,
        FilterSmoothing,
        ParameterESMDA,
    )
    from data_assimilation.inflation import InflationScheme

    rng = np.random.default_rng(seed)
    draws = rng.normal(size=_N_E_LARGE)
    # Exact prior moments, so the tolerances below measure the MDA's sampling
    # error alone.
    draws = (draws - draws.mean()) / draws.std(ddof=1)
    prior_a = _PRIOR_MEAN + math.sqrt(_P) * draws
    prior = xarray.Dataset(
        {"a": (("ensemble",), prior_a)}, coords={"ensemble": np.arange(_N_E_LARGE)}
    )
    state = xarray.Dataset(
        {"u": (("ensemble", "x"), prior_a[:, None])},
        coords={"ensemble": np.arange(_N_E_LARGE), "x": [0]},
    )
    batches = [
        xarray.DataArray(
            np.array([[_Y]]), dims=("time", "obs"), coords={"time": [1.0], "obs": [0]}
        )
    ]
    obs_op: Any = _IdentityObsOp()  # ONE instance: the shared budget checks it
    smoother = ParameterESMDA(
        observation_operator=obs_op,
        forward_model=_IdentityEnsembleModel(),
        C_D=jnp.diag(jnp.full(1, _R)),
        num_steps=num_steps,
        rng_key=jax.random.PRNGKey(seed),
        likelihood_weight=policy.smoother_weight,
    )
    enkf = EnsembleKalmanFilter(
        observation_operator=obs_op,
        forward_model=_IdentityEnsembleModel(),
        C_D=jnp.full(1, _R),
        analysis=ETKFAnalysis(),
        mode=mode,  # type: ignore[arg-type]
        # Joint mode demands spread maintenance; the base scheme is the
        # identity, as the exact-posterior tests require.
        inflation=InflationScheme() if mode == "joint" else None,
        rng_key=jax.random.PRNGKey(seed + 1),
        beta=policy.beta,
    )
    hybrid = FilterSmoothing(smoother=smoother, filter=enkf, tempering=policy)
    result = hybrid.run(state=state, params=prior, observations=batches)
    return result, prior_a


def _ensemble_moments(values: Any) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float).ravel()
    return float(arr.mean()), float(arr.var(ddof=1))


@pytest.mark.parametrize("beta", [2.0, 4.0])  # type: ignore[misc]
def test_kernel_shared_budget_joint_recovers_once_conditioned(beta: float) -> None:
    policy = resolve_tempering_policy(beta, "shared_budget")
    result, _ = _kernel_hybrid(policy, "joint")

    # ESMDA phase (stochastic): theta ~ the w-tempered posterior.
    p_w = 1.0 / (1.0 / _P + policy.smoother_weight / _R)
    m_w = p_w * (_PRIOR_MEAN / _P + policy.smoother_weight * _Y / _R)
    esmda_mean, esmda_var = _ensemble_moments(result.esmda_params["a"].values)
    assert esmda_var == pytest.approx(p_w, rel=0.12)
    assert esmda_mean == pytest.approx(m_w, abs=4.0 * math.sqrt(p_w / _N_E_LARGE))

    # Filter phase (deterministic ETKF with beta R): EXACT given its input.
    x_mean, x_var = _ensemble_moments(result.state["u"].values)
    expected_var = 1.0 / (1.0 / esmda_var + 1.0 / (beta * _R))
    gain = esmda_var / (esmda_var + beta * _R)
    assert x_var == pytest.approx(expected_var, rel=1e-4)
    assert x_mean == pytest.approx(esmda_mean + gain * (_Y - esmda_mean), abs=1e-4)
    # Joint: theta moves with x, member for member (the posterior stays rank 1).
    np.testing.assert_allclose(
        np.asarray(result.params["a"].values).ravel(),
        np.asarray(result.state["u"].values).ravel(),
        rtol=1e-5,
        atol=1e-5,
    )

    # End to end: the once-conditioned posterior, within sampling error, and
    # clearly separated from the filter-only over-count.
    v = 1.0 / (1.0 / _P + 1.0 / _R)
    v_filter_only = 1.0 / (1.0 / _P + (1.0 + 1.0 / beta) / _R)
    assert x_var == pytest.approx(v, rel=0.12)
    assert abs(x_var - v) < abs(x_var - v_filter_only)


def test_kernel_filter_only_joint_overcounts() -> None:
    beta = 2.0
    policy = resolve_tempering_policy(beta, "filter_only")
    result, _ = _kernel_hybrid(policy, "joint")

    _, x_var = _ensemble_moments(result.state["u"].values)
    v_filter_only = 1.0 / (1.0 / _P + policy.nominal_combined_exponent / _R)
    v = 1.0 / (1.0 / _P + 1.0 / _R)
    assert x_var == pytest.approx(v_filter_only, rel=0.12)
    assert abs(x_var - v_filter_only) < abs(x_var - v)


def test_kernel_state_only_leaves_theta_at_the_esmda_share() -> None:
    """mode="state": x reaches ~V, theta keeps ESMDA's P_w — the joint is off.

    The cross-covariance is KERNEL-dependent here, which is itself the symptom:
    the ideal stochastic oracle above gives correlation ``sqrt(V / P_w)``, while
    the deterministic ETKF rescales x's anomalies by ``sqrt(V / P_w)`` and so
    keeps x perfectly correlated with theta — but at a different variance, so
    x != theta member by member although the model says x = theta. Neither is
    the joint posterior; a state-only update is not a joint Bayesian update.
    """
    beta = 2.0
    policy = resolve_tempering_policy(beta, "shared_budget")
    result, _ = _kernel_hybrid(policy, "state")

    assert result.params is None  # the filter never touches theta in state mode
    theta = np.asarray(result.esmda_params["a"].values, dtype=float).ravel()
    x = np.asarray(result.state["u"].values, dtype=float).ravel()
    _, theta_var = _ensemble_moments(theta)
    _, x_var = _ensemble_moments(x)
    v = 1.0 / (1.0 / _P + 1.0 / _R)
    p_w = 1.0 / (1.0 / _P + policy.smoother_weight / _R)
    assert x_var == pytest.approx(v, rel=0.12)
    assert theta_var == pytest.approx(p_w, rel=0.12)
    # The analytical excess P_w^2/(P_w + beta R), measured on this ensemble.
    excess = theta_var - x_var
    assert excess == pytest.approx(theta_var**2 / (theta_var + beta * _R), rel=1e-3)
    # ETKF: anomalies scaled, not mixed — correlation 1 ...
    assert float(np.corrcoef(x, theta)[0, 1]) == pytest.approx(1.0, abs=1e-5)
    x_anom, theta_anom = x - x.mean(), theta - theta.mean()
    np.testing.assert_allclose(
        x_anom, math.sqrt(x_var / theta_var) * theta_anom, rtol=1e-3, atol=1e-5
    )
    # ... yet x and theta no longer coincide, although x = theta by construction.
    mismatch = float(np.std(x_anom - theta_anom, ddof=1))
    assert mismatch == pytest.approx(math.sqrt(theta_var) - math.sqrt(x_var), rel=1e-3)
    assert mismatch > 0.1 * math.sqrt(x_var)
