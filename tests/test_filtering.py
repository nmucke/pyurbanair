"""Unit tests for the sequential filtering package (data_assimilation.filtering).

Covers the Phase 1 deliverables of docs/temp/da_filtering_module_plan.md:
the linear-Gaussian cycle against the exact Kalman filter, scalar parameter
convergence on a toy forward model, joint-mode localization equivalence, the
parameter-collapse construction guard, and the inflation / parameter-evolution
schemes. Everything runs on toy in-memory forward models — no CFD solver.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray
from data_assimilation.filtering import EnsembleKalmanFilter, RandomWalkEvolution
from data_assimilation.inflation import RTPP, RTPS, MultiplicativeInflation
from data_assimilation.localization.base import BaseLocalization


class _ToyLinearModel:
    """x_{k+1} = A x_k (+ effect * a), wrapped in the ensemble-model interface.

    The forecast Dataset carries a two-frame time dimension whose final frame
    is the propagated state, so the filter's end-of-segment selection
    (``isel(time=-1)``) is exercised.
    """

    save_on_disk = False
    results_dir = None

    def __init__(self, A: np.ndarray, param_effect: float = 0.0) -> None:
        self.A = jnp.asarray(A)
        self.param_effect = param_effect

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> xarray.Dataset:
        assert state is not None
        x = jnp.asarray(state["u"].values)  # (N_e, nx)
        x_new = x @ self.A.T
        if params is not None and self.param_effect != 0.0:
            a = jnp.asarray(params["a"].values)  # (N_e,)
            x_new = x_new + self.param_effect * a[:, None]
        frames = jnp.stack([x, x_new], axis=1)  # (N_e, 2, nx)
        return xarray.Dataset(
            {"u": (("ensemble", "time", "x"), frames)},
            coords={
                "ensemble": np.arange(x.shape[0]),
                "time": np.array([0.0, 1.0]),
                "x": np.arange(x.shape[1]),
            },
        )

    def apply_failure_substitutions_to_params(
        self, params: Optional[xarray.Dataset]
    ) -> Optional[xarray.Dataset]:
        return params

    def apply_failure_substitutions_to_state(
        self, state: Optional[xarray.Dataset]
    ) -> Optional[xarray.Dataset]:
        return state


class _ParamOnlyModel(_ToyLinearModel):
    """Forecast is a direct broadcast of the scalar parameter: u = a."""

    def __init__(self, nx: int = 2) -> None:
        super().__init__(np.eye(nx))
        self.nx = nx

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> xarray.Dataset:
        assert params is not None
        a = jnp.asarray(params["a"].values)  # (N_e,)
        frames = jnp.broadcast_to(
            a[:, None, None], (a.shape[0], 2, self.nx)
        )  # (N_e, 2, nx)
        return xarray.Dataset(
            {"u": (("ensemble", "time", "x"), frames)},
            coords={
                "ensemble": np.arange(a.shape[0]),
                "time": np.array([0.0, 1.0]),
                "x": np.arange(self.nx),
            },
        )


class _ToyObsOp:
    """Linear observation of the forecast's final frame: y = H x_T."""

    def __init__(self, H: np.ndarray) -> None:
        self.H = jnp.asarray(H)  # (N_d, nx)

    def __call__(self, state: xarray.Dataset) -> jnp.ndarray:
        x = jnp.asarray(state["u"].isel(time=-1).values)  # (N_e, nx)
        return x @ self.H.T  # (N_e, N_d)


class _AllOnesLocalization(BaseLocalization):
    """Keeps every observation at full weight for every row.

    ``localized_update`` documents that an all-ones inflation row reduces the
    per-row solve to the exact global update, so a filter with this strategy
    must reproduce the unlocalized filter (same rng) — the equivalence check
    for the filter's localization plumbing.
    """

    requires_coordinates = False
    block_grouping = False

    def inflation_factors(
        self,
        aug_dev: jnp.ndarray,
        pred_obs_dev: jnp.ndarray,
        row_coords: Optional[jnp.ndarray] = None,
        obs_coords: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return jnp.ones((aug_dev.shape[0], pred_obs_dev.shape[0]))


def _initial_state(
    key: jax.Array, n_e: int, mean: np.ndarray, cov: np.ndarray
) -> xarray.Dataset:
    L = np.linalg.cholesky(cov)
    x0 = mean[None, :] + jax.random.normal(key, (n_e, mean.size)) @ L.T
    return xarray.Dataset(
        {"u": (("ensemble", "x"), x0)},
        coords={"ensemble": np.arange(n_e), "x": np.arange(mean.size)},
    )


def _params_dataset(values: np.ndarray) -> xarray.Dataset:
    return xarray.Dataset(
        {"a": (("ensemble",), jnp.asarray(values))},
        coords={"ensemble": np.arange(values.shape[0])},
    )


# ---------------------------------------------------------------------------
# (a) Linear-Gaussian cycling against the exact Kalman filter
# ---------------------------------------------------------------------------


def test_state_mode_matches_exact_kalman_filter() -> None:
    """State-mode EnKF cycling converges to the exact KF (N_e large).

    Deterministic linear dynamics, linear H, Gaussian initial ensemble: the
    stochastic EnKF's analysis mean and covariance must match the exact
    Kalman filter recursion to O(1/sqrt(N_e)) after several cycles.
    """
    A = np.array([[0.9, 0.2], [-0.1, 0.8]])
    H = np.array([[1.0, 0.0]])
    r = 0.05  # observation-error variance
    m0 = np.array([1.0, -0.5])
    P0 = np.array([[0.5, 0.1], [0.1, 0.3]])
    observations = np.array([[1.2], [0.7], [0.4], [0.3]])

    n_e = 4000
    state = _initial_state(jax.random.PRNGKey(1), n_e, m0, P0)
    enkf = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(H),
        forward_model=_ToyLinearModel(A),
        C_D=jnp.array([r]),
        mode="state",
        rng_key=jax.random.PRNGKey(2),
    )
    result = enkf.run(state=state, observations=jnp.asarray(observations))

    # Exact Kalman filter recursion on the same sequence.
    m, P = m0.copy(), P0.copy()
    for y in observations:
        m, P = A @ m, A @ P @ A.T
        S = H @ P @ H.T + r * np.eye(1)
        K = P @ H.T @ np.linalg.inv(S)
        m = m + (K @ (y - H @ m)).ravel()
        P = (np.eye(2) - K @ H) @ P

    assert result.state is not None
    ens = np.asarray(result.state["u"].values)  # (N_e, nx)
    np.testing.assert_allclose(ens.mean(axis=0), m, atol=0.05)
    np.testing.assert_allclose(np.cov(ens.T), P, atol=0.02)

    # Diagnostics: the analysis must not degrade the observation-space fit,
    # and a consistent filter has innovation chi2 of order one.
    for diag in result.diagnostics:
        assert diag.obs_posterior_rmse <= diag.obs_prior_rmse + 1e-8
        assert 0.0 < diag.innovation_chi2 < 20.0
        assert diag.param_spread_prior is None


# ---------------------------------------------------------------------------
# (b) Scalar parameter convergence on a toy forward model
# ---------------------------------------------------------------------------


def test_parameter_mode_converges_to_truth() -> None:
    """Parameter-only filtering pulls the ensemble toward the true scalar."""
    truth = 2.0
    n_e, num_cycles = 100, 10
    H = np.array([[1.0, 0.0]])
    rng = np.random.default_rng(0)
    observations = truth + 0.05 * rng.standard_normal((num_cycles, 1))

    prior = 0.0 + 1.0 * rng.standard_normal(n_e)
    params = _params_dataset(prior)
    state = _initial_state(jax.random.PRNGKey(3), n_e, np.zeros(2), np.eye(2))

    enkf = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(H),
        forward_model=_ParamOnlyModel(),
        C_D=jnp.array([0.05**2]),
        mode="parameter",
        parameter_evolution=RandomWalkEvolution(std=0.02),
        rng_key=jax.random.PRNGKey(4),
    )
    result = enkf.run(
        state=state,
        params=params,
        observations=jnp.asarray(observations),
        return_history=True,
    )

    assert result.params is not None
    posterior = np.asarray(result.params["a"].values)
    assert abs(posterior.mean() - truth) < 0.1
    assert posterior.std() < prior.std() / 3
    # History: prior + one entry per cycle, concatenated over "cycle".
    assert result.params_history is not None
    assert result.params_history.sizes["cycle"] == num_cycles + 1
    # Spread maintenance keeps the posterior spread strictly positive.
    final_spread = result.diagnostics[-1].param_spread_posterior
    assert final_spread is not None and final_spread > 0.0
    assert result.diagnostics[-1].state_spread_prior is None


# ---------------------------------------------------------------------------
# (c) Joint mode and localization plumbing
# ---------------------------------------------------------------------------


def _run_joint(localization: Optional[BaseLocalization]) -> tuple:
    A = np.array([[0.9, 0.2], [-0.1, 0.8]])
    H = np.array([[1.0, 0.0]])
    n_e = 40
    rng = np.random.default_rng(5)
    state = _initial_state(
        jax.random.PRNGKey(6), n_e, np.array([1.0, -0.5]), 0.4 * np.eye(2)
    )
    params = _params_dataset(rng.standard_normal(n_e))
    observations = np.array([[1.0], [0.8]])

    enkf = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(H),
        forward_model=_ToyLinearModel(A, param_effect=0.5),
        C_D=jnp.array([0.1]),
        mode="joint",
        localization=localization,
        rng_key=jax.random.PRNGKey(7),
    )
    result = enkf.run(
        state=state, params=params, observations=jnp.asarray(observations)
    )
    assert result.state is not None and result.params is not None
    return np.asarray(result.state["u"].values), np.asarray(result.params["a"].values)


def test_joint_mode_all_ones_localization_matches_global() -> None:
    """All-ones localization reproduces the global joint update exactly.

    Exercises the filter's localize_mask/group_ids plumbing end to end: with
    every observation kept at full weight, the per-row local analyses (and
    the globally-updated parameter rows) must equal the unlocalized filter
    run with the same rng key.
    """
    state_glob, params_glob = _run_joint(localization=None)
    state_loc, params_loc = _run_joint(localization=_AllOnesLocalization())

    np.testing.assert_allclose(state_loc, state_glob, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(params_loc, params_glob, rtol=1e-4, atol=1e-5)


def test_joint_mode_updates_both_blocks() -> None:
    """The joint analysis moves both the state and the parameters."""
    A = np.array([[0.9, 0.2], [-0.1, 0.8]])
    H = np.array([[1.0, 0.0]])
    n_e = 40
    rng = np.random.default_rng(8)
    state = _initial_state(jax.random.PRNGKey(9), n_e, np.zeros(2), np.eye(2))
    params = _params_dataset(rng.standard_normal(n_e))

    enkf = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(H),
        forward_model=_ToyLinearModel(A, param_effect=0.5),
        C_D=jnp.array([0.1]),
        mode="joint",
        rng_key=jax.random.PRNGKey(10),
    )
    result = enkf.run(state=state, params=params, observations=jnp.array([[2.0]]))
    assert result.params is not None
    assert not np.allclose(
        np.asarray(result.params["a"].values), np.asarray(params["a"].values)
    )
    diag = result.diagnostics[0]
    assert diag.state_spread_posterior is not None
    assert diag.param_spread_posterior is not None
    assert diag.state_spread_prior is not None
    assert diag.state_spread_posterior <= diag.state_spread_prior + 1e-8


# ---------------------------------------------------------------------------
# (d) Construction guards
# ---------------------------------------------------------------------------


def _dummy_filter_kwargs() -> dict:
    return {
        "observation_operator": _ToyObsOp(np.array([[1.0, 0.0]])),
        "forward_model": _ToyLinearModel(np.eye(2)),
        "C_D": jnp.array([0.1]),
    }


def test_parameter_mode_without_spread_maintenance_raises() -> None:
    with pytest.raises(ValueError, match="spread maintenance"):
        EnsembleKalmanFilter(mode="parameter", **_dummy_filter_kwargs())


def test_parameter_mode_with_inflation_only_is_accepted() -> None:
    EnsembleKalmanFilter(
        mode="parameter", inflation=RTPS(alpha=0.5), **_dummy_filter_kwargs()
    )


def test_state_mode_with_parameter_evolution_raises() -> None:
    with pytest.raises(ValueError, match="no effect"):
        EnsembleKalmanFilter(
            mode="state",
            parameter_evolution=RandomWalkEvolution(std=0.1),
            **_dummy_filter_kwargs(),
        )


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode"):
        EnsembleKalmanFilter(
            mode="smoother",  # type: ignore[arg-type]
            **_dummy_filter_kwargs(),
        )


def test_c_d_matrix_accepted_and_off_diagonal_rejected() -> None:
    kwargs = _dummy_filter_kwargs()
    kwargs["C_D"] = jnp.diag(jnp.array([0.1]))
    enkf = EnsembleKalmanFilter(mode="state", **kwargs)
    assert enkf.C_D_diag.shape == (1,)

    kwargs["C_D"] = jnp.array([[0.1, 0.01], [0.01, 0.1]])
    with pytest.raises(ValueError, match="diagonal"):
        EnsembleKalmanFilter(mode="state", **kwargs)


def test_one_d_observations_rejected() -> None:
    enkf = EnsembleKalmanFilter(mode="state", **_dummy_filter_kwargs())
    state = _initial_state(jax.random.PRNGKey(0), 5, np.zeros(2), np.eye(2))
    with pytest.raises(ValueError, match="num_cycles"):
        enkf.run(state=state, observations=jnp.ones(3))


def test_time_varying_params_rejected() -> None:
    enkf = EnsembleKalmanFilter(
        mode="parameter",
        parameter_evolution=RandomWalkEvolution(std=0.1),
        **_dummy_filter_kwargs(),
    )
    params = xarray.Dataset(
        {"a": (("time", "ensemble"), np.zeros((3, 5)))},
        coords={"time": np.arange(3.0), "ensemble": np.arange(5)},
    )
    with pytest.raises(NotImplementedError, match="Time-varying"):
        enkf.run(params=params, observations=jnp.ones((2, 1)))


# ---------------------------------------------------------------------------
# Inflation schemes
# ---------------------------------------------------------------------------


def test_multiplicative_inflation_scales_prior_anomalies() -> None:
    dev = jnp.asarray(np.random.default_rng(0).standard_normal((4, 10)))
    inflated = MultiplicativeInflation(1.5).inflate_prior(dev)
    np.testing.assert_allclose(np.asarray(inflated), 1.5 * np.asarray(dev))
    with pytest.raises(ValueError, match="positive"):
        MultiplicativeInflation(0.0)


def test_rtps_restores_prior_spread_at_alpha_one() -> None:
    rng = np.random.default_rng(1)
    dev_prior = jnp.asarray(rng.standard_normal((4, 30)))
    dev_prior = dev_prior - dev_prior.mean(axis=1, keepdims=True)
    dev_post = 0.3 * dev_prior  # analysis shrank the spread
    restored = RTPS(alpha=1.0).inflate_posterior(dev_prior, dev_post)
    np.testing.assert_allclose(
        np.std(np.asarray(restored), axis=1, ddof=1),
        np.std(np.asarray(dev_prior), axis=1, ddof=1),
        rtol=1e-6,
    )
    # alpha=0 is a no-op; zero-spread rows are left unchanged.
    unchanged = RTPS(alpha=0.0).inflate_posterior(dev_prior, dev_post)
    np.testing.assert_allclose(np.asarray(unchanged), np.asarray(dev_post))
    zero_post = jnp.zeros_like(dev_post)
    np.testing.assert_allclose(
        np.asarray(RTPS(alpha=1.0).inflate_posterior(dev_prior, zero_post)), 0.0
    )
    with pytest.raises(ValueError, match="alpha"):
        RTPS(alpha=1.5)


def test_rtpp_blends_anomalies() -> None:
    rng = np.random.default_rng(2)
    dev_prior = jnp.asarray(rng.standard_normal((3, 8)))
    dev_post = jnp.asarray(rng.standard_normal((3, 8)))
    blended = RTPP(alpha=0.25).inflate_posterior(dev_prior, dev_post)
    np.testing.assert_allclose(
        np.asarray(blended), 0.25 * np.asarray(dev_prior) + 0.75 * np.asarray(dev_post)
    )


# ---------------------------------------------------------------------------
# Parameter evolution
# ---------------------------------------------------------------------------


def test_random_walk_evolution_adds_configured_noise() -> None:
    n_e = 2000
    params = xarray.Dataset(
        {
            "a": (("ensemble",), jnp.zeros(n_e)),
            "b": (("ensemble",), jnp.ones(n_e)),
        },
        coords={"ensemble": np.arange(n_e)},
    )
    evolved = RandomWalkEvolution(std={"a": 0.5}).evolve(params, jax.random.PRNGKey(11))
    # 'a' gets ~N(0, 0.5^2) noise; 'b' (absent from the mapping) is unchanged.
    assert np.std(np.asarray(evolved["a"].values)) == pytest.approx(0.5, rel=0.1)
    np.testing.assert_array_equal(
        np.asarray(evolved["b"].values), np.asarray(params["b"].values)
    )
    with pytest.raises(ValueError, match=">= 0"):
        RandomWalkEvolution(std=-0.1)
