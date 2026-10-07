"""Unit tests for the sequential filtering package (data_assimilation.filtering).

Covers the Phase 1 deliverables of docs/plans/implemented/da_filtering_module_plan.md:
the linear-Gaussian cycle against the exact Kalman filter, scalar parameter
convergence on a toy forward model, joint-mode localization equivalence, the
parameter-collapse construction guard, and the inflation / parameter-evolution
schemes. Everything runs on toy in-memory forward models — no CFD solver.
"""

import pathlib
import types
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray
from data_assimilation.filtering import (
    EnsembleKalmanFilter,
    ETKFAnalysis,
    FilterResult,
    LETKFAnalysis,
    ObservationTSVD,
    RandomWalkEvolution,
)
from data_assimilation.filtering.analysis import AnalysisScheme, StochasticEnKFAnalysis
from data_assimilation.inflation import RTPP, RTPS, MultiplicativeInflation
from data_assimilation.localization.base import BaseLocalization
from data_assimilation.localization.correlation import CorrelationLocalization
from data_assimilation.localization.distance import DistanceLocalization
from data_assimilation.reduction import OnlineStateReduction, StreamingStateReduction


class _ToyLinearModel:
    """x_{k+1} = A x_k (+ effect * a), wrapped in the ensemble-model interface.

    The forecast Dataset carries a two-frame time dimension whose final frame
    is the propagated state, so the filter's end-of-segment selection
    (``isel(time=-1)``) is exercised.
    """

    save_on_disk = False
    results_dir: Optional[pathlib.Path] = None

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


class _OnDiskToyLinearModel(_ToyLinearModel):
    """Disk-writing twin of ``_ToyLinearModel`` for filter I/O equivalence."""

    save_on_disk = True

    def __init__(self, A: np.ndarray, results_dir: pathlib.Path) -> None:
        super().__init__(A)
        self.results_dir = results_dir

    def set_results_dir(self, results_dir: pathlib.Path) -> None:
        self.results_dir = results_dir

    def run_ensemble(  # type: ignore[override, unused-ignore]
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> None:
        forecast = super().run_ensemble(state=state, params=params)
        assert forecast is not None
        results_dir = self.results_dir
        assert results_dir is not None
        for member in range(forecast.sizes["ensemble"]):
            forecast.isel(ensemble=member, drop=True).to_netcdf(
                results_dir / f"state_{member}.nc"
            )
        return None


class _ToyObsOp:
    """Linear observation of the forecast's final frame: y = H x_T."""

    def __init__(self, H: np.ndarray) -> None:
        self.H = jnp.asarray(H)  # (N_d, nx)

    def __call__(self, state: xarray.Dataset) -> jnp.ndarray:
        x = jnp.asarray(state["u"].isel(time=-1).values)  # (N_e, nx)
        return x @ self.H.T  # (N_e, N_d)


class _TemporalToyObsOp:
    """Time-resolved twin of ``_ToyObsOp``: y_t = H x_t for EVERY frame.

    Returns the labelled DataArray a ``TemporalObservationOperator`` produces
    (dims ``("ensemble", "time", "obs")``, or ``("time", "obs")`` for one
    member read back from disk), so the filter's per-frame path — and with it
    the serial sweep — is exercised end to end.
    """

    def __init__(self, H: np.ndarray) -> None:
        self.H = jnp.asarray(H)  # (N_d, nx)

    def __call__(self, state: xarray.Dataset) -> xarray.DataArray:
        x = jnp.asarray(state["u"].values)  # (..., time, nx)
        values = np.asarray(jnp.einsum("...i,di->...d", x, self.H))
        dims = (
            ("ensemble", "time", "obs") if "ensemble" in state.dims else ("time", "obs")
        )
        return xarray.DataArray(
            values,
            dims=dims,
            coords={"time": np.asarray(state["time"].values, dtype=float)},
        )


class _CoordinateToyObsOp(_ToyObsOp):
    """Toy operator exposing one physical sensor for distance localization."""

    use_interpolation = True
    obs_x = np.array([0.0])
    obs_y = np.array([0.0])
    obs_z = np.array([0.0])
    num_sensors = 1


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
        assert diag.reduction_rank is None
        assert diag.reduction_basis_time is None
        # Timing and provenance are recorded on the unreduced path too, so a
        # reduced run can be compared against this one.
        assert diag.analysis_time is not None
        assert diag.obs_posterior_rmse_kind == "exact"


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
        parameter_evolution=RandomWalkEvolution(std={"a": 0.02}),
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


def test_parameter_mode_correlation_localizes_each_parameter() -> None:
    """Correlation localization updates the informed param and rejects noise."""
    n_e = 40
    rng = np.random.default_rng(12)
    signal = rng.standard_normal(n_e)
    nuisance = rng.standard_normal(n_e)
    params = xarray.Dataset(
        {
            "a": ("ensemble", signal),
            "b": ("ensemble", nuisance),
        },
        coords={"ensemble": np.arange(n_e)},
    )

    enkf = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(np.array([[1.0, 0.0]])),
        forward_model=_ParamOnlyModel(),
        C_D=jnp.array([0.1]),
        mode="parameter",
        localization=CorrelationLocalization(
            truncation_correlation=0.999, max_inflation=1.0
        ),
        inflation=MultiplicativeInflation(1.0),
        rng_key=jax.random.PRNGKey(13),
    )
    result = enkf.run(params=params, observations=jnp.array([[2.0]]))

    assert result.params is not None
    assert not np.allclose(result.params["a"], params["a"])
    np.testing.assert_allclose(result.params["b"], params["b"], atol=1e-7)


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
        # Joint mode requires spread maintenance; identical in both runs, so
        # the localization equivalence below is unaffected.
        inflation=RTPS(alpha=0.5),
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
    every observation kept at full weight, every per-row local analysis must
    equal the unlocalized filter run with the same rng key.
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
        inflation=RTPS(alpha=0.5),
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


def test_joint_correlation_localizes_parameter_rows() -> None:
    """Joint correlation localization applies to parameters, not only state."""
    n_e = 40
    rng = np.random.default_rng(14)
    state = _initial_state(jax.random.PRNGKey(15), n_e, np.zeros(2), np.eye(2))
    params = _params_dataset(rng.standard_normal(n_e))
    common: Any = dict(
        observation_operator=_ToyObsOp(np.array([[1.0, 0.0]])),
        forward_model=_ToyLinearModel(np.eye(2)),
        C_D=jnp.array([0.1]),
        mode="joint",
        inflation=MultiplicativeInflation(1.0),
        rng_key=jax.random.PRNGKey(16),
    )
    localized = EnsembleKalmanFilter(
        localization=CorrelationLocalization(
            truncation_correlation=0.999, max_inflation=1.0
        ),
        **common,
    ).run(state=state, params=params, observations=jnp.array([[2.0]]))
    global_result = EnsembleKalmanFilter(localization=None, **common).run(
        state=state, params=params, observations=jnp.array([[2.0]])
    )

    assert localized.params is not None and global_result.params is not None
    np.testing.assert_allclose(localized.params["a"], params["a"], atol=1e-7)
    assert not np.allclose(global_result.params["a"], params["a"])


def test_joint_distance_localizes_state_but_keeps_parameter_update_global() -> None:
    """Joint distance localization changes state support, not parameter math."""
    n_e = 40
    signal = jax.random.normal(jax.random.PRNGKey(17), (n_e,))
    state = xarray.Dataset(
        {"u": (("ensemble", "x"), jnp.stack([signal, signal], axis=1))},
        coords={"ensemble": np.arange(n_e), "x": [0.0, 10.0]},
    )
    params = _params_dataset(np.asarray(signal))
    common: Any = dict(
        observation_operator=_CoordinateToyObsOp(np.array([[1.0, 0.0]])),
        forward_model=_ToyLinearModel(np.eye(2)),
        C_D=jnp.array([0.1]),
        mode="joint",
        inflation=MultiplicativeInflation(1.0),
        rng_key=jax.random.PRNGKey(18),
    )
    localized = EnsembleKalmanFilter(
        localization=DistanceLocalization(
            localization_radius=0.1, max_inflation=1.0, block_grouping=False
        ),
        **common,
    ).run(state=state, params=params, observations=jnp.array([[2.0]]))
    global_result = EnsembleKalmanFilter(localization=None, **common).run(
        state=state, params=params, observations=jnp.array([[2.0]])
    )

    assert localized.state is not None and global_result.state is not None
    assert localized.params is not None and global_result.params is not None
    np.testing.assert_allclose(localized.state["u"][:, 1], state["u"][:, 1], atol=1e-6)
    assert not np.allclose(global_result.state["u"][:, 1], state["u"][:, 1])
    np.testing.assert_allclose(
        localized.params["a"], global_result.params["a"], atol=1e-6
    )


# ---------------------------------------------------------------------------
# (d) Construction guards
# ---------------------------------------------------------------------------


def _dummy_filter_kwargs() -> dict:
    return {
        "observation_operator": _ToyObsOp(np.array([[1.0, 0.0]])),
        "forward_model": _ToyLinearModel(np.eye(2)),
        "C_D": jnp.array([0.1]),
    }


@pytest.mark.parametrize("mode", ["parameter", "joint"])  # type: ignore[misc, unused-ignore]
def test_parameter_updating_modes_without_spread_maintenance_raise(
    mode: str,
) -> None:
    """Both parameter-updating modes refuse silently-collapsing configs."""
    with pytest.raises(ValueError, match="spread maintenance"):
        EnsembleKalmanFilter(mode=mode, **_dummy_filter_kwargs())  # type: ignore[arg-type]


def test_parameter_mode_with_inflation_only_is_accepted() -> None:
    EnsembleKalmanFilter(
        mode="parameter", inflation=RTPS(alpha=0.5), **_dummy_filter_kwargs()
    )


def test_parameter_mode_rejects_distance_localization() -> None:
    with pytest.raises(ValueError, match="parameter-only"):
        EnsembleKalmanFilter(
            mode="parameter",
            localization=DistanceLocalization(localization_radius=1.0),
            inflation=MultiplicativeInflation(1.0),
            **_dummy_filter_kwargs(),
        )


def test_state_mode_with_parameter_evolution_raises() -> None:
    with pytest.raises(ValueError, match="no effect"):
        EnsembleKalmanFilter(
            mode="state",
            parameter_evolution=RandomWalkEvolution(std={"a": 0.1}),
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
        parameter_evolution=RandomWalkEvolution(std={"a": 0.1}),
        **_dummy_filter_kwargs(),
    )
    params = xarray.Dataset(
        {"a": (("time", "ensemble"), np.zeros((3, 5)))},
        coords={"time": np.arange(3.0), "ensemble": np.arange(5)},
    )
    with pytest.raises(NotImplementedError, match="Time-varying"):
        enkf.run(params=params, observations=jnp.ones((2, 1)))


# ---------------------------------------------------------------------------
# Labelled (time-resolved) observations and the serial per-frame sweep
# ---------------------------------------------------------------------------


def _labelled_truth_obs(values: np.ndarray, times: np.ndarray) -> xarray.DataArray:
    """One cycle's time-resolved truth observations, ("time", "obs")."""
    return xarray.DataArray(
        values, dims=("time", "obs"), coords={"time": np.asarray(times, dtype=float)}
    )


class _FinalFrameTemporalToyObsOp(_TemporalToyObsOp):
    """Labelled operator emitting exactly ONE frame: the segment's last.

    The single-frame twin of ``_ToyObsOp``: same numbers, but returned as the
    ``("ensemble", "time", "obs")`` DataArray a ``TemporalObservationOperator``
    produces, so the labelled path can be held to the flat one bit for bit.
    """

    def __call__(self, state: xarray.Dataset) -> xarray.DataArray:
        return super().__call__(state.isel(time=[-1]))


def _single_frame_filter(
    forward_model: Any, observation_operator: Any
) -> EnsembleKalmanFilter:
    return EnsembleKalmanFilter(
        observation_operator=observation_operator,
        forward_model=forward_model,
        C_D=jnp.array([0.2]),
        mode="state",
        rng_key=jax.random.PRNGKey(30),
    )


def test_one_frame_per_cycle_matches_the_legacy_flat_run(
    tmp_path: pathlib.Path,
) -> None:
    """T = 1 is the legacy path, bit for bit.

    A one-frame ``("time", "obs")`` DataArray per cycle against a one-frame
    labelled operator must reproduce — exactly, RNG draws included — the run
    fed the flat ``(num_cycles, N_d)`` array by an array-returning operator.
    That equality is what makes the serial sweep a strict generalization: every
    existing configuration takes the ``T = 1`` branch. The on-disk path (per-
    member DataArrays concatenated along "ensemble") is held to the same
    result, to float tolerance rather than bit-identity because it round-trips
    through NetCDF.
    """
    n_e, num_cycles = 12, 3
    H = np.array([[1.0, 0.5]])
    state = _initial_state(jax.random.PRNGKey(29), n_e, np.zeros(2), np.eye(2))
    values = np.array([[0.5], [0.2], [-0.4]])
    labelled = [
        _labelled_truth_obs(values[k][None, :], np.array([1.0]))
        for k in range(num_cycles)
    ]

    frames = _single_frame_filter(
        _ToyLinearModel(np.eye(2)), _FinalFrameTemporalToyObsOp(H)
    )
    frames.collect_pred_obs = True
    labelled_result = frames.run(state=state, observations=labelled)

    legacy = _single_frame_filter(_ToyLinearModel(np.eye(2)), _ToyObsOp(H))
    legacy.collect_pred_obs = True
    legacy_result = legacy.run(state=state, observations=jnp.asarray(values))

    assert labelled_result.state is not None and legacy_result.state is not None
    np.testing.assert_array_equal(
        np.asarray(labelled_result.state["u"]), np.asarray(legacy_result.state["u"])
    )
    for labelled_diag, legacy_diag in zip(
        labelled_result.diagnostics, legacy_result.diagnostics
    ):
        assert labelled_diag.obs_prior_rmse == legacy_diag.obs_prior_rmse
        assert labelled_diag.obs_posterior_rmse == legacy_diag.obs_posterior_rmse
        assert labelled_diag.innovation_chi2 == legacy_diag.innovation_chi2
        assert (
            labelled_diag.state_spread_posterior == legacy_diag.state_spread_posterior
        )
    # Recorded predicted observations keep the flat (N_d, N_e) shape.
    assert len(frames.pred_obs_history) == num_cycles
    for recorded, expected in zip(frames.pred_obs_history, legacy.pred_obs_history):
        assert recorded.shape == (1, n_e)
        np.testing.assert_array_equal(recorded, expected)

    on_disk = _single_frame_filter(
        _OnDiskToyLinearModel(np.eye(2), tmp_path), _FinalFrameTemporalToyObsOp(H)
    ).run(state=state, observations=labelled)
    assert on_disk.state is not None
    np.testing.assert_allclose(
        on_disk.state["u"], legacy_result.state["u"], rtol=1e-5, atol=2e-5
    )


def test_every_frame_of_a_segment_is_assimilated_serially() -> None:
    """T frames = T full-weight analyses, so information accumulates.

    ``A = I`` makes both frames of the segment carry the SAME predicted
    observation, and both are observed with the same ``y``: assimilating the
    pair must therefore leave less posterior spread than assimilating one of
    them, which a single analysis of a length-2 stacked vector would not do
    either (that would be one update against two perfectly correlated rows).
    """
    n_e = 400
    H = np.array([[1.0, 0.5]])
    state = _initial_state(jax.random.PRNGKey(31), n_e, np.zeros(2), np.eye(2))
    y = np.array([[0.2]])

    serial = EnsembleKalmanFilter(
        observation_operator=_TemporalToyObsOp(H),  # both frames of the segment
        forward_model=_ToyLinearModel(np.eye(2)),
        C_D=jnp.array([0.2]),
        mode="state",
        rng_key=jax.random.PRNGKey(32),
    )
    serial.collect_pred_obs = True
    serial_result = serial.run(
        state=state,
        observations=[
            _labelled_truth_obs(np.repeat(y, 2, axis=0), np.array([0.0, 1.0]))
        ],
    )
    single = _single_frame_filter(_ToyLinearModel(np.eye(2)), _ToyObsOp(H))
    single_result = single.run(state=state, observations=jnp.asarray(y))

    assert serial_result.state is not None and single_result.state is not None
    serial_spread = float(np.std(np.asarray(serial_result.state["u"]), axis=0).mean())
    single_spread = float(np.std(np.asarray(single_result.state["u"]), axis=0).mean())
    assert serial_spread < single_spread
    # Both frames' rows ride along the whole sweep, so the history holds the
    # segment's frames stacked frame-major.
    assert serial.pred_obs_history[0].shape == (2, n_e)


def test_an_uninformative_frame_leaves_the_ensemble_unchanged() -> None:
    """obs == the ensemble-mean prediction with a huge C_D is a no-op.

    The deterministic transform makes the statement exact (no perturbation
    draw): each frame's analysis must reduce to the identity, so a whole
    multi-frame sweep of such frames returns the forecast untouched. It is the
    guard against a sweep that "assimilates" the same information repeatedly.
    """
    n_e = 16
    H = np.array([[1.0, 0.5]])
    state = _initial_state(jax.random.PRNGKey(35), n_e, np.zeros(2), np.eye(2))
    forecast = _ToyLinearModel(np.eye(2)).run_ensemble(state=state)
    pred_obs_mean = float(np.mean(np.asarray(_ToyObsOp(H)(forecast))))

    enkf = EnsembleKalmanFilter(
        observation_operator=_TemporalToyObsOp(H),
        forward_model=_ToyLinearModel(np.eye(2)),
        C_D=jnp.array([1e10]),
        analysis=ETKFAnalysis(),
        mode="state",
        rng_key=jax.random.PRNGKey(36),
    )
    result = enkf.run(
        state=state,
        observations=[
            _labelled_truth_obs(np.full((2, 1), pred_obs_mean), np.array([0.0, 1.0]))
        ],
    )

    assert result.state is not None
    np.testing.assert_allclose(
        np.asarray(result.state["u"]),
        np.asarray(forecast["u"].isel(time=-1)),
        rtol=1e-5,
        atol=1e-6,
    )


def test_pred_obs_history_holds_the_raw_stacked_frames() -> None:
    """``(T*N_obs, N_e)``, frame-major, and BEFORE prior inflation."""
    n_e = 8
    H = np.array([[1.0, 0.5]])
    state = _initial_state(jax.random.PRNGKey(37), n_e, np.zeros(2), np.eye(2))

    enkf = EnsembleKalmanFilter(
        observation_operator=_TemporalToyObsOp(H),
        forward_model=_ToyLinearModel(np.eye(2)),
        C_D=jnp.array([0.2]),
        mode="state",
        inflation=MultiplicativeInflation(1.5),
        rng_key=jax.random.PRNGKey(38),
    )
    enkf.collect_pred_obs = True
    enkf.run(
        state=state,
        observations=[
            _labelled_truth_obs(np.array([[0.2], [0.8]]), np.array([0.0, 1.0]))
        ],
    )

    forecast = _ToyLinearModel(np.eye(2)).run_ensemble(state=state)
    expected = np.asarray(_TemporalToyObsOp(H)(forecast))  # (N_e, T, N_obs)
    recorded = enkf.pred_obs_history[0]
    assert recorded.shape == (2, n_e)
    np.testing.assert_allclose(
        recorded, expected.transpose(1, 2, 0).reshape(-1, n_e), rtol=1e-6, atol=1e-6
    )


def test_pred_obs_post_history_parallels_pred_obs_history() -> None:
    """The posterior ride-along rows, entry for entry and shape for shape.

    ``run_filtering.py`` pairs the two histories into the ESMDA-schema
    ``window_{w}_pred_obs.nc`` (step 0 = prior, step 1 = posterior), so they
    must be recorded under the SAME gate, one entry per cycle, in the same
    ``(T*N_obs, N_e)`` layout. The posterior entry must also differ from the
    prior one — recording the pre-analysis rows twice would make the data
    mismatch look flat while raising no error anywhere.
    """
    n_e, num_cycles = 8, 3
    H = np.array([[1.0, 0.5]])
    state = _initial_state(jax.random.PRNGKey(51), n_e, np.zeros(2), np.eye(2))
    observations = [
        _labelled_truth_obs(np.array([[0.2], [0.8]]), np.array([0.0, 1.0]))
        for _ in range(num_cycles)
    ]

    def _filter() -> EnsembleKalmanFilter:
        return EnsembleKalmanFilter(
            observation_operator=_TemporalToyObsOp(H),
            forward_model=_ToyLinearModel(np.eye(2)),
            C_D=jnp.array([0.2]),
            mode="state",
            rng_key=jax.random.PRNGKey(52),
        )

    # Default-off: neither history is touched, so the flag stays the only cost.
    quiet = _filter()
    quiet.run(state=state, observations=observations)
    assert quiet.pred_obs_history == []
    assert quiet.pred_obs_post_history == []

    enkf = _filter()
    enkf.collect_pred_obs = True
    enkf.run(state=state, observations=observations)

    assert len(enkf.pred_obs_post_history) == num_cycles
    for prior, posterior in zip(enkf.pred_obs_history, enkf.pred_obs_post_history):
        assert posterior.shape == prior.shape == (2, n_e)
        assert not np.allclose(posterior, prior)

    # Rebound per run() call like ``pred_obs_history``, so a caller running the
    # same filter over several passes reads that pass's entries alone.
    enkf.run(state=state, observations=observations[:1])
    assert len(enkf.pred_obs_post_history) == 1


@pytest.mark.parametrize(
    "spread",
    [
        {"parameter_evolution": RandomWalkEvolution(std={"a": 0.05})},
        {"inflation": RTPS(alpha=0.5)},
    ],
    ids=["random_walk", "rtps"],
)  # type: ignore[misc, unused-ignore]
def test_windowing_the_cycle_chain_is_mathematically_inert(
    spread: dict[str, Any],
) -> None:
    """The same horizon as ONE run() call or as W, identically.

    The window loop in ``scripts/run_filtering.py`` is computational
    chunking: it splits the horizon into ``run()`` calls so RAM and peak disk
    stay bounded, carrying the analyzed state and parameters into the next call
    exactly as ESMDA carries its own, and relying on ``BaseFilter.rng_key``
    being an instance attribute the cycle loop mutates in place (``run()``
    never reseeds it). This pins that whole claim: the posterior state, the
    posterior parameters and every per-cycle diagnostic must be bit-identical.

    The observations are built ONCE for the horizon, in global cycle order —
    the script's discipline, and the reason its noise draws do not depend on
    the window count either. The random-walk case pins that the parameter
    evolution, applied before every forecast after the first analysis, also
    carries across ``run()`` calls.
    """
    n_e, num_cycles, cycles_per_window = 10, 6, 3
    H = np.array([[1.0, 0.5]])
    state = _initial_state(jax.random.PRNGKey(61), n_e, np.zeros(2), np.eye(2))
    params = _params_dataset(
        np.asarray(0.5 + 0.1 * jax.random.normal(jax.random.PRNGKey(62), (n_e,)))
    )

    # One frame per cycle, exactly as the windowed script builds them.
    obs_key = jax.random.PRNGKey(63)
    observations = []
    for cycle in range(num_cycles):
        obs_key, subkey = jax.random.split(obs_key)
        values = 0.4 + 0.1 * np.asarray(jax.random.normal(subkey, (1, 1)))
        observations.append(_labelled_truth_obs(values, np.array([1.0])))

    def _filter() -> EnsembleKalmanFilter:
        return EnsembleKalmanFilter(
            observation_operator=_FinalFrameTemporalToyObsOp(H),
            forward_model=_ToyLinearModel(np.eye(2), param_effect=0.7),
            C_D=jnp.array([0.2]),
            mode="joint",
            rng_key=jax.random.PRNGKey(64),
            **spread,
        )

    single = _filter()
    single.collect_pred_obs = True
    whole = single.run(state=state, params=params, observations=observations)

    windowed = _filter()
    windowed.collect_pred_obs = True
    carried_state: Optional[xarray.Dataset] = state
    carried_params: Optional[xarray.Dataset] = params
    window_results = []
    windowed_pred_obs: list[np.ndarray] = []
    windowed_pred_obs_post: list[np.ndarray] = []
    for first in range(0, num_cycles, cycles_per_window):
        result = windowed.run(
            state=carried_state,
            params=carried_params,
            observations=observations[first : first + cycles_per_window],
            return_history=True,
        )
        window_results.append(result)
        # Both histories are rebound per call, which is exactly what per-window
        # saving wants — read them before the next window overwrites them.
        windowed_pred_obs.extend(windowed.pred_obs_history)
        windowed_pred_obs_post.extend(windowed.pred_obs_post_history)
        carried_state, carried_params = result.state, result.params

    assert whole.state is not None and window_results[-1].state is not None
    np.testing.assert_array_equal(
        np.asarray(window_results[-1].state["u"]), np.asarray(whole.state["u"])
    )
    assert whole.params is not None and window_results[-1].params is not None
    np.testing.assert_array_equal(
        np.asarray(window_results[-1].params["a"]), np.asarray(whole.params["a"])
    )

    # Per-cycle diagnostics, in global cycle order (the script renumbers the
    # per-call `cycle` field the same way).
    windowed_diagnostics = [d for r in window_results for d in r.diagnostics]
    assert len(windowed_diagnostics) == num_cycles
    for chunked, reference in zip(windowed_diagnostics, whole.diagnostics):
        assert chunked.obs_prior_rmse == reference.obs_prior_rmse
        assert chunked.obs_posterior_rmse == reference.obs_posterior_rmse
        assert chunked.innovation_chi2 == reference.innovation_chi2
        assert chunked.state_spread_posterior == reference.state_spread_posterior
        assert chunked.param_spread_posterior == reference.param_spread_posterior

    # The observation-space histories the per-window artifacts stack are the
    # same rows in the same order, merely split at the window boundary.
    assert len(windowed_pred_obs) == num_cycles
    for chunked_obs, reference_obs in zip(windowed_pred_obs, single.pred_obs_history):
        np.testing.assert_array_equal(chunked_obs, reference_obs)
    for chunked_obs, reference_obs in zip(
        windowed_pred_obs_post, single.pred_obs_post_history
    ):
        np.testing.assert_array_equal(chunked_obs, reference_obs)


def test_C_D_is_checked_against_the_per_frame_observation_vector() -> None:
    """C_D sized for the whole segment (T frames) is rejected.

    The serial sweep assimilates one frame at a time, so ``C_D`` is that one
    frame's error covariance — a vector sized for the stacked segment is the
    natural mistake and must fail loudly instead of misaligning the sweep.
    """
    state = _initial_state(jax.random.PRNGKey(33), 6, np.zeros(2), np.eye(2))
    enkf = EnsembleKalmanFilter(
        observation_operator=_TemporalToyObsOp(np.array([[1.0, 0.5]])),
        forward_model=_ToyLinearModel(np.eye(2)),
        C_D=jnp.array([0.2, 0.2]),
        mode="state",
        rng_key=jax.random.PRNGKey(34),
    )
    with pytest.raises(ValueError, match="C_D"):
        enkf.run(
            state=state,
            observations=[
                _labelled_truth_obs(np.array([[0.2], [0.8]]), np.array([0.0, 1.0]))
            ],
        )


# ---------------------------------------------------------------------------
# assimilate_every_n_step: thin the ANALYSES, not the observations
# ---------------------------------------------------------------------------


class _MultiFrameToyModel(_ToyLinearModel):
    """``_ToyLinearModel`` emitting ``num_frames`` frames per forecast segment.

    One propagation step per output frame, so the frames genuinely differ and
    a test can tell WHICH of them an analysis used.
    """

    def __init__(self, A: np.ndarray, num_frames: int) -> None:
        super().__init__(A)
        self.num_frames = num_frames

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> xarray.Dataset:
        assert state is not None
        x = jnp.asarray(state["u"].values)  # (N_e, nx)
        frames = []
        for _ in range(self.num_frames):
            x = x @ self.A.T
            frames.append(x)
        stacked = jnp.stack(frames, axis=1)  # (N_e, T, nx)
        return xarray.Dataset(
            {"u": (("ensemble", "time", "x"), stacked)},
            coords={
                "ensemble": np.arange(stacked.shape[0]),
                "time": np.arange(1, self.num_frames + 1, dtype=float),
                "x": np.arange(stacked.shape[2]),
            },
        )


class _StridedObservationOperator:
    """The archived filtering script's ORIGINAL stride mechanism.

    Kept here as a fixture — the script drops it in favour of
    ``BaseFilter.assimilate_every_n_step`` — so the library knob can be pinned
    against the mechanism it replaces: wrap the operator, subset its labelled
    output to ``[n-1::n]``, and hand the filter the truth's matching strided
    frames. Anything the knob computes differently would show up as a
    difference against this.
    """

    def __init__(self, operator: Any, stride: int) -> None:
        self._operator = operator
        self._stride = int(stride)

    def __call__(self, state: xarray.Dataset) -> Any:
        obs = self._operator(state)
        if isinstance(obs, xarray.DataArray) and "time" in obs.dims:
            return obs.isel(time=slice(self._stride - 1, None, self._stride))
        return obs

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._operator, name)


_STRIDE_A = np.array([[1.0, 0.2], [0.0, 1.0]])


def _stride_filter(
    num_frames: int,
    stride: int,
    *,
    wrapped: bool = False,
    seed: int = 70,
) -> EnsembleKalmanFilter:
    """An EnKF over a ``num_frames``-frame segment, strided by knob or wrapper."""
    operator: Any = _TemporalToyObsOp(np.array([[1.0, 0.5]]))
    if wrapped:
        operator = _StridedObservationOperator(operator, stride)
    enkf = EnsembleKalmanFilter(
        observation_operator=operator,
        forward_model=_MultiFrameToyModel(_STRIDE_A, num_frames),
        C_D=jnp.array([0.2]),
        mode="state",
        rng_key=jax.random.PRNGKey(seed),
    )
    if not wrapped:
        enkf.assimilate_every_n_step = stride
    return enkf


def _stride_batches(values: np.ndarray) -> list[xarray.DataArray]:
    """``(num_cycles, T, 1)`` values as one labelled batch per cycle."""
    return [
        _labelled_truth_obs(cycle, np.arange(1, cycle.shape[0] + 1, dtype=float))
        for cycle in values
    ]


def test_a_stride_matches_the_wrapped_operator_mechanism() -> None:
    """The knob == the script's old wrapper + double-sliced truth, bit for bit.

    The stride used to live in ``run_filtering.py`` as an operator wrapper
    (predicted side) plus an ``isel(time=slice(n-1, None, n))`` on the truth
    (real side). Lifting it into the filter must change nothing about the
    analysis: same frames, same order, same PRNG draws — which is what lets the
    script delete its wrapper without re-tuning a single run.
    """
    n_e, num_cycles, num_frames, stride = 12, 3, 4, 2
    state = _initial_state(jax.random.PRNGKey(71), n_e, np.zeros(2), np.eye(2))
    values = np.asarray(
        jax.random.normal(jax.random.PRNGKey(72), (num_cycles, num_frames, 1))
    )

    knob = _stride_filter(num_frames, stride)
    knob.collect_pred_obs = True
    knob_result = knob.run(state=state, observations=_stride_batches(values))

    wrapper = _stride_filter(num_frames, stride, wrapped=True)
    wrapper.collect_pred_obs = True
    wrapper_result = wrapper.run(
        # The truth-side half of the old mechanism.
        state=state,
        observations=_stride_batches(values[:, stride - 1 :: stride]),
    )

    assert knob_result.state is not None and wrapper_result.state is not None
    np.testing.assert_array_equal(
        np.asarray(knob_result.state["u"]), np.asarray(wrapper_result.state["u"])
    )
    for analysed, reference in zip(knob_result.diagnostics, wrapper_result.diagnostics):
        assert analysed.obs_prior_rmse == reference.obs_prior_rmse
        assert analysed.obs_posterior_rmse == reference.obs_posterior_rmse
        assert analysed.innovation_chi2 == reference.innovation_chi2
    for analysed_obs, reference_obs in zip(
        knob.pred_obs_history, wrapper.pred_obs_history
    ):
        np.testing.assert_array_equal(analysed_obs, reference_obs)

    # Negative control: the OTHER subset of the same frames is a different
    # filter, so the equality above is not "any stride matches any stride".
    leading = _stride_filter(num_frames, stride, wrapped=True)
    leading_result = leading.run(
        state=state, observations=_stride_batches(values[:, 0::stride])
    )
    assert leading_result.state is not None
    assert not np.allclose(
        np.asarray(leading_result.state["u"]), np.asarray(knob_result.state["u"])
    )


def test_a_stride_records_full_frames_but_analyses_the_strided_rows() -> None:
    """Frames history full-resolution, flat histories on the analysed rows.

    The split ``run_filtering``'s per-window artifacts depend on: the labelled
    frames keep every frame the operator produced (so a caller reading them
    back does not lose frames to the stride), while the flat prior/posterior
    blocks hold only the rows an analysis actually touched (so they stay
    row-alignable with each other).
    """
    n_e, num_cycles, num_frames, stride = 8, 2, 4, 2
    state = _initial_state(jax.random.PRNGKey(73), n_e, np.zeros(2), np.eye(2))
    values = np.asarray(
        jax.random.normal(jax.random.PRNGKey(74), (num_cycles, num_frames, 1))
    )

    enkf = _stride_filter(num_frames, stride)
    enkf.collect_pred_obs = True
    result = enkf.run(state=state, observations=_stride_batches(values))

    assert len(enkf.pred_obs_frames_history) == num_cycles
    for frames in enkf.pred_obs_frames_history:
        assert frames is not None and frames.sizes["time"] == num_frames
    assert len(enkf.pred_obs_history) == num_cycles
    for prior, posterior in zip(enkf.pred_obs_history, enkf.pred_obs_post_history):
        assert prior.shape == (num_frames // stride, n_e)
        assert posterior.shape == prior.shape
    # The recorded prior rows ARE the strided subset of the full frames.
    for frames, prior in zip(enkf.pred_obs_frames_history, enkf.pred_obs_history):
        assert frames is not None
        full = np.asarray(frames.transpose("ensemble", "time", "obs").values)
        np.testing.assert_allclose(
            prior,
            full[:, stride - 1 :: stride, :].transpose(1, 2, 0).reshape(-1, n_e),
            rtol=1e-6,
            atol=1e-6,
        )
    # One diagnostic per cycle, over the analysed frames only.
    assert len(result.diagnostics) == num_cycles


def test_a_stride_of_one_is_the_unstrided_filter() -> None:
    """The default is the identity, down to the PRNG state."""
    n_e, num_cycles, num_frames = 8, 2, 3
    state = _initial_state(jax.random.PRNGKey(75), n_e, np.zeros(2), np.eye(2))
    values = np.asarray(
        jax.random.normal(jax.random.PRNGKey(76), (num_cycles, num_frames, 1))
    )

    strided = _stride_filter(num_frames, 1)
    strided_result = strided.run(state=state, observations=_stride_batches(values))
    plain = _stride_filter(num_frames, 1)
    del plain.assimilate_every_n_step  # back to the class default
    plain_result = plain.run(state=state, observations=_stride_batches(values))

    assert strided_result.state is not None and plain_result.state is not None
    np.testing.assert_array_equal(
        np.asarray(strided_result.state["u"]), np.asarray(plain_result.state["u"])
    )
    np.testing.assert_array_equal(
        np.asarray(strided.rng_key), np.asarray(plain.rng_key)
    )


def test_a_stride_that_does_not_divide_the_cycles_frames_is_rejected() -> None:
    """Loud, up front, and naming both numbers.

    A stride that does not tile the batch would leave the last analysis on some
    interior frame instead of the segment's own final one — the frame the state
    analysis actually updates — which is invisible in the output.
    """
    state = _initial_state(jax.random.PRNGKey(77), 6, np.zeros(2), np.eye(2))
    enkf = _stride_filter(3, 2)
    with pytest.raises(ValueError, match="assimilate_every_n_step=2"):
        enkf.run(
            state=state,
            observations=_stride_batches(np.zeros((1, 3, 1))),
        )

    enkf.assimilate_every_n_step = 0
    with pytest.raises(ValueError, match=">= 1"):
        enkf.run(state=state, observations=_stride_batches(np.zeros((1, 4, 1))))


def test_pred_obs_frames_history_keeps_the_operators_labelled_output() -> None:
    """The third history: ``H(x_f)`` with its TIME axis still attached.

    The flat ``(T*N_obs, N_e)`` block has lost the time coordinate, so a caller
    that has to re-express the predicted observations in another space cannot
    reconstruct them from it. Recorded under the same gate as the other two, one entry per
    cycle, and ``None`` for an operator with nothing to label — so the indices
    stay aligned entry for entry.
    """
    n_e, num_cycles = 8, 2
    H = np.array([[1.0, 0.5]])
    state = _initial_state(jax.random.PRNGKey(70), n_e, np.zeros(2), np.eye(2))
    observations = [
        _labelled_truth_obs(np.array([[0.2], [0.8]]), np.array([0.0, 1.0]))
        for _ in range(num_cycles)
    ]

    def _filter(observation_operator: Any) -> EnsembleKalmanFilter:
        return EnsembleKalmanFilter(
            observation_operator=observation_operator,
            forward_model=_ToyLinearModel(np.eye(2)),
            C_D=jnp.array([0.2]),
            mode="state",
            rng_key=jax.random.PRNGKey(71),
        )

    # Default-off: the history stays empty, so the flag remains the only cost.
    quiet = _filter(_TemporalToyObsOp(H))
    quiet.run(state=state, observations=observations)
    assert quiet.pred_obs_frames_history == []

    labelled = _filter(_TemporalToyObsOp(H))
    labelled.collect_pred_obs = True
    labelled.run(state=state, observations=observations)

    assert len(labelled.pred_obs_frames_history) == num_cycles
    forecast = _ToyLinearModel(np.eye(2)).run_ensemble(state=state)
    expected = _TemporalToyObsOp(H)(forecast)
    frames = labelled.pred_obs_frames_history[0]
    assert isinstance(frames, xarray.DataArray)
    assert frames.dims == ("ensemble", "time", "obs")
    np.testing.assert_allclose(
        np.asarray(frames.transpose("ensemble", "time", "obs").values),
        np.asarray(expected),
        rtol=1e-6,
        atol=1e-6,
    )
    # The same rows the flat history holds, before the time axis was folded in.
    np.testing.assert_allclose(
        np.asarray(frames.values).transpose(1, 2, 0).reshape(-1, n_e),
        labelled.pred_obs_history[0],
        rtol=1e-6,
        atol=1e-6,
    )
    # Rebound per call, like the other histories.
    labelled.run(state=state, observations=observations[:1])
    assert len(labelled.pred_obs_frames_history) == 1

    # An operator returning a plain array has no time labels to keep: the
    # placeholder is what holds the index alignment.
    plain = _filter(_ToyObsOp(H))
    plain.collect_pred_obs = True
    plain.run(state=state, observations=jnp.asarray(np.array([[0.2], [0.8]])))
    assert plain.pred_obs_frames_history == [None, None]
    assert len(plain.pred_obs_history) == num_cycles


# ---------------------------------------------------------------------------
# collect_forecast_frames: the frames BETWEEN the analyses
# ---------------------------------------------------------------------------


def test_forecast_frames_hold_the_free_run_between_analyses() -> None:
    """Every output frame of every segment, and they really are the FORECAST.

    ``state_history`` shows the ensemble at the analysis times alone — under a
    stride, one frame in every ``n`` — so "the ensemble drifts away from the
    truth between analyses, then the analysis pulls it back" is invisible in it.
    The recorded frames are what makes that readable, which requires two things
    of them: cycle ``k``'s segment continues from cycle ``k-1``'s ANALYSIS (it
    is the free run the analysis launched, not an independent rollout), and its
    last frame is the PRIOR of cycle ``k``'s analysis, not the posterior
    ``state_history`` already holds.
    """
    n_e, num_cycles, num_frames, stride = 8, 3, 4, 4
    state = _initial_state(jax.random.PRNGKey(78), n_e, np.zeros(2), np.eye(2))
    values = np.asarray(
        jax.random.normal(jax.random.PRNGKey(79), (num_cycles, num_frames, 1))
    )
    batches = _stride_batches(values)

    plain = _stride_filter(num_frames, stride)
    plain_result = plain.run(state=state, observations=batches, return_history=True)
    recording = _stride_filter(num_frames, stride)
    recording.collect_forecast_frames = True
    result = recording.run(state=state, observations=batches, return_history=True)

    # Recording is a pure observer: same analyses, same PRNG stream.
    assert plain_result.forecast_history is None
    assert plain_result.state is not None and result.state is not None
    np.testing.assert_array_equal(
        np.asarray(plain_result.state["u"]), np.asarray(result.state["u"])
    )
    np.testing.assert_array_equal(
        np.asarray(plain.rng_key), np.asarray(recording.rng_key)
    )

    frames = result.forecast_history
    assert frames is not None
    # Full resolution: every frame of every segment, stride or no stride.
    assert frames.sizes["time"] == num_cycles * num_frames
    assert frames.sizes["ensemble"] == n_e
    assert result.state_history is not None
    assert result.state_history.sizes["cycle"] == num_cycles

    forecast = np.asarray(frames.transpose("ensemble", "time", "x")["u"].values)
    analysed = np.asarray(
        result.state_history.transpose("cycle", "ensemble", "x")["u"].values
    )
    for cycle in range(num_cycles):
        prior = forecast[:, (cycle + 1) * num_frames - 1]
        assert not np.allclose(prior, analysed[cycle])
        if cycle:
            np.testing.assert_allclose(
                forecast[:, cycle * num_frames],
                analysed[cycle - 1] @ np.asarray(_STRIDE_A).T,
                rtol=1e-6,
                atol=1e-6,
            )


def test_forecast_frames_are_rebound_per_run_and_off_by_default() -> None:
    """Default: nothing recorded. On: one call's frames, not two calls' worth."""
    n_e, num_frames = 6, 2
    state = _initial_state(jax.random.PRNGKey(80), n_e, np.zeros(2), np.eye(2))
    values = np.asarray(jax.random.normal(jax.random.PRNGKey(81), (2, num_frames, 1)))

    enkf = _stride_filter(num_frames, 1)
    assert (
        enkf.run(state=state, observations=_stride_batches(values)).forecast_history
        is None
    )

    enkf.collect_forecast_frames = True
    first = enkf.run(state=state, observations=_stride_batches(values))
    second = enkf.run(state=state, observations=_stride_batches(values[:1]))
    assert first.forecast_history is not None and second.forecast_history is not None
    assert first.forecast_history.sizes["time"] == 2 * num_frames
    assert second.forecast_history.sizes["time"] == num_frames


def test_forecast_frames_agree_on_the_in_memory_and_on_disk_paths(
    tmp_path: pathlib.Path,
) -> None:
    """The segment reader keeps the same frames however the forecast was held.

    On-disk mode is where the frames would otherwise be lost to pruning, and it
    is the mode production runs use, so its reader is the one that has to agree
    with the in-memory path frame for frame — not just at the analysis times.
    """
    n_e = 10
    state = _initial_state(jax.random.PRNGKey(82), n_e, np.zeros(2), np.eye(2))
    common: Any = dict(
        observation_operator=_ToyObsOp(np.array([[1.0, 0.5]])),
        C_D=jnp.array([0.2]),
        mode="state",
        rng_key=jax.random.PRNGKey(83),
    )
    observations = jnp.array([[0.5], [0.3]])

    in_memory = EnsembleKalmanFilter(forward_model=_ToyLinearModel(_STRIDE_A), **common)
    in_memory.collect_forecast_frames = True
    memory_result = in_memory.run(state=state, observations=observations)
    on_disk = EnsembleKalmanFilter(
        forward_model=_OnDiskToyLinearModel(_STRIDE_A, tmp_path), **common
    )
    on_disk.collect_forecast_frames = True
    disk_result = on_disk.run(state=state, observations=observations)

    assert memory_result.forecast_history is not None
    assert disk_result.forecast_history is not None
    memory_frames = memory_result.forecast_history.transpose("ensemble", "time", "x")
    disk_frames = disk_result.forecast_history.transpose("ensemble", "time", "x")
    # Two cycles of the toy model's two-frame segment.
    assert memory_frames.sizes["time"] == 4
    np.testing.assert_allclose(
        np.asarray(disk_frames["u"].values),
        np.asarray(memory_frames["u"].values),
        rtol=1e-5,
        atol=2e-5,
    )


# ---------------------------------------------------------------------------
# Reduced physical-state analyses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["state", "joint"])  # type: ignore[misc, unused-ignore]
@pytest.mark.parametrize(
    "inflation",
    [
        None,
        MultiplicativeInflation(1.25),
        RTPS(alpha=0.65),
        RTPP(alpha=0.4),
    ],
    ids=["none", "multiplicative", "rtps", "rtpp"],
)  # type: ignore[misc, unused-ignore]
def test_full_rank_current_reduction_matches_physical_update(
    mode: str, inflation: Optional[Any]
) -> None:
    """Full-rank POD preserves state/joint EnKF and inflation semantics.

    The ``rtps`` case is also the regression against relaxing *coefficient*
    rows: RTPS rescales each row by its own prior/posterior spread ratio, which
    does not commute with a basis rotation, so a posterior hook applied in
    modal coordinates would break this equality even at full rank. (RTPP is
    linear and multiplicative inflation is uniform, so those cases check the
    plumbing rather than the non-commutation.)
    """
    n_e, n_x = 32, 4
    rng = np.random.default_rng(20)
    state = _initial_state(jax.random.PRNGKey(21), n_e, np.zeros(n_x), np.eye(n_x))
    params = _params_dataset(rng.standard_normal(n_e))
    common: Any = dict(
        observation_operator=_ToyObsOp(
            np.array([[1.0, 0.2, -0.1, 0.0], [0.0, 0.3, 0.5, 1.0]])
        ),
        C_D=jnp.array([0.15, 0.3]),
        mode=mode,
        inflation=inflation,
        parameter_evolution=(
            RandomWalkEvolution(std={"a": 0.0})
            if mode == "joint" and inflation is None
            else None
        ),
        rng_key=jax.random.PRNGKey(22),
    )
    A = np.array(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.8, 0.2, 0.0],
            [0.1, 0.0, 0.7, 0.1],
            [0.0, 0.0, 0.2, 0.9],
        ]
    )
    observations = jnp.array([[0.7, -0.2]])

    full = EnsembleKalmanFilter(
        forward_model=_ToyLinearModel(A, param_effect=0.2), **common
    ).run(state=state, params=params, observations=observations)
    reduced = EnsembleKalmanFilter(
        forward_model=_ToyLinearModel(A, param_effect=0.2),
        state_reduction=OnlineStateReduction(energy_fraction=1.0, whiten=False),
        **common,
    ).run(state=state, params=params, observations=observations)

    assert full.state is not None and reduced.state is not None
    np.testing.assert_allclose(
        reduced.state["u"], full.state["u"], rtol=1e-5, atol=2e-5
    )
    if mode == "joint":
        assert full.params is not None and reduced.params is not None
        np.testing.assert_allclose(
            reduced.params["a"], full.params["a"], rtol=1e-5, atol=2e-5
        )
    else:
        assert reduced.params is params

    diag = reduced.diagnostics[0]
    full_diag = full.diagnostics[0]
    assert diag.state_spread_prior == pytest.approx(full_diag.state_spread_prior)
    assert diag.state_spread_posterior == pytest.approx(
        full_diag.state_spread_posterior
    )
    if mode == "joint":
        assert diag.param_spread_prior == pytest.approx(full_diag.param_spread_prior)
        assert diag.param_spread_posterior == pytest.approx(
            full_diag.param_spread_posterior
        )
    else:
        assert diag.param_spread_prior is None
        assert diag.param_spread_posterior is None
    assert diag.reduction_rank is not None
    assert diag.reduction_rank <= n_e - 1
    assert diag.reduction_available_rank is not None
    assert diag.reduction_retained_energy == pytest.approx(1.0, abs=2e-6)
    assert diag.reduction_discarded_increment_fraction == pytest.approx(0.0, abs=2e-5)
    assert diag.reduction_basis_time is not None
    assert diag.analysis_time is not None
    assert diag.reduction_basis_updated is True
    assert diag.obs_posterior_rmse_kind == "unreduced_ride_along"


@pytest.mark.parametrize(
    "make_reduction",
    [
        lambda: OnlineStateReduction(energy_fraction=0.5, max_rank=1, whiten=False),
        lambda: StreamingStateReduction(energy_fraction=0.5, max_rank=1),
    ],
    ids=["current", "streaming"],
)  # type: ignore[misc, unused-ignore]
def test_reduced_zero_gain_preserves_every_physical_member(
    make_reduction: Any,
) -> None:
    """An observation with zero spread cannot erase projection residuals."""
    reduction = make_reduction()
    n_e = 12
    state = _initial_state(
        jax.random.PRNGKey(23), n_e, np.array([1.0, -1.0]), np.eye(2)
    )
    result = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(np.zeros((1, 2))),
        forward_model=_ToyLinearModel(np.eye(2)),
        C_D=jnp.array([0.1]),
        mode="state",
        state_reduction=reduction,
        rng_key=jax.random.PRNGKey(24),
    ).run(state=state, observations=jnp.array([[5.0]]))

    assert result.state is not None
    np.testing.assert_allclose(result.state["u"], state["u"], atol=2e-6)
    assert result.diagnostics[0].reduction_increment_norm == pytest.approx(
        0.0, abs=1e-7
    )


def test_truncated_reduction_preserves_state_contract_and_is_finite() -> None:
    state = _initial_state(jax.random.PRNGKey(25), 16, np.zeros(3), np.eye(3)).astype(
        np.float32
    )
    result = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(np.array([[1.0, 0.0, 0.0]])),
        forward_model=_ToyLinearModel(np.eye(3)),
        C_D=jnp.array([0.2]),
        mode="state",
        state_reduction=OnlineStateReduction(max_rank=1, whiten=False),
        rng_key=jax.random.PRNGKey(26),
    ).run(state=state, observations=jnp.array([[1.0]]))

    assert result.state is not None
    assert result.state.sizes == state.sizes
    assert result.state["u"].dims == state["u"].dims
    assert result.state["u"].dtype == state["u"].dtype
    np.testing.assert_array_equal(result.state.coords["x"], state.coords["x"])
    assert bool(np.isfinite(result.state["u"]).all())
    diag = result.diagnostics[0]
    assert diag.reduction_rank == 1
    assert diag.reduction_available_rank is not None
    assert diag.reduction_rank < diag.reduction_available_rank
    # Truncation really did throw part of the full-space update away — the
    # control measurement is not silently reporting "nothing discarded".
    assert diag.reduction_projection_residual is not None
    assert diag.reduction_projection_residual > 1e-3
    assert diag.reduction_discarded_increment_fraction is not None
    assert diag.reduction_discarded_increment_fraction > 1e-3


@pytest.mark.parametrize("mode", ["state", "joint"])  # type: ignore[misc, unused-ignore]
def test_streaming_reduction_matches_full_update_across_cycles(mode: str) -> None:
    """A full-rank streaming basis keeps spanning the current ensemble.

    The EnKF increment already lies in the span of the current forecast
    anomalies, and an untruncated accumulated basis contains that span, so a
    multi-cycle streaming run must stay identical to the unreduced filter.
    """
    # N_s exceeds even the ACCUMULATED rank (n_cycles * (N_e - 1)), so the
    # basis stays a genuine projector on every cycle rather than becoming a
    # square rotation (which would make this test unfalsifiable).
    n_e, n_x, n_cycles = 6, 24, 3
    rng = np.random.default_rng(31)
    state = _initial_state(jax.random.PRNGKey(32), n_e, np.zeros(n_x), np.eye(n_x))
    params = _params_dataset(rng.standard_normal(n_e))
    A = 0.9 * np.eye(n_x) + 0.1 * np.eye(n_x, k=1)
    common: Any = dict(
        observation_operator=_ToyObsOp(np.eye(1, n_x)),
        C_D=jnp.array([0.2]),
        mode=mode,
        inflation=RTPS(alpha=0.5),
        rng_key=jax.random.PRNGKey(33),
    )
    observations = jnp.asarray(rng.standard_normal((n_cycles, 1)))

    full = EnsembleKalmanFilter(
        forward_model=_ToyLinearModel(A, param_effect=0.3), **common
    ).run(state=state, params=params, observations=observations)
    streaming = EnsembleKalmanFilter(
        forward_model=_ToyLinearModel(A, param_effect=0.3),
        state_reduction=StreamingStateReduction(energy_fraction=1.0),
        **common,
    ).run(state=state, params=params, observations=observations)

    assert full.state is not None and streaming.state is not None
    np.testing.assert_allclose(
        streaming.state["u"], full.state["u"], rtol=1e-5, atol=2e-5
    )
    if mode == "joint":
        assert full.params is not None and streaming.params is not None
        np.testing.assert_allclose(
            streaming.params["a"], full.params["a"], rtol=1e-5, atol=2e-5
        )
    else:
        assert streaming.params is params
    for cycle, diag in enumerate(streaming.diagnostics):
        assert diag.reduction_rank is not None and diag.reduction_rank < n_x
        # The split is always available: the incremental fit must not fall back
        # to its re-orthogonalization branch (which cannot report one).
        assert diag.reduction_discarded_increment_fraction == pytest.approx(
            0.0, abs=2e-5
        )
        if cycle:
            # Consecutive bases of a slowly evolving system stay close; a rank
            # that merely grows must not be reported as a 90-degree rotation.
            assert diag.reduction_subspace_drift is not None
            assert diag.reduction_subspace_drift < 0.4 * np.pi / 2


class _TwoVariableToyModel(_ToyLinearModel):
    """Persistence forecast of a two-variable state with very different scales."""

    def __init__(self) -> None:
        super().__init__(np.eye(1))

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> xarray.Dataset:
        assert state is not None
        frames = {
            name: (
                ("ensemble", "time", "x"),
                jnp.stack([jnp.asarray(state[name].values)] * 2, axis=1),
            )
            for name in ("u", "v")
        }
        return xarray.Dataset(
            frames,
            coords={
                "ensemble": state.coords["ensemble"],
                "time": np.array([0.0, 1.0]),
                "x": state.coords["x"],
            },
        )


class _TwoVariableObsOp:
    """Observe both variables of the forecast's final frame."""

    def __init__(self, H_u: np.ndarray, H_v: np.ndarray) -> None:
        self.H_u = jnp.asarray(H_u)
        self.H_v = jnp.asarray(H_v)

    def __call__(self, state: xarray.Dataset) -> jnp.ndarray:
        u = jnp.asarray(state["u"].isel(time=-1).values)
        v = jnp.asarray(state["v"].isel(time=-1).values)
        return u @ self.H_u.T + v @ self.H_v.T


class _DonorSubstitutingModel(_ToyLinearModel):
    """Clones member 0 over member 1, as the ensemble model's repair path does."""

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> xarray.Dataset:
        forecast = super().run_ensemble(state=state, params=params)
        values = np.array(forecast["u"].values, copy=True)
        values[1] = values[0]
        forecast["u"] = (forecast["u"].dims, jnp.asarray(values))
        return forecast


def test_reduced_update_is_exact_when_the_basis_is_a_real_projector() -> None:
    """``N_s > N_e - 1``: ``U_r`` truly projects, and the EnKF stays exact.

    The stochastic EnKF increment lies in the span of the forecast anomalies,
    which a full statistical-rank basis reproduces exactly — so equality here
    tests the projection, not an orthogonal change of basis (which is all a
    square ``U`` can be).
    """
    n_e, n_x = 6, 12
    state = _initial_state(jax.random.PRNGKey(40), n_e, np.zeros(n_x), np.eye(n_x))
    common: Any = dict(
        observation_operator=_ToyObsOp(np.eye(2, n_x)),
        C_D=jnp.array([0.2, 0.3]),
        mode="state",
        rng_key=jax.random.PRNGKey(41),
    )
    observations = jnp.array([[0.8, -0.4]])
    full = EnsembleKalmanFilter(
        forward_model=_ToyLinearModel(np.eye(n_x)), **common
    ).run(state=state, observations=observations)
    reduced = EnsembleKalmanFilter(
        forward_model=_ToyLinearModel(np.eye(n_x)),
        state_reduction=OnlineStateReduction(energy_fraction=1.0, whiten=False),
        **common,
    ).run(state=state, observations=observations)

    diag = reduced.diagnostics[0]
    assert diag.reduction_rank == n_e - 1 < n_x  # a rank-5 basis in a 12-D state
    assert full.state is not None and reduced.state is not None
    np.testing.assert_allclose(
        reduced.state["u"], full.state["u"], rtol=1e-5, atol=2e-5
    )
    assert diag.reduction_discarded_increment_fraction == pytest.approx(0.0, abs=2e-5)
    # The basis does NOT span the state space: residual energy stays on members.
    assert diag.reduction_projection_residual == pytest.approx(0.0, abs=1e-5)


def test_variable_scales_are_inverted_when_the_increment_is_decoded() -> None:
    """A non-unit state norm must leave a full-rank analysis unchanged."""
    n_e, n_x = 10, 3
    key_u, key_v = jax.random.split(jax.random.PRNGKey(42))
    state = xarray.Dataset(
        {
            "u": (("ensemble", "x"), jax.random.normal(key_u, (n_e, n_x))),
            # 100x larger units: a dropped or mis-applied scale shows up as a
            # large error in the decoded increment.
            "v": (("ensemble", "x"), 100.0 * jax.random.normal(key_v, (n_e, n_x))),
        },
        coords={"ensemble": np.arange(n_e), "x": np.arange(n_x)},
    )
    reduction = OnlineStateReduction(
        energy_fraction=1.0, whiten=False, variable_scales={"v": 100.0}
    )
    common: Any = dict(
        observation_operator=_TwoVariableObsOp(
            np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.01, 0.0]])
        ),
        C_D=jnp.array([0.25]),
        mode="state",
        rng_key=jax.random.PRNGKey(43),
    )
    observations = jnp.array([[1.5]])
    full = EnsembleKalmanFilter(forward_model=_TwoVariableToyModel(), **common).run(
        state=state, observations=observations
    )
    scaled = EnsembleKalmanFilter(
        forward_model=_TwoVariableToyModel(), state_reduction=reduction, **common
    ).run(state=state, observations=observations)

    assert full.state is not None and scaled.state is not None
    assert reduction.resolved_variable_scales == {"u": 1.0, "v": 100.0}
    for name in ("u", "v"):
        np.testing.assert_allclose(
            scaled.state[name], full.state[name], rtol=1e-5, atol=2e-4
        )


def test_joint_reduction_actually_moves_both_blocks() -> None:
    """Joint mode must update parameters as well as the reduced state."""
    n_e = 20
    rng = np.random.default_rng(44)
    state = _initial_state(jax.random.PRNGKey(45), n_e, np.zeros(2), np.eye(2))
    params = _params_dataset(rng.standard_normal(n_e))
    result = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(np.array([[1.0, 0.5]])),
        forward_model=_ToyLinearModel(np.eye(2), param_effect=0.8),
        C_D=jnp.array([0.1]),
        mode="joint",
        inflation=RTPS(alpha=0.5),
        state_reduction=OnlineStateReduction(max_rank=1, whiten=False),
        rng_key=jax.random.PRNGKey(46),
    ).run(state=state, params=params, observations=jnp.array([[2.5]]))

    assert result.params is not None and result.state is not None
    assert not np.allclose(result.params["a"], params["a"])
    assert not np.allclose(result.state["u"], state["u"])


def test_donor_substituted_forecast_stays_valid_reduction_input() -> None:
    """A repaired (duplicated) member is rank-deficient, not invalid."""
    n_e, n_x = 5, 8  # N_s > N_e so the ensemble, not the state, bounds the rank
    state = _initial_state(jax.random.PRNGKey(47), n_e, np.zeros(n_x), np.eye(n_x))
    result = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(np.eye(1, n_x)),
        forward_model=_DonorSubstitutingModel(np.eye(n_x)),
        C_D=jnp.array([0.2]),
        mode="state",
        state_reduction=OnlineStateReduction(energy_fraction=1.0, whiten=False),
        rng_key=jax.random.PRNGKey(48),
    ).run(state=state, observations=jnp.array([[1.0]]))

    assert result.state is not None
    assert bool(np.isfinite(result.state["u"]).all())
    # N_e - 1 = 4 distinct anomaly directions; the cloned member costs one.
    assert result.diagnostics[0].reduction_available_rank == 3


def test_shipped_streaming_defaults_run_through_the_filter() -> None:
    """configs/assimilation_settings/state_reduction.yaml's ``svd_streaming``."""
    n_e, n_cycles = 12, 4
    state = _initial_state(jax.random.PRNGKey(49), n_e, np.zeros(4), np.eye(4))
    reduction = StreamingStateReduction(forgetting_factor=0.9, energy_fraction=0.99)
    result = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(np.array([[1.0, 0.2, 0.0, -0.3]])),
        forward_model=_ToyLinearModel(0.95 * np.eye(4)),
        C_D=jnp.array([0.2]),
        mode="state",
        state_reduction=reduction,
        rng_key=jax.random.PRNGKey(50),
    ).run(state=state, observations=jnp.zeros((n_cycles, 1)))

    assert result.state is not None
    assert bool(np.isfinite(result.state["u"]).all())
    assert reduction.covariance_half_life == pytest.approx(np.log(0.5) / np.log(0.9))
    for cycle, diag in enumerate(result.diagnostics):
        assert diag.reduction_basis_updated is True
        assert diag.reduction_rank is not None and 0 < diag.reduction_rank <= 4
        assert diag.reduction_spectrum_max is not None
        # Cycle 0 has no previous basis to drift from; later cycles do.
        assert (diag.reduction_subspace_drift is None) == (cycle == 0)


def test_reduction_construction_guards_are_actionable() -> None:
    reduction = OnlineStateReduction(whiten=False)
    with pytest.raises(ValueError, match="mode='parameter'"):
        EnsembleKalmanFilter(
            mode="parameter",
            inflation=MultiplicativeInflation(1.0),
            state_reduction=reduction,
            **_dummy_filter_kwargs(),
        )
    with pytest.raises(ValueError, match="incompatible with localization"):
        EnsembleKalmanFilter(
            mode="state",
            localization=_AllOnesLocalization(),
            state_reduction=reduction,
            **_dummy_filter_kwargs(),
        )
    # ESMDA-only snapshot knobs would be silently ignored by the filter.
    with pytest.raises(ValueError, match="basis_source/snapshot_stride"):
        EnsembleKalmanFilter(
            mode="state",
            state_reduction=OnlineStateReduction(
                whiten=False, basis_source="window_snapshots"
            ),
            **_dummy_filter_kwargs(),
        )
    with pytest.raises(ValueError, match="basis_source/snapshot_stride"):
        EnsembleKalmanFilter(
            mode="state",
            state_reduction=OnlineStateReduction(whiten=False, snapshot_stride=4),
            **_dummy_filter_kwargs(),
        )


def test_nonfinite_forecast_does_not_mutate_streaming_basis() -> None:
    state = _initial_state(jax.random.PRNGKey(27), 10, np.zeros(2), np.eye(2))
    reduction = StreamingStateReduction(forgetting_factor=0.9, energy_fraction=1.0)
    enkf = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(np.array([[1.0, 0.0]])),
        forward_model=_ToyLinearModel(np.eye(2)),
        C_D=jnp.array([0.1]),
        mode="state",
        state_reduction=reduction,
        rng_key=jax.random.PRNGKey(28),
    )
    enkf.run(state=state, observations=jnp.array([[0.0]]))
    modes_before = np.asarray(reduction.modes).copy()
    singular_before = np.asarray(reduction.singular_values).copy()

    bad_state = state.copy(deep=True)
    bad_values = np.array(bad_state["u"].values, copy=True)
    bad_values[3, 0] = np.nan
    bad_state["u"] = (bad_state["u"].dims, bad_values)
    with pytest.raises(ValueError, match=r"member/sample columns \[3\]"):
        enkf.run(state=bad_state, observations=jnp.array([[0.0]]))
    np.testing.assert_array_equal(reduction.modes, modes_before)
    np.testing.assert_array_equal(reduction.singular_values, singular_before)


def test_reduced_in_memory_and_on_disk_paths_are_equivalent(
    tmp_path: pathlib.Path,
) -> None:
    n_e = 14
    state = _initial_state(jax.random.PRNGKey(29), n_e, np.zeros(2), np.eye(2))
    common: Any = dict(
        observation_operator=_ToyObsOp(np.array([[1.0, 0.5]])),
        C_D=jnp.array([0.2]),
        mode="state",
        rng_key=jax.random.PRNGKey(30),
    )
    observations = jnp.array([[0.5]])
    in_memory = EnsembleKalmanFilter(
        forward_model=_ToyLinearModel(np.eye(2)),
        state_reduction=OnlineStateReduction(energy_fraction=1.0, whiten=False),
        **common,
    ).run(state=state, observations=observations)
    on_disk = EnsembleKalmanFilter(
        forward_model=_OnDiskToyLinearModel(np.eye(2), tmp_path),
        state_reduction=OnlineStateReduction(energy_fraction=1.0, whiten=False),
        **common,
    ).run(state=state, observations=observations)

    assert in_memory.state is not None and on_disk.state is not None
    np.testing.assert_allclose(
        on_disk.state["u"], in_memory.state["u"], rtol=1e-5, atol=2e-5
    )
    assert (
        on_disk.diagnostics[0].reduction_rank == in_memory.diagnostics[0].reduction_rank
    )


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
        RandomWalkEvolution(std={"a": -0.1})
    # One scalar for every parameter ignores their units: refused.
    with pytest.raises(ValueError, match="per-parameter"):
        RandomWalkEvolution(std=0.1)  # type: ignore[arg-type]


class _ParamRecordingModel(_ParamOnlyModel):
    """``_ParamOnlyModel`` that records the parameters of every forecast."""

    def __init__(self) -> None:
        super().__init__()
        self.received: list[np.ndarray] = []

    def run_ensemble(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> xarray.Dataset:
        assert params is not None
        self.received.append(np.asarray(params["a"].values))
        return super().run_ensemble(state=state, params=params)


def _evolving_param_filter(std: float) -> EnsembleKalmanFilter:
    return EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(np.array([[1.0, 0.0]])),
        forward_model=_ParamRecordingModel(),
        C_D=jnp.array([0.05**2]),
        mode="parameter",
        parameter_evolution=RandomWalkEvolution(std={"a": std}),
        rng_key=jax.random.PRNGKey(12),
    )


def test_evolution_is_applied_before_the_next_forecast_not_to_the_posterior() -> None:
    """Posterior params are pure analyses; the NEXT forecast gets them evolved."""
    n_e = 20
    params = _params_dataset(
        np.asarray(jax.random.normal(jax.random.PRNGKey(13), (n_e,)))
    )
    observations = jnp.array([[1.0], [1.2]])

    enkf = _evolving_param_filter(std=0.3)
    model = cast(_ParamRecordingModel, enkf.forward_model)
    result = enkf.run(params=params, observations=observations, return_history=True)
    assert result.params is not None and result.params_history is not None
    posterior = np.asarray(result.params_history["a"].isel(cycle=1).values)

    # The first forecast uses the prior untouched; the second uses the cycle-0
    # analysis plus random-walk noise, while the saved analysis carries none.
    np.testing.assert_array_equal(model.received[0], np.asarray(params["a"].values))
    assert not np.allclose(model.received[1], posterior)
    np.testing.assert_array_equal(
        np.asarray(result.params["a"].values),
        np.asarray(result.params_history["a"].isel(cycle=-1).values),
    )

    # Same rng consumption without noise: the first analysis is identical, so
    # the noise of the evolution never reached it.
    noiseless = _evolving_param_filter(std=0.0).run(
        params=params, observations=observations, return_history=True
    )
    assert noiseless.params_history is not None
    np.testing.assert_array_equal(
        np.asarray(noiseless.params_history["a"].isel(cycle=1).values), posterior
    )

    # A later run() on the same instance continues the chain: its first
    # forecast is evolved from the parameters it is handed.
    enkf.run(params=result.params, observations=observations[:1])
    assert len(model.received) == 3
    assert not np.allclose(model.received[2], np.asarray(result.params["a"].values))


# ---------------------------------------------------------------------------
# Ensemble-transform cycle diagnostics
#
# The transform diagnostics exist only as attributes of the last analysis call;
# without the filter reading them back, nothing outside the scheme can see the
# resource-gate quantities of docs/plans/implemented/filtering_state_reduction_and_
# transforms.md §6. These tests pin the additive/nullable contract: a field is
# populated exactly on the path where it means something, and None everywhere
# else, so cycle_diagnostics.yaml has one schema for every analysis.
# ---------------------------------------------------------------------------

_TRANSFORM_FIELDS = (
    "transform_available_rank",
    "transform_retained_rank",
    "transform_retained_energy",
    "transform_discarded_spectrum_max",
)
_LOCAL_FIELDS = (
    "local_num_blocks",
    "local_num_active_blocks",
    "local_num_updated_rows",
    "local_active_obs_min",
    "local_active_obs_median",
    "local_active_obs_max",
    "local_retained_rank_min",
    "local_retained_rank_mean",
    "local_retained_rank_max",
    "local_available_rank_max",
    "local_retained_energy_min",
    "local_retained_energy_mean",
    "local_discarded_spectrum_max",
    "local_chunk_size",
)


def test_the_declared_diagnostic_groups_cover_every_field() -> None:
    """The two tuples above must stay the whole of their prefix groups.

    They drive every ``_assert_all_none`` below, so a field added to
    ``CycleDiagnostics`` but not listed here would silently lose its null-path
    coverage — which is how the per-block energy readouts were once absent from
    both this file and docs/data_assimilation.md while the rank fields kept the
    suite green.
    """
    import dataclasses

    from data_assimilation.filtering.base import CycleDiagnostics

    names = [f.name for f in dataclasses.fields(CycleDiagnostics)]
    assert tuple(n for n in names if n.startswith("transform_")) == _TRANSFORM_FIELDS
    assert tuple(n for n in names if n.startswith("local_")) == _LOCAL_FIELDS


def _assert_all_none(diag: Any, fields: tuple[str, ...]) -> None:
    unset = {name: getattr(diag, name) for name in fields}
    assert unset == dict.fromkeys(fields, None)


def _transform_diag_filter(analysis: Any, localization: Any, **overrides: Any) -> Any:
    """One state-mode cycle on the toy model; returns its CycleDiagnostics.

    ``H`` deliberately repeats one row, so the whitened observation anomalies
    have available rank 1 out of a fixed ``min(N_d, N_e) = 2`` — the rank
    diagnostics then distinguish "kept every thin direction" from "truncated to
    the informative one" instead of both reading 2.
    """
    n_e = 16
    state = _initial_state(
        jax.random.PRNGKey(101), n_e, np.array([1.0, -0.5]), 0.4 * np.eye(2)
    )
    enkf = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(np.array([[1.0, 0.0], [1.0, 0.0]])),
        forward_model=_ToyLinearModel(np.array([[0.9, 0.2], [-0.1, 0.8]])),
        C_D=jnp.array([0.1, 0.1]),
        mode="state",
        analysis=analysis,
        localization=localization,
        rng_key=jax.random.PRNGKey(102),
        **overrides,
    )
    result = enkf.run(state=state, observations=jnp.array([[1.0, 1.0]]))
    return result.diagnostics[0]


def test_stochastic_analysis_leaves_every_transform_diagnostic_null() -> None:
    """The default scheme forms no transform, so both groups stay None.

    This is the "one stable schema" half of the contract: a stochastic run's
    cycle_diagnostics.yaml carries the same keys as an ETKF's, all null.
    """
    diag = _transform_diag_filter(analysis=None, localization=None)
    _assert_all_none(diag, _TRANSFORM_FIELDS)
    _assert_all_none(diag, _LOCAL_FIELDS)


def test_global_etkf_records_untruncated_transform_diagnostics() -> None:
    """TSVD off: every thin direction retained, and available_rank says so."""
    diag = _transform_diag_filter(analysis=ETKFAnalysis(), localization=None)

    # Fixed rank min(N_d, N_e) = 2 retained; only one direction is informative.
    assert diag.transform_retained_rank == 2
    assert diag.transform_available_rank == 1
    assert diag.transform_retained_energy == pytest.approx(1.0, abs=1e-5)
    # 0.0, not None: the truncation ran and discarded nothing.
    assert diag.transform_discarded_spectrum_max == 0.0
    _assert_all_none(diag, _LOCAL_FIELDS)


def test_global_etkf_tsvd_records_the_truncation_it_applied() -> None:
    """TSVD on: retained rank drops to the available rank, and the largest
    discarded singular value is recorded (here the round-off direction)."""
    diag = _transform_diag_filter(
        analysis=ETKFAnalysis(tsvd=ObservationTSVD(enabled=True)), localization=None
    )

    assert diag.transform_retained_rank == 1
    assert diag.transform_available_rank == 1
    assert diag.transform_retained_energy == pytest.approx(1.0, abs=1e-5)
    assert diag.transform_discarded_spectrum_max == pytest.approx(0.0, abs=1e-4)
    _assert_all_none(diag, _LOCAL_FIELDS)


def test_letkf_records_local_block_diagnostics() -> None:
    """All-ones localization: one block, every row updated, all obs active."""
    diag = _transform_diag_filter(
        analysis=LETKFAnalysis(), localization=_AllOnesLocalization()
    )

    # Two state rows plus the two appended predicted-observation rows; every one
    # of them sees the same all-ones selection, so they share a single block.
    assert diag.local_num_blocks == 1
    assert diag.local_num_active_blocks == 1
    assert diag.local_num_updated_rows == 4
    assert diag.local_active_obs_min == 2
    assert diag.local_active_obs_median == 2.0
    assert diag.local_active_obs_max == 2
    assert diag.local_retained_rank_min == diag.local_retained_rank_max == 2
    assert diag.local_retained_rank_mean == pytest.approx(2.0)
    assert diag.local_available_rank_max == 1
    # The per-block energy readouts, the local counterparts of
    # transform_retained_energy / transform_discarded_spectrum_max: with the
    # TSVD off the single block keeps everything, so 1.0 and 0.0 — values that
    # mean "the truncation ran and discarded nothing", not "not collected".
    assert diag.local_retained_energy_min == pytest.approx(1.0, abs=1e-5)
    assert diag.local_retained_energy_mean == pytest.approx(1.0, abs=1e-5)
    assert diag.local_discarded_spectrum_max == pytest.approx(0.0, abs=1e-4)
    assert diag.local_chunk_size is not None and diag.local_chunk_size >= 1
    # A localized transform is not a global one; the global group stays null.
    _assert_all_none(diag, _TRANSFORM_FIELDS)


def test_letkf_local_summaries_exclude_blocks_with_no_active_observation() -> None:
    """A row outside every localization radius forms its own inactive block.

    It computes no transform and is returned untouched, so it must not enter the
    active-observation or rank summaries — otherwise every recorded minimum
    would be zero and the gate's worst-block question unanswerable.
    """
    n_e = 24
    signal = jax.random.normal(jax.random.PRNGKey(103), (n_e,))
    state = xarray.Dataset(
        {"u": (("ensemble", "x"), jnp.stack([signal, 0.5 * signal], axis=1))},
        coords={"ensemble": np.arange(n_e), "x": [0.0, 10.0]},
    )
    enkf = EnsembleKalmanFilter(
        observation_operator=_CoordinateToyObsOp(np.array([[1.0, 0.0]])),
        forward_model=_ToyLinearModel(np.eye(2)),
        C_D=jnp.array([0.1]),
        mode="state",
        analysis=LETKFAnalysis(),
        localization=DistanceLocalization(
            localization_radius=0.1, max_inflation=1.0, block_grouping=False
        ),
        rng_key=jax.random.PRNGKey(104),
    )
    result = enkf.run(state=state, observations=jnp.array([[2.0]]))
    diag = result.diagnostics[0]

    # The far row is excluded; the near row and the appended observation row
    # (masked to the global update) both see the single sensor.
    assert diag.local_num_blocks == 2
    assert diag.local_num_active_blocks == 1
    assert diag.local_num_updated_rows == 2
    assert diag.local_active_obs_min == 1  # not 0: the inactive block is excluded
    assert diag.local_active_obs_max == 1
    assert result.state is not None
    np.testing.assert_allclose(
        np.asarray(result.state["u"][:, 1]), np.asarray(state["u"][:, 1]), atol=1e-6
    )


class _CountingTransformAnalysis:
    """Zero-gain analysis stub publishing a DIFFERENT transform on each call.

    Only the diagnostic contract matters here — the stub returns the augmented
    ensemble unchanged — so the two calls per reduced cycle become
    distinguishable, which no real scheme's are.
    """

    localization_policy = "optional"

    def __init__(self) -> None:
        self.calls = 0
        self.last_transform: Any = None

    def __call__(self, augmented: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.last_transform = types.SimpleNamespace(
            available_rank=self.calls,
            retained_rank=self.calls,
            retained_energy=float(self.calls),
            discarded_spectrum=jnp.zeros(0),
        )
        return augmented


def test_transform_diagnostics_come_from_the_posterior_producing_call() -> None:
    """Under a state reduction the analysis runs TWICE per cycle.

    ``_record_reduction_diagnostics`` calls it a second time on the fit
    coordinates, overwriting ``last_transform``. In production both calls see
    the same observations and recompute the same transform, so the ordering
    inside ``_analysis_cycle`` is invisible to every other test; the stub above
    makes it visible, so the recorded values are pinned to the call that
    actually produced the cycle's posterior rather than to a diagnostic's
    implementation detail.
    """
    analysis = _CountingTransformAnalysis()
    diag = _transform_diag_filter(
        analysis=analysis,
        localization=None,
        state_reduction=OnlineStateReduction(energy_fraction=1.0, whiten=False),
    )

    # The second (diagnostic) call really happens — otherwise this test would
    # pass for the wrong reason.
    assert analysis.calls == 2
    assert diag.transform_retained_rank == 1
    assert diag.transform_available_rank == 1
    assert diag.transform_retained_energy == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Beta tempering: R_filter = beta * R (docs/plans/implemented/hybrid_beta_tempering.md)
# ---------------------------------------------------------------------------


def _beta_filter(beta: Optional[float] = None, **overrides: Any) -> Any:
    """A small joint filter, built with or WITHOUT an explicit ``beta``.

    ``beta=None`` omits the argument entirely, so the legacy-equivalence test
    below compares against the constructor exactly as every pre-beta caller
    invoked it rather than against another explicit value.
    """
    kwargs: dict[str, Any] = dict(
        observation_operator=_TemporalToyObsOp(np.array([[1.0, 0.5], [0.0, 1.0]])),
        forward_model=_ToyLinearModel(
            np.array([[0.9, 0.2], [-0.1, 0.8]]), param_effect=0.5
        ),
        C_D=jnp.array([0.2, 0.35]),
        mode="joint",
        inflation=RTPS(alpha=0.5),
        parameter_evolution=RandomWalkEvolution(std={"a": 0.05}),
        rng_key=jax.random.PRNGKey(71),
    )
    kwargs.update(overrides)
    if beta is not None:
        kwargs["beta"] = beta
    return EnsembleKalmanFilter(**kwargs)


def _beta_problem(
    n_e: int = 16, num_cycles: int = 3
) -> tuple[xarray.Dataset, xarray.Dataset, list[xarray.DataArray]]:
    """Joint-mode inputs with TWO frames per cycle, so the serial sweep runs."""
    state = _initial_state(jax.random.PRNGKey(72), n_e, np.zeros(2), np.eye(2))
    params = _params_dataset(
        np.asarray(0.3 + 0.2 * jax.random.normal(jax.random.PRNGKey(73), (n_e,)))
    )
    values = np.asarray(
        0.5 + 0.3 * jax.random.normal(jax.random.PRNGKey(74), (num_cycles, 2, 2))
    )
    observations = [
        _labelled_truth_obs(values[k], np.array([0.0, 1.0])) for k in range(num_cycles)
    ]
    return state, params, observations


@pytest.mark.parametrize(  # type: ignore[misc]
    "beta",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        0.5,
        0.0,
        -2.0,
        True,
        False,
        np.bool_(True),
        "2.0",
        None,
        jnp.asarray(2.0),
    ],
    ids=[
        "nan",
        "inf",
        "-inf",
        "below-one",
        "zero",
        "negative",
        "True",
        "False",
        "np-bool",
        "string",
        "None",
        "jax-array",
    ],
)
def test_invalid_beta_is_rejected_at_construction(beta: Any) -> None:
    """beta must be a real, finite number >= 1 — and never a bool.

    ``None`` is rejected rather than defaulted: the Hydra config always spells
    the value, so a ``null`` there is a config error, not "use the default".
    """
    with pytest.raises(ValueError, match="beta"):
        EnsembleKalmanFilter(mode="state", beta=beta, **_dummy_filter_kwargs())


def test_beta_that_overflows_the_covariance_dtype_is_rejected() -> None:
    """A FINITE beta can still overflow float32 variances to inf.

    ``1e39`` is a perfectly finite Python float but ``1e39 * 0.1`` is not
    representable in float32; the effective covariance is validated in the dtype
    the analyses will actually use, so this fails at construction rather than as
    a NaN ensemble in cycle 0.
    """
    kwargs = _dummy_filter_kwargs()
    kwargs["C_D"] = jnp.array([0.1], dtype=jnp.float32)
    with pytest.raises(ValueError, match="overflows"):
        EnsembleKalmanFilter(mode="state", beta=1e39, **kwargs)


@pytest.mark.parametrize(  # type: ignore[misc]
    "beta", [1, 2, np.float32(3.0), np.float64(2.5), np.int64(4)]
)
def test_valid_beta_is_stored_as_a_python_float(beta: Any) -> None:
    enkf = EnsembleKalmanFilter(mode="state", beta=beta, **_dummy_filter_kwargs())
    assert type(enkf.beta) is float
    assert enkf.beta == float(beta)
    # C_D_diag stays PHYSICAL; only the effective vector is tempered.
    np.testing.assert_array_equal(np.asarray(enkf.C_D_diag), [np.float32(0.1)])
    np.testing.assert_array_equal(
        np.asarray(enkf.effective_C_D_diag),
        np.asarray(float(beta) * jnp.array([0.1])),
    )


def test_default_beta_is_the_legacy_filter_bit_for_bit() -> None:
    """No ``beta`` argument, and ``beta=1.0``, give the SAME run, bitwise.

    Joint mode, prior/posterior inflation, parameter evolution and a two-frame
    serial sweep, over several cycles: every path through the cycle loop that
    touches the covariance or the PRNG stream. The RNG key left on the
    instance is compared too, so an extra split for beta (which would shift
    every later draw) cannot hide behind equal posteriors.
    """
    state, params, observations = _beta_problem()
    legacy = _beta_filter()
    explicit = _beta_filter(beta=1.0)
    assert legacy.beta == 1.0
    np.testing.assert_array_equal(
        np.asarray(legacy.effective_C_D_diag), np.asarray(legacy.C_D_diag)
    )

    legacy_result = legacy.run(state=state, params=params, observations=observations)
    explicit_result = explicit.run(
        state=state, params=params, observations=observations
    )

    assert legacy_result.state is not None and explicit_result.state is not None
    np.testing.assert_array_equal(
        np.asarray(explicit_result.state["u"]), np.asarray(legacy_result.state["u"])
    )
    assert legacy_result.params is not None and explicit_result.params is not None
    np.testing.assert_array_equal(
        np.asarray(explicit_result.params["a"]), np.asarray(legacy_result.params["a"])
    )
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(explicit.rng_key)),
        np.asarray(jax.random.key_data(legacy.rng_key)),
    )
    for ours, reference in zip(explicit_result.diagnostics, legacy_result.diagnostics):
        # Everything but the wall-clock timing is identical.
        ours.analysis_time = reference.analysis_time = None
        assert ours == reference


def _pre_scaled_pair(beta: float, **overrides: Any) -> tuple[Any, Any, jnp.ndarray]:
    """``(beta, C_D)`` and ``(1, beta * C_D)`` twins of the same filter.

    Scaling the covariance exactly once means the tempered filter IS the
    untempered filter handed ``beta * C_D``: same kernel inputs, same keys, so
    the two runs must agree bit for bit. A second scaling anywhere (``alpha``
    in the stochastic kernel, a re-multiplication per window) or a site that
    still reads the physical ``C_D`` would break the equality.
    """
    C_D = overrides.pop("C_D", jnp.array([0.2, 0.35]))
    tempered = _beta_filter(beta=beta, C_D=C_D, **overrides)
    pre_scaled = _beta_filter(beta=1.0, C_D=beta * C_D, **overrides)
    return tempered, pre_scaled, C_D


def _assert_same_run(tempered: Any, pre_scaled: Any, **run_kwargs: Any) -> tuple:
    ours = tempered.run(**run_kwargs)
    reference = pre_scaled.run(**run_kwargs)
    assert ours.state is not None and reference.state is not None
    np.testing.assert_array_equal(
        np.asarray(ours.state["u"]), np.asarray(reference.state["u"])
    )
    if reference.params is not None:
        assert ours.params is not None
        np.testing.assert_array_equal(
            np.asarray(ours.params["a"]), np.asarray(reference.params["a"])
        )
    return ours, reference


@pytest.mark.parametrize("beta", [2.0, 4.0, 8.0])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "localization",
    [None, _AllOnesLocalization(), CorrelationLocalization(max_inflation=4.0)],
    ids=["global", "all-ones", "correlation"],
)
def test_stochastic_beta_equals_pre_scaling_the_covariance(
    beta: float, localization: Optional[BaseLocalization]
) -> None:
    """Joint mode, multi-frame sweep, global and localized stochastic updates.

    Both blocks of the joint update are tempered alike (the parameter rows are
    compared too), which is automatic because R is scaled rather than rows.
    """
    tempered, pre_scaled, _ = _pre_scaled_pair(beta, localization=localization)
    state, params, observations = _beta_problem()
    ours, reference = _assert_same_run(
        tempered, pre_scaled, state=state, params=params, observations=observations
    )
    # ...while the physical-covariance diagnostic is NOT the pre-scaled one:
    # the tempered run's chi2 is still measured against the physical C_D.
    assert ours.diagnostics[0].innovation_chi2 != pytest.approx(
        reference.diagnostics[0].innovation_chi2
    )
    # And beta genuinely changed the analysis (the comparison is not vacuous).
    untempered = _beta_filter(localization=localization).run(
        state=state, params=params, observations=observations
    )
    assert untempered.state is not None and ours.state is not None
    assert not np.allclose(
        np.asarray(untempered.state["u"]), np.asarray(ours.state["u"])
    )
    # The first cycle's forecast is identical, so its NIS is too: the chi2 is
    # physical whatever beta is.
    assert ours.diagnostics[0].innovation_chi2 == (
        untempered.diagnostics[0].innovation_chi2
    )


class _RecordingAnalysis(AnalysisScheme):
    """The stochastic analysis, recording the covariance of every call."""

    def __init__(self) -> None:
        self.inner = StochasticEnKFAnalysis()
        self.covariances: list[np.ndarray] = []

    def __call__(  # type: ignore[override]
        self,
        augmented: jnp.ndarray,
        pred_obs: jnp.ndarray,
        obs: jnp.ndarray,
        C_D_diag: jnp.ndarray,
        rng_key: jax.Array,
        **kwargs: Any,
    ) -> jnp.ndarray:
        self.covariances.append(np.asarray(C_D_diag))
        return self.inner(augmented, pred_obs, obs, C_D_diag, rng_key, **kwargs)


def test_the_sweep_and_the_reduction_replay_see_the_effective_covariance() -> None:
    """Every analysis call of a cycle — each frame of the serial sweep AND the
    reduction's discarded-increment replay — is handed ``beta * C_D``.

    Two frames per cycle and a truncated state reduction, so each cycle makes
    four calls (two sweep frames, two replay frames). Across two ``run()``
    calls — the window pattern — none of them sees the physical covariance or a
    compounded one, and the instance's vectors are exactly as constructed.
    """
    beta = 3.0
    C_D = jnp.array([0.2, 0.35])
    analysis = _RecordingAnalysis()
    enkf = _beta_filter(
        beta=beta,
        C_D=C_D,
        analysis=analysis,
        mode="state",
        parameter_evolution=None,
        state_reduction=OnlineStateReduction(max_rank=1, whiten=False),
    )
    state, _, observations = _beta_problem(num_cycles=4)
    constructed = np.asarray(enkf.effective_C_D_diag).copy()
    first = enkf.run(state=state, observations=observations[:2])
    enkf.run(state=first.state, observations=observations[2:])

    expected = np.asarray(beta * C_D)
    assert len(analysis.covariances) == 4 * 2 * 2  # cycles x (sweep+replay) x frames
    for covariance in analysis.covariances:
        np.testing.assert_array_equal(covariance, expected)
    np.testing.assert_array_equal(np.asarray(enkf.effective_C_D_diag), constructed)
    np.testing.assert_array_equal(np.asarray(enkf.C_D_diag), np.asarray(C_D))


def test_reduction_replay_is_tempered_like_the_sweep() -> None:
    """The discarded-increment fraction of a tempered run is the pre-scaled one.

    Were the replay to read the PHYSICAL ``C_D`` while the sweep used the
    effective one, its weight matrix would describe a different analysis than
    the posterior it is supposed to decompose — and the fraction would differ
    from the pre-scaled twin's (it does differ from the untempered run's, which
    is what shows the diagnostic is sensitive to the covariance at all).
    """
    beta = 4.0
    reduction_kwargs: dict[str, Any] = dict(
        mode="state", parameter_evolution=None, inflation=None
    )
    tempered, pre_scaled, _ = _pre_scaled_pair(
        beta,
        state_reduction=OnlineStateReduction(max_rank=1, whiten=False),
        **reduction_kwargs,
    )
    state, _, observations = _beta_problem(num_cycles=2)
    ours, reference = _assert_same_run(
        tempered, pre_scaled, state=state, observations=observations
    )
    untempered = _beta_filter(
        state_reduction=OnlineStateReduction(max_rank=1, whiten=False),
        **reduction_kwargs,
    ).run(state=state, observations=observations)
    for tempered_diag, reference_diag, untempered_diag in zip(
        ours.diagnostics, reference.diagnostics, untempered.diagnostics
    ):
        fraction = tempered_diag.reduction_discarded_increment_fraction
        assert fraction is not None and fraction > 1e-3
        assert fraction == reference_diag.reduction_discarded_increment_fraction
        assert fraction != pytest.approx(
            untempered_diag.reduction_discarded_increment_fraction
        )


def test_windowing_a_tempered_filter_is_still_inert() -> None:
    """Repeated ``run()`` calls never re-multiply the covariance.

    The same horizon as one call or as two windows on ONE instance must be
    bit-identical (the pre-beta windowing contract), which it could not be if
    any call scaled ``C_D`` again.
    """
    state, params, observations = _beta_problem(num_cycles=4)
    whole = _beta_filter(beta=4.0).run(
        state=state, params=params, observations=observations
    )
    windowed = _beta_filter(beta=4.0)
    first = windowed.run(state=state, params=params, observations=observations[:2])
    second = windowed.run(
        state=first.state, params=first.params, observations=observations[2:]
    )
    assert whole.state is not None and second.state is not None
    np.testing.assert_array_equal(
        np.asarray(second.state["u"]), np.asarray(whole.state["u"])
    )
    assert whole.params is not None and second.params is not None
    np.testing.assert_array_equal(
        np.asarray(second.params["a"]), np.asarray(whole.params["a"])
    )
    np.testing.assert_array_equal(
        np.asarray(windowed.effective_C_D_diag), np.asarray(4.0 * windowed.C_D_diag)
    )


@pytest.mark.parametrize("beta", [1.0, 2.0, 4.0, 8.0])  # type: ignore[misc]
def test_stochastic_beta_targets_the_kalman_analysis_with_beta_R(beta: float) -> None:
    """Fixed linear prior: posterior moments match the Kalman update with beta R.

    One analysis of a large ensemble (identity forecast, linear ``H``); the
    reference is built from the prior ensemble's own SAMPLE moments, so what
    remains is the perturbed-observation sampling error alone — hence a
    statistical tolerance here and an exact one for the ETKF
    (test_filtering_etkf.py). The tolerance is well inside the gap to the
    wrong targets: ``R`` (beta ignored) and ``beta**2 R`` (beta applied twice,
    e.g. once through ``C_D`` and again as the kernel's ``alpha``).
    """
    n_e = 20000
    H = np.array([[1.0, 0.5], [0.0, 1.0]])
    C_D = np.array([0.3, 0.5])
    state = _initial_state(
        jax.random.PRNGKey(75),
        n_e,
        np.array([0.4, -0.2]),
        np.array([[1.0, 0.3], [0.3, 0.8]]),
    )
    y = np.array([1.1, 0.2])
    result = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(H),
        forward_model=_ToyLinearModel(np.eye(2)),
        C_D=jnp.asarray(C_D),
        mode="state",
        beta=beta,
        rng_key=jax.random.PRNGKey(76),
    ).run(state=state, observations=jnp.asarray(y)[None, :])

    prior = np.asarray(state["u"].values, dtype=np.float64)  # (N_e, nx)
    m_f, P_f = prior.mean(axis=0), np.cov(prior.T)

    def _kalman(R_scale: float) -> tuple[np.ndarray, np.ndarray]:
        S = H @ P_f @ H.T + R_scale * np.diag(C_D)
        K = P_f @ H.T @ np.linalg.inv(S)
        return m_f + K @ (y - H @ m_f), (np.eye(2) - K @ H) @ P_f

    assert result.state is not None
    posterior = np.asarray(result.state["u"].values, dtype=np.float64)
    m_a, P_a = _kalman(beta)
    # The perturbations are CENTRED, so the analysis mean is the sample Kalman
    # mean up to float32 round-off; only the covariance carries sampling error
    # (~5e-3 at this size, against a >= 0.05 gap to either wrong target).
    np.testing.assert_allclose(posterior.mean(axis=0), m_a, atol=1e-4)
    np.testing.assert_allclose(np.cov(posterior.T), P_a, atol=0.015)
    if beta > 1.0:
        for wrong in (1.0, beta**2):
            _, P_wrong = _kalman(wrong)
            assert np.abs(P_wrong - P_a).max() > 0.05


def test_per_cycle_covariances_match_chained_calls_and_preserve_default() -> None:
    state = _initial_state(jax.random.PRNGKey(11), 20, np.zeros(1), np.eye(1))

    def make_filter() -> EnsembleKalmanFilter:
        return EnsembleKalmanFilter(
            observation_operator=_ToyObsOp(np.eye(1)),
            forward_model=cast(Any, _ToyLinearModel(np.eye(1))),
            C_D=jnp.array([0.5]),
            analysis=ETKFAnalysis(),
            rng_key=jax.random.PRNGKey(8),
            mode="state",
        )

    observations = jnp.array([[1.0], [2.0]])
    varying = make_filter()
    result = varying.run(
        state=state,
        observations=observations,
        observation_covariances=jnp.array([[0.1], [2.0]]),
    )
    chained = make_filter()
    carry = state
    for y, variance in zip(observations, [0.1, 2.0]):
        chained.set_observation_covariance(jnp.array([variance]))
        chained_result = chained.run(state=carry, observations=y[None, :])
        assert chained_result.state is not None
        carry = chained_result.state
    assert result.state is not None
    np.testing.assert_allclose(result.state.u, carry.u, atol=1e-6)
    np.testing.assert_array_equal(varying.C_D_diag, [0.5])
    assert result.diagnostics[-1].innovation_chi2 == pytest.approx(
        chained_result.diagnostics[-1].innovation_chi2
    )


def test_covariance_is_validated_before_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ToyLinearModel(np.eye(1))

    def forbidden(**kwargs: Any) -> xarray.Dataset:
        pytest.fail("forecast happened before covariance validation")

    monkeypatch.setattr(model, "run_ensemble", forbidden)
    filt = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(np.eye(1)),
        forward_model=cast(Any, model),
        C_D=jnp.ones(1),
        mode="state",
    )
    with pytest.raises(ValueError, match="finite"):
        filt.run(
            observations=jnp.ones((2, 1)),
            observation_covariances=jnp.array([[1.0], [np.inf]]),
        )


def test_actual_analyzed_observations_include_posterior_inflation() -> None:
    state = _initial_state(jax.random.PRNGKey(11), 20, np.zeros(1), np.eye(1))
    filt = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(np.eye(1)),
        forward_model=cast(Any, _ToyLinearModel(np.eye(1))),
        C_D=jnp.array([0.1]),
        analysis=ETKFAnalysis(),
        inflation=RTPP(alpha=0.8),
        mode="state",
    )
    filt.collect_pred_obs = True
    filt.collect_analyzed_observations = True
    result = filt.run(state=state, observations=jnp.array([[1.0]]))
    actual = filt.analyzed_pred_obs_history[0]
    assert result.state is not None
    np.testing.assert_allclose(actual, np.asarray(result.state.u).T)
    assert not np.allclose(actual, filt.pred_obs_post_history[0])
    assert result.diagnostics[0].obs_analyzed_final_rmse is not None


def test_varying_frame_covariances_follow_analysis_stride() -> None:
    state = _initial_state(jax.random.PRNGKey(71), 12, np.zeros(2), np.eye(2))
    values = np.arange(8, dtype=float).reshape(2, 4, 1) / 10
    variances = np.array([0.1, 0.2, 0.4, 0.8, 0.3, 0.6, 0.9, 1.2]).reshape(2, 4, 1)
    full = _stride_filter(4, 2)
    subset = _stride_filter(4, 2, wrapped=True)
    result = full.run(
        state=state,
        observations=_stride_batches(values),
        observation_covariances=variances,
    )
    reference = subset.run(
        state=state,
        observations=_stride_batches(values[:, 1::2]),
        observation_covariances=variances[:, 1::2],
    )
    assert result.state is not None and reference.state is not None
    np.testing.assert_array_equal(result.state.u, reference.state.u)
    for actual, expected in zip(result.diagnostics, reference.diagnostics):
        assert actual.innovation_chi2 == expected.innovation_chi2


# ---------------------------------------------------------------------------
# Beta tempering x per-window physical observation covariances
# ---------------------------------------------------------------------------


def _toy_state_filter(
    beta: float = 1.0, analysis: Optional[AnalysisScheme] = None, **overrides: Any
) -> EnsembleKalmanFilter:
    kwargs: dict[str, Any] = dict(
        observation_operator=_ToyObsOp(np.eye(1)),
        forward_model=cast(Any, _ToyLinearModel(np.eye(1))),
        C_D=jnp.array([0.5]),
        analysis=ETKFAnalysis() if analysis is None else analysis,
        rng_key=jax.random.PRNGKey(8),
        mode="state",
        beta=beta,
    )
    kwargs.update(overrides)
    return EnsembleKalmanFilter(**kwargs)


@pytest.mark.parametrize("scheme", ["etkf", "stochastic"])  # type: ignore[misc]
def test_beta_tempers_per_cycle_observation_covariances(scheme: str) -> None:
    """``run(observation_covariances=R_k)`` at beta analyses with ``beta R_k``.

    Bitwise the beta-1 filter handed the pre-scaled covariances, while the chi2
    diagnostic keeps reading the PHYSICAL ones.
    """
    state = _initial_state(jax.random.PRNGKey(11), 20, np.zeros(1), np.eye(1))
    observations = jnp.array([[1.0], [2.0]])
    physical = jnp.array([[0.1], [2.0]])
    beta = 4.0

    def run(filter_beta: float, covariances: jnp.ndarray) -> FilterResult:
        analysis = ETKFAnalysis() if scheme == "etkf" else StochasticEnKFAnalysis()
        return _toy_state_filter(beta=filter_beta, analysis=analysis).run(
            state=state, observations=observations, observation_covariances=covariances
        )

    tempered = run(beta, physical)
    prescaled = run(1.0, beta * physical)
    untempered = run(1.0, physical)
    assert tempered.state is not None and prescaled.state is not None
    assert untempered.state is not None
    np.testing.assert_array_equal(tempered.state.u, prescaled.state.u)
    assert not np.allclose(tempered.state.u, untempered.state.u)
    # Cycle 0 forecasts the same ensemble in both runs, so the chi2 against the
    # physical covariance must agree exactly; the pre-scaled run's does not.
    assert tempered.diagnostics[0].innovation_chi2 == pytest.approx(
        untempered.diagnostics[0].innovation_chi2, rel=0, abs=0
    )
    assert tempered.diagnostics[0].innovation_chi2 != pytest.approx(
        prescaled.diagnostics[0].innovation_chi2
    )


def test_set_observation_covariance_retempers_from_the_new_physical_one() -> None:
    """A replacement is PHYSICAL: effective = beta * new, never compounded."""
    filt = _toy_state_filter(beta=3.0)
    new = jnp.array([0.2])
    for _ in range(2):  # repeated windows installing the same covariance
        filt.set_observation_covariance(new)
        np.testing.assert_array_equal(filt.C_D_diag, new)
        np.testing.assert_array_equal(filt.effective_C_D_diag, 3.0 * new)
    # A diagonal matrix is accepted and reduced, like the constructor's.
    filt.set_observation_covariance(jnp.diag(jnp.array([0.4])))
    np.testing.assert_array_equal(filt.effective_C_D_diag, 3.0 * jnp.array([0.4]))


def test_rejected_covariance_replacement_leaves_the_filter_unchanged() -> None:
    """``beta * C_D`` overflowing the dtype fails before anything is assigned."""
    fmax = float(np.finfo(np.asarray(jnp.ones(1)).dtype).max)
    filt = _toy_state_filter(beta=1e30, C_D=jnp.array([1e-9]))
    physical, effective = filt.C_D_diag, filt.effective_C_D_diag
    with pytest.raises(ValueError, match="overflows"):
        filt.set_observation_covariance(jnp.array([fmax / 1e20]))
    assert filt.C_D_diag is physical
    assert filt.effective_C_D_diag is effective


def test_tempered_window_covariance_overflow_is_rejected_before_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-window covariances are tempered (and checked) before cycle 0."""
    model = _ToyLinearModel(np.eye(1))

    def forbidden(**kwargs: Any) -> xarray.Dataset:
        pytest.fail("forecast happened before the tempered covariance check")

    monkeypatch.setattr(model, "run_ensemble", forbidden)
    fmax = float(np.finfo(np.asarray(jnp.ones(1)).dtype).max)
    filt = _toy_state_filter(
        beta=1e30, forward_model=cast(Any, model), C_D=jnp.array([1e-9])
    )
    with pytest.raises(ValueError, match="overflows"):
        filt.run(
            observations=jnp.ones((2, 1)),
            observation_covariances=jnp.array([[1e-9], [fmax / 1e20]]),
        )


def test_beta_is_read_only_after_construction() -> None:
    """Reassigning beta would leave the cached effective covariance stale."""
    filt = _toy_state_filter(beta=2.0)
    with pytest.raises(AttributeError, match="fixed at construction"):
        filt.beta = 4.0
    assert filt.beta == 2.0
    np.testing.assert_array_equal(filt.effective_C_D_diag, 2.0 * filt.C_D_diag)


def test_default_beta_hands_the_kernel_the_constructors_covariance() -> None:
    """Pinned against the INPUT, not another run through the new code path.

    At the default beta every analysis must receive exactly the covariance the
    caller passed — same values, same dtype — which is what the pre-beta filter
    handed the kernel, so the kernel's output is the legacy output.
    """
    physical = jnp.array([0.2, 0.35])
    seen: list[jnp.ndarray] = []
    inner = StochasticEnKFAnalysis()

    class _Recording(AnalysisScheme):
        localization_policy = inner.localization_policy

        def __call__(  # type: ignore[override]
            self,
            augmented: jnp.ndarray,
            pred_obs: jnp.ndarray,
            obs: jnp.ndarray,
            C_D_diag: jnp.ndarray,
            rng_key: jax.Array,
            **kwargs: Any,
        ) -> jnp.ndarray:
            seen.append(C_D_diag)
            return inner(augmented, pred_obs, obs, C_D_diag, rng_key, **kwargs)

    state, params, observations = _beta_problem()
    _beta_filter(analysis=_Recording(), C_D=physical).run(
        state=state, params=params, observations=observations
    )
    assert seen, "the analysis never ran"
    for received in seen:
        assert received.dtype == physical.dtype
        np.testing.assert_array_equal(np.asarray(received), np.asarray(physical))
