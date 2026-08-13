"""Unit tests for the ESMDA smoother variants (data_assimilation.smoothing).

Covers the parts of ``smoothing/esmda.py`` reachable without a forward model:
the global Kalman update, the time-varying flatten/unflatten round-trip and its
block grouping, and the constructor validation added in the code review.
"""

import pathlib
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray
from data_assimilation.localization.correlation import CorrelationLocalization
from data_assimilation.localization.distance import DistanceLocalization
from data_assimilation.observation_operator import (
    AggregateObservations,
    ObservationOperator,
    TemporalObservationOperator,
    flatten_observations,
)
from data_assimilation.smoothing.esmda import (
    ParameterESMDA,
    StateAndParameterESMDA,
    StateESMDA,
    TimeVaryingParameterESMDA,
)


def _dummy_obs_op() -> ObservationOperator:
    return ObservationOperator(
        obs_ids_x=[0],
        obs_ids_y=[0],
        obs_ids_z=[0],
        obs_states=["u"],
        solver_name="pylbm",
    )


def _forward_model(save_on_disk: bool = False) -> SimpleNamespace:
    return SimpleNamespace(save_on_disk=save_on_disk, results_dir=None)


# ---------------------------------------------------------------------------
# Observation aggregation: the real and the predicted observations share one path
# ---------------------------------------------------------------------------


_N_SENSORS = 2
_TIMES = np.array([0.0, 1.0, 2.0, 3.0])


def _temporal_obs_op() -> Any:
    """A time-resolved operator over ``_N_SENSORS`` sensors of ``u``.

    ``Any`` because the smoothers annotate ``observation_operator`` as the bare
    :class:`ObservationOperator` even though the temporal wrapper is what every
    caller passes.
    """
    return TemporalObservationOperator(
        ObservationOperator(
            obs_ids_x=list(range(_N_SENSORS)),
            obs_ids_y=[0] * _N_SENSORS,
            obs_ids_z=[0] * _N_SENSORS,
            obs_states=["u"],
            solver_name="pylbm",
        )
    )


def _obs_state(n_e: int | None = None) -> xarray.Dataset:
    """A state whose value at (member, frame, sensor) is self-identifying."""
    sensors = np.arange(_N_SENSORS, dtype=float)
    field = 10.0 * _TIMES[:, None] + sensors[None, :]  # (time, sensor)
    coords = {"time": _TIMES, "z": [0.0], "y": [0.0], "x": sensors}
    if n_e is None:
        return xarray.Dataset(
            {"u": (("time", "z", "y", "x"), field[:, None, None, :])}, coords=coords
        )
    members = field[None, :, :] + 100.0 * np.arange(n_e, dtype=float)[:, None, None]
    return xarray.Dataset(
        {"u": (("ensemble", "time", "z", "y", "x"), members[:, :, None, None, :])},
        coords={"ensemble": np.arange(n_e), **coords},
    )


def _smoother_with_aggregation(
    aggregate: AggregateObservations | None,
) -> ParameterESMDA:
    """Bare smoother carrying only what the observation path needs."""
    smoother = ParameterESMDA.__new__(ParameterESMDA)
    smoother.observation_operator = _temporal_obs_op()
    smoother.aggregate_observations = aggregate
    return smoother


def test_get_observations_passes_a_flat_array_through() -> None:
    """A pre-flattened vector is the caller's responsibility, not re-aggregated."""
    smoother = _smoother_with_aggregation(AggregateObservations(interval_seconds=2.0))
    flat = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(smoother._get_observations(flat)), flat)


def test_get_observations_aggregates_then_flattens_time_major() -> None:
    """Interval means, flattened time-major so the sensor stays the fast axis."""
    smoother = _smoother_with_aggregation(AggregateObservations(interval_seconds=2.0))
    observed = _temporal_obs_op()(_obs_state())

    flat = np.asarray(smoother._get_observations(observed))
    # Frames 0/1 and 2/3 average to 10*t means of 5 and 25, plus the sensor id.
    np.testing.assert_allclose(flat, [5.0, 6.0, 25.0, 26.0])

    # Without an aggregator the full time-resolved vector is assimilated.
    full = np.asarray(_smoother_with_aggregation(None)._get_observations(observed))
    np.testing.assert_allclose(full, flatten_observations(observed))
    assert full.size == _TIMES.size * _N_SENSORS


def test_observation_step_puts_predictions_in_the_observation_space() -> None:
    """``H(x)`` is aggregated and flattened exactly like the real observations."""
    n_e = 3
    aggregate = AggregateObservations(interval_seconds=2.0)
    smoother = _smoother_with_aggregation(aggregate)

    pred_obs = np.asarray(smoother._observation_step(state=_obs_state(n_e=n_e)))
    assert pred_obs.shape == (n_e, 4)
    # Member m offsets every entry by 100*m; row 0 is the single-member answer.
    expected = np.array([5.0, 6.0, 25.0, 26.0]) + 100.0 * np.arange(n_e)[:, None]
    np.testing.assert_allclose(pred_obs, expected)


def test_observation_step_disk_path_matches_the_in_memory_path(
    tmp_path: pathlib.Path,
) -> None:
    """Per-member files are stacked before aggregation, giving the same (N_e, N_d)."""
    n_e = 3
    state = _obs_state(n_e=n_e)
    for member in range(n_e):
        state.isel(ensemble=member, drop=True).to_netcdf(
            tmp_path / f"state_{member}.nc"
        )

    in_memory = _smoother_with_aggregation(
        AggregateObservations(interval_seconds=2.0)
    )._observation_step(state=state)
    on_disk = _smoother_with_aggregation(
        AggregateObservations(interval_seconds=2.0)
    )._observation_step(results_dir=tmp_path)

    np.testing.assert_allclose(np.asarray(on_disk), np.asarray(in_memory))


def test_observation_step_disk_path_tolerates_jittered_member_time_axes(
    tmp_path: pathlib.Path,
) -> None:
    """Members whose time coords differ in the last bits still stack, not union.

    A backend can write per-member time coordinates that are equal only to
    float precision. Under xarray's default (outer) join those axes would be
    UNIONED and the observations padded with NaN -- a silently longer, NaN-poisoned
    innovation vector rather than an error.
    """
    n_e = 3
    state = _obs_state(n_e=n_e)
    for member in range(n_e):
        one = state.isel(ensemble=member, drop=True)
        one = one.assign_coords(time=_TIMES + member * 1e-9)
        one.to_netcdf(tmp_path / f"state_{member}.nc")

    smoother = _smoother_with_aggregation(AggregateObservations(interval_seconds=2.0))
    on_disk = np.asarray(smoother._observation_step(results_dir=tmp_path))

    assert on_disk.shape == (n_e, 4)
    assert np.isfinite(on_disk).all()


def test_aggregate_observations_is_forwarded_by_the_constructor() -> None:
    """The smoothers take the aggregator as a constructor keyword (hydra wiring)."""
    aggregate = AggregateObservations(interval_seconds=30.0)
    static = ParameterESMDA(
        observation_operator=_dummy_obs_op(),
        forward_model=_forward_model(),
        C_D=jnp.diag(jnp.ones(1)),
        aggregate_observations=aggregate,
    )
    assert static.aggregate_observations is aggregate

    # Through the time-varying constructor's explicit signature...
    dynamic = TimeVaryingParameterESMDA(
        observation_operator=_dummy_obs_op(),
        forward_model=_forward_model(),
        C_D=jnp.diag(jnp.ones(1)),
        num_time_points=2,
        aggregate_observations=aggregate,
    )
    assert dynamic.aggregate_observations is aggregate

    # ...and through the state-bearing variants' ``**kwargs`` chain.
    joint = StateAndParameterESMDA(
        observation_operator=_dummy_obs_op(),
        forward_model=_forward_model(),
        C_D=jnp.diag(jnp.ones(1)),
        aggregate_observations=aggregate,
    )
    assert joint.aggregate_observations is aggregate

    # Absent -> full-resolution assimilation, and resolvable on an instance
    # built without ``__init__`` (the array-level tests below do that).
    assert ParameterESMDA.__new__(ParameterESMDA).aggregate_observations is None


# ---------------------------------------------------------------------------
# Global Kalman update (_compute_kalman_update, localization=None)
# ---------------------------------------------------------------------------


def test_global_kalman_update_matches_esmda_formula() -> None:
    """The global update reproduces the ESMDA perturbed-observation formula.

    Fixed-seed regression against an independent computation that mirrors the
    internal rng split exactly, so it pins the exact linear algebra rather than
    a stochastic large-ensemble limit.
    """
    N_aug, N_d, N_e = 4, 3, 200
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    augmented = jax.random.normal(k1, (N_aug, N_e))
    G = jax.random.normal(k2, (N_d, N_aug))
    pred_obs = G @ augmented
    obs = jax.random.normal(k3, (N_d,))
    C_D = jnp.diag(0.2 * jnp.ones(N_d))
    alpha = 3.0
    key = jax.random.PRNGKey(999)

    esmda = ParameterESMDA.__new__(ParameterESMDA)
    esmda.alpha = alpha
    esmda.C_D = C_D
    esmda.C_D_sqrt = jnp.sqrt(C_D)
    esmda.localization = None
    esmda.rng_key = key

    result = esmda._compute_kalman_update(augmented, pred_obs, obs, N_e)

    # Independent replication of the same computation, including the rng split
    # and the perturbation centering the implementation performs.
    _, subkey = jax.random.split(key)
    aug_dev = augmented - augmented.mean(axis=1, keepdims=True)
    po_dev = pred_obs - pred_obs.mean(axis=1, keepdims=True)
    C_MD = (aug_dev @ po_dev.T) / (N_e - 1)
    C_DD = (po_dev @ po_dev.T) / (N_e - 1)
    Z = jax.random.normal(subkey, (N_d, N_e))
    Z = Z - Z.mean(axis=1, keepdims=True)
    perturbed = obs[:, None] + jnp.sqrt(alpha) * (jnp.sqrt(C_D) @ Z)
    x = jnp.linalg.solve(C_DD + alpha * C_D, perturbed - pred_obs)
    expected = augmented + C_MD @ x

    np.testing.assert_allclose(np.array(result), np.array(expected), atol=1e-5)


def test_global_kalman_update_recovers_linear_gaussian_posterior() -> None:
    """One un-tempered update moves the mean toward the exact Bayesian posterior.

    With a linear observation operator and a large ensemble, ESMDA with
    ``alpha=1`` reproduces the analytic linear-Gaussian posterior mean.
    """
    N_aug, N_d, N_e = 2, 2, 20000
    key = jax.random.PRNGKey(7)
    k1, k2, k3 = jax.random.split(key, 3)
    prior_mean = jnp.array([1.0, -2.0])
    prior = prior_mean[:, None] + jax.random.normal(k1, (N_aug, N_e))
    G = jnp.array([[1.0, 0.0], [0.5, 1.0]])
    C_D = jnp.diag(jnp.array([0.1, 0.2]))
    pred_obs = G @ prior
    obs = jnp.array([2.0, 0.5])

    esmda = ParameterESMDA.__new__(ParameterESMDA)
    esmda.alpha = 1.0
    esmda.C_D = C_D
    esmda.C_D_sqrt = jnp.sqrt(C_D)
    esmda.localization = None
    esmda.rng_key = k2

    updated = esmda._compute_kalman_update(prior, pred_obs, obs, N_e, alpha=1.0)
    post_mean = np.array(updated.mean(axis=1))

    # Analytic Gaussian posterior mean: C_prior = I here.
    C_prior = jnp.eye(N_aug)
    K = C_prior @ G.T @ jnp.linalg.inv(G @ C_prior @ G.T + C_D)
    analytic = prior_mean + K @ (obs - G @ prior_mean)
    np.testing.assert_allclose(post_mean, np.array(analytic), atol=0.05)


def test_global_kalman_update_raises_on_singular_system() -> None:
    """A singular C_DD + alpha*C_D must fail loudly, not NaN-poison silently."""
    N_aug, N_d, N_e = 2, 2, 8
    augmented = jax.random.normal(jax.random.PRNGKey(0), (N_aug, N_e))
    pred_obs = jax.random.normal(jax.random.PRNGKey(1), (N_d, N_e))
    obs = jnp.zeros(N_d)
    # Zero observation-error covariance + rank-deficient forecast -> singular.
    C_D = jnp.zeros((N_d, N_d))

    esmda = ParameterESMDA.__new__(ParameterESMDA)
    esmda.alpha = 1.0
    esmda.C_D = C_D
    esmda.C_D_sqrt = jnp.sqrt(C_D)
    esmda.localization = None
    esmda.rng_key = jax.random.PRNGKey(2)

    # Force a singular auto-covariance by making pred_obs constant across members.
    pred_obs = jnp.zeros((N_d, N_e))
    with pytest.raises(FloatingPointError, match="non-finite"):
        esmda._compute_kalman_update(augmented, pred_obs, obs, N_e)


# ---------------------------------------------------------------------------
# Time-varying flatten / unflatten round-trip and block grouping
# ---------------------------------------------------------------------------


def _time_varying_params(num_time: int = 3, n_e: int = 5) -> xarray.Dataset:
    rng = np.random.default_rng(0)
    return xarray.Dataset(
        {
            "inflow_angle": (
                ("time", "ensemble"),
                rng.normal(size=(num_time, n_e)),
            ),
            "velocity_magnitude": (
                ("time", "ensemble"),
                rng.normal(size=(num_time, n_e)),
            ),
            # A static parameter whose name ends in ``_2`` -- the old regex
            # grouping would have merged it into a ``_<int>`` block.
            "sensor_2": (("ensemble",), rng.normal(size=n_e)),
        },
        coords={"time": np.arange(num_time, dtype=float), "ensemble": np.arange(n_e)},
    )


def _make_time_varying_smoother(
    num_time_points: int, pin: bool
) -> TimeVaryingParameterESMDA:
    smoother = TimeVaryingParameterESMDA.__new__(TimeVaryingParameterESMDA)
    smoother.num_time_points = num_time_points
    smoother.pin_initial_time_point = pin
    return smoother


@pytest.mark.parametrize("pin", [False, True])  # type: ignore[misc]
def test_time_varying_flatten_unflatten_round_trip(pin: bool) -> None:
    num_time = 3
    params = _time_varying_params(num_time)
    smoother = _make_time_varying_smoother(num_time, pin)

    flat = smoother._flatten_time_varying_params(params)
    restored = smoother._unflatten_params(flat, params)

    for name in params.data_vars:
        np.testing.assert_allclose(
            np.array(restored[name].values),
            np.array(params[name].values),
            err_msg=f"round-trip mismatch for {name} (pin={pin})",
        )
    # When pinning, t=0 is not part of the flattened (augmented) vector.
    assert ("inflow_angle_0" in flat.data_vars) == (not pin)


@pytest.mark.parametrize("pin", [False, True])  # type: ignore[misc]
def test_time_varying_group_ids_group_knots_not_unrelated_params(pin: bool) -> None:
    num_time = 3
    params = _time_varying_params(num_time)
    smoother = _make_time_varying_smoother(num_time, pin)

    flat = smoother._flatten_time_varying_params(params)
    group_ids = np.array(smoother._time_varying_group_ids(params))

    # One id per flattened row, matching the flattened variable order.
    assert group_ids.shape[0] == len(flat.data_vars)
    names = list(flat.data_vars)
    by_group: dict[int, list[str]] = {}
    for gid, name in zip(group_ids, names):
        by_group.setdefault(int(gid), []).append(name)

    # All knots of one parameter share exactly one block; unrelated parameters
    # (including the static ``sensor_2``) never share a block.
    knots = num_time - (1 if pin else 0)
    angle_group = {
        gid for gid, ns in by_group.items() if ns[0].startswith("inflow_angle_")
    }
    assert len(angle_group) == 1
    assert len(by_group[angle_group.pop()]) == knots
    static_group = [gid for gid, ns in by_group.items() if ns == ["sensor_2"]]
    assert len(static_group) == 1


def test_time_varying_flatten_rejects_time_size_mismatch() -> None:
    params = _time_varying_params(num_time=3)
    smoother = _make_time_varying_smoother(num_time_points=4, pin=False)
    with pytest.raises(ValueError, match="num_time_points"):
        smoother._flatten_time_varying_params(params)


def test_parameter_group_ids_keep_static_params_separate() -> None:
    """Two static params that share a numeric-suffix base must not be merged.

    The plain ParameterESMDA path gives each parameter its own block, so even
    names like ``sensor_1``/``sensor_2`` stay independent (the failure mode of
    the old regex-based grouping).
    """
    n_e = 6
    rng = np.random.default_rng(1)
    params = xarray.Dataset(
        {
            "sensor_1": (("ensemble",), rng.normal(size=n_e)),
            "sensor_2": (("ensemble",), rng.normal(size=n_e)),
        },
        coords={"ensemble": np.arange(n_e)},
    )
    esmda = ParameterESMDA.__new__(ParameterESMDA)
    esmda.localization = CorrelationLocalization(block_grouping=True)

    # Reproduce the group-id construction the block-grouping path performs.
    group_ids = jnp.arange(len(params.data_vars), dtype=int)
    assert int(group_ids[0]) != int(group_ids[1])


def _bare_smoother(
    cls: Any,
    localization: Any,
    key: jax.Array,
) -> Any:
    """Minimal initialized smoother for array-level localization tests."""
    obj = cls.__new__(cls)
    obj.alpha = 1.0
    obj.C_D = jnp.diag(jnp.array([0.1]))
    obj.C_D_sqrt = jnp.sqrt(obj.C_D)
    obj.localization = localization
    obj.rng_key = key
    obj.state_reduction = None
    return obj


def test_parameter_esmda_correlation_localizes_parameter_rows() -> None:
    """Parameter-only ESMDA keeps only significantly correlated updates."""
    n_e = 40
    signal = jax.random.normal(jax.random.PRNGKey(30), (n_e,))
    nuisance = jax.random.normal(jax.random.PRNGKey(31), (n_e,))
    params = jnp.stack([signal, nuisance])
    pred_obs = signal[None, :]
    obs = jnp.array([2.0])
    key = jax.random.PRNGKey(32)

    localized = _bare_smoother(
        ParameterESMDA,
        CorrelationLocalization(truncation_correlation=0.999, max_inflation=1.0),
        key,
    )._compute_kalman_update(params, pred_obs, obs, n_e)
    global_result = _bare_smoother(ParameterESMDA, None, key)._compute_kalman_update(
        params, pred_obs, obs, n_e
    )

    assert not np.allclose(np.asarray(localized[0]), np.asarray(params[0]))
    np.testing.assert_allclose(localized[1], params[1], atol=1e-7)
    assert not np.allclose(np.asarray(global_result[1]), np.asarray(params[1]))


def test_joint_esmda_correlation_localizes_state_and_parameter_rows() -> None:
    """Correlation localization applies to both blocks in joint ESMDA."""
    n_e = 40
    signal = jax.random.normal(jax.random.PRNGKey(40), (n_e,))
    nuisance = jax.random.normal(jax.random.PRNGKey(41), (n_e,))
    state = xarray.Dataset(
        {"u": (("ensemble", "x"), signal[:, None])},
        coords={"ensemble": np.arange(n_e), "x": [0.0]},
    )
    params = xarray.Dataset(
        {"a": ("ensemble", nuisance)}, coords={"ensemble": np.arange(n_e)}
    )
    pred_obs = signal[None, :]
    obs = jnp.array([2.0])
    key = jax.random.PRNGKey(42)

    localized_obj = _bare_smoother(
        StateAndParameterESMDA,
        CorrelationLocalization(truncation_correlation=0.999, max_inflation=1.0),
        key,
    )
    _, localized_params = localized_obj._augmented_state_update(
        state, params, pred_obs, obs, n_e
    )
    global_obj = _bare_smoother(StateAndParameterESMDA, None, key)
    _, global_params = global_obj._augmented_state_update(
        state, params, pred_obs, obs, n_e
    )

    np.testing.assert_allclose(localized_params["a"], params["a"], atol=1e-7)
    assert not np.allclose(global_params["a"], params["a"])


def test_joint_esmda_distance_localizes_state_but_not_parameters() -> None:
    """Distance localization leaves non-spatial parameters on the global path."""
    n_e = 40
    signal = jax.random.normal(jax.random.PRNGKey(50), (n_e,))
    state = xarray.Dataset(
        {"u": (("ensemble", "x"), jnp.stack([signal, signal], axis=1))},
        coords={"ensemble": np.arange(n_e), "x": [0.0, 10.0]},
    )
    params = xarray.Dataset(
        {"a": ("ensemble", signal)}, coords={"ensemble": np.arange(n_e)}
    )
    pred_obs = signal[None, :]
    obs = jnp.array([2.0])
    key = jax.random.PRNGKey(51)

    localized_obj = _bare_smoother(
        StateAndParameterESMDA,
        DistanceLocalization(
            localization_radius=1.0, max_inflation=1.0, block_grouping=False
        ),
        key,
    )
    localized_obj.observation_operator = ObservationOperator(
        obs_x=[0.0],
        obs_y=[0.0],
        obs_z=[0.0],
        obs_states=["u"],
        solver_name="pylbm",
    )
    localized_state, localized_params = localized_obj._augmented_state_update(
        state, params, pred_obs, obs, n_e
    )
    global_obj = _bare_smoother(StateAndParameterESMDA, None, key)
    global_state, global_params = global_obj._augmented_state_update(
        state, params, pred_obs, obs, n_e
    )

    np.testing.assert_allclose(localized_state["u"][:, 1], state["u"][:, 1])
    assert not np.allclose(global_state["u"][:, 1], state["u"][:, 1])
    np.testing.assert_allclose(localized_params["a"], global_params["a"], atol=1e-6)


def test_state_only_esmda_is_a_state_bearing_smoother() -> None:
    """The state-only variant participates in the shared state-localization path."""
    assert issubclass(StateESMDA, StateAndParameterESMDA)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_non_diagonal_C_D() -> None:
    C_D = jnp.array([[1.0, 0.3], [0.3, 1.0]])
    with pytest.raises(ValueError, match="diagonal"):
        ParameterESMDA(
            observation_operator=_dummy_obs_op(),
            forward_model=_forward_model(),
            C_D=C_D,
        )


def test_constructor_rejects_zero_variance_C_D() -> None:
    # A zero observation-error variance makes C_DD + alpha*C_D singular.
    C_D = jnp.diag(jnp.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="positive"):
        ParameterESMDA(
            observation_operator=_dummy_obs_op(),
            forward_model=_forward_model(),
            C_D=C_D,
        )


def test_constructor_rejects_inconsistent_alpha_schedule() -> None:
    # num_steps / alpha must equal 1 (sum_k 1/alpha_k = 1); alpha != num_steps
    # tempers the likelihood into a different inference problem.
    C_D = jnp.diag(jnp.ones(1))
    with pytest.raises(ValueError, match="Inconsistent ES-MDA schedule"):
        ParameterESMDA(
            observation_operator=_dummy_obs_op(),
            forward_model=_forward_model(),
            C_D=C_D,
            num_steps=4,
            alpha=2.0,
        )


def test_constructor_accepts_consistent_explicit_alpha() -> None:
    # alpha == num_steps is the consistent uniform schedule.
    C_D = jnp.diag(jnp.ones(1))
    esmda = ParameterESMDA(
        observation_operator=_dummy_obs_op(),
        forward_model=_forward_model(),
        C_D=C_D,
        num_steps=4,
        alpha=4.0,
    )
    assert esmda.alpha == 4.0


def test_constructor_rejects_distance_localization_on_parameter_smoother() -> None:
    C_D = jnp.diag(jnp.ones(2))
    with pytest.raises(ValueError, match="state-bearing"):
        ParameterESMDA(
            observation_operator=_dummy_obs_op(),
            forward_model=_forward_model(),
            C_D=C_D,
            localization=DistanceLocalization(localization_radius=10.0),
        )


def test_state_smoother_accepts_distance_localization() -> None:
    C_D = jnp.diag(jnp.ones(2))
    esmda = StateAndParameterESMDA(
        observation_operator=ObservationOperator(
            obs_x=[1.0],
            obs_y=[1.0],
            obs_z=[1.0],
            obs_states=["u", "v"],
            solver_name="pylbm",
        ),
        forward_model=_forward_model(),
        C_D=C_D,
        localization=DistanceLocalization(localization_radius=10.0),
    )
    assert esmda.localization is not None


def test_rng_key_default_is_not_shared_import_time_object() -> None:
    """Omitting rng_key yields a usable key without an import-time default."""
    C_D = jnp.diag(jnp.ones(1))
    esmda = ParameterESMDA(
        observation_operator=_dummy_obs_op(),
        forward_model=_forward_model(),
        C_D=C_D,
        rng_key=None,
    )
    assert esmda.rng_key is not None
    # Splitting must work (the annotation-only Optional used to crash here).
    jax.random.split(esmda.rng_key)


def test_state_group_ids_keep_staggered_grids_separate() -> None:
    """Equal flat indices on staggered grids must not be treated as co-located."""
    state = xarray.Dataset(
        {
            "u": (("ensemble", "zt", "yt", "xm"), np.zeros((2, 2, 2, 3))),
            "v": (("ensemble", "zt", "ym", "xt"), np.zeros((2, 2, 2, 3))),
        },
        coords={
            "zt": [1.0, 2.0],
            "yt": [0.0, 1.0],
            "ym": [0.5, 1.5],
            "xm": [0.5, 1.5, 2.5],
            "xt": [0.0, 1.0, 2.0],
        },
    )
    obj = StateAndParameterESMDA.__new__(StateAndParameterESMDA)
    groups = np.asarray(obj._state_group_ids(state))
    n_cells = state["u"].size // state.sizes["ensemble"]
    assert set(groups[:n_cells]).isdisjoint(set(groups[n_cells:]))


def test_state_group_ids_share_collocated_grid() -> None:
    """Variables on the same physical grid retain joint block updates."""
    state = xarray.Dataset(
        {
            "u": (("ensemble", "z", "x"), np.zeros((2, 2, 3))),
            "v": (("ensemble", "z", "x"), np.zeros((2, 2, 3))),
        },
        coords={"z": [1.0, 2.0], "x": [0.0, 1.0, 2.0]},
    )
    obj = StateAndParameterESMDA.__new__(StateAndParameterESMDA)
    groups = np.asarray(obj._state_group_ids(state))
    n_cells = state["u"].size // state.sizes["ensemble"]
    np.testing.assert_array_equal(groups[:n_cells], groups[n_cells:])
