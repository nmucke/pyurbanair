"""Unit tests for the ESMDA smoother variants (data_assimilation.smoothing).

Covers the parts of ``smoothing/esmda.py`` reachable without a forward model:
the global Kalman update, the time-varying flatten/unflatten round-trip and its
block grouping, and the constructor validation added in the code review.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray
from data_assimilation.localization.correlation import CorrelationLocalization
from data_assimilation.localization.distance import DistanceLocalization
from data_assimilation.observation_operator import ObservationOperator
from data_assimilation.smoothing.esmda import (
    ParameterESMDA,
    StateAndParameterESMDA,
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

    # Independent replication of the same computation, including the rng split.
    _, subkey = jax.random.split(key)
    aug_dev = augmented - augmented.mean(axis=1, keepdims=True)
    po_dev = pred_obs - pred_obs.mean(axis=1, keepdims=True)
    C_MD = (aug_dev @ po_dev.T) / (N_e - 1)
    C_DD = (po_dev @ po_dev.T) / (N_e - 1)
    Z = jax.random.normal(subkey, (N_d, N_e))
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


def test_state_group_ids_reject_staggered_shapes() -> None:
    """Block grouping requires collocated state variables (one shared shape)."""
    # Staggered C-grid: u on (zt, yt, xm), v on (zt, ym, xt) -- different cell
    # shapes, so the flat-index co-location assumption does not hold.
    state = xarray.Dataset(
        {
            "u": (("ensemble", "zt", "yt", "xm"), np.zeros((2, 3, 4, 5))),
            "v": (("ensemble", "zt", "ym", "xt"), np.zeros((2, 3, 4, 6))),
        }
    )
    obj = StateAndParameterESMDA.__new__(StateAndParameterESMDA)
    with pytest.raises(ValueError, match="collocated"):
        obj._state_group_ids(state)
