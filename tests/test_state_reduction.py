"""Tests for the online reduced SVD/KL state update (data_assimilation.reduction)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray


class _DummyEnsembleModel:
    """Minimal stand-in: the unit tests never run a forecast."""

    def __init__(self, save_on_disk: bool = False, results_dir=None) -> None:
        self.save_on_disk = save_on_disk
        self.results_dir = results_dir


def _make_state(key, n_e: int, nt: int = 1, nx: int = 4, ny: int = 3, nz: int = 2):
    shape = (n_e, nt, nx, ny, nz)
    keys = jax.random.split(key, 3)
    data_vars = {
        name: (
            ("ensemble", "time", "x", "y", "z"),
            jax.random.normal(k, shape),
        )
        for name, k in zip(("u", "v", "w"), keys)
    }
    return xarray.Dataset(
        data_vars,
        coords={
            "ensemble": np.arange(n_e),
            "time": np.arange(nt, dtype=float),
            "x": np.linspace(0.0, 10.0, nx),
            "y": np.linspace(0.0, 8.0, ny),
            "z": np.linspace(0.0, 5.0, nz),
        },
    )


def _make_params(key, n_e: int):
    k1, k2 = jax.random.split(key)
    return xarray.Dataset(
        {
            "inflow_angle": ("ensemble", 45.0 + jax.random.normal(k1, (n_e,))),
            "velocity_magnitude": (
                "ensemble",
                2.0 + 0.1 * jax.random.normal(k2, (n_e,)),
            ),
        },
        coords={"ensemble": np.arange(n_e)},
    )


def _make_smoother(n_d: int, rng_seed: int = 0, **kwargs):
    from data_assimilation.smoothing.esmda import StateAndParameterESMDA

    return StateAndParameterESMDA(
        observation_operator=kwargs.pop("observation_operator", None),
        forward_model=kwargs.pop("forward_model", _DummyEnsembleModel()),
        C_D=0.1 * jnp.eye(n_d),
        num_steps=1,
        rng_key=jax.random.PRNGKey(rng_seed),
        **kwargs,
    )


def test_full_rank_ic_source_matches_full_space_update() -> None:
    """At full rank, the IC-source reduced update equals the full-space update.

    The full-space ESMDA increment lives in the ensemble anomaly span, which an
    untruncated IC-source basis represents exactly; with identical rng keys both
    paths draw the same observation perturbations, so the equality is exact up
    to float round-off.
    """
    from data_assimilation.reduction import OnlineStateReduction

    n_e, n_d = 8, 5
    key = jax.random.PRNGKey(42)
    k_state, k_params, k_pred, k_obs = jax.random.split(key, 4)
    state = _make_state(k_state, n_e).isel(time=0)
    params = _make_params(k_params, n_e)
    pred_obs = jax.random.normal(k_pred, (n_d, n_e))
    obs = jax.random.normal(k_obs, (n_d,))

    full = _make_smoother(n_d)
    reduced = _make_smoother(
        n_d, state_reduction=OnlineStateReduction(energy_fraction=1.0)
    )

    full_states, full_params = full._augmented_state_update(
        state, params, pred_obs, obs, n_e
    )
    red_states, red_params = reduced._augmented_state_update(
        state, params, pred_obs, obs, n_e
    )

    for var in full_states.data_vars:
        np.testing.assert_allclose(
            red_states[var].values, full_states[var].values, rtol=1e-3, atol=1e-4
        )
    for var in full_params.data_vars:
        np.testing.assert_allclose(
            red_params[var].values, full_params[var].values, rtol=1e-3, atol=1e-4
        )


def test_truncation_energy_criterion_and_max_rank() -> None:
    from data_assimilation.reduction import OnlineStateReduction

    # Exactly rank-2 anomalies with singular values 10 and 1 (energies 100, 1):
    # cumulative fractions are 100/101 ~ 0.9901 and 1.0. Built with zero row
    # means (zero-column-sum right factor), so fit()'s centering is a no-op and
    # its scaled anomalies have exactly these singular values.
    n_s, n_samples = 30, 6
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(n_s, 2)))
    v_raw = rng.normal(size=(n_samples, 2))
    v, _ = np.linalg.qr(v_raw - v_raw.mean(axis=0, keepdims=True))
    snapshots = jnp.asarray(np.sqrt(n_samples - 1) * q @ np.diag([10.0, 1.0]) @ v.T)

    reduction = OnlineStateReduction(energy_fraction=0.99)
    reduction.fit(snapshots)
    assert reduction.rank == 1

    reduction = OnlineStateReduction(energy_fraction=0.999)
    reduction.fit(snapshots)
    assert reduction.rank == 2

    reduction = OnlineStateReduction(energy_fraction=0.999, max_rank=1)
    reduction.fit(snapshots)
    assert reduction.rank == 1

    # energy_fraction=1.0 keeps every nonzero mode (here exactly 2).
    reduction = OnlineStateReduction(energy_fraction=1.0)
    reduction.fit(snapshots)
    assert reduction.rank == 2


def test_encode_is_whitened_for_ic_source() -> None:
    from data_assimilation.reduction import OnlineStateReduction

    n_s, n_e = 50, 10
    states_flat = jax.random.normal(jax.random.PRNGKey(1), (n_s, n_e))
    reduction = OnlineStateReduction(energy_fraction=1.0)
    reduction.fit(states_flat)
    xi = reduction.encode(states_flat)

    assert xi.shape[0] == reduction.rank <= n_e - 1
    cov = np.asarray(xi @ xi.T) / (n_e - 1)
    np.testing.assert_allclose(cov, np.eye(reduction.rank), atol=1e-4)
    np.testing.assert_allclose(np.asarray(xi.mean(axis=1)), 0.0, atol=1e-5)


def test_invalid_construction_args_raise() -> None:
    from data_assimilation.reduction import OnlineStateReduction

    with pytest.raises(ValueError, match="energy_fraction"):
        OnlineStateReduction(energy_fraction=0.0)
    with pytest.raises(ValueError, match="basis_source"):
        OnlineStateReduction(basis_source="offline")
    with pytest.raises(ValueError, match="max_rank"):
        OnlineStateReduction(max_rank=0)
    with pytest.raises(ValueError, match="snapshot_stride"):
        OnlineStateReduction(snapshot_stride=0)


def test_window_snapshot_flattening_matches_state_flattening() -> None:
    from data_assimilation.reduction import OnlineStateReduction

    n_e, nt = 3, 4
    state = _make_state(jax.random.PRNGKey(2), n_e, nt=nt)
    smoother = _make_smoother(
        4,
        state_reduction=OnlineStateReduction(basis_source="window_snapshots"),
    )

    snapshots = smoother._flatten_window_snapshots(state)
    assert snapshots.shape[1] == n_e * nt

    # Column e*nt + t is member e's frame t, with _flatten_state's row order.
    for e in range(n_e):
        for t in range(nt):
            frame = smoother._flatten_state(state.isel(time=t, drop=True))
            np.testing.assert_array_equal(
                np.asarray(snapshots[:, e * nt + t]), np.asarray(frame[:, e])
            )

    # Stride thins the time frames.
    smoother.state_reduction.snapshot_stride = 2
    strided = smoother._flatten_window_snapshots(state)
    assert strided.shape[1] == n_e * 2


def test_zero_gain_leaves_state_unchanged_for_snapshot_source() -> None:
    """With zero predicted-observation spread the Kalman gain is zero; the
    increment decoding must then return each member's state exactly (the
    window_snapshots projection residual is never discarded)."""
    from data_assimilation.reduction import OnlineStateReduction

    n_e, n_d, nt = 5, 4, 3
    full_state = _make_state(jax.random.PRNGKey(3), n_e, nt=nt)
    params = _make_params(jax.random.PRNGKey(4), n_e)
    pred_obs = jnp.ones((n_d, n_e))  # zero ensemble deviation -> zero gain
    obs = jnp.zeros((n_d,))

    smoother = _make_smoother(
        n_d,
        state_reduction=OnlineStateReduction(
            energy_fraction=0.9, basis_source="window_snapshots"
        ),
    )
    snapshots = smoother._flatten_window_snapshots(full_state)
    state_ic = full_state.isel(time=0)

    updated_states, updated_params = smoother._augmented_state_update(
        state_ic, params, pred_obs, obs, n_e, snapshots_flat=snapshots
    )

    for var in state_ic.data_vars:
        np.testing.assert_allclose(
            updated_states[var].values, state_ic[var].values, atol=1e-6
        )
    for var in params.data_vars:
        np.testing.assert_allclose(
            updated_params[var].values, params[var].values, atol=1e-6
        )


def test_state_reduction_incompatible_with_localization() -> None:
    from data_assimilation.localization.correlation import CorrelationLocalization
    from data_assimilation.reduction import OnlineStateReduction
    from data_assimilation.smoothing.esmda import StateAndTimeVaryingParameterESMDA

    with pytest.raises(ValueError, match="localization"):
        _make_smoother(
            3,
            localization=CorrelationLocalization(),
            state_reduction=OnlineStateReduction(),
        )

    with pytest.raises(ValueError, match="localization"):
        StateAndTimeVaryingParameterESMDA(
            observation_operator=None,
            forward_model=_DummyEnsembleModel(),
            C_D=jnp.eye(3),
            num_time_points=3,
            localization=CorrelationLocalization(),
            state_reduction=OnlineStateReduction(),
        )


def test_final_time_smoothing_requires_reduction_and_in_memory_mode(
    tmp_path,
) -> None:
    from data_assimilation.reduction import OnlineStateReduction

    with pytest.raises(ValueError, match="state_reduction"):
        _make_smoother(3, final_time_smoothing=True)

    with pytest.raises(ValueError, match="on-disk"):
        _make_smoother(
            3,
            forward_model=_DummyEnsembleModel(save_on_disk=True, results_dir=tmp_path),
            state_reduction=OnlineStateReduction(),
            final_time_smoothing=True,
        )


def test_final_time_smoothing_updates_every_frame_not_params() -> None:
    from data_assimilation.reduction import OnlineStateReduction

    n_e, n_d, nt = 6, 4, 3

    def dummy_obs_op(ds: xarray.Dataset) -> jnp.ndarray:
        flat = jnp.asarray(
            ds["u"].transpose("ensemble", ...).values.reshape(ds.sizes["ensemble"], -1)
        )
        return flat[:, :n_d]  # (N_e, N_d)

    smoother = _make_smoother(
        n_d,
        observation_operator=dummy_obs_op,
        state_reduction=OnlineStateReduction(energy_fraction=1.0),
        final_time_smoothing=True,
    )

    state = _make_state(jax.random.PRNGKey(5), n_e, nt=nt)
    obs = jax.random.normal(jax.random.PRNGKey(6), (n_d,))

    smoothed = smoother._final_time_smoothing_step(state, obs)

    assert smoothed.sizes == state.sizes
    for t in range(nt):
        diff = np.abs(
            smoothed["u"].isel(time=t).values - state["u"].isel(time=t).values
        )
        assert diff.max() > 0.0, f"time frame {t} was not updated"

    # The step never touches parameters (they are not in its augmented vector)
    # and is a no-op when the flag is off.
    smoother.final_time_smoothing = False
    assert smoother._final_time_smoothing_step(state, obs) is state
