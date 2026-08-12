"""Tests for the online reduced SVD/KL state update (data_assimilation.reduction)."""

import math
import warnings
from typing import Any

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


def test_orthonormal_coefficients_and_row_scaling_round_trip() -> None:
    from data_assimilation.reduction import OnlineStateReduction

    n_s, n_e = 12, 6
    key = jax.random.PRNGKey(101)
    scaled_states = jax.random.normal(key, (n_s, n_e))
    row_scales = jnp.linspace(0.2, 5.0, n_s)
    offsets = jnp.linspace(-3.0, 4.0, n_s)[:, None]
    physical_states = offsets + row_scales[:, None] * scaled_states

    reduction = OnlineStateReduction(energy_fraction=1.0, whiten=False)
    reduction.fit(physical_states, row_scales=row_scales)
    coefficients = reduction.encode(physical_states)

    # Orthonormal coefficients carry the retained covariance rather than being
    # whitened. Their covariance eigenvalues are the fitted sigma^2 values.
    covariance = np.asarray(coefficients @ coefficients.T) / (n_e - 1)
    np.testing.assert_allclose(
        covariance,
        np.diag(np.asarray(reduction.singular_values) ** 2),
        rtol=2e-5,
        atol=2e-6,
    )
    physical_anomalies = physical_states - physical_states.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(
        reduction.decode_increment(coefficients),
        physical_anomalies,
        rtol=2e-5,
        atol=2e-5,
    )
    assert reduction.rank == reduction.available_rank <= n_e - 1
    assert reduction.projection_residual(physical_states) < 2e-5


def test_variable_scales_resolve_in_flatten_order_and_warn_for_mixed_units() -> None:
    from data_assimilation.reduction import OnlineStateReduction

    state = xarray.Dataset(
        {
            # Deliberately reverse alphabetic insertion order: state flattening
            # and scale expansion both sort variable names.
            "v": (("ensemble", "x"), np.zeros((3, 2))),
            "u": (("ensemble", "x"), np.zeros((3, 2))),
        },
        coords={"ensemble": np.arange(3), "x": np.arange(2)},
    )
    state["u"].attrs["units"] = "m s-1"
    state["v"].attrs["units"] = "K"

    unscaled = OnlineStateReduction()
    with pytest.warns(UserWarning, match="conflicting units"):
        # The unit default skips row scaling entirely rather than multiplying
        # every (N_s, N_e) array by ones each cycle.
        assert unscaled.resolve_row_scales(state) is None
    assert unscaled.resolved_variable_scales == {"u": 1.0, "v": 1.0}
    # Warn only once per reduction instance.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unscaled.resolve_row_scales(state)

    reduction = OnlineStateReduction(variable_scales={"u": 2.0})
    np.testing.assert_array_equal(
        reduction.resolve_row_scales(state), np.array([2.0, 2.0, 1.0, 1.0])
    )
    assert reduction.resolved_variable_scales == {"u": 2.0, "v": 1.0}

    with pytest.raises(ValueError, match="absent from the state"):
        OnlineStateReduction(variable_scales={"temperature": 1.0}).resolve_row_scales(
            state
        )


def test_invalid_construction_args_raise() -> None:
    from data_assimilation.reduction import (
        OnlineStateReduction,
        StreamingStateReduction,
    )

    with pytest.raises(ValueError, match="energy_fraction"):
        OnlineStateReduction(energy_fraction=0.0)
    with pytest.raises(ValueError, match="basis_source"):
        OnlineStateReduction(basis_source="offline")
    with pytest.raises(ValueError, match="max_rank"):
        OnlineStateReduction(max_rank=0)
    with pytest.raises(ValueError, match="snapshot_stride"):
        OnlineStateReduction(snapshot_stride=0)
    with pytest.raises(ValueError, match="variable_scales"):
        OnlineStateReduction(variable_scales={"u": 0.0})
    with pytest.raises(ValueError, match="forgetting_factor"):
        StreamingStateReduction(forgetting_factor=0.0)
    with pytest.raises(ValueError, match="update_every_n_cycles"):
        StreamingStateReduction(update_every_n_cycles=0)


def _centered_block(array: np.ndarray) -> np.ndarray:
    return (array - array.mean(axis=1, keepdims=True)) / np.sqrt(array.shape[1] - 1)


def _represented_covariance(reduction: Any) -> np.ndarray:
    modes = np.asarray(reduction.modes)
    singular_values = np.asarray(reduction.singular_values)
    return (modes * singular_values[None, :]) @ (modes * singular_values[None, :]).T


@pytest.mark.parametrize("forgetting_factor", [1.0, 0.4])  # type: ignore[misc]
def test_streaming_cycle_zero_exactly_matches_current_batch_svd(
    forgetting_factor: float,
) -> None:
    """Cycle 0 fits B_0 directly — no spurious lambda-dependent scale.

    Parametrized over the forgetting factor precisely because a lambda applied
    to the first block would be invisible at lambda=1.
    """
    from data_assimilation.reduction import (
        OnlineStateReduction,
        StreamingStateReduction,
    )

    block = jax.random.normal(jax.random.PRNGKey(110), (15, 7))
    current = OnlineStateReduction(energy_fraction=1.0, whiten=False)
    streaming = StreamingStateReduction(
        energy_fraction=1.0, forgetting_factor=forgetting_factor
    )
    current.fit(block)
    streaming.fit(block)

    np.testing.assert_allclose(
        streaming.singular_values, current.singular_values, rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(
        np.asarray(streaming.modes @ streaming.modes.T),
        np.asarray(current.modes @ current.modes.T),
        rtol=2e-5,
        atol=2e-5,
    )
    assert streaming.accumulator_weight == 1.0
    assert streaming.subspace_drift is None
    assert streaming.rank == streaming.available_rank <= block.shape[1] - 1


def test_streaming_no_forgetting_matches_concatenated_anomaly_blocks() -> None:
    from data_assimilation.reduction import StreamingStateReduction

    rng = np.random.default_rng(111)
    blocks = [rng.normal(size=(18, n_e)).astype(np.float32) for n_e in (5, 7, 6)]
    reduction = StreamingStateReduction(energy_fraction=1.0)
    for block in blocks:
        reduction.fit(block)

    concatenated = np.concatenate([_centered_block(block) for block in blocks], axis=1)
    expected_covariance = concatenated @ concatenated.T
    np.testing.assert_allclose(
        _represented_covariance(reduction),
        expected_covariance,
        rtol=8e-5,
        atol=8e-5,
    )
    expected_singular_values = np.linalg.svd(concatenated, compute_uv=False)
    expected_rank = np.linalg.matrix_rank(concatenated.astype(np.float32))
    np.testing.assert_allclose(
        reduction.singular_values,
        expected_singular_values[:expected_rank],
        rtol=8e-5,
        atol=8e-5,
    )
    assert reduction.accumulator_weight == len(blocks)


def test_streaming_forgetting_matches_explicit_weighted_covariance() -> None:
    from data_assimilation.reduction import StreamingStateReduction

    forgetting_factor = 0.4
    rng = np.random.default_rng(112)
    blocks = [rng.normal(size=(10, 5)).astype(np.float32) for _ in range(4)]
    reduction = StreamingStateReduction(
        energy_fraction=1.0, forgetting_factor=forgetting_factor
    )
    for block in blocks:
        reduction.fit(block)

    expected = np.zeros((10, 10), dtype=np.float32)
    for block in blocks:
        anomaly_block = _centered_block(block)
        expected = forgetting_factor * expected + anomaly_block @ anomaly_block.T
    np.testing.assert_allclose(
        _represented_covariance(reduction), expected, rtol=1e-4, atol=8e-5
    )
    expected_weight = sum(forgetting_factor**i for i in range(len(blocks)))
    assert reduction.accumulator_weight == pytest.approx(expected_weight)
    assert reduction.covariance_half_life == pytest.approx(
        np.log(0.5) / np.log(forgetting_factor)
    )
    assert reduction.spectrum_max == pytest.approx(
        float(reduction.singular_values[0]) / np.sqrt(expected_weight)
    )


def test_streaming_is_deterministic_and_rank_stays_bounded() -> None:
    from data_assimilation.reduction import StreamingStateReduction

    blocks = [
        np.random.default_rng(seed).normal(size=(25, 6)).astype(np.float32)
        for seed in range(30)
    ]
    first = StreamingStateReduction(energy_fraction=1.0, max_rank=3)
    second = StreamingStateReduction(energy_fraction=1.0, max_rank=3)
    for block in blocks:
        first.fit(block)
        second.fit(block)
        assert first.rank <= 3
        assert first.modes.shape == (25, first.rank)

    np.testing.assert_allclose(first.singular_values, second.singular_values)
    np.testing.assert_allclose(
        np.asarray(first.modes @ first.modes.T),
        np.asarray(second.modes @ second.modes.T),
    )
    from data_assimilation.reduction import _orthogonality_error

    assert _orthogonality_error(first.modes) < 1e-5


def test_streaming_forgetting_adapts_to_rotating_subspace() -> None:
    from data_assimilation.reduction import StreamingStateReduction

    x_direction = np.array([[1.0], [0.0]], dtype=np.float32)
    y_direction = np.array([[0.0], [1.0]], dtype=np.float32)
    member_pattern = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    old_block = x_direction @ member_pattern
    current_block = y_direction @ member_pattern

    equal_history = StreamingStateReduction(
        energy_fraction=1.0, max_rank=1, forgetting_factor=1.0
    )
    forgotten = StreamingStateReduction(
        energy_fraction=1.0, max_rank=1, forgetting_factor=0.1
    )
    for _ in range(8):
        equal_history.fit(old_block)
        forgotten.fit(old_block)
    for _ in range(2):
        equal_history.fit(current_block)
        forgotten.fit(current_block)

    assert forgotten.projection_residual(current_block) < 0.1
    assert equal_history.projection_residual(current_block) > 0.9


def test_streaming_update_cadence_reuses_then_updates_basis() -> None:
    from data_assimilation.reduction import StreamingStateReduction

    first = np.array([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    second = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    reduction = StreamingStateReduction(
        energy_fraction=1.0, max_rank=2, update_every_n_cycles=2
    )
    reduction.fit(first)
    covariance_after_first = _represented_covariance(reduction)
    reduction.fit(second)  # skipped by cadence, but validated and re-centered
    np.testing.assert_array_equal(
        _represented_covariance(reduction), covariance_after_first
    )
    assert reduction.accumulator_weight == 1.0
    reduction.fit(second)
    assert reduction.rank == 2
    assert reduction.accumulator_weight == 2.0


def test_nonfinite_streaming_fit_is_transactional() -> None:
    from data_assimilation.reduction import StreamingStateReduction

    reduction = StreamingStateReduction(energy_fraction=1.0)
    valid = jax.random.normal(jax.random.PRNGKey(113), (9, 5))
    reduction.fit(valid)
    modes_before = np.asarray(reduction.modes).copy()
    singular_values_before = np.asarray(reduction.singular_values).copy()
    mean_before = np.asarray(reduction._mean).copy()
    weight_before = reduction.accumulator_weight
    drift_before = reduction.subspace_drift

    invalid = np.asarray(valid).copy()
    invalid[2, 1] = np.nan
    invalid[7, 4] = np.inf
    with pytest.raises(ValueError, match=r"columns \[1, 4\]"):
        reduction.fit(invalid)

    np.testing.assert_array_equal(reduction.modes, modes_before)
    np.testing.assert_array_equal(reduction.singular_values, singular_values_before)
    np.testing.assert_array_equal(reduction._mean, mean_before)
    assert reduction.accumulator_weight == weight_before
    assert reduction.subspace_drift == drift_before

    # A valid repaired block remains admissible immediately after the failure.
    repaired = invalid
    repaired[:, 1] = repaired[:, 0]
    repaired[:, 4] = repaired[:, 3]
    reduction.fit(repaired)
    assert np.isfinite(reduction.singular_values).all()


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


def test_row_scales_follow_state_augmentation_flatten_order() -> None:
    """Scales land on the rows ``StateAugmentation.flatten`` actually produces.

    The reduction must not re-derive the flattening order: a multi-variable
    state whose variables have *different* shapes would silently misalign
    scales onto the wrong rows if the two orders ever diverged.
    """
    from data_assimilation.augmentation import StateAugmentation
    from data_assimilation.reduction import OnlineStateReduction

    n_e = 3
    state = xarray.Dataset(
        {
            "w": (("ensemble", "x", "y"), np.full((n_e, 2, 3), 30.0)),
            "u": (("ensemble", "s"), np.full((n_e, 4), 10.0)),
            "v": (("ensemble", "x", "y"), np.full((n_e, 2, 3), 20.0)),
        },
        coords={
            "ensemble": np.arange(n_e),
            "x": np.arange(2),
            "y": np.arange(3),
            "s": np.arange(4),
        },
    )

    scales = {"u": 10.0, "v": 20.0, "w": 30.0}
    row_scales = OnlineStateReduction(variable_scales=scales).resolve_row_scales(state)
    flat = np.asarray(StateAugmentation().flatten(state))

    assert row_scales is not None
    assert flat.shape[0] == np.asarray(row_scales).shape[0]
    # Every variable's constant value equals its configured scale, so a
    # correctly ordered scale vector normalises the flat state to exactly 1.
    np.testing.assert_allclose(flat / np.asarray(row_scales)[:, None], 1.0, rtol=1e-6)


@pytest.mark.parametrize(  # type: ignore[misc]
    "n_s,n_e", [(2_000, 2), (60_000, 20), (172_800, 50)]
)
def test_numerical_rank_survives_production_state_sizes(n_s: int, n_e: int) -> None:
    """A "full rank" basis must stay full rank across the supported shapes.

    Both ends matter in float32. numpy's ``matrix_rank`` tolerance scales with
    ``max(shape)``, so at ``N_s ~ 1e5`` it would cut every mode below ~1 percent
    of sigma_max — silently truncating the correctness control while still
    reporting full retained energy. Scaling by the sample dimension alone breaks
    the other end: at the 2-member smoke shape it would fall below the centering
    round-off floor and admit the null direction as a real mode.
    """
    from data_assimilation.reduction import OnlineStateReduction

    rng = np.random.default_rng(220)
    decay = np.linspace(1.0, 0.002, n_e).astype(np.float32)
    states = jnp.asarray(
        (rng.normal(size=(n_s, n_e)).astype(np.float32) * decay[None, :])
        @ rng.normal(size=(n_e, n_e)).astype(np.float32)
    )

    reduction = OnlineStateReduction(energy_fraction=1.0, whiten=False)
    reduction.fit(states)
    # Centering removes exactly one direction; every other mode is genuine.
    assert reduction.available_rank == n_e - 1
    assert reduction.rank == n_e - 1
    assert reduction.projection_residual(states) < 1e-5
    assert reduction.retained_energy_fraction == pytest.approx(1.0, abs=1e-6)


def test_retained_energy_is_measured_against_the_full_spectrum() -> None:
    """Truncation must show up as retained energy below one, not be normalised away."""
    from data_assimilation.reduction import OnlineStateReduction

    rng = np.random.default_rng(221)
    decay = np.geomspace(1.0, 1e-3, 10).astype(np.float32)
    states = jnp.asarray(
        (rng.normal(size=(40, 10)).astype(np.float32) * decay[None, :])
        @ rng.normal(size=(10, 10)).astype(np.float32)
    )
    reduction = OnlineStateReduction(energy_fraction=1.0, max_rank=3, whiten=False)
    reduction.fit(states)

    spectrum = np.asarray(
        jnp.linalg.svd(
            (np.asarray(states) - np.asarray(states).mean(axis=1, keepdims=True))
            / np.sqrt(states.shape[1] - 1),
            compute_uv=False,
        )
    )
    expected = float(np.sum(spectrum[:3] ** 2) / np.sum(spectrum**2))
    assert reduction.rank == 3
    assert reduction.retained_energy_fraction == pytest.approx(expected, rel=1e-4)
    assert reduction.retained_energy_fraction < 1.0


def test_fit_coordinates_reconstruct_the_fitted_anomalies() -> None:
    """``anomalies = U_full @ fit_coordinates`` — the free truncation split."""
    from data_assimilation.reduction import (
        OnlineStateReduction,
        StreamingStateReduction,
    )

    rng = np.random.default_rng(222)
    block = jnp.asarray(rng.normal(size=(30, 8)).astype(np.float32))
    anomalies = np.asarray(
        (block - block.mean(axis=1, keepdims=True)) / np.sqrt(block.shape[1] - 1)
    )

    for reduction in (
        OnlineStateReduction(energy_fraction=1.0, max_rank=2, whiten=False),
        StreamingStateReduction(energy_fraction=1.0, max_rank=2),
    ):
        reduction.fit(block)
        coordinates = reduction.fit_coordinates
        assert coordinates is not None
        assert reduction.rank == 2 and coordinates.shape[0] > reduction.rank
        # The retained rows are exactly what the truncated basis represents.
        retained = np.asarray(reduction.modes @ coordinates[: reduction.rank])
        projected = anomalies - (
            anomalies
            - np.asarray(reduction.modes) @ (np.asarray(reduction.modes).T @ anomalies)
        )
        np.testing.assert_allclose(retained, projected, rtol=2e-4, atol=2e-5)
        # ...and the full set reconstructs the anomalies' energy.
        np.testing.assert_allclose(
            float(np.linalg.norm(np.asarray(coordinates))),
            float(np.linalg.norm(anomalies)),
            rtol=2e-4,
        )


def test_streaming_skipped_cycle_reports_no_stale_split() -> None:
    from data_assimilation.reduction import StreamingStateReduction

    rng = np.random.default_rng(223)
    reduction = StreamingStateReduction(energy_fraction=1.0, update_every_n_cycles=2)
    observed = []
    for _ in range(2):
        reduction.fit(jnp.asarray(rng.normal(size=(12, 5)).astype(np.float32)))
        observed.append(
            (
                reduction.basis_updated,
                reduction.fit_coordinates is None,
                reduction.subspace_drift,
            )
        )

    # The second cycle is skipped by the cadence: it reuses the basis and
    # reports no coordinate split and no drift rather than stale values.
    assert [updated for updated, _, _ in observed] == [True, False]
    assert [no_split for _, no_split, _ in observed] == [False, True]
    assert [drift for _, _, drift in observed] == [None, None]


def test_shipped_streaming_defaults_keep_rank_and_orthogonality_bounded() -> None:
    """The shipped streaming configuration must survive a long run.

    An energy criterion alone cannot bound the accumulator: it keeps absorbing
    new directions, so the retained rank (and the ``(N_s, rank)`` basis) grows
    every cycle until it dwarfs the ensemble. The rank cap in
    ``conf/filtering/state_reduction/svd_streaming.yaml`` is what makes the
    streaming option usable, and a single Gram-Schmidt pass would let
    orthogonality decay until the expensive re-orthogonalization branch fires
    (which cannot report a coordinate split).
    """
    from data_assimilation.reduction import (
        StreamingStateReduction,
        _orthogonality_error,
    )
    from omegaconf import OmegaConf

    shipped = OmegaConf.load(
        "conf/filtering/state_reduction/svd_streaming.yaml"
    ).state_reduction
    assert shipped.max_rank is not None, "the shipped default must bound the rank"
    reduction = StreamingStateReduction(
        energy_fraction=shipped.energy_fraction,
        max_rank=shipped.max_rank,
        forgetting_factor=shipped.forgetting_factor,
        update_every_n_cycles=shipped.update_every_n_cycles,
    )

    n_s, n_e = 300, 12
    rng = np.random.default_rng(224)
    for _ in range(30):
        reduction.fit(jnp.asarray(rng.normal(size=(n_s, n_e)).astype(np.float32)))
        assert reduction.rank <= shipped.max_rank
        assert reduction.modes.shape == (n_s, reduction.rank)
        # Stay below the implementation's own (rank-scaled) correction trigger:
        # the re-orthogonalization branch must never have to fire.
        assert _orthogonality_error(reduction.modes) < 1e-5 * math.sqrt(reduction.rank)
        # The coordinate split survives every cycle, so the discarded-increment
        # diagnostic is never silently unavailable.
        assert reduction.fit_coordinates is not None


def test_subspace_drift_is_meaningful_when_the_rank_changes() -> None:
    """A growing rank is not a 90-degree rotation."""
    from data_assimilation.reduction import StreamingStateReduction

    rng = np.random.default_rng(225)
    basis = np.linalg.qr(rng.normal(size=(20, 20)))[0].astype(np.float32)

    # Cycle 1 adds directions inside a subspace the basis already spans, so the
    # rank grows while the subspace barely tilts.
    nested = StreamingStateReduction(energy_fraction=1.0)
    nested.fit(jnp.asarray(basis[:, :4] @ rng.normal(size=(4, 6)).astype(np.float32)))
    rank_before = nested.rank
    nested.fit(jnp.asarray(basis[:, :8] @ rng.normal(size=(8, 6)).astype(np.float32)))
    assert nested.rank > rank_before
    assert nested.subspace_drift is not None
    # The drift is an acos of a float32 cosine near 1, which resolves nothing
    # below a few times 1e-3 and moves with the platform's LAPACK: this same
    # basis reads 3.5e-4 on macOS/Accelerate and 1.5e-3 on Linux/OpenBLAS. The
    # claim under test is that a rank change alone is not reported as a
    # rotation, so the threshold sits above that floor and two orders of
    # magnitude below the pi/2 it has to rule out.
    assert nested.subspace_drift < 1e-2

    # A block in an orthogonal subspace does tilt it, and drift is reported as a
    # real angle rather than being clipped to pi/2 by the rank change alone.
    rotated = StreamingStateReduction(energy_fraction=1.0, max_rank=3)
    rotated.fit(jnp.asarray(basis[:, :3] @ rng.normal(size=(3, 6)).astype(np.float32)))
    rotated.fit(
        jnp.asarray(10.0 * basis[:, 10:13] @ rng.normal(size=(3, 6)).astype(np.float32))
    )
    assert rotated.subspace_drift is not None
    assert rotated.subspace_drift > 1.0


def test_esmda_whitened_rank_rule_is_unchanged() -> None:
    """The whitened (ESMDA) path keeps numpy's conservative ``max(shape)`` cut.

    ``encode`` divides by the retained singular values there, so admitting a
    mode just above the round-off floor amplifies it by ``1/sigma``. The
    float32-safe floor is scoped to the non-whitened filtering path.
    """
    from data_assimilation.reduction import _numerical_rank

    n_s, n_e = 200_000, 50
    singular_values = jnp.asarray(
        np.concatenate([[1.0], np.full(n_e - 1, 2e-2)]).astype(np.float32)
    )
    eps = float(np.finfo(np.float32).eps)
    assert n_s * eps > 2e-2 > n_e * eps  # the two rules disagree here by design

    assert _numerical_rank(singular_values, (n_s, n_e), conservative=True) == 1
    assert _numerical_rank(singular_values, (n_s, n_e), conservative=False) == n_e
