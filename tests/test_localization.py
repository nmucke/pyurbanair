"""Tests for ESMDA localization (data_assimilation.localization)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _correlation_matrix(aug_dev: jnp.ndarray, pred_obs_dev: jnp.ndarray) -> np.ndarray:
    N_e = aug_dev.shape[1]
    cov = (aug_dev @ pred_obs_dev.T) / (N_e - 1)
    denom = jnp.outer(aug_dev.std(axis=1), pred_obs_dev.std(axis=1))
    return np.array(cov / denom)


def test_correlation_excludes_below_threshold_and_keeps_above() -> None:
    from data_assimilation.localization.correlation import CorrelationLocalization

    N_e = 200
    base = jax.random.normal(jax.random.PRNGKey(0), (N_e,))
    aug = jnp.stack(
        [
            base,  # correlated with obs 0 and 2
            jax.random.normal(jax.random.PRNGKey(1), (N_e,)),  # uncorrelated
        ]
    )
    pred = jnp.stack(
        [
            base,  # strongly correlated with row 0
            jax.random.normal(jax.random.PRNGKey(2), (N_e,)),  # noise
            base + 0.1 * jax.random.normal(jax.random.PRNGKey(3), (N_e,)),
        ]
    )
    aug_dev = aug - aug.mean(axis=1, keepdims=True)
    pred_dev = pred - pred.mean(axis=1, keepdims=True)

    rho = np.abs(_correlation_matrix(aug_dev, pred_dev))
    loc = CorrelationLocalization(truncation_correlation=0.3, max_inflation=8.0)
    inflation = np.array(loc.inflation_factors(aug_dev, pred_dev))

    assert np.all(np.isinf(inflation[rho < 0.3]))
    assert np.all(np.isfinite(inflation[rho >= 0.3]))
    assert np.all(inflation[np.isfinite(inflation)] >= 1.0 - 1e-6)


def test_correlation_inflation_reaches_max_at_truncation_distance() -> None:
    from data_assimilation.localization.correlation import CorrelationLocalization

    rho_t, e_max, beta = 0.3, 8.0, 0.5
    loc = CorrelationLocalization(
        truncation_correlation=rho_t, tapering_beta=beta, max_inflation=e_max
    )

    # Construct a single (row, obs) pair whose correlation sits just inside
    # the threshold so the correlation distance ~ truncation distance.
    N_e = 4000
    g = jax.random.normal(jax.random.PRNGKey(0), (N_e,))
    noise = jax.random.normal(jax.random.PRNGKey(1), (N_e,))
    # mix to target |corr| ~ rho_t (=> d_c ~ d_t => inflation ~ e_max)
    target = rho_t
    obs = target * g + jnp.sqrt(1 - target**2) * noise
    aug_dev = (g - g.mean())[None, :]
    pred_dev = (obs - obs.mean())[None, :]

    rho = abs(_correlation_matrix(aug_dev, pred_dev)[0, 0])
    inflation = float(loc.inflation_factors(aug_dev, pred_dev)[0, 0])

    # Only assert when the sampled correlation stayed above the threshold.
    if rho >= rho_t:
        assert inflation == pytest.approx(e_max, rel=0.25)


def test_max_inflation_one_disables_tapering() -> None:
    from data_assimilation.localization.correlation import CorrelationLocalization

    N_e = 300
    aug_dev = jax.random.normal(jax.random.PRNGKey(0), (4, N_e))
    pred_dev = jax.random.normal(jax.random.PRNGKey(1), (3, N_e))
    loc = CorrelationLocalization(truncation_correlation=1e-6, max_inflation=1.0)
    inflation = np.array(loc.inflation_factors(aug_dev, pred_dev))
    # Nothing excluded (threshold ~0), nothing tapered (E_max == 1).
    assert np.allclose(inflation, 1.0)


def _global_update(augmented, pred_obs, obs, C_D, C_D_sqrt, alpha, rng_key):
    N_e = augmented.shape[1]
    aug_dev = augmented - augmented.mean(axis=1, keepdims=True)
    po_dev = pred_obs - pred_obs.mean(axis=1, keepdims=True)
    C_MD = (aug_dev @ po_dev.T) / (N_e - 1)
    C_DD = (po_dev @ po_dev.T) / (N_e - 1)
    Z = jax.random.normal(rng_key, (obs.shape[0], N_e))
    perturbed = obs[:, None] + jnp.sqrt(alpha) * (C_D_sqrt @ Z)
    x = jnp.linalg.solve(C_DD + alpha * C_D, perturbed - pred_obs)
    return augmented + C_MD @ x


def test_localized_update_matches_global_when_nothing_localized() -> None:
    from data_assimilation.localization.correlation import CorrelationLocalization

    N_aug, N_d, N_e = 6, 5, 80
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(7), 4)
    augmented = jax.random.normal(k1, (N_aug, N_e))
    M = jax.random.normal(k2, (N_d, N_aug))
    pred_obs = M @ augmented + 0.3 * jax.random.normal(k3, (N_d, N_e))
    obs = jax.random.normal(k4, (N_d,))
    C_D = jnp.diag(0.25 * jnp.ones(N_d))
    C_D_sqrt = jnp.sqrt(C_D)
    alpha = 2.0
    rng = jax.random.PRNGKey(123)

    aug_dev = augmented - augmented.mean(axis=1, keepdims=True)
    po_dev = pred_obs - pred_obs.mean(axis=1, keepdims=True)

    loc = CorrelationLocalization(truncation_correlation=1e-6, max_inflation=1.0)
    localized = loc.localized_update(
        augmented=augmented,
        aug_dev=aug_dev,
        pred_obs=pred_obs,
        pred_obs_dev=po_dev,
        obs=obs,
        C_D=C_D,
        C_D_sqrt=C_D_sqrt,
        alpha=alpha,
        rng_key=rng,
    )
    global_result = _global_update(augmented, pred_obs, obs, C_D, C_D_sqrt, alpha, rng)
    assert jnp.allclose(localized, global_result, atol=1e-5)


def test_excluded_row_is_left_unchanged() -> None:
    from data_assimilation.localization.correlation import CorrelationLocalization

    N_aug, N_d, N_e = 4, 5, 80
    k1, k3, k4 = jax.random.split(jax.random.PRNGKey(11), 3)
    augmented = jax.random.normal(k1, (N_aug, N_e))
    # Predicted observations independent of the state -> only sampling-noise
    # correlations (~1/sqrt(N_e)), all far below the 0.999 threshold.
    pred_obs = jax.random.normal(k3, (N_d, N_e))
    obs = jax.random.normal(k4, (N_d,))
    C_D = jnp.diag(0.25 * jnp.ones(N_d))
    C_D_sqrt = jnp.sqrt(C_D)

    aug_dev = augmented - augmented.mean(axis=1, keepdims=True)
    po_dev = pred_obs - pred_obs.mean(axis=1, keepdims=True)

    # Threshold so high that every observation is excluded for every row.
    loc = CorrelationLocalization(truncation_correlation=0.999, max_inflation=8.0)
    updated = loc.localized_update(
        augmented=augmented,
        aug_dev=aug_dev,
        pred_obs=pred_obs,
        pred_obs_dev=po_dev,
        obs=obs,
        C_D=C_D,
        C_D_sqrt=C_D_sqrt,
        alpha=2.0,
        rng_key=jax.random.PRNGKey(0),
    )
    assert jnp.allclose(updated, augmented, atol=1e-6)


def test_block_grouping_shares_selection_and_transition() -> None:
    """Co-located rows in one block get an identical (joint) update."""
    from data_assimilation.localization.correlation import CorrelationLocalization

    N_aug, N_d, N_e = 6, 5, 80
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(3), 4)
    # Build co-located rows: rows {0,1} are near-duplicates (same cell), as are
    # {2,3} and {4,5}. They individually correlate with different observations,
    # so without grouping they would select different obs.
    cells = jax.random.normal(k1, (3, N_e))
    augmented = jnp.stack(
        [cells[0], cells[0] + 1e-3 * jax.random.normal(k2, (N_e,)),
         cells[1], cells[1] + 1e-3 * jax.random.normal(k3, (N_e,)),
         cells[2], cells[2]]
    )
    M = jax.random.normal(k4, (N_d, 3))
    pred_obs = M @ cells + 0.3 * jax.random.normal(jax.random.PRNGKey(9), (N_d, N_e))
    obs = jax.random.normal(jax.random.PRNGKey(10), (N_d,))
    C_D = jnp.diag(0.25 * jnp.ones(N_d))
    C_D_sqrt = jnp.sqrt(C_D)

    aug_dev = augmented - augmented.mean(axis=1, keepdims=True)
    po_dev = pred_obs - pred_obs.mean(axis=1, keepdims=True)
    group_ids = jnp.array([0, 0, 1, 1, 2, 2])

    loc = CorrelationLocalization(truncation_correlation=0.2, max_inflation=8.0)
    inflation = np.array(loc.inflation_factors(aug_dev, po_dev))

    # Members of a block share an identical inflation row after grouping, so
    # the active-observation set and transition are shared across the block.
    from data_assimilation.localization.base import _group_inflation

    grouped = np.array(_group_inflation(jnp.asarray(inflation), group_ids))
    for a, b in [(0, 1), (2, 3), (4, 5)]:
        assert np.array_equal(
            np.isinf(grouped[a]), np.isinf(grouped[b])
        ), "block members must select the same observations"
        finite = np.isfinite(grouped[a])
        assert np.allclose(grouped[a][finite], grouped[b][finite])
    # The block min is <= each member's own inflation (strongest correlation).
    assert np.all(grouped[0][np.isfinite(grouped[0])] <= inflation[0][np.isfinite(grouped[0])] + 1e-6)


def _build_localization_problem(seed: int = 21):
    """Small augmented problem where state rows (0..3) correlate with obs and
    param rows (4..5) are independent (so localization would taper/exclude obs
    for them). Returns the inputs needed for ``localized_update``/``_global``.
    """
    N_state, N_param, N_d, N_e = 4, 2, 5, 120
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(seed), 5)
    state = jax.random.normal(k1, (N_state, N_e))
    # Params independent of everything -> only sampling-noise correlations.
    params = jax.random.normal(k2, (N_param, N_e))
    augmented = jnp.concatenate([state, params], axis=0)
    # Predicted obs driven by the state only.
    M = jax.random.normal(k3, (N_d, N_state))
    pred_obs = M @ state + 0.3 * jax.random.normal(k4, (N_d, N_e))
    obs = jax.random.normal(k5, (N_d,))
    C_D = jnp.diag(0.25 * jnp.ones(N_d))
    C_D_sqrt = jnp.sqrt(C_D)
    return augmented, pred_obs, obs, C_D, C_D_sqrt, N_state, N_param


def test_localize_mask_gives_global_update_for_masked_rows() -> None:
    """Masked-out (param) rows get the exact global update; some state rows
    differ from theirs because localization tapers/excludes observations."""
    from data_assimilation.localization.correlation import CorrelationLocalization

    augmented, pred_obs, obs, C_D, C_D_sqrt, N_state, N_param = (
        _build_localization_problem()
    )
    alpha = 2.0
    rng = jax.random.PRNGKey(99)

    aug_dev = augmented - augmented.mean(axis=1, keepdims=True)
    po_dev = pred_obs - pred_obs.mean(axis=1, keepdims=True)

    localize_mask = jnp.concatenate(
        [jnp.ones(N_state, dtype=bool), jnp.zeros(N_param, dtype=bool)]
    )

    # Threshold high enough to exclude/taper observations for the masked rows.
    loc = CorrelationLocalization(truncation_correlation=0.3, max_inflation=8.0)
    localized = loc.localized_update(
        augmented=augmented,
        aug_dev=aug_dev,
        pred_obs=pred_obs,
        pred_obs_dev=po_dev,
        obs=obs,
        C_D=C_D,
        C_D_sqrt=C_D_sqrt,
        alpha=alpha,
        rng_key=rng,
        localize_mask=localize_mask,
    )
    global_result = _global_update(
        augmented, pred_obs, obs, C_D, C_D_sqrt, alpha, rng
    )

    # Masked-out rows (params) match the global update exactly.
    assert jnp.allclose(
        localized[N_state:], global_result[N_state:], atol=1e-5
    )
    # At least one localized (state) row differs from its global update.
    assert not jnp.allclose(
        localized[:N_state], global_result[:N_state], atol=1e-5
    )


def test_localize_mask_none_unchanged() -> None:
    """Passing ``localize_mask=None`` reproduces the call without the arg."""
    from data_assimilation.localization.correlation import CorrelationLocalization

    augmented, pred_obs, obs, C_D, C_D_sqrt, _, _ = _build_localization_problem(
        seed=5
    )
    alpha = 2.0
    rng = jax.random.PRNGKey(7)

    aug_dev = augmented - augmented.mean(axis=1, keepdims=True)
    po_dev = pred_obs - pred_obs.mean(axis=1, keepdims=True)

    loc = CorrelationLocalization(truncation_correlation=0.3, max_inflation=8.0)
    kwargs = dict(
        augmented=augmented,
        aug_dev=aug_dev,
        pred_obs=pred_obs,
        pred_obs_dev=po_dev,
        obs=obs,
        C_D=C_D,
        C_D_sqrt=C_D_sqrt,
        alpha=alpha,
        rng_key=rng,
    )
    without_arg = loc.localized_update(**kwargs)
    with_none = loc.localized_update(**kwargs, localize_mask=None)
    assert jnp.array_equal(without_arg, with_none)


def test_localize_mask_with_state_grouping() -> None:
    """State rows grouped + params masked: block members get identical updates,
    and the masked param rows still receive the global update."""
    from data_assimilation.localization.correlation import CorrelationLocalization

    N_d, N_e = 5, 80
    k1, kp = jax.random.split(jax.random.PRNGKey(31), 2)
    # State: cells {0,1} are exact duplicates (one block), {2,3} another block.
    # Identical anomalies AND identical cross-covariance rows -> the joint block
    # update is byte-for-byte identical across block members.
    cells = jax.random.normal(k1, (2, N_e))
    state = jnp.stack([cells[0], cells[0], cells[1], cells[1]])
    params = jax.random.normal(kp, (2, N_e))  # independent param rows
    augmented = jnp.concatenate([state, params], axis=0)
    N_state, N_param = 4, 2

    M = jax.random.normal(jax.random.PRNGKey(8), (N_d, 2))
    pred_obs = M @ cells + 0.3 * jax.random.normal(jax.random.PRNGKey(9), (N_d, N_e))
    obs = jax.random.normal(jax.random.PRNGKey(10), (N_d,))
    C_D = jnp.diag(0.25 * jnp.ones(N_d))
    C_D_sqrt = jnp.sqrt(C_D)
    alpha = 2.0
    rng = jax.random.PRNGKey(123)

    aug_dev = augmented - augmented.mean(axis=1, keepdims=True)
    po_dev = pred_obs - pred_obs.mean(axis=1, keepdims=True)

    # State rows grouped by cell; each param gets its own unique block id.
    group_ids = jnp.array([0, 0, 1, 1, 2, 3])
    localize_mask = jnp.concatenate(
        [jnp.ones(N_state, dtype=bool), jnp.zeros(N_param, dtype=bool)]
    )

    loc = CorrelationLocalization(truncation_correlation=0.2, max_inflation=8.0)
    updated = loc.localized_update(
        augmented=augmented,
        aug_dev=aug_dev,
        pred_obs=pred_obs,
        pred_obs_dev=po_dev,
        obs=obs,
        C_D=C_D,
        C_D_sqrt=C_D_sqrt,
        alpha=alpha,
        rng_key=rng,
        group_ids=group_ids,
        localize_mask=localize_mask,
    )

    # Block members share an identical (joint) update.
    assert jnp.allclose(updated[0], updated[1], atol=1e-5)
    assert jnp.allclose(updated[2], updated[3], atol=1e-5)

    # The masked param rows still get the global update.
    global_result = _global_update(
        augmented, pred_obs, obs, C_D, C_D_sqrt, alpha, rng
    )
    assert jnp.allclose(updated[N_state:], global_result[N_state:], atol=1e-5)


def test_parameter_esmda_runs_with_correlation_localization(compose_test_cfg) -> None:
    """End-to-end: parameter ESMDA composes and runs with localization on."""
    from scripts.run_esmda import run

    cfg = compose_test_cfg(
        [
            "model@truth_model=pyudales",
            "model@assim_model=pyudales",
            "esmda/smoother=static",
            "params@prior_params=static",
            "params@truth_params=static_truth",
            # Switch on adaptive correlation localization via the config group
            # (`esmda/localization` defaults to `none`), then lower its threshold.
            "esmda/localization=correlation",
            "esmda.localization.truncation_correlation=0.2",
            "ensemble.ensemble_size=4",
            "ensemble.num_parallel_processes=2",
            "esmda.num_steps=1",
            "esmda.num_assimilation_windows=1",
            "run.skip_viz=true",
            # Inline truth + in-bounds test-grid sensors + a domain-appropriate
            # nnudge_meters (see tests/test_run_esmda.py for the same reasons).
            "run.truth_dir=null",
            "obs.x_points=[2.5,2.5,18.0,18.0]",
            "obs.y_points=[5.0,15.0,5.0,15.0]",
            "obs.z_points=[3.0,3.0,3.0,3.0]",
            "obs.interval_seconds=3.0",
            "truth_model.forward_model.nudging_config.nnudge_meters=4.0",
            "assim_model.forward_model.nudging_config.nnudge_meters=4.0",
        ],
        config_name="run_esmda",
    )
    run(cfg)


def test_invalid_parameters_raise() -> None:
    from data_assimilation.localization.correlation import CorrelationLocalization

    with pytest.raises(ValueError):
        CorrelationLocalization(truncation_correlation=1.5)
    with pytest.raises(ValueError):
        CorrelationLocalization(tapering_beta=0.0)
    with pytest.raises(ValueError):
        CorrelationLocalization(max_inflation=0.5)


# ---------------------------------------------------------------------------
# Distance-based localization
# ---------------------------------------------------------------------------

def test_distance_excludes_beyond_radius_keeps_within() -> None:
    from data_assimilation.localization.distance import DistanceLocalization

    loc = DistanceLocalization(
        localization_radius=10.0, tapering_beta=0.5, max_inflation=4.0
    )
    row = jnp.zeros((1, 3))  # one grid point at the origin
    # Sensors at horizontal distance 3 (un-tapered), 8 (tapered), 20 (excluded).
    obs = jnp.array([[3.0, 0.0, 0.0], [8.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    inflation = np.array(loc.inflation_factors(None, None, row_coords=row, obs_coords=obs))

    assert inflation[0, 0] == pytest.approx(1.0)  # within beta*radius -> no taper
    assert np.isfinite(inflation[0, 1]) and inflation[0, 1] > 1.0  # tapered
    assert np.isinf(inflation[0, 2])  # beyond the radius -> excluded


def test_distance_inflation_reaches_max_at_radius() -> None:
    from data_assimilation.localization.distance import DistanceLocalization

    loc = DistanceLocalization(
        localization_radius=10.0, tapering_beta=0.5, max_inflation=4.0
    )
    row = jnp.zeros((1, 3))
    obs = jnp.array([[10.0, 0.0, 0.0]])  # exactly at the radius
    inflation = float(loc.inflation_factors(None, None, row_coords=row, obs_coords=obs)[0, 0])
    assert inflation == pytest.approx(4.0, rel=1e-5)


def test_distance_requires_coordinates() -> None:
    from data_assimilation.localization.distance import DistanceLocalization

    loc = DistanceLocalization(localization_radius=10.0)
    with pytest.raises(ValueError):
        loc.inflation_factors(jnp.zeros((2, 5)), jnp.zeros((3, 5)))


def test_distance_horizontal_only_ignores_vertical() -> None:
    from data_assimilation.localization.distance import DistanceLocalization

    row = jnp.zeros((1, 3))
    obs = jnp.array([[3.0, 0.0, 100.0]])  # 3 m horizontally, 100 m vertically

    horiz = DistanceLocalization(localization_radius=10.0, horizontal_only=True)
    assert np.isfinite(
        float(horiz.inflation_factors(None, None, row_coords=row, obs_coords=obs)[0, 0])
    )  # horizontal distance 3 < radius -> kept

    full = DistanceLocalization(localization_radius=10.0, horizontal_only=False)
    assert np.isinf(
        float(full.inflation_factors(None, None, row_coords=row, obs_coords=obs)[0, 0])
    )  # 3-D distance ~100 > radius -> excluded


def test_distance_invalid_parameters_raise() -> None:
    from data_assimilation.localization.distance import DistanceLocalization

    with pytest.raises(ValueError):
        DistanceLocalization(localization_radius=0.0)
    with pytest.raises(ValueError):
        DistanceLocalization(localization_radius=10.0, tapering_beta=1.0)
    with pytest.raises(ValueError):
        DistanceLocalization(localization_radius=10.0, max_inflation=0.5)


def test_observation_coords_tile_sensor_locations() -> None:
    """obs j maps to sensor j % num_sensors (sensor is the innermost axis)."""
    from data_assimilation.observation_operator import ObservationOperator
    from data_assimilation.smoothing.esmda import StateAndParameterESMDA

    op = ObservationOperator(
        obs_x=[1.0, 2.0, 3.0],
        obs_y=[4.0, 5.0, 6.0],
        obs_z=[7.0, 8.0, 9.0],
        obs_states=["u", "v", "w"],
        solver_name="pylbm",
    )
    obj = StateAndParameterESMDA.__new__(StateAndParameterESMDA)
    obj.observation_operator = op

    n_d = op.num_obs  # 3 sensors * 3 states = 9
    coords = np.array(obj._observation_coords(n_d))
    sensors = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]], dtype=float)
    assert coords.shape == (9, 3)
    for j in range(n_d):
        assert np.allclose(coords[j], sensors[j % 3])


def test_state_row_coords_match_flatten_order() -> None:
    """`_state_row_coords` row order matches `_flatten_state` exactly.

    Build a state whose ``u`` equals its x-coordinate and ``v`` equals its
    y-coordinate at every cell; flattening then yields the coordinate per row,
    which must equal the coordinate `_state_row_coords` reports for that row.
    """
    import xarray
    from data_assimilation.smoothing.esmda import StateAndParameterESMDA

    x = np.array([0.0, 10.0, 20.0, 30.0])
    y = np.array([0.0, 5.0])
    z = np.array([1.0, 2.0, 3.0])
    ens = 2
    shape = (ens, z.size, y.size, x.size)
    u = np.broadcast_to(x[None, None, None, :], shape).astype(float)
    v = np.broadcast_to(y[None, None, :, None], shape).astype(float)
    state = xarray.Dataset(
        {"u": (("ensemble", "z", "y", "x"), u.copy()),
         "v": (("ensemble", "z", "y", "x"), v.copy())},
        coords={"x": x, "y": y, "z": z},
    )
    obj = StateAndParameterESMDA.__new__(StateAndParameterESMDA)

    flat = np.array(obj._flatten_state(state))  # (N_s, ens); u rows then v rows
    coords = np.array(obj._state_row_coords(state))  # (N_s, 3)
    n_cells = z.size * y.size * x.size

    # u rows carry the x coordinate; v rows carry the y coordinate.
    assert np.allclose(flat[:n_cells, 0], coords[:n_cells, 0])
    assert np.allclose(flat[n_cells:, 0], coords[n_cells:, 1])


def test_state_row_coords_raise_on_missing_coordinates() -> None:
    """A spatial dim without coordinate values must raise, not fall back to
    grid indices (which would be compared against sensor coords in metres)."""
    import xarray
    from data_assimilation.smoothing.esmda import StateAndParameterESMDA

    state = xarray.Dataset(
        {"u": (("ensemble", "z", "y", "x"), np.zeros((2, 3, 2, 4)))},
        coords={"x": np.arange(4.0), "y": np.arange(2.0)},  # no z coordinate
    )
    obj = StateAndParameterESMDA.__new__(StateAndParameterESMDA)
    with pytest.raises(ValueError, match="'z'.*no.*coordinate"):
        obj._state_row_coords(state)


def test_get_states_on_disk_selects_initial_frame(tmp_path) -> None:
    """The on-disk branch must read the same frame as the in-memory branch
    (time=0, the window's initial condition) — `_analysis` feeds the analyzed
    result forward as the next forecast's warm start."""
    import xarray
    from data_assimilation.smoothing.esmda import StateAndParameterESMDA

    time = np.array([0.0, 1.0])
    members = []
    for i in range(2):
        u = np.full((2, 3), float(i))  # frame t: value i + 10*t
        u[1, :] += 10.0
        ds = xarray.Dataset(
            {"u": (("time", "x"), u)},
            coords={"time": time, "x": np.arange(3.0)},
        )
        ds.to_netcdf(tmp_path / f"state_{i}.nc")
        members.append(ds)

    obj = StateAndParameterESMDA.__new__(StateAndParameterESMDA)
    on_disk = obj._get_states(results_dir=tmp_path)
    in_memory = obj._get_states(
        state=xarray.concat(members, dim="ensemble", join="override")
    )

    assert "time" not in on_disk.dims
    np.testing.assert_allclose(on_disk["u"].values, in_memory["u"].values)
    np.testing.assert_allclose(on_disk["u"].values, [[0.0] * 3, [1.0] * 3])


def test_return_state_history_on_disk_raises() -> None:
    """Silently returning params-only while the caller expects a state history
    crashes downstream; the unsupported combination must fail loudly instead."""
    from types import SimpleNamespace

    from data_assimilation.smoothing.esmda import ParameterESMDA

    obj = ParameterESMDA.__new__(ParameterESMDA)
    obj.forward_model = SimpleNamespace(save_on_disk=True)
    with pytest.raises(ValueError, match="return_state_history"):
        obj._analysis(params=None, observations=None, return_state_history=True)
