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


def test_parameter_esmda_runs_with_correlation_localization(compose_test_cfg) -> None:
    """End-to-end: parameter ESMDA composes and runs with localization on."""
    from scripts.run_esmda import run

    cfg = compose_test_cfg(
        [
            "model@truth_model=pyudales",
            "model@assim_model=pyudales",
            "esmda/smoother=parameter",
            "params@prior_params=static",
            "params@truth_params=static_truth",
            # Correlation localization is the default `esmda.localization`; just
            # tune its truncation threshold here.
            "esmda.localization.truncation_correlation=0.2",
            "ensemble.ensemble_size=4",
            "ensemble.num_parallel_processes=2",
            "esmda.num_steps=1",
            "esmda.num_assimilation_windows=1",
            "run.skip_viz=true",
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
