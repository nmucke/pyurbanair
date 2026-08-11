"""Unit tests for the localized deterministic ensemble transform (LETKF).

The reference throughout is ``_reference_letkf``: a deliberately naive
row-by-row loop that whitens with that row's inflation, calls the shared
:func:`ensemble_transform` kernel and applies it to that one row. It has no
deduplication, no chunking, no inactive-block skipping and no gather/scatter —
exactly the four optimizations :class:`LETKFAnalysis` adds. Every structural
test below therefore compares the optimized implementation against arithmetic
that cannot share a bug with it.

Member identity is checked explicitly rather than implied. The mean and
covariance identities that dominate ETKF testing are blind to a permutation of
the ensemble columns, while ``RTPP`` blends each posterior member against *its
own* prior anomaly, so a scrambled gather would be silently wrong in exactly
the way that matters. The reference loop preserves column order by
construction, which is what gives the comparisons their bite.
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from data_assimilation.filtering import EnsembleKalmanFilter
from data_assimilation.filtering.etkf import (
    _CHUNK_ELEMENT_BUDGET,
    _MAX_CHUNK,
    ETKFAnalysis,
    LETKFAnalysis,
    ObservationTSVD,
    _resolve_chunk_size,
    apply_ensemble_transform,
    ensemble_transform,
    whiten_observations,
)
from data_assimilation.localization.base import BaseLocalization
from data_assimilation.localization.correlation import CorrelationLocalization
from data_assimilation.localization.distance import DistanceLocalization
from data_assimilation.reduction import OnlineStateReduction

from tests.test_filtering import _AllOnesLocalization, _dummy_filter_kwargs
from tests.test_filtering_etkf import ATOL, RTOL, _observation_problem

# "Unchanged" for rows the analysis provably skips is an exact claim here (the
# implementation returns the input values, it does not recompute them), so the
# corresponding assertions use exact equality rather than this tolerance.


class _PrescribedLocalization(BaseLocalization):
    """A localization strategy that returns a caller-supplied inflation matrix.

    Nothing in the package offers this: the shipped strategies derive their
    inflation from coordinates or from ensemble correlations, so a test that
    wants a *specific* pattern of kept/tapered/excluded observations has to
    reach through them. Driving :class:`LETKFAnalysis` directly with the matrix
    isolates the analysis from whichever strategy produced it — which is the
    real contract, since the analysis only ever sees ``inflation_factors``.
    """

    def __init__(
        self,
        inflation: Any,
        block_grouping: bool = False,
        localizes_parameters: bool = True,
    ) -> None:
        self._inflation = jnp.asarray(inflation)
        self.block_grouping = block_grouping
        self.localizes_parameters = localizes_parameters

    def inflation_factors(
        self,
        aug_dev: jnp.ndarray,
        pred_obs_dev: jnp.ndarray,
        row_coords: Optional[jnp.ndarray] = None,
        obs_coords: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return self._inflation


def _reference_letkf(
    augmented: jnp.ndarray,
    pred_obs: jnp.ndarray,
    obs: jnp.ndarray,
    C_D_diag: jnp.ndarray,
    row_inflation: jnp.ndarray,
    tsvd: Optional[ObservationTSVD] = None,
) -> jnp.ndarray:
    """One kernel call per row, in row order and member order.

    The literal definition of a local analysis, at the cost of one dispatch per
    row. Column order is inherited from :func:`apply_ensemble_transform`, so a
    member permutation anywhere in :class:`LETKFAnalysis`'s gather/scatter path
    shows up as a mismatch.
    """
    rows = []
    for index in range(augmented.shape[0]):
        transform = ensemble_transform(
            *whiten_observations(
                pred_obs, obs, C_D_diag, error_inflation=row_inflation[index]
            ),
            tsvd=tsvd,
        )
        rows.append(apply_ensemble_transform(augmented[index][None, :], transform)[0])
    return jnp.stack(rows)


def _analyze(
    analysis: LETKFAnalysis,
    problem: tuple[Any, ...],
    localization: BaseLocalization,
    **kwargs: Any,
) -> jnp.ndarray:
    augmented, _, pred_obs, obs, C_D_diag = problem
    return analysis(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=localization,
        **kwargs,
    )


def _joint_problem(
    seed: int = 0,
    n_state: int = 4,
    n_param: int = 2,
    n_d: int = 3,
    n_e: int = 60,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """A joint-mode ensemble shaped like what ``BaseFilter`` hands the scheme.

    ``[state rows | parameter rows | the N_d appended predicted-observation
    diagnostic rows]``. Parameter row ``i`` is deliberately built from
    predicted observation ``i`` plus noise, so correlation localization keeps a
    *subset* of observations for it — the interesting case, distinct both from
    "no observation applies" (which would leave the row untouched) and from
    "all observations apply" (which would be the global update).
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    state = 3.0 + 0.8 * jax.random.normal(keys[0], (n_state, n_e))
    operator = jax.random.normal(keys[1], (n_d, n_state))
    pred_obs = operator @ state + 0.2 * jax.random.normal(keys[2], (n_d, n_e))
    params = jnp.stack(
        [
            pred_obs[index % n_d] + 0.05 * jax.random.normal(keys[3], (n_e,))
            for index in range(n_param)
        ]
    )
    augmented = jnp.concatenate([state, params, pred_obs], axis=0)
    obs = jnp.mean(pred_obs, axis=1) + 0.5 * jax.random.normal(keys[4], (n_d,))
    C_D_diag = 0.05 * (1.0 + jnp.arange(n_d, dtype=state.dtype))
    return augmented, pred_obs, obs, C_D_diag, jnp.asarray([n_state, n_param, n_d])


# ---------------------------------------------------------------------------
# (a) The localization limits: all-ones is global, infinite is absent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tsvd",
    [None, ObservationTSVD(enabled=True, max_rank=2)],
    ids=["untruncated", "truncated"],
)  # type: ignore[misc]
def test_all_ones_localization_equals_the_global_etkf(
    tsvd: Optional[ObservationTSVD],
) -> None:
    """Every observation at full weight for every row *is* the global ETKF.

    Near-tautological by construction — an all-ones inflation vector collapses
    into a single block whose whitening is the unlocalized whitening — which is
    the point: the globally updated rows (diagnostic rows always, parameter
    rows under distance localization) ride the same code path rather than a
    parallel one that could drift. The truncated variant additionally pins that
    the traced per-block rank criterion agrees with the host-side one.
    """
    problem = _observation_problem(seed=1)
    analysis = LETKFAnalysis(tsvd=tsvd)
    localized = _analyze(analysis, problem, _AllOnesLocalization())
    global_etkf = ETKFAnalysis(tsvd=tsvd)(
        problem[0], problem[2], problem[3], problem[4], jax.random.PRNGKey(0)
    )

    np.testing.assert_allclose(
        np.asarray(localized), np.asarray(global_etkf), rtol=RTOL, atol=ATOL
    )
    diagnostics = analysis.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.num_blocks == 1  # one selection, deduplicated
    assert diagnostics.num_updated_rows == problem[0].shape[0]
    # Not vacuous: the analysis has to actually move the ensemble.
    assert not np.allclose(np.asarray(localized), np.asarray(problem[0]), atol=1e-3)


def test_an_infinitely_inflated_observation_has_no_effect() -> None:
    """``E_inf = inf`` on an observation == that observation not existing.

    The exclusion half of the localization contract, checked end to end through
    the analysis rather than on the whitening helper alone: an excluded
    observation must not leak in through the block partition, the rank
    criterion or the transform.
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(seed=2, n_d=3)
    n_aug = augmented.shape[0]
    excluded = jnp.tile(jnp.array([jnp.inf, 1.0, 1.0]), (n_aug, 1))

    with_exclusion = LETKFAnalysis()(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_PrescribedLocalization(excluded),
    )
    without_the_observation = LETKFAnalysis()(
        augmented,
        pred_obs[1:],
        obs[1:],
        C_D_diag[1:],
        jax.random.PRNGKey(0),
        localization=_PrescribedLocalization(jnp.ones((n_aug, 2))),
    )
    np.testing.assert_allclose(
        np.asarray(with_exclusion),
        np.asarray(without_the_observation),
        rtol=RTOL,
        atol=ATOL,
    )
    # Not vacuous: observation 0 genuinely carries information.
    all_three = LETKFAnalysis()(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_PrescribedLocalization(jnp.ones((n_aug, 3))),
    )
    assert not np.allclose(np.asarray(with_exclusion), np.asarray(all_three), atol=1e-3)


def test_uniform_inflation_equals_scaling_the_observation_variances() -> None:
    """``R_eff = C_D * E_inf**2`` — the taper is a *standard deviation* factor.

    The convention the localized stochastic update already uses
    (``localization/base.py``, where ``E_inf`` multiplies the perturbation
    matrix ``E``), pinned here without reusing the whitening helper on both
    sides: a uniform inflation of ``c`` must be indistinguishable from having
    declared observation-error variances ``c**2`` larger. Getting the exponent
    wrong would rescale every taper in the domain while leaving every
    equality-of-two-code-paths test happy.
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(seed=10, n_d=3)
    factor = 2.0
    inflated = LETKFAnalysis()(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_PrescribedLocalization(jnp.full((augmented.shape[0], 3), factor)),
    )
    rescaled = LETKFAnalysis()(
        augmented,
        pred_obs,
        obs,
        C_D_diag * factor**2,
        jax.random.PRNGKey(0),
        localization=_AllOnesLocalization(),
    )
    np.testing.assert_allclose(
        np.asarray(inflated), np.asarray(rescaled), rtol=RTOL, atol=ATOL
    )
    # Not vacuous: the taper genuinely weakens the update.
    untapered = LETKFAnalysis()(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_AllOnesLocalization(),
    )
    assert float(jnp.mean(jnp.abs(inflated - augmented))) < float(
        jnp.mean(jnp.abs(untapered - augmented))
    )


def test_rows_without_any_active_observation_are_bit_identical() -> None:
    """A row no observation reaches is returned untouched, not recomputed.

    Exact equality, not a tolerance: those rows are skipped entirely, because a
    ``mean + (row - mean)`` round trip is *not* the identity in float32 and a
    localized filter spends most of its rows on this path (at the shipped
    radius roughly 95 percent of the domain). A drifting "unchanged" row would
    accumulate over a 60-cycle run.
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(seed=3, n_aug=6, n_d=2)
    inflation = jnp.asarray(
        [
            [1.0, 1.0],
            [jnp.inf, jnp.inf],
            [1.0, jnp.inf],
            [jnp.inf, jnp.inf],
            [2.0, 1.0],
            [jnp.inf, jnp.inf],
        ]
    )
    analysis = LETKFAnalysis()
    posterior = analysis(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_PrescribedLocalization(inflation),
    )
    inactive = np.asarray([1, 3, 5])
    np.testing.assert_array_equal(
        np.asarray(posterior[inactive]), np.asarray(augmented[inactive])
    )
    active = np.asarray([0, 2, 4])
    assert not np.allclose(
        np.asarray(posterior[active]), np.asarray(augmented[active]), atol=1e-3
    )

    diagnostics = analysis.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.num_updated_rows == 3
    # The three all-excluded rows share ONE deduplicated block, and it is not
    # counted as active.
    assert diagnostics.num_active_blocks == 3
    assert len(set(diagnostics.row_block[inactive].tolist())) == 1


def test_all_rows_excluded_returns_the_ensemble_untouched() -> None:
    """The degenerate localization: no transform is computed at all."""
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(seed=4, n_d=2)
    analysis = LETKFAnalysis()
    posterior = analysis(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_PrescribedLocalization(
            jnp.full((augmented.shape[0], 2), jnp.inf)
        ),
    )
    np.testing.assert_array_equal(np.asarray(posterior), np.asarray(augmented))
    diagnostics = analysis.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.num_active_blocks == 0
    assert diagnostics.num_updated_rows == 0


# ---------------------------------------------------------------------------
# (b) Block grouping, deduplication and chunking are optimizations only
# ---------------------------------------------------------------------------


def test_block_grouping_equals_the_repeated_row_calculation() -> None:
    """Grouping == giving every row of a block the block's shared inflation.

    ``group_ids`` only changes *which* inflation vector a row sees (the
    block-wise ``segment_min``); once that is fixed, a grouped analysis and an
    ungrouped one on the pre-shared vectors must be the same arithmetic. This
    also pins that grouping happens before the analysis, not as a post-hoc
    averaging of per-row results.
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(seed=5, n_aug=6, n_d=3)
    inflation = jnp.asarray(
        [
            [1.0, 3.0, jnp.inf],
            [2.0, jnp.inf, jnp.inf],  # block 0 with the row above
            [jnp.inf, 1.0, 4.0],
            [1.5, 2.0, jnp.inf],  # block 1 with the row above
            [jnp.inf, jnp.inf, 1.0],
            [1.0, 1.0, 1.0],  # block 2 with the row above
        ]
    )
    group_ids = jnp.asarray([0, 0, 1, 1, 2, 2])
    grouped = LETKFAnalysis()(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_PrescribedLocalization(inflation, block_grouping=True),
        group_ids=group_ids,
    )

    # The block minimum, spelled out by hand rather than reusing the helper
    # under test.
    raw = np.asarray(inflation)
    shared = np.repeat(np.minimum(raw[0::2], raw[1::2]), 2, axis=0)
    repeated = LETKFAnalysis()(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_PrescribedLocalization(jnp.asarray(shared)),
    )
    np.testing.assert_allclose(
        np.asarray(grouped), np.asarray(repeated), rtol=RTOL, atol=ATOL
    )
    # Block members receive an identical transform, so identical rows in and
    # identical rows out — and grouping is not a no-op relative to the raw
    # per-row analysis.
    ungrouped = LETKFAnalysis()(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_PrescribedLocalization(inflation),
    )
    assert not np.allclose(np.asarray(grouped), np.asarray(ungrouped), atol=1e-3)


def test_a_global_row_cannot_disable_localization_for_its_block() -> None:
    """Masked rows are excluded from the ``segment_min``, not folded into it.

    The deterministic twin of
    ``tests/test_localization.py::\
test_masked_row_cannot_disable_localization_for_shared_group``: if the mask
    were applied *after* grouping, the all-ones global row would drag its
    block's minimum to one and quietly turn a localized state block into a
    global update.
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(seed=6, n_aug=2, n_d=2)
    inflation = jnp.asarray([[3.0, jnp.inf], [1.0, 1.0]])
    posterior = LETKFAnalysis()(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_PrescribedLocalization(inflation, block_grouping=True),
        group_ids=jnp.asarray([0, 0]),
        localize_mask=jnp.asarray([True, False]),
    )
    # Row 0 keeps its own (3, inf) selection; row 1 takes the global update.
    reference = _reference_letkf(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jnp.asarray([[3.0, jnp.inf], [1.0, 1.0]]),
    )
    np.testing.assert_allclose(
        np.asarray(posterior), np.asarray(reference), rtol=RTOL, atol=ATOL
    )
    assert not np.allclose(np.asarray(posterior[0]), np.asarray(posterior[1]))


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, None])  # type: ignore[misc]
def test_chunked_and_unchunked_calculations_agree(chunk_size: Optional[int]) -> None:
    """Chunking is a memory bound, never a numerical choice.

    Every chunk size must produce the same answer as the row-by-row reference,
    including sizes that do not divide the block count (so the ``lax.map``
    remainder path is exercised) and the default, which fits every block into
    one chunk at this size.
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(seed=7, n_aug=9, n_d=3)
    keys = jax.random.split(jax.random.PRNGKey(70), 1)
    # Nine distinct selections: a mix of full weight, taper and exclusion.
    inflation = 1.0 + 3.0 * jax.random.uniform(keys[0], (9, 3))
    inflation = inflation.at[jnp.asarray([1, 4, 7]), 2].set(jnp.inf)
    inflation = inflation.at[3, :].set(jnp.inf)

    posterior = LETKFAnalysis(block_chunk_size=chunk_size)(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_PrescribedLocalization(inflation),
    )
    reference = _reference_letkf(augmented, pred_obs, obs, C_D_diag, inflation)
    np.testing.assert_allclose(
        np.asarray(posterior), np.asarray(reference), rtol=RTOL, atol=ATOL
    )


def test_deduplication_does_not_change_the_answer() -> None:
    """Rows sharing an inflation vector share a transform — and only that.

    Deduplication is what makes LETKF affordable at ``N_s ~ 230k`` (it collapses
    every observation-free row into one block), so it must be provably
    invisible: the answer has to equal the naive per-row loop while the block
    count drops below the row count.
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(seed=8, n_aug=8, n_d=3)
    pattern = jnp.asarray(
        [
            [1.0, 2.0, jnp.inf],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, jnp.inf],  # duplicate of row 0
            [jnp.inf, jnp.inf, jnp.inf],
            [1.0, 1.0, 1.0],  # duplicate of row 1
            [jnp.inf, jnp.inf, jnp.inf],  # duplicate of row 3
            [3.0, 1.0, 1.0],
            [1.0, 2.0, jnp.inf],  # duplicate of row 0
        ]
    )
    analysis = LETKFAnalysis()
    posterior = analysis(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_PrescribedLocalization(pattern),
    )
    reference = _reference_letkf(augmented, pred_obs, obs, C_D_diag, pattern)
    np.testing.assert_allclose(
        np.asarray(posterior), np.asarray(reference), rtol=RTOL, atol=ATOL
    )
    diagnostics = analysis.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.num_blocks == 4  # {row0, row1, all-excluded, row6}
    assert diagnostics.num_active_blocks == 3
    # Duplicated selections really do land in one block.
    assert len(set(diagnostics.row_block[np.asarray([0, 2, 7])].tolist())) == 1


def test_member_order_survives_the_gather_and_scatter() -> None:
    """Posterior columns stay attached to their forecast members.

    The gather/scatter path (rows -> block index -> transform -> back into the
    full array) is the one place a member permutation could be introduced, and
    the mean/covariance identities cannot see it. ``RTPP`` can: it blends each
    posterior member against its own prior anomaly. The reference loop below
    never reorders columns, and the ensemble is built so that reversing the
    members changes the answer, which is what makes the comparison able to fail.
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(
        seed=9, n_aug=6, n_d=2, n_e=12
    )
    inflation = jnp.asarray(
        [
            [1.0, jnp.inf],
            [1.0, jnp.inf],
            [2.0, 1.0],
            [jnp.inf, 1.0],
            [1.0, 1.0],
            [jnp.inf, jnp.inf],
        ]
    )
    posterior = LETKFAnalysis()(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_PrescribedLocalization(inflation),
    )
    reference = _reference_letkf(augmented, pred_obs, obs, C_D_diag, inflation)
    np.testing.assert_allclose(
        np.asarray(posterior), np.asarray(reference), rtol=RTOL, atol=ATOL
    )
    # Without this the comparison above would also pass for a reversed member
    # order, since a member-symmetric ensemble is its own mirror image.
    assert not np.allclose(
        np.asarray(posterior), np.asarray(posterior)[:, ::-1], atol=1e-3
    )
    # And the increment is member-specific, not a shared shift.
    increment = np.asarray(posterior - augmented)
    assert np.std(increment[4]) > 1e-3


# ---------------------------------------------------------------------------
# (c) The shipped strategies: which rows stay global
# ---------------------------------------------------------------------------


def _distance_setup() -> tuple[Any, ...]:
    """Joint problem with 1-D state coordinates and two well-separated sensors.

    Sensor 0 sits at the origin next to state rows 0-2 and sensor 1 sits far
    away next to state row 3, so with ``localization_radius=3`` every state row
    has a *different* observation selection: full weight, full weight, tapered,
    and the other sensor.
    """
    augmented, pred_obs, obs, C_D_diag, sizes = _joint_problem(seed=20, n_d=2)
    n_state, n_param, n_d = (int(value) for value in sizes)
    state_coords = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 0.0, 0.0]]
    )
    row_coords = jnp.asarray(
        np.concatenate([state_coords, np.zeros((n_param + n_d, 3))], axis=0)
    )
    obs_coords = jnp.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    # Exactly the mask ``BaseFilter._localization_plumbing`` builds: state rows
    # localized, parameter rows only if the strategy supports them, appended
    # diagnostic rows never.
    localize_mask = jnp.concatenate(
        [
            jnp.ones(n_state, dtype=bool),
            jnp.full((n_param,), DistanceLocalization.localizes_parameters, dtype=bool),
            jnp.zeros(n_d, dtype=bool),
        ]
    )
    return (
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        row_coords,
        obs_coords,
        localize_mask,
        n_state,
        n_param,
        n_d,
    )


def test_distance_localization_keeps_parameter_and_diagnostic_rows_global() -> None:
    """Non-spatial rows ride the global ETKF, spatial rows do not.

    Distance localization cannot place a scalar parameter, so
    ``localizes_parameters`` is ``False`` and those rows are masked; the
    appended predicted-observation diagnostic rows are always masked (which is
    why ``obs_posterior_rmse`` is labelled a global ride-along proxy under
    localization). Both must equal the *unlocalized* ETKF applied to the same
    rows — and they must land in one shared block, since an all-ones inflation
    vector is a single selection however many rows carry it.
    """
    (
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        row_coords,
        obs_coords,
        localize_mask,
        n_state,
        n_param,
        n_d,
    ) = _distance_setup()
    assert DistanceLocalization.localizes_parameters is False

    analysis = LETKFAnalysis()
    posterior = analysis(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=DistanceLocalization(localization_radius=3.0),
        localize_mask=localize_mask,
        row_coords=row_coords,
        obs_coords=obs_coords,
    )
    global_etkf = ETKFAnalysis()(
        augmented, pred_obs, obs, C_D_diag, jax.random.PRNGKey(0)
    )

    global_rows = slice(n_state, n_state + n_param + n_d)
    np.testing.assert_allclose(
        np.asarray(posterior[global_rows]),
        np.asarray(global_etkf[global_rows]),
        rtol=RTOL,
        atol=ATOL,
    )
    # The state rows are genuinely localized (row 3 sees only the far sensor).
    assert not np.allclose(
        np.asarray(posterior[:n_state]),
        np.asarray(global_etkf[:n_state]),
        atol=1e-3,
    )
    diagnostics = analysis.last_diagnostics
    assert diagnostics is not None
    global_blocks = set(diagnostics.row_block[n_state:].tolist())
    assert len(global_blocks) == 1
    assert not global_blocks & set(diagnostics.row_block[:n_state].tolist())


def test_correlation_localization_acts_on_state_and_parameter_blocks() -> None:
    """Correlation localization localizes both joint blocks, not just the state.

    ``localizes_parameters`` is ``True`` there because the strategy needs only
    ensemble correlations, so the parameter rows must get a *selection*: not the
    global update (all observations) and not the identity (none of them). The
    parameter rows here are built from one predicted observation each, so the
    expected outcome is a strict subset.
    """
    augmented, pred_obs, obs, C_D_diag, sizes = _joint_problem(seed=21, n_d=3)
    n_state, n_param, n_d = (int(value) for value in sizes)
    assert CorrelationLocalization.localizes_parameters is True
    localize_mask = jnp.concatenate(
        [
            jnp.ones(n_state + n_param, dtype=bool),
            jnp.zeros(n_d, dtype=bool),
        ]
    )

    analysis = LETKFAnalysis()
    posterior = analysis(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=CorrelationLocalization(
            truncation_correlation=0.7, max_inflation=8.0
        ),
        localize_mask=localize_mask,
    )
    global_etkf = ETKFAnalysis()(
        augmented, pred_obs, obs, C_D_diag, jax.random.PRNGKey(0)
    )

    diagnostics = analysis.last_diagnostics
    assert diagnostics is not None
    counts = diagnostics.active_observation_counts[diagnostics.row_block]
    # Both blocks are localized: a strict subset of the observations applies.
    assert np.all(counts[:n_state] < n_d)
    assert np.all(counts[n_state : n_state + n_param] < n_d)
    assert np.all(counts[n_state : n_state + n_param] > 0)
    # ... and the diagnostic rows still see everything.
    assert np.all(counts[n_state + n_param :] == n_d)

    for block in (slice(0, n_state), slice(n_state, n_state + n_param)):
        assert not np.allclose(
            np.asarray(posterior[block]), np.asarray(global_etkf[block]), atol=1e-3
        )
        assert not np.allclose(
            np.asarray(posterior[block]), np.asarray(augmented[block]), atol=1e-3
        )


# ---------------------------------------------------------------------------
# (d) Per-block observation TSVD
# ---------------------------------------------------------------------------


def test_disabled_tsvd_equals_the_full_rank_local_transforms() -> None:
    """Disabled truncation == every thin-SVD direction of each local ``Y_w``.

    The per-block statement of the kernel's contract: with the TSVD off, a
    block retains ``min(N_d, N_e)`` directions, including any that its own
    exclusions made numerically zero. Compared against the untruncated kernel
    per row, and against an ``energy_fraction=1.0`` truncation, which drops only
    those numerically zero directions and must therefore agree to float32.
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(
        seed=30, n_aug=6, n_d=4, n_e=20
    )
    inflation = jnp.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, jnp.inf, 1.0],
            [jnp.inf, 1.0, 1.0, jnp.inf],
            [1.0, jnp.inf, jnp.inf, jnp.inf],
            [2.0, 2.0, 2.0, 2.0],
            [1.5, jnp.inf, 1.0, 3.0],
        ]
    )
    localization = _PrescribedLocalization(inflation)
    analysis = LETKFAnalysis()
    disabled = analysis(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=localization,
    )
    np.testing.assert_allclose(
        np.asarray(disabled),
        np.asarray(_reference_letkf(augmented, pred_obs, obs, C_D_diag, inflation)),
        rtol=RTOL,
        atol=ATOL,
    )
    diagnostics = analysis.last_diagnostics
    assert diagnostics is not None
    assert np.all(np.asarray(diagnostics.retained_ranks) == 4)  # min(N_d, N_e)

    full_energy = LETKFAnalysis(
        tsvd=ObservationTSVD(enabled=True, energy_fraction=1.0)
    )(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=localization,
    )
    np.testing.assert_allclose(
        np.asarray(disabled), np.asarray(full_energy), rtol=RTOL, atol=ATOL
    )


def test_local_tsvd_rank_never_exceeds_the_active_or_ensemble_rank() -> None:
    """Per block: ``retained <= available <= min(N_d_active, N_e - 1)``.

    An excluded observation contributes a zero row to ``Y_w``, so it cannot
    raise the local rank; the whitened anomalies are centred across the
    ensemble, so ``N_e - 1`` caps it from the other side. The check runs on
    every block of a problem that deliberately mixes selection sizes, and with
    ``energy_fraction=1.0`` so the retained rank is the *available* rank rather
    than a smaller scientific cut.
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(
        seed=31, n_aug=5, n_d=6, n_e=5
    )
    inflation = jnp.asarray(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 6 active, but N_e - 1 = 4
            [1.0, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf],  # 1 active
            [1.0, 2.0, jnp.inf, jnp.inf, jnp.inf, jnp.inf],  # 2 active
            [1.0, 1.0, 3.0, jnp.inf, jnp.inf, jnp.inf],  # 3 active
            [jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf],  # none
        ]
    )
    analysis = LETKFAnalysis(tsvd=ObservationTSVD(enabled=True, energy_fraction=1.0))
    analysis(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_PrescribedLocalization(inflation),
    )
    diagnostics = analysis.last_diagnostics
    assert diagnostics is not None
    counts = diagnostics.active_observation_counts
    active_counts = counts[counts > 0]
    retained = np.asarray(diagnostics.retained_ranks)
    available = np.asarray(diagnostics.available_ranks)
    assert retained.shape == active_counts.shape
    assert np.all(available <= np.minimum(active_counts, 5 - 1))
    assert np.all(retained <= available)
    assert np.all(retained >= 1)
    # Non-vacuous: the fully observed block is capped by the ensemble, not by
    # the observation count.
    assert available.max() == 4


def test_numerical_tolerance_caps_the_local_rank_below_the_energy_cut() -> None:
    """The available-rank cap binds independently of the energy criterion.

    ``energy_fraction=1.0`` asks for every direction while a relative floor
    above the second singular value declares all but the first numerically
    zero; the retained rank must be the *smaller* of the two. Without the cap
    the local transform would assimilate directions the same configuration
    discards globally. Both paths now share :func:`_retention_mask`, so this
    additionally pins that the shared criterion honours the tolerance cap on
    the local side as well as the global one.
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(
        seed=33, n_aug=5, n_d=4, n_e=20
    )
    spectrum = jnp.linalg.svd(
        whiten_observations(pred_obs, obs, C_D_diag)[0], compute_uv=False
    )
    floor = float(spectrum[1] / spectrum[0]) * 1.01
    tsvd = ObservationTSVD(enabled=True, energy_fraction=1.0, numerical_tolerance=floor)

    analysis = LETKFAnalysis(tsvd=tsvd)
    localized = _analyze(
        analysis, (augmented, None, pred_obs, obs, C_D_diag), _AllOnesLocalization()
    )
    diagnostics = analysis.last_diagnostics
    assert diagnostics is not None
    assert np.all(np.asarray(diagnostics.retained_ranks) == 1)
    assert np.all(np.asarray(diagnostics.available_ranks) == 1)
    # And the host-side global policy agrees, on the same spectrum.
    np.testing.assert_allclose(
        np.asarray(localized),
        np.asarray(
            ETKFAnalysis(tsvd=tsvd)(
                augmented, pred_obs, obs, C_D_diag, jax.random.PRNGKey(0)
            )
        ),
        rtol=RTOL,
        atol=ATOL,
    )


def test_tsvd_truncation_changes_the_answer_per_block() -> None:
    """A rank cap must actually bite, and only through the observation modes."""
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(
        seed=32, n_aug=6, n_d=4, n_e=20
    )
    inflation = jnp.asarray([[1.0, 1.0, 1.0, 1.0]] * 3 + [[1.0, 2.0, jnp.inf, 1.0]] * 3)
    localization = _PrescribedLocalization(inflation)
    untruncated = LETKFAnalysis()(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=localization,
    )
    analysis = LETKFAnalysis(tsvd=ObservationTSVD(enabled=True, max_rank=1))
    truncated = analysis(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=localization,
    )
    assert not np.allclose(np.asarray(truncated), np.asarray(untruncated), atol=1e-3)
    diagnostics = analysis.last_diagnostics
    assert diagnostics is not None
    assert np.all(np.asarray(diagnostics.retained_ranks) == 1)
    # Truncation removes observation directions; it never touches R, so the
    # posterior spread can only be larger than the untruncated one.
    assert float(jnp.mean(jnp.std(truncated, axis=1))) > float(
        jnp.mean(jnp.std(untruncated, axis=1))
    )


def test_untruncated_blocks_report_full_energy_and_nothing_discarded() -> None:
    """The TSVD-off baseline for the two spectral diagnostics.

    Plan step 5 asks the shared kernel for available rank, retained rank,
    retained energy AND discarded spectrum. The first two existed per block;
    these are the other two. With the truncation disabled every block keeps
    every thin-SVD direction, so the honest report is "all of the energy, none
    of the spectrum discarded" — ``1.0`` and ``0.0``, not ``None``, because a
    transform genuinely ran and genuinely discarded nothing. (``None`` is
    reserved for "no local transform ran at all", which is expressed by these
    arrays being empty.)
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(
        seed=50, n_aug=6, n_d=4, n_e=20
    )
    inflation = jnp.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, jnp.inf, 1.0],
            [jnp.inf, 1.0, 1.0, jnp.inf],
            [1.0, jnp.inf, jnp.inf, jnp.inf],
            [2.0, 2.0, 2.0, 2.0],
            [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
        ]
    )
    analysis = LETKFAnalysis()
    analysis(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_PrescribedLocalization(inflation),
    )
    diagnostics = analysis.last_diagnostics
    assert diagnostics is not None
    # One entry per ACTIVE block: the all-excluded row contributes none.
    assert diagnostics.retained_energies.shape == (diagnostics.num_active_blocks,)
    assert diagnostics.discarded_spectrum_maxima.shape == (
        diagnostics.num_active_blocks,
    )
    np.testing.assert_allclose(diagnostics.retained_energies, 1.0, atol=1e-6)
    np.testing.assert_array_equal(
        diagnostics.discarded_spectrum_maxima,
        np.zeros(diagnostics.num_active_blocks),
    )


def test_truncated_blocks_report_the_energy_and_singular_value_they_dropped() -> None:
    """Both diagnostics against the block's own spectrum, computed by hand.

    ``max_rank=1`` on an all-ones localization gives one block whose spectrum is
    the *global* whitened spectrum, so the expected retained energy
    (``s_0^2 / sum(s^2)``) and the expected largest discarded singular value
    (``s_1``) can be written down from a plain ``svd`` of the whitened anomalies
    — arithmetic that shares no code with the traced reductions under test.
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(
        seed=51, n_aug=5, n_d=4, n_e=20
    )
    spectrum = np.asarray(
        jnp.linalg.svd(
            whiten_observations(pred_obs, obs, C_D_diag)[0], compute_uv=False
        ),
        dtype=np.float64,
    )
    analysis = LETKFAnalysis(tsvd=ObservationTSVD(enabled=True, max_rank=1))
    analysis(
        augmented,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_AllOnesLocalization(),
    )
    diagnostics = analysis.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.num_active_blocks == 1
    assert int(diagnostics.retained_ranks[0]) == 1

    expected_energy = spectrum[0] ** 2 / np.sum(spectrum**2)
    assert float(diagnostics.retained_energies[0]) == pytest.approx(
        expected_energy, rel=1e-5
    )
    assert float(diagnostics.discarded_spectrum_maxima[0]) == pytest.approx(
        spectrum[1], rel=1e-5
    )
    # Not vacuous: a rank-1 cut of a rank-4 spectrum must lose real energy and
    # discard a singular value comparable to the leading one.
    assert 0.0 < expected_energy < 0.99
    assert spectrum[1] > 0.05 * spectrum[0]

    # The global ETKF reports the same two numbers for the same problem: the
    # per-block diagnostics are the local readout of one criterion, not a
    # parallel measurement.
    global_analysis = ETKFAnalysis(tsvd=ObservationTSVD(enabled=True, max_rank=1))
    global_analysis(augmented, pred_obs, obs, C_D_diag, jax.random.PRNGKey(0))
    transform = global_analysis.last_transform
    assert transform is not None and transform.retained_energy is not None
    assert float(diagnostics.retained_energies[0]) == pytest.approx(
        transform.retained_energy, rel=1e-5
    )
    assert float(diagnostics.discarded_spectrum_maxima[0]) == pytest.approx(
        float(jnp.max(transform.discarded_spectrum)), rel=1e-5
    )


def test_a_zero_spread_block_reports_zero_energy_rather_than_nan() -> None:
    """The degenerate spectrum must not put a NaN in ``run_info.yaml``.

    A block whose predicted observations have no ensemble spread has an
    all-zero whitened spectrum, so "what fraction of the energy did we retain"
    is ``0/0``. The transform itself is well defined there (weights ``0`` and
    ``1``, i.e. the identity — nothing is assimilated), so the analysis does
    *not* fail; without an explicit guard the diagnostic alone would carry a
    NaN into the persisted per-cycle record. ``0.0`` is the same answer the
    global :func:`_retained_energy` gives for the same spectrum, keeping the two
    readouts consistent.
    """
    augmented, _, _, _, C_D_diag = _observation_problem(seed=53, n_aug=4, n_d=3)
    n_e = augmented.shape[1]
    collapsed = jnp.tile(jnp.array([[1.0], [2.0], [3.0]]), (1, n_e))
    obs = jnp.array([4.0, 5.0, 6.0])

    analysis = LETKFAnalysis()
    posterior = analysis(
        augmented,
        collapsed,
        obs,
        C_D_diag,
        jax.random.PRNGKey(0),
        localization=_AllOnesLocalization(),
    )
    diagnostics = analysis.last_diagnostics
    assert diagnostics is not None
    assert np.all(np.isfinite(diagnostics.retained_energies))
    np.testing.assert_array_equal(
        diagnostics.retained_energies, np.zeros(diagnostics.num_active_blocks)
    )
    np.testing.assert_array_equal(
        diagnostics.discarded_spectrum_maxima,
        np.zeros(diagnostics.num_active_blocks),
    )
    # No information, so the transform really was the identity.
    np.testing.assert_allclose(np.asarray(posterior), np.asarray(augmented), atol=1e-6)
    # And the global path agrees on the same degenerate spectrum.
    global_analysis = ETKFAnalysis()
    global_analysis(augmented, collapsed, obs, C_D_diag, jax.random.PRNGKey(0))
    assert global_analysis.last_transform is not None
    assert global_analysis.last_transform.retained_energy == 0.0


def test_local_diagnostic_arrays_are_numpy_and_yaml_safe() -> None:
    """Every per-block array leaves as NumPy, never as a JAX scalar source.

    ``write_yaml``'s ``_to_native`` converts NumPy scalars but not JAX ones, so
    a device array reaching ``run_info.yaml`` through
    ``BaseFilter._record_transform_diagnostics`` is a persistence bug that only
    shows up at write time. Pinning the type here keeps that failure impossible
    rather than merely unlikely, for the empty (no active block) case too.
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(seed=52, n_d=2)
    fields = (
        "row_block",
        "active_observation_counts",
        "retained_ranks",
        "available_ranks",
        "retained_energies",
        "discarded_spectrum_maxima",
    )
    for strategy in (
        _AllOnesLocalization(),
        _PrescribedLocalization(jnp.full((augmented.shape[0], 2), jnp.inf)),
    ):
        analysis = LETKFAnalysis()
        analysis(
            augmented,
            pred_obs,
            obs,
            C_D_diag,
            jax.random.PRNGKey(0),
            localization=strategy,
        )
        diagnostics = analysis.last_diagnostics
        assert diagnostics is not None
        for name in fields:
            value = getattr(diagnostics, name)
            assert isinstance(value, np.ndarray), f"{name} is {type(value)}"
        # And the scalars the summary casts are plain Python numbers.
        assert isinstance(float(diagnostics.retained_energies.sum()), float)


# ---------------------------------------------------------------------------
# (e) Policy: LETKF without localization is a configuration error
# ---------------------------------------------------------------------------


def test_letkf_without_localization_raises_at_construction() -> None:
    """No strategy means no localization to do — fail before cycle 0."""
    with pytest.raises(ValueError, match="localization_policy='required'") as excinfo:
        EnsembleKalmanFilter(
            mode="state", analysis=LETKFAnalysis(), **_dummy_filter_kwargs()
        )
    assert "LETKFAnalysis" in str(excinfo.value)


def test_letkf_call_refuses_a_missing_localization_strategy() -> None:
    """Defense in depth for a direct caller bypassing the constructor."""
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(seed=40)
    with pytest.raises(ValueError, match="requires a localization strategy"):
        LETKFAnalysis()(augmented, pred_obs, obs, C_D_diag, jax.random.PRNGKey(0))


def test_letkf_declares_the_required_policy_and_constructs_with_one() -> None:
    assert LETKFAnalysis.localization_policy == "required"
    EnsembleKalmanFilter(
        mode="state",
        analysis=LETKFAnalysis(),
        localization=_AllOnesLocalization(),
        **_dummy_filter_kwargs(),
    )


def test_letkf_with_state_reduction_is_rejected_structurally() -> None:
    """No new guard needed: LETKF requires localization, which bans reduction.

    PR 1's reduction fits one global basis with no spatial support, so it
    cannot be localized. ``BaseFilter`` already refuses ``state_reduction``
    together with any localization strategy; this pins that the two rules
    compose into "LETKF and state reduction never run together" without a
    third, separately maintained check.
    """
    with pytest.raises(ValueError, match="incompatible with localization"):
        EnsembleKalmanFilter(
            mode="state",
            analysis=LETKFAnalysis(),
            localization=_AllOnesLocalization(),
            state_reduction=OnlineStateReduction(whiten=False),
            **_dummy_filter_kwargs(),
        )


def test_invalid_block_chunk_size_raises() -> None:
    with pytest.raises(ValueError, match="block_chunk_size"):
        LETKFAnalysis(block_chunk_size=0)


def test_wrong_length_obs_raises_before_the_traced_block_loop() -> None:
    """A bad ``obs`` gets the ETKF's clear message, not a tracer TypeError.

    Every other shape is validated before the loop; ``obs`` used to reach
    ``_whiten`` inside the traced ``lax.map``, where the mismatch surfaced as a
    broadcasting error from the middle of an unrelated function.
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(seed=41, n_d=3)
    for analysis, kwargs in (
        (LETKFAnalysis(), {"localization": _AllOnesLocalization()}),
        (ETKFAnalysis(), {}),
    ):
        with pytest.raises(ValueError, match="obs must be a 1-D vector"):
            analysis(
                augmented,
                pred_obs,
                obs[:-1],
                C_D_diag,
                jax.random.PRNGKey(0),
                **kwargs,  # type: ignore[arg-type]
            )


def test_non_finite_observations_raise_even_when_no_block_is_active() -> None:
    """Skipping a provably-identity analysis must not skip the failure contract.

    With every observation excluded the analysis returns the ensemble
    untouched, which is correct — but a diverged forecast must still be
    reported. Every other path in the package raises ``FloatingPointError``
    because the non-finite values reach the transform weights; this pins that
    the shortcut does too, instead of silently reporting a successful no-op
    cycle for a NaN ensemble.
    """
    augmented, _, pred_obs, obs, C_D_diag = _observation_problem(
        seed=42, n_aug=4, n_d=2
    )
    poisoned = pred_obs.at[0, 0].set(jnp.nan)

    with pytest.raises(FloatingPointError, match="non-finite"):
        LETKFAnalysis()(
            augmented,
            poisoned,
            obs,
            C_D_diag,
            jax.random.PRNGKey(0),
            localization=_PrescribedLocalization(
                jnp.full((augmented.shape[0], 2), jnp.inf)
            ),
        )
    # Same contract with an active block, which reaches the weights instead.
    with pytest.raises(FloatingPointError, match="non-finite"):
        LETKFAnalysis()(
            augmented,
            poisoned,
            obs,
            C_D_diag,
            jax.random.PRNGKey(0),
            localization=_AllOnesLocalization(),
        )
    # And a finite ensemble with no active block still returns quietly.
    np.testing.assert_array_equal(
        np.asarray(
            LETKFAnalysis()(
                augmented,
                pred_obs,
                obs,
                C_D_diag,
                jax.random.PRNGKey(0),
                localization=_PrescribedLocalization(
                    jnp.full((augmented.shape[0], 2), jnp.inf)
                ),
            )
        ),
        np.asarray(augmented),
    )


# ---------------------------------------------------------------------------
# (f) One rank criterion, two readouts
# ---------------------------------------------------------------------------


def test_the_chunk_budget_counts_the_arrays_that_dominate() -> None:
    """The chunk size must shrink as ``N_d`` grows, because the buffers do.

    ``_resolve_chunk_size`` is a *memory* budget, and the two largest per-item
    arrays in the block loop — the whitened anomalies ``(N_d, N_e)`` and the
    thin SVD's ``left`` ``(N_d, r)`` — are both linear in ``N_d``. An earlier
    accounting used ``N_e * r`` alone, which is independent of ``N_d`` once
    ``N_d >= N_e``: at ``N_d = 400, N_e = 50`` it assumed 41 MB per chunk
    against 655 MB actually allocated, and at ``N_d = 2000``, 3277 MB. The
    assertion that catches exactly that is the last one: the budget must bound
    the arrays the loop really builds.
    """
    n_e = 50
    dimensions = (12, 50, 100, 400, 1000, 2000)
    sizes = [_resolve_chunk_size(n_e, n_d) for n_d in dimensions]

    # The shipped shape (N_d = 12, N_e = 50) is unchanged and still capped by
    # the count bound, so LETKF stays fast where it is actually run.
    assert sizes[0] == _MAX_CHUNK
    # Past the count cap the budget binds, and it binds harder as N_d grows.
    beyond_the_cap = [size for size in sizes if size < _MAX_CHUNK]
    assert len(beyond_the_cap) >= 4
    assert all(a > b for a, b in zip(beyond_the_cap, beyond_the_cap[1:]))

    for n_d, chunk in zip(dimensions, sizes):
        rank = min(n_d, n_e)
        # whitened anomalies + left + right_t + the returned modes.
        elements = chunk * (n_d * n_e + n_d * rank + 2 * n_e * rank)
        assert elements <= _CHUNK_ELEMENT_BUDGET


def test_an_explicit_chunk_size_bypasses_the_budget() -> None:
    """The override is for tests and benchmarks; it is still validated."""
    assert _resolve_chunk_size(50, 2000, override=9999) == 9999
    with pytest.raises(ValueError, match="block_chunk_size"):
        _resolve_chunk_size(50, 12, override=0)


#: ``(N_d, N_e)`` shapes for the rank sweep below, spanning ``N_d`` in [2, 12]
#: and ``N_e`` in [5, 55] — the range the shipped configurations live in. Held
#: to a fixed list rather than drawn at random because ``lax.map`` re-traces per
#: shape: 9 shapes make the 216-problem sweep an 9-second test instead of a
#: two-minute one, and the *spectrum*, not the shape, is what the rank criterion
#: is sensitive to.
_SWEEP_SHAPES = (
    (2, 5),
    (2, 20),
    (3, 55),
    (5, 5),
    (5, 20),
    (8, 55),
    (12, 5),
    (12, 20),
    (12, 55),
)


def _random_transform_problem(
    rng: np.random.Generator, n_d: int, n_e: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """A random ``(augmented, pred_obs, obs, C_D_diag)`` with a random decay.

    The spectral decay is what makes the rank criterion non-trivial: a flat
    spectrum puts the energy cut in the middle of a near-tie, a steep one puts
    it in the round-off tail. Everything is float32, the dtype the filter
    actually runs in and the one in which a cumulative energy sum saturates.
    """
    decay = float(rng.choice([0.0, 0.3, 0.7, 1.5]))

    anomalies = rng.standard_normal((n_d, n_e))
    anomalies -= anomalies.mean(axis=1, keepdims=True)
    pred_obs = np.exp(-decay * np.arange(n_d))[:, None] * anomalies
    augmented = 3.0 + 0.7 * rng.standard_normal((4, n_e))
    obs = pred_obs.mean(axis=1) + 0.3 * rng.standard_normal(n_d)
    C_D_diag = 0.05 * (1.0 + np.arange(n_d))
    return (
        jnp.asarray(augmented, dtype=jnp.float32),
        jnp.asarray(pred_obs, dtype=jnp.float32),
        jnp.asarray(obs, dtype=jnp.float32),
        jnp.asarray(C_D_diag, dtype=jnp.float32),
    )


@pytest.mark.parametrize("energy_fraction", [0.9, 0.99, 0.999, 1.0])  # type: ignore[misc]
def test_global_and_local_rank_policies_agree_on_a_random_sweep(
    energy_fraction: float,
) -> None:
    """One criterion, two readouts — over 54 random problems per fraction.

    ``test_all_ones_localization_equals_the_global_etkf`` makes this claim on a
    single problem; the rank criterion is a *threshold*, so a single problem is
    exactly the wrong shape of evidence. This sweep is the regression for a
    measured defect: the ETKF used to select its rank from a float64
    ``np.cumsum`` while each LETKF block used a dtype-native ``jnp.cumsum``, and
    at ``energy_fraction = 1.0`` a float32 cumulative sum saturates at one after
    ~7 significant digits and stops counting. Over 600 random problems the two
    disagreed on 36 percent of them, by up to 4 modes, moving the posterior by
    1.07e-4 — five times ``ATOL``. Every fraction below 1.0 passed, which is why
    ``1.0`` is in this list and is the point of it.

    The fix is structural (both paths call ``_retention_mask``), so this test
    fails for any reintroduced second implementation, not just that one.

    **Why the ranks are compared to within one mode rather than exactly.** The
    two paths call the same criterion but not the same LAPACK kernel: the global
    ETKF factorizes one ``(N_d, N_e)`` matrix while the block loop factorizes a
    batch under ``lax.map``. Their singular values agree to round-off, which is
    exactly the scale at which the ``sigma > sigma_max * eps * 64`` floor lives,
    so a spectrum whose smallest mode sits near that floor can legitimately land
    on either side depending on the BLAS (macOS/Accelerate here, Linux/OpenBLAS
    in CI). A one-mode window at the floor costs the test nothing: a mode that
    close to round-off is damped to nothing in the transform, which the
    posterior comparison below confirms. What it must not lose is the bite — a
    genuine policy fork moved 36 percent of problems by up to 4 modes — so the
    per-problem window is paired with a budget on how many problems may differ
    at all.
    """
    rng = np.random.default_rng(20260811 + int(energy_fraction * 1000))
    tsvd = ObservationTSVD(enabled=True, energy_fraction=energy_fraction)
    localization = _AllOnesLocalization()
    truncating_problems = 0
    disagreements = 0
    num_problems = 0

    for n_d, n_e in _SWEEP_SHAPES:
        for _ in range(6):
            augmented, pred_obs, obs, C_D_diag = _random_transform_problem(
                rng, n_d, n_e
            )
            global_analysis = ETKFAnalysis(tsvd=tsvd)
            global_posterior = global_analysis(
                augmented, pred_obs, obs, C_D_diag, jax.random.PRNGKey(0)
            )
            local_analysis = LETKFAnalysis(tsvd=tsvd)
            local_posterior = local_analysis(
                augmented,
                pred_obs,
                obs,
                C_D_diag,
                jax.random.PRNGKey(0),
                localization=localization,
            )

            transform = global_analysis.last_transform
            diagnostics = local_analysis.last_diagnostics
            assert transform is not None and diagnostics is not None
            assert diagnostics.num_active_blocks == 1
            local_rank = int(np.asarray(diagnostics.retained_ranks)[0])
            local_available = int(np.asarray(diagnostics.available_ranks)[0])

            assert transform.available_rank is not None
            retained_gap = abs(transform.retained_rank - local_rank)
            available_gap = abs(transform.available_rank - local_available)
            assert max(retained_gap, available_gap) <= 1, (
                f"rank disagreement at N_d={n_d}, N_e={n_e}, "
                f"energy_fraction={energy_fraction}: global "
                f"{transform.retained_rank}/{transform.available_rank} vs local "
                f"{local_rank}/{local_available}"
            )
            num_problems += 1
            disagreements += int(max(retained_gap, available_gap) > 0)
            np.testing.assert_allclose(
                np.asarray(local_posterior),
                np.asarray(global_posterior),
                rtol=RTOL,
                atol=ATOL,
            )
            if local_rank < min(n_d, n_e):
                truncating_problems += 1

    # Non-vacuous: the criterion has to actually cut somewhere in the sweep,
    # otherwise "both paths retain everything" would satisfy every assertion
    # above. At energy_fraction=1.0 the cut is the round-off floor, which the
    # centred anomalies' null direction always trips.
    assert truncating_problems > 0
    # The bite the one-mode window would otherwise give away: round-off ties at
    # the floor are rare, a forked criterion is not. The measured fork moved 36
    # percent of problems; 10 percent is comfortably above the tie rate (0 on
    # this platform) and far below that.
    assert disagreements <= num_problems // 10, (
        f"{disagreements} of {num_problems} problems disagree on the rank at "
        f"energy_fraction={energy_fraction}: that is a forked criterion, not "
        "round-off at the numerical floor"
    )
