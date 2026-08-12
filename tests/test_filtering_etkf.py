"""Unit tests for the deterministic ensemble transform (ETKF) and its TSVD.

The load-bearing test here is the exact linear Kalman comparison: the ETKF is
*exact* for a linear observation operator given the ensemble's own SAMPLE
forecast mean and covariance, so the comparison is against the Kalman update
built from ``np.mean``/``np.cov`` of the forecast ensemble — not against a
population covariance, and not to a sampling-error tolerance. That makes it a
float32-precision equality on a 24-member ensemble instead of a statistical
check that a subtly wrong transform would also pass.

The rest of the file pins the structural contracts the surrounding machinery
depends on: pure right-multiplication (PR 1's reduction diagnostic calls the
analysis twice and subtracts), determinism/key-independence, the symmetric
square root (RTPP blends member-wise against the prior), structural mean
preservation under truncation, and that the TSVD only ever removes whitened
*observation* directions.
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from data_assimilation.filtering import EnsembleKalmanFilter, RandomWalkEvolution
from data_assimilation.filtering.analysis import AnalysisScheme, StochasticEnKFAnalysis
from data_assimilation.filtering.etkf import (
    ETKFAnalysis,
    LETKFAnalysis,
    ObservationTSVD,
    _retention_mask,
    apply_ensemble_transform,
    ensemble_transform,
    whiten_observations,
)
from data_assimilation.inflation import RTPP, RTPS
from data_assimilation.reduction import OnlineStateReduction, _select_rank

from tests.test_filtering import (
    _AllOnesLocalization,
    _dummy_filter_kwargs,
    _initial_state,
    _ParamOnlyModel,
    _params_dataset,
    _ToyLinearModel,
    _ToyObsOp,
)

# Realistic float32 JAX tolerances (matching the conventions already used in
# tests/test_filtering.py for reduced-vs-full equality); the ETKF identities
# below are exact in exact arithmetic, so anything looser would hide a bug and
# anything tighter would be platform noise (macOS/Accelerate vs Linux/OpenBLAS).
RTOL = 1e-5
ATOL = 2e-5


def _observation_problem(
    seed: int = 0,
    n_aug: int = 8,
    n_d: int = 3,
    n_e: int = 24,
) -> tuple[jnp.ndarray, np.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """A linear-Gaussian problem with ``pred_obs = H @ augmented`` exactly.

    The exact-Kalman comparison needs the sample cross-covariance to *be*
    ``P_f H^T``, which holds only when the predicted observations are the
    linear operator applied to the very same ensemble. The mean is deliberately
    much larger than the anomalies, which is what physical state rows look like
    and what makes the explicit centering in ``apply_ensemble_transform``
    matter.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    mean = 5.0 * jax.random.normal(keys[0], (n_aug, 1))
    augmented = mean + 0.7 * jax.random.normal(keys[1], (n_aug, n_e))
    H = np.asarray(jax.random.normal(keys[2], (n_d, n_aug)))
    pred_obs = jnp.asarray(H) @ augmented
    C_D_diag = 0.05 * (1.0 + jnp.arange(n_d, dtype=augmented.dtype))
    obs = (jnp.asarray(H) @ mean).ravel() + 0.3 * jax.random.normal(keys[3], (n_d,))
    return augmented, H, pred_obs, obs, C_D_diag


def _analyze(
    analysis: AnalysisScheme,
    problem: tuple[Any, ...],
    rows: Optional[jnp.ndarray] = None,
    key_seed: int = 0,
) -> jnp.ndarray:
    """Run one analysis on ``rows`` (default: the problem's augmented array)."""
    augmented, _, pred_obs, obs, C_D_diag = problem
    return analysis(
        augmented if rows is None else rows,
        pred_obs,
        obs,
        C_D_diag,
        jax.random.PRNGKey(key_seed),
    )


# ---------------------------------------------------------------------------
# (a) The exact linear Kalman filter, from the SAMPLED forecast moments
# ---------------------------------------------------------------------------


def test_matches_exact_kalman_update_from_sampled_moments() -> None:
    """ETKF posterior mean AND covariance equal the exact KF update.

    The reference is built in float64 from the ensemble's own sample moments:
    ``P_f = cov(augmented)``, ``K = P_f H^T (H P_f H^T + R)^{-1}``,
    ``m_a = m_f + K d``, ``P_a = (I - K H) P_f``. Because the ETKF transform is
    the symmetric square root of ``[(N_e-1) I + Y^T R^{-1} Y]^{-1}``, both
    identities hold exactly (Woodbury / push-through), so this is an arithmetic
    equality rather than an O(1/sqrt(N_e)) statistical one.
    """
    problem = _observation_problem(seed=1)
    augmented, H, _, obs, C_D_diag = problem
    posterior = _analyze(ETKFAnalysis(), problem)

    forecast = np.asarray(augmented, dtype=np.float64)
    m_f = forecast.mean(axis=1)
    P_f = np.cov(forecast, ddof=1)
    R = np.diag(np.asarray(C_D_diag, dtype=np.float64))
    S = H @ P_f @ H.T + R
    K = P_f @ H.T @ np.linalg.inv(S)
    m_a = m_f + K @ (np.asarray(obs, dtype=np.float64) - H @ m_f)
    P_a = (np.eye(P_f.shape[0]) - K @ H) @ P_f

    analyzed = np.asarray(posterior, dtype=np.float64)
    np.testing.assert_allclose(analyzed.mean(axis=1), m_a, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(np.cov(analyzed, ddof=1), P_a, rtol=RTOL, atol=ATOL)

    # Guard against a vacuous comparison: the update must actually move the
    # ensemble and must actually contract the covariance.
    assert not np.allclose(analyzed.mean(axis=1), m_f, atol=1e-3)
    assert np.trace(P_a) < 0.95 * np.trace(P_f)


def test_transform_is_deterministic_and_ignores_the_rng_key() -> None:
    """Same input -> bit-identical output, for any key.

    Key-independence is a contract, not an accident: ``BaseFilter`` splits a key
    every cycle regardless of scheme, and PR 1's reduction diagnostic relies on
    two calls with the same key producing the same weights. Testing two
    *different* keys pins "accepted and ignored" rather than merely "consumed
    reproducibly".
    """
    problem = _observation_problem(seed=2)
    analysis = ETKFAnalysis()
    first = _analyze(analysis, problem, key_seed=0)
    again = _analyze(analysis, problem, key_seed=0)
    other_key = _analyze(analysis, problem, key_seed=99)

    np.testing.assert_array_equal(np.asarray(first), np.asarray(again))
    np.testing.assert_array_equal(np.asarray(first), np.asarray(other_key))


# ---------------------------------------------------------------------------
# (b) Structural contracts the filter machinery depends on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tsvd", [None, ObservationTSVD(enabled=True, max_rank=1)])  # type: ignore[misc]
def test_increment_is_a_pure_row_blockwise_right_multiplication(
    tsvd: Optional[ObservationTSVD],
) -> None:
    """A row subset's increment is unchanged by the rows around it.

    This is exactly what ``BaseFilter._record_reduction_diagnostics`` assumes
    when it re-runs the analysis on the reduction's tiny ``(k, N_e)``
    coordinate array and subtracts: the weights must come only from
    ``(pred_obs, obs, C_D_diag)``. A transform that peeked at the analyzed rows
    (their count, their spread, a row-shaped random draw) would break here.
    """
    problem = _observation_problem(seed=3)
    augmented = problem[0]
    analysis = ETKFAnalysis(tsvd=tsvd)

    full_increment = _analyze(analysis, problem) - augmented
    subset = augmented[2:5]
    subset_increment = _analyze(analysis, problem, rows=subset) - subset

    np.testing.assert_allclose(
        np.asarray(subset_increment),
        np.asarray(full_increment[2:5]),
        rtol=RTOL,
        atol=ATOL,
    )

    # The same property on an already-centred array with a different row count,
    # which is the literal shape of the reduction diagnostic's second call.
    centred = augmented - jnp.mean(augmented, axis=1, keepdims=True)
    np.testing.assert_allclose(
        np.asarray(_analyze(analysis, problem, rows=centred) - centred),
        np.asarray(full_increment),
        rtol=RTOL,
        atol=ATOL,
    )


def test_anomaly_transform_is_the_symmetric_mean_preserving_root() -> None:
    """``W_a`` is symmetric positive definite with ``W_a @ 1 = 1``.

    Symmetry is what keeps member identity intact for RTPP (which blends each
    posterior member with *its own* prior anomaly); ``W_a 1 = 1`` follows from
    ``Y_w 1 = 0`` and is what makes the posterior mean exactly
    ``mean + X wbar``. A Cholesky or randomly rotated root would satisfy the
    covariance identity above and fail both assertions here.
    """
    problem = _observation_problem(seed=4)
    _, _, pred_obs, obs, C_D_diag = problem
    transform = ensemble_transform(*whiten_observations(pred_obs, obs, C_D_diag))
    W_a = np.asarray(transform.anomaly_weights(), dtype=np.float64)

    np.testing.assert_allclose(W_a, W_a.T, rtol=RTOL, atol=ATOL)
    assert np.min(np.linalg.eigvalsh(W_a)) > 0.0
    ones = np.ones(W_a.shape[0])
    np.testing.assert_allclose(W_a @ ones, ones, rtol=RTOL, atol=ATOL)


def test_factored_application_matches_the_assembled_weight_matrix() -> None:
    """``apply_ensemble_transform`` really applies ``wbar 1^T + W_a``.

    The transform is stored factored (``V``, per-mode scales) so LETKF never
    has to materialize an ``N_e x N_e`` matrix per block; this pins that the
    factored application is the documented matrix and nothing else. In
    particular it catches a member-order scramble in the application path,
    which the mean/covariance identities above are blind to but which RTPP and
    every member-wise diagnostic are not.
    """
    problem = _observation_problem(seed=13)
    _, _, pred_obs, obs, C_D_diag = problem
    transform = ensemble_transform(*whiten_observations(pred_obs, obs, C_D_diag))

    rows = jax.random.normal(jax.random.PRNGKey(14), (5, transform.num_members))
    mean = jnp.mean(rows, axis=1, keepdims=True)
    expected = mean + (rows - mean) @ (
        transform.mean_weights[:, None] + transform.anomaly_weights()
    )
    np.testing.assert_allclose(
        np.asarray(apply_ensemble_transform(rows, transform)),
        np.asarray(expected),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "tsvd",
    [None, ObservationTSVD(enabled=True, max_rank=1)],
    ids=["untruncated", "truncated"],
)  # type: ignore[misc]
def test_zero_innovation_leaves_the_mean_exactly_unchanged(
    tsvd: Optional[ObservationTSVD],
) -> None:
    """With ``obs == mean(pred_obs)``, ``wbar = 0`` and the mean cannot move.

    Mean preservation is structural (``W_a 1 = 1``), so it survives truncation
    — the sharp version of "the transform is mean-preserving".
    """
    augmented, H, pred_obs, _, C_D_diag = _observation_problem(seed=5)
    obs = jnp.mean(pred_obs, axis=1)
    problem = (augmented, H, pred_obs, obs, C_D_diag)
    posterior = _analyze(ETKFAnalysis(tsvd=tsvd), problem)

    np.testing.assert_allclose(
        np.asarray(jnp.mean(posterior, axis=1)),
        np.asarray(jnp.mean(augmented, axis=1)),
        rtol=RTOL,
        atol=ATOL,
    )
    # Still a real analysis: the spread contracts even though the mean does not.
    assert float(jnp.mean(jnp.std(posterior, axis=1))) < 0.95 * float(
        jnp.mean(jnp.std(augmented, axis=1))
    )


def test_zero_predicted_observation_spread_leaves_the_ensemble_unchanged() -> None:
    """No observable spread -> no information -> the identity transform.

    Also the numerical claim behind "disabled TSVD keeps every direction": with
    ``s = 0`` the retained weights are exactly ``0`` and ``1``, so retaining
    round-off directions cannot perturb the ensemble.
    """
    augmented, H, _, _, C_D_diag = _observation_problem(seed=6)
    pred_obs = jnp.tile(jnp.array([[1.0], [2.0], [3.0]]), (1, augmented.shape[1]))
    problem = (augmented, H, pred_obs, jnp.array([4.0, 5.0, 6.0]), C_D_diag)

    posterior = _analyze(ETKFAnalysis(), problem)
    np.testing.assert_allclose(np.asarray(posterior), np.asarray(augmented), atol=1e-6)


# ---------------------------------------------------------------------------
# (c) Observation-space TSVD
# ---------------------------------------------------------------------------


def test_disabled_tsvd_reproduces_the_untruncated_transform() -> None:
    """Disabled == retain everything, including the round-off null direction.

    ``N_d > N_e - 1`` here, so the thin SVD carries at least one numerically
    zero singular value that the ``energy_fraction=1.0`` cut drops and the
    disabled path keeps. Agreement to float32 tolerance is the claim that
    keeping it is harmless.
    """
    problem = _observation_problem(seed=7, n_aug=5, n_d=8, n_e=6)
    disabled = _analyze(ETKFAnalysis(), problem)
    full_energy = _analyze(
        ETKFAnalysis(tsvd=ObservationTSVD(enabled=True, energy_fraction=1.0)), problem
    )
    np.testing.assert_allclose(
        np.asarray(disabled), np.asarray(full_energy), rtol=RTOL, atol=ATOL
    )

    _, _, pred_obs, obs, C_D_diag = problem
    whitened = whiten_observations(pred_obs, obs, C_D_diag)
    assert ensemble_transform(*whitened).retained_rank == 6
    assert ensemble_transform(*whitened).available_rank == 5


def test_truncation_only_removes_whitened_observation_directions() -> None:
    """Truncating rank ``r`` == running untruncated on a rank-``r`` ``Y_w``.

    Reconstructing the predicted observations from the truncated whitened
    anomalies and analyzing them with the SAME ``C_D`` reproduces the truncated
    analysis exactly. That pins both halves of the plan's claim: TSVD removes
    weak linear combinations of the observation anomalies, and it does *not*
    regularize by touching the physical observation-error variances (an
    implementation that inflated ``R`` instead would not match).
    """
    problem = _observation_problem(seed=8, n_d=4, n_e=20)
    augmented, H, pred_obs, obs, C_D_diag = problem
    rank = 2

    truncated = _analyze(
        ETKFAnalysis(tsvd=ObservationTSVD(enabled=True, max_rank=rank)), problem
    )

    whitened_anomalies, _ = whiten_observations(pred_obs, obs, C_D_diag)
    left, spectrum, right_t = jnp.linalg.svd(whitened_anomalies, full_matrices=False)
    truncated_whitened = (left[:, :rank] * spectrum[:rank]) @ right_t[
        :rank
    ]  # rank-r Y_w
    rebuilt_pred_obs = (
        jnp.mean(pred_obs, axis=1, keepdims=True)
        + jnp.sqrt(C_D_diag)[:, None] * truncated_whitened
    )
    reference_problem = (augmented, H, rebuilt_pred_obs, obs, C_D_diag)
    reference = _analyze(ETKFAnalysis(), reference_problem)

    np.testing.assert_allclose(
        np.asarray(truncated), np.asarray(reference), rtol=1e-4, atol=1e-4
    )
    # Not vacuous: at rank 2 of 4 the truncation must change the answer.
    assert not np.allclose(
        np.asarray(truncated), np.asarray(_analyze(ETKFAnalysis(), problem)), atol=1e-3
    )


def test_retained_rank_never_exceeds_the_active_or_ensemble_rank() -> None:
    """``available_rank <= min(N_d, N_e - 1)`` and the retained rank <= that."""
    for n_d, n_e in ((3, 24), (8, 6), (12, 12)):
        _, _, pred_obs, obs, C_D_diag = _observation_problem(
            seed=9, n_aug=5, n_d=n_d, n_e=n_e
        )
        whitened = whiten_observations(pred_obs, obs, C_D_diag)
        transform = ensemble_transform(
            *whitened, tsvd=ObservationTSVD(enabled=True, energy_fraction=1.0)
        )
        assert transform.available_rank is not None
        assert transform.available_rank <= min(n_d, n_e - 1)
        assert transform.retained_rank <= transform.available_rank
        assert transform.discarded_spectrum.shape[0] == (
            min(n_d, n_e) - transform.retained_rank
        )


def test_energy_fraction_and_numerical_tolerance_cut_the_spectrum() -> None:
    """Both knobs cut, independently, and report consistent diagnostics."""
    _, _, pred_obs, obs, C_D_diag = _observation_problem(seed=10, n_d=4, n_e=20)
    whitened = whiten_observations(pred_obs, obs, C_D_diag)

    full = ensemble_transform(*whitened)
    assert full.retained_rank == 4

    energy_cut = ensemble_transform(
        *whitened, tsvd=ObservationTSVD(enabled=True, energy_fraction=0.5)
    )
    assert 0 < energy_cut.retained_rank < 4
    assert energy_cut.retained_energy is not None
    assert energy_cut.retained_energy >= 0.5

    # The numerical floor is independent of the scientific truncation: a
    # relative floor above the second singular value keeps exactly one mode
    # even though energy_fraction alone would keep more.
    spectrum = jnp.linalg.svd(whitened[0], compute_uv=False)
    floor = float(spectrum[1] / spectrum[0]) * 1.01
    tolerance_cut = ensemble_transform(
        *whitened,
        tsvd=ObservationTSVD(
            enabled=True, energy_fraction=1.0, numerical_tolerance=floor
        ),
    )
    assert tolerance_cut.retained_rank == 1
    assert tolerance_cut.available_rank == 1


def _prescribed_spectrum(
    singular_values: np.ndarray, n_e: int, seed: int = 0, n_d: Optional[int] = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Whitened anomalies/innovation whose spectrum is exactly ``singular_values``.

    Built from two random orthonormal factors, so the rank criterion can be
    checked against an arithmetic expectation instead of against whatever a
    random matrix happens to produce. Returned in float32 — the dtype the
    filter runs in, and the whole point of the tests below.

    ``n_d`` (default: one row per singular value) pads the observation
    dimension with exactly-zero directions, which is how a *shape* — the other
    input to the rank policy — can be varied without changing the spectrum.
    """
    rng = np.random.default_rng(seed)
    rank = singular_values.size
    n_d = rank if n_d is None else n_d
    left, _ = np.linalg.qr(rng.standard_normal((n_d, rank)))
    right, _ = np.linalg.qr(rng.standard_normal((n_e, rank)))
    anomalies = (left * singular_values) @ right.T
    innovation = rng.standard_normal(n_d)
    return (
        jnp.asarray(anomalies, dtype=jnp.float32),
        jnp.asarray(innovation, dtype=jnp.float32),
    )


@pytest.mark.parametrize(
    "energy_fraction, expected_rank",
    [(0.6, 1), (0.9, 2), (0.95, 3), (0.99, 3), (0.999, 4), (1.0, 4)],
)  # type: ignore[misc]
def test_the_energy_criterion_keeps_the_smallest_sufficient_prefix(
    energy_fraction: float, expected_rank: int
) -> None:
    """The rank policy itself, on a spectrum with an arithmetic answer.

    ``s = (1, 0.6, 0.3, 0.1)`` has cumulative energy fractions
    ``0.6849, 0.9315, 0.9932, 1.0``, so "the smallest prefix holding
    ``energy_fraction`` of the spectrum" is a table, not an opinion. The
    ETKF/LETKF share one implementation of this criterion
    (``_retention_mask``), so a sweep comparing the two paths cannot see a
    change to the criterion itself — this is the test that can. The fractions
    straddle each prefix (``0.99`` falls just below the third, ``0.999`` just
    above it); the *exact* boundary, where a prefix holds precisely the
    requested share, needs a tied spectrum and is pinned by
    ``test_the_energy_criterion_breaks_an_exact_tie_toward_the_shorter_prefix``.
    """
    spectrum = np.array([1.0, 0.6, 0.3, 0.1])
    whitened = _prescribed_spectrum(spectrum, n_e=10, seed=1)
    transform = ensemble_transform(
        *whitened,
        tsvd=ObservationTSVD(enabled=True, energy_fraction=energy_fraction),
    )
    assert transform.retained_rank == expected_rank
    assert transform.available_rank == 4  # nothing here is numerically zero
    energy = spectrum**2
    assert transform.retained_energy == pytest.approx(
        energy[:expected_rank].sum() / energy.sum(), abs=1e-6
    )


@pytest.mark.parametrize(
    "spectrum, energy_fraction, expected_rank",
    [
        ((1.0, 1.0, 1.0, 1.0), 0.25, 1),
        ((1.0, 1.0, 1.0, 1.0), 0.5, 2),
        ((1.0, 1.0, 1.0, 1.0), 0.75, 3),
        ((1.0, 1.0, 1.0, 1.0), 1.0, 4),
        ((2.0, 1.0, 1.0), 2.0 / 3.0, 1),
    ],
    ids=["flat-0.25", "flat-0.5", "flat-0.75", "flat-1.0", "stepped-2/3"],
)  # type: ignore[misc]
def test_the_energy_criterion_breaks_an_exact_tie_toward_the_shorter_prefix(
    spectrum: tuple[float, ...], energy_fraction: float, expected_rank: int
) -> None:
    """The ``>`` in the suffix comparison, on spectra that land exactly on it.

    A flat spectrum is not a corner case: it is what a well-conditioned sensor
    network produces, and it is where a prefix holds *precisely* the requested
    share of the energy. ``(1, 1, 1, 1)`` at ``0.5`` must keep 2 modes, not 3 —
    the prefix that leaves exactly the tolerated tail is already sufficient.

    Pinned on ``_retention_mask`` directly rather than through
    ``ensemble_transform`` because an exact tie cannot survive an SVD: a
    float32 factorization of a flat matrix returns ``1 +- 1e-7``, which puts the
    comparison on the wrong side of the tie at random. The spectrum, not the
    matrix, is what this criterion consumes.

    The second half is the claim the docstrings make: this is the *same*
    criterion as the state reduction's host-side ``_select_rank`` (float64,
    prefix cumulative sum, ``searchsorted``). Ties are exactly where two
    implementations of "smallest sufficient prefix" drift apart, so agreeing
    here is what makes "one policy, two implementations" checkable.
    """
    values = jnp.asarray(spectrum, dtype=jnp.float32)
    shape = (len(spectrum), 10)
    tsvd = ObservationTSVD(enabled=True, energy_fraction=energy_fraction)

    retain, rank, available = _retention_mask(values, shape, tsvd)
    assert int(rank) == expected_rank
    assert int(available) == len(spectrum)
    np.testing.assert_array_equal(
        np.asarray(retain), np.arange(len(spectrum)) < expected_rank
    )

    reduction_rank, _, _ = _select_rank(
        values, shape, energy_fraction, None, conservative_rank=False
    )
    assert reduction_rank == expected_rank


@pytest.mark.parametrize(
    "spectrum, n_d, n_e, expected_rank",
    [
        ((1.0, 1e-4), 2000, 50, 2),
        ((1.0, 1e-6), 2, 2, 1),
    ],
    ids=["tall-matrix", "tiny-matrix"],
)  # type: ignore[misc]
def test_the_round_off_floor_is_the_non_conservative_one(
    spectrum: tuple[float, ...], n_d: int, n_e: int, expected_rank: int
) -> None:
    """``max(min(shape), 64) * eps``, both halves of it.

    The module docstring argues at length that the *conservative* cut
    (``numpy.linalg.matrix_rank``'s ``max(shape) * eps``, right for ESMDA's
    whitened encode, which divides by sigma) would discard genuine observation
    directions here, where nothing divides by a singular value. Every other
    shape in this suite has ``max(N_d, N_e) < 64``, so the two conventions
    coincide and the argument goes untested. These two shapes separate them:

    * ``(2000, 50)`` — a dense sensor network. The floors are ``64 * eps``
      (7.6e-6) against ``2000 * eps`` (2.4e-4), 31x apart, and a mode at
      ``1e-4 * sigma_max`` is real information the conservative cut would drop;
    * ``(2, 2)`` — the ``_RANK_TOLERANCE_FLOOR`` half. Taking ``min(shape)``
      alone would put the floor at ``2 * eps`` (2.4e-7) and admit a mode at
      ``1e-6 * sigma_max`` that is round-off, not information.
    """
    whitened = _prescribed_spectrum(np.array(spectrum), n_e=n_e, seed=5, n_d=n_d)
    transform = ensemble_transform(
        *whitened, tsvd=ObservationTSVD(enabled=True, energy_fraction=1.0)
    )
    assert transform.available_rank == expected_rank
    assert transform.retained_rank == expected_rank


def test_a_numerical_tolerance_applies_with_the_scientific_cut_disabled() -> None:
    """``enabled=False`` disables the *scientific* cut, not the floor.

    ``numerical_tolerance`` says what "numerically zero" means; it is not a
    scientific choice, so it applies either way (plan §6 step 5 calls it
    "independent of the scientific truncation"). Previously it was validated on
    construction and then never consulted when disabled, so
    ``ObservationTSVD(enabled=False, numerical_tolerance=0.9)`` silently
    retained everything.

    The default — disabled with no explicit floor — is unchanged and still
    retains *every* thin-SVD direction, round-off ones included, which is what
    makes the disabled path the untruncated ETKF bit for bit.
    """
    spectrum = np.array([1.0, 0.5, 1e-3, 1e-4])
    whitened = _prescribed_spectrum(spectrum, n_e=10, seed=6)

    everything = ensemble_transform(*whitened, tsvd=ObservationTSVD(enabled=False))
    assert everything.retained_rank == 4

    floored = ensemble_transform(
        *whitened, tsvd=ObservationTSVD(enabled=False, numerical_tolerance=1e-2)
    )
    assert floored.retained_rank == 2
    assert floored.available_rank == 2
    # And it really truncated: the two disagree on the rows they analyze.
    rows = _observation_problem(seed=12, n_aug=6, n_e=10)[0]
    assert not np.allclose(
        np.asarray(apply_ensemble_transform(rows, everything)),
        np.asarray(apply_ensemble_transform(rows, floored)),
        rtol=RTOL,
        atol=ATOL,
    )


def test_a_max_rank_that_cannot_take_effect_is_rejected() -> None:
    """The other half of finding "no silently-ignored knob".

    ``max_rank`` *is* part of the scientific truncation, so unlike
    ``numerical_tolerance`` it cannot quietly start applying when the
    truncation is off — a block spelled ``enabled: false`` must not truncate.
    Ignoring it would be the silent no-op this rejects instead.
    """
    with pytest.raises(ValueError, match="max_rank"):
        ObservationTSVD(enabled=False, max_rank=2)
    # Still fine in both consistent spellings.
    assert ObservationTSVD(enabled=True, max_rank=2).max_rank == 2
    assert ObservationTSVD(enabled=False).max_rank is None


def test_energy_fraction_one_keeps_modes_a_float32_cumulative_sum_cannot_see() -> None:
    """``energy_fraction=1.0`` means the round-off floor, not a cumulative sum.

    The direct regression for the defect this policy was rewritten to fix. With
    ``s = (1, 1e-3, 1e-4)`` the trailing modes hold ``1e-6`` and ``1e-8`` of the
    energy, so a float32 cumulative sum reaches exactly ``1.0`` after the second
    mode and a ``cumulative < 1 - 1e-12`` comparison stops counting — while the
    third mode is four orders of magnitude *above* the numerical floor and is
    real, assimilable information. Retaining every numerically nonzero
    direction is what ``energy_fraction=1.0`` means, and it is a statement
    about the floor that no cumulative comparison in the array's own dtype can
    express.

    It is a statement about the *outcome*, not about a special case in the
    code: the suffix criterion tolerates a zero tail at ``1.0``, counts every
    strictly nonzero singular value, and is capped at the numerically nonzero
    rank. An explicit ``energy_fraction >= 1.0`` short-circuit used to sit in
    front of that and was provably dead (0 differences over 108,324 swept
    cases), so it is gone; this assertion is unaffected either way, which is the
    point.
    """
    spectrum = np.array([1.0, 1e-3, 1e-4])
    whitened = _prescribed_spectrum(spectrum, n_e=10, seed=2)
    transform = ensemble_transform(
        *whitened, tsvd=ObservationTSVD(enabled=True, energy_fraction=1.0)
    )
    assert transform.available_rank == 3
    assert transform.retained_rank == 3

    # The trap, spelled out: the criterion this replaced drops a mode here.
    energy = np.square(np.asarray(spectrum, dtype=np.float32))
    cumulative = np.cumsum(energy) / np.sum(energy)
    assert int(np.sum(cumulative < 1.0 - 1e-12)) + 1 < 3


def test_infinite_error_inflation_excludes_one_observation() -> None:
    """The LETKF exclusion contract: ``E_inf = inf`` zeroes a row, no NaNs.

    Whitening with an infinite inflation on observation 0 must give exactly the
    analysis that drops that observation entirely.
    """
    problem = _observation_problem(seed=11, n_d=3, n_e=20)
    augmented, _, pred_obs, obs, C_D_diag = problem

    excluded = ensemble_transform(
        *whiten_observations(
            pred_obs, obs, C_D_diag, error_inflation=jnp.array([jnp.inf, 1.0, 1.0])
        )
    )
    dropped = ensemble_transform(
        *whiten_observations(pred_obs[1:], obs[1:], C_D_diag[1:])
    )
    np.testing.assert_allclose(
        np.asarray(apply_ensemble_transform(augmented, excluded)),
        np.asarray(apply_ensemble_transform(augmented, dropped)),
        rtol=RTOL,
        atol=ATOL,
    )
    assert bool(jnp.all(jnp.isfinite(excluded.mean_weights)))
    # Unit inflation must be a no-op relative to plain whitening.
    unit = whiten_observations(
        pred_obs, obs, C_D_diag, error_inflation=jnp.ones(3, dtype=pred_obs.dtype)
    )
    plain = whiten_observations(pred_obs, obs, C_D_diag)
    np.testing.assert_allclose(np.asarray(unit[0]), np.asarray(plain[0]), rtol=RTOL)
    np.testing.assert_allclose(np.asarray(unit[1]), np.asarray(plain[1]), rtol=RTOL)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"energy_fraction": 0.0},
        {"energy_fraction": 1.5},
        {"max_rank": 0},
        {"numerical_tolerance": 1.0},
        {"numerical_tolerance": float("nan")},
    ],
)  # type: ignore[misc]
def test_invalid_tsvd_settings_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ObservationTSVD(enabled=True, **kwargs)


# ---------------------------------------------------------------------------
# (d) Localization policy: fail fast at construction
# ---------------------------------------------------------------------------


def test_etkf_with_localization_raises_at_construction() -> None:
    """ETKF + localization must fail before the first forecast, actionably."""
    with pytest.raises(ValueError, match="localization_policy='forbidden'") as excinfo:
        EnsembleKalmanFilter(
            mode="state",
            analysis=ETKFAnalysis(),
            localization=_AllOnesLocalization(),
            **_dummy_filter_kwargs(),
        )
    message = str(excinfo.value)
    assert "ETKFAnalysis" in message and "_AllOnesLocalization" in message
    assert "LETKF" in message  # tells the user what to do instead


def test_etkf_call_refuses_a_localization_strategy() -> None:
    """Defense in depth for a direct caller bypassing the filter constructor."""
    problem = _observation_problem(seed=12)
    with pytest.raises(ValueError, match="cannot localize"):
        ETKFAnalysis()(
            problem[0],
            problem[2],
            problem[3],
            problem[4],
            jax.random.PRNGKey(0),
            localization=_AllOnesLocalization(),
        )


def test_required_policy_without_localization_raises() -> None:
    """The policy a future LETKF declares fails fast the other way round."""

    class _RequiresLocalization(ETKFAnalysis):
        localization_policy = "required"

    with pytest.raises(ValueError, match="localization_policy='required'"):
        EnsembleKalmanFilter(
            mode="state", analysis=_RequiresLocalization(), **_dummy_filter_kwargs()
        )


def test_unknown_localization_policy_raises() -> None:
    """A typo in the declaration must fail loudly, not disable the check."""

    class _Typo(ETKFAnalysis):
        localization_policy = "forbiden"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="localization_policy='forbiden'"):
        EnsembleKalmanFilter(mode="state", analysis=_Typo(), **_dummy_filter_kwargs())


def test_stochastic_scheme_keeps_the_optional_policy() -> None:
    """The default scheme's behavior is unchanged by the new declaration."""
    assert StochasticEnKFAnalysis().localization_policy == "optional"
    EnsembleKalmanFilter(
        mode="state", localization=_AllOnesLocalization(), **_dummy_filter_kwargs()
    )


# ---------------------------------------------------------------------------
# (e) The ETKF inside the filter: modes, inflation, PR 1's state reduction
# ---------------------------------------------------------------------------


class _ForecastOnlyAnalysis(AnalysisScheme):
    """An analysis that does nothing, so a filter run exposes its forecast.

    ``localization_policy`` stays ``"optional"`` so the same filter kwargs work
    with and without a strategy.
    """

    def __call__(
        self,
        augmented: jnp.ndarray,
        pred_obs: jnp.ndarray,
        obs: jnp.ndarray,
        C_D_diag: jnp.ndarray,
        rng_key: jax.Array,
        localization: Any = None,
        group_ids: Any = None,
        localize_mask: Any = None,
        row_coords: Any = None,
        obs_coords: Any = None,
    ) -> jnp.ndarray:
        return augmented


def _etkf_filter(**overrides: Any) -> EnsembleKalmanFilter:
    kwargs: dict[str, Any] = dict(
        observation_operator=_ToyObsOp(np.array([[1.0, 0.2, -0.1, 0.0]])),
        forward_model=_ToyLinearModel(
            np.array(
                [
                    [0.9, 0.1, 0.0, 0.0],
                    [0.0, 0.8, 0.2, 0.0],
                    [0.1, 0.0, 0.7, 0.1],
                    [0.0, 0.0, 0.2, 0.9],
                ]
            ),
            param_effect=0.3,
        ),
        C_D=jnp.array([0.15]),
        analysis=ETKFAnalysis(),
        mode="state",
        rng_key=jax.random.PRNGKey(31),
    )
    kwargs.update(overrides)
    return EnsembleKalmanFilter(**kwargs)


def test_etkf_joint_mode_updates_both_blocks_and_state_mode_does_not() -> None:
    n_e = 24
    state = _initial_state(jax.random.PRNGKey(32), n_e, np.zeros(4), np.eye(4))
    params = _params_dataset(np.random.default_rng(33).standard_normal(n_e))
    observations = jnp.array([[1.5], [1.2]])

    joint = _etkf_filter(mode="joint", inflation=RTPS(alpha=0.5)).run(
        state=state, params=params, observations=observations
    )
    assert joint.params is not None and joint.state is not None
    assert not np.allclose(joint.params["a"], params["a"])
    assert not np.allclose(joint.state["u"], state["u"])

    state_only = _etkf_filter(mode="state").run(
        state=state, params=params, observations=observations
    )
    assert state_only.params is params  # carried through untouched
    assert state_only.state is not None
    assert not np.allclose(state_only.state["u"], state["u"])


def test_etkf_parameter_mode_moves_the_parameter_block() -> None:
    """The third filter mode, and the scheme's rank diagnostics after a run."""
    n_e = 24
    params = _params_dataset(np.random.default_rng(36).standard_normal(n_e))
    analysis = ETKFAnalysis()
    result = EnsembleKalmanFilter(
        observation_operator=_ToyObsOp(np.array([[1.0, 0.0]])),
        forward_model=_ParamOnlyModel(),
        C_D=jnp.array([0.05]),
        analysis=analysis,
        mode="parameter",
        parameter_evolution=RandomWalkEvolution(std=0.02),
        rng_key=jax.random.PRNGKey(37),
    ).run(params=params, observations=jnp.array([[2.0], [2.0]]))

    assert result.params is not None
    assert abs(float(np.mean(result.params["a"])) - 2.0) < abs(
        float(np.mean(params["a"])) - 2.0
    )
    # One observation -> at most one informative whitened direction.
    assert analysis.last_transform is not None
    assert analysis.last_transform.available_rank == 1
    assert analysis.last_transform.retained_rank == 1


def test_transform_diagnostics_track_every_cycle_not_only_the_first() -> None:
    """``last_transform`` is *this* cycle's transform, on every cycle.

    ``BaseFilter._record_transform_diagnostics`` reads the scheme's
    ``last_transform`` after each analysis, so the ``transform_*`` block of
    ``cycle_diagnostics.yaml`` is only a per-cycle resource-gate quantity if the
    attribute is actually rewritten every cycle. A scheme that assigned it once
    would still satisfy every single-cycle assertion in this suite — and would
    write a constant block over a 60-cycle production run, which is precisely
    the failure the fields exist to make visible.

    Two readings of the same property: directly on the scheme (two calls with
    different observations must not share a transform) and through the filter
    (the recorded retained energy must actually move between cycles). The
    observation operator has three rows so the whitened spectrum has something
    to move, and the TSVD is on so the energy is a truncation diagnostic rather
    than a constant ``1.0``.
    """
    H = np.array([[1.0, 0.2, -0.1, 0.0], [0.0, 1.0, 0.3, 0.1], [0.2, 0.0, 0.5, 1.0]])
    tsvd = ObservationTSVD(enabled=True, energy_fraction=0.9)

    # (a) directly on the scheme.
    analysis = ETKFAnalysis(tsvd=tsvd)
    first = _observation_problem(seed=51, n_d=3, n_e=24)
    _analyze(analysis, first)
    after_first = analysis.last_transform
    assert after_first is not None
    second = _observation_problem(seed=52, n_d=3, n_e=24)
    _analyze(analysis, second)
    after_second = analysis.last_transform
    assert after_second is not None
    assert not np.allclose(
        np.asarray(after_first.mean_weights), np.asarray(after_second.mean_weights)
    )

    # (b) through the filter, where the numbers are persisted per cycle.
    n_e = 24
    state = _initial_state(jax.random.PRNGKey(53), n_e, np.zeros(4), np.eye(4))
    params = _params_dataset(np.random.default_rng(54).standard_normal(n_e))
    result = _etkf_filter(
        observation_operator=_ToyObsOp(H),
        C_D=jnp.array([0.15, 0.4, 0.25]),
        analysis=ETKFAnalysis(tsvd=tsvd),
        mode="state",
    ).run(
        state=state,
        params=params,
        observations=jnp.array([[1.5, -0.4, 0.9], [1.2, 0.8, -0.3], [0.4, 1.6, 0.2]]),
    )

    energies = [diag.transform_retained_energy for diag in result.diagnostics]
    assert len(energies) == 3
    assert all(energy is not None for energy in energies)
    assert len(set(energies)) == len(energies), (
        "transform_retained_energy repeats across cycles, so the recorded "
        f"transform is not this cycle's: {energies}"
    )
    # The rest of the block is populated every cycle too, not just cycle 0.
    for diag in result.diagnostics:
        assert diag.transform_retained_rank is not None
        assert diag.transform_available_rank is not None
        assert diag.transform_discarded_spectrum_max is not None


@pytest.mark.parametrize("mode", ["state", "joint"])  # type: ignore[misc]
def test_full_rank_state_reduction_reproduces_the_unreduced_etkf(mode: str) -> None:
    """PR 1's reduced representation is transparent to the ETKF.

    The reduced path encodes the state block (a LEFT multiplication) and the
    transform is a RIGHT multiplication, so at full rank the two commute and
    the analyzed state must be unchanged. This also exercises the reduction's
    discarded-increment diagnostic, which calls the analysis a second time.
    """
    n_e = 24
    state = _initial_state(jax.random.PRNGKey(34), n_e, np.zeros(4), np.eye(4))
    params = _params_dataset(np.random.default_rng(35).standard_normal(n_e))
    observations = jnp.array([[0.9]])
    common: Any = dict(mode=mode, inflation=RTPS(alpha=0.6))

    full = _etkf_filter(**common).run(
        state=state, params=params, observations=observations
    )
    reduced = _etkf_filter(
        state_reduction=OnlineStateReduction(energy_fraction=1.0, whiten=False),
        **common,
    ).run(state=state, params=params, observations=observations)

    assert full.state is not None and reduced.state is not None
    np.testing.assert_allclose(
        reduced.state["u"], full.state["u"], rtol=RTOL, atol=ATOL
    )
    if mode == "joint":
        assert full.params is not None and reduced.params is not None
        np.testing.assert_allclose(
            reduced.params["a"], full.params["a"], rtol=RTOL, atol=ATOL
        )
    diag = reduced.diagnostics[0]
    assert diag.reduction_rank is not None and diag.reduction_rank <= n_e - 1
    # No assertion on ``reduction_discarded_increment_fraction`` here: at full
    # rank ``increment_coordinates[rank:]`` is an empty slice whose norm is
    # ``0.0`` by construction, so the check could not fail whatever the ETKF
    # did. The diagnostic is exercised under real truncation by
    # ``test_truncated_state_reduction_reports_a_discarded_increment`` below.


def test_truncated_state_reduction_reports_a_discarded_increment() -> None:
    """The discarded-increment diagnostic, where it can actually be nonzero.

    ``BaseFilter._record_reduction_diagnostics`` measures the increment the
    truncation threw away by calling the analysis a *second* time on the tiny
    ``(k, N_e)`` coordinate array and comparing the retained coordinates with
    the full ones. That subtraction is only meaningful because the ETKF
    transform depends on ``(pred_obs, obs, C_D_diag)`` alone (the
    pure-right-multiplication contract at the top of this file), so this is the
    end-to-end check of that contract on the reduced path: with a hard rank cap
    of 1 on a 4-dimensional state, the discarded fraction must be strictly
    positive and a proper fraction.
    """
    n_e = 24
    state = _initial_state(jax.random.PRNGKey(34), n_e, np.zeros(4), np.eye(4))
    params = _params_dataset(np.random.default_rng(35).standard_normal(n_e))
    reduced = _etkf_filter(
        mode="state",
        state_reduction=OnlineStateReduction(max_rank=1, whiten=False),
    ).run(state=state, params=params, observations=jnp.array([[0.9]]))

    diag = reduced.diagnostics[0]
    assert diag.reduction_rank == 1
    fraction = diag.reduction_discarded_increment_fraction
    assert fraction is not None
    assert 0.0 < fraction < 1.0


@pytest.mark.parametrize(
    "make_analysis, localization",
    [
        (ETKFAnalysis, None),
        (LETKFAnalysis, _AllOnesLocalization()),
    ],
    ids=["etkf", "letkf"],
)  # type: ignore[misc]
def test_rtpp_blends_each_member_against_its_own_prior(
    make_analysis: Any, localization: Any
) -> None:
    """RTPP end to end under both deterministic schemes.

    RTPP member identity is the stated reason all three docstrings in
    ``etkf.py`` insist on the *symmetric* square root — any root ``W_a Q`` with
    orthogonal ``Q`` gives the same posterior covariance, so the covariance
    identities elsewhere in this file cannot tell them apart — yet nothing
    exercised the two together. This does, through the filter, and checks the
    properties that make the pairing meaningful:

    * the blend is member-wise and linear, so ``alpha = 0.5`` is exactly the
      midpoint of the ``alpha = 0`` and ``alpha = 1`` ensembles member by
      member. A posterior that re-centred or reordered members between the
      analysis and the relaxation would break this;
    * ``alpha = 1`` returns the *physical forecast* anomalies exactly, which
      pins that ``dev_prior`` really is the forecast this cycle produced;
    * RTPP moves spread only: the analyzed mean is the same for every alpha;
    * every member's posterior anomaly is positively aligned with **its own**
      prior anomaly. That is the property the symmetric (minimum-distance) root
      provides and a rotated root does not, and the last assertion shows the
      claim has bite by checking that a mismatched pairing fails it.

    The forecast is recovered by running the identical filter with an identity
    analysis: the toy model is deterministic and the ETKF/LETKF ignore the RNG
    key, so the runs share a forecast bit for bit.
    """
    n_e = 24
    state = _initial_state(jax.random.PRNGKey(44), n_e, np.zeros(4), np.eye(4))
    params = _params_dataset(np.random.default_rng(45).standard_normal(n_e))
    observations = jnp.array([[1.4]])

    def _run(analysis: AnalysisScheme, inflation: Any) -> np.ndarray:
        result = _etkf_filter(
            mode="state",
            analysis=analysis,
            inflation=inflation,
            localization=localization,
        ).run(state=state, params=params, observations=observations)
        assert result.state is not None
        return np.asarray(result.state["u"].values)  # (N_e, nx)

    def _anomalies(ensemble: np.ndarray) -> np.ndarray:
        return ensemble - ensemble.mean(axis=0, keepdims=True)

    forecast = _run(_ForecastOnlyAnalysis(), None)
    analyzed = _run(make_analysis(), RTPP(alpha=0.0))
    blended = _run(make_analysis(), RTPP(alpha=0.5))
    relaxed = _run(make_analysis(), RTPP(alpha=1.0))

    prior_anomalies = _anomalies(forecast)
    posterior_anomalies = _anomalies(analyzed)

    # The analysis genuinely contracts the anomalies, so everything below is a
    # statement about two different ensembles.
    assert np.std(posterior_anomalies) < 0.9 * np.std(prior_anomalies)

    # (a) member-wise linear blend.
    np.testing.assert_allclose(
        _anomalies(blended),
        0.5 * (posterior_anomalies + _anomalies(relaxed)),
        rtol=RTOL,
        atol=ATOL,
    )
    # (b) alpha = 1 restores the physical forecast anomalies, member by member.
    np.testing.assert_allclose(
        _anomalies(relaxed), prior_anomalies, rtol=RTOL, atol=ATOL
    )
    assert not np.allclose(_anomalies(relaxed), prior_anomalies[::-1], atol=1e-3)
    # (c) spread only: the analyzed mean is untouched by the relaxation, and it
    #     is not the forecast mean.
    for ensemble in (blended, relaxed):
        np.testing.assert_allclose(
            ensemble.mean(axis=0), analyzed.mean(axis=0), rtol=RTOL, atol=ATOL
        )
    assert not np.allclose(analyzed.mean(axis=0), forecast.mean(axis=0), atol=1e-3)
    # ... and the spread grows monotonically with alpha.
    spreads = [float(np.std(_anomalies(e))) for e in (analyzed, blended, relaxed)]
    assert spreads[0] < spreads[1] < spreads[2]

    # (d) each posterior member stays aligned with its OWN prior member.
    def _alignment(pairing: np.ndarray) -> np.ndarray:
        numerator = np.sum(prior_anomalies * pairing, axis=1)
        norms = np.linalg.norm(prior_anomalies, axis=1) * np.linalg.norm(
            pairing, axis=1
        )
        return numerator / norms

    assert np.all(_alignment(posterior_anomalies) > 0.5)
    # Non-vacuous: the same statement is false for a scrambled pairing, which
    # is exactly what a rotated square root would produce.
    assert np.min(_alignment(posterior_anomalies[::-1])) < 0.5
