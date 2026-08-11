"""Unit and e2e tests for the moving window (Phase 2 of filter smoothing).

Covers the contract in docs/plans/filter_smoothing_windowed_esmda.md §8a: the
orchestrator ``data_assimilation.filter_smoothing.moving_window.run_moving_window``
slides a length-``L`` window over a ``T``-cycle horizon in ``num_windows`` steps
of ``window_shift`` cycles, carrying the *state* forward from each window's
final consistency pass, carrying the overlapping *parameter* ensemble forward as
the next window's prior, appending a leading-edge knot with the library's
``ParameterEvolution``, and finalizing the knots that leave the window.

Everything except the solver-backed e2e test runs on the toy models of
tests/test_filter_smoothing.py — the semantics under test are about *which*
ensemble and *which* observation rows each window is handed, so the tests read
a call log rather than reverse-engineering the numbers.

The stubs are imported from tests/test_filter_smoothing.py rather than copied
(the convention of tests/test_filtering_etkf.py): the Phase 1 toy model and its
observation operator already pin the shapes the inner pass demands, and a
divergent copy here would let the two phases drift apart. The one new stub is
:class:`_RecordingSmoother`, which records what the *orchestrator* hands each
window — the only surface Phase 2 adds.

Like the other e2e suites, the solver-backed test shares the pylbm build tree
and ``.temp``, so pytest sessions must not run concurrently (auto-memory:
serial e2e sessions).
"""

import pathlib
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray
from data_assimilation.filter_smoothing import (
    FilterSmoothingESMDA,
    MovingWindowResult,
    run_moving_window,
)
from data_assimilation.filtering.parameter_evolution import (
    IdentityEvolution,
    RandomWalkEvolution,
)

from tests.test_filter_smoothing import (
    _scalar_state,
    _smoothing_kwargs,
    _trajectory_params,
    _TrajectoryToyModel,
)
from tests.test_run_filter_smoothing import (
    _overrides,
    _stub_observation_operator,
    _StubEnsembleModel,
)

# The toy models are float32 (JAX default), so "unchanged value" comparisons get
# a float32-aware tolerance. The two identity-shaped claims — a single window
# reproducing Phase 1, and the assembled trajectory being a concatenation of
# slices already computed — are asserted EXACTLY on purpose: they are copies,
# not recomputations, and any drift there means the orchestrator recomputed
# something it should have carried.
RTOL = 1e-6
ATOL = 1e-6


# ---------------------------------------------------------------------------
# Stubs and builders
# ---------------------------------------------------------------------------


class _RecordingSmoother(FilterSmoothingESMDA):
    """A real smoother that logs what the orchestrator handed each window.

    Subclass rather than a duck-typed proxy: the orchestrator may reach for any
    public attribute of the smoother it is given (``num_steps``, ``rng_key``,
    ``C_D_diag``, ...), and a stand-in would silently pass a test the real class
    would fail. Only ``run`` is intercepted, and it delegates — so the recorded
    window results are genuine Phase 1 results, which is exactly what the carry
    assertions need to compare against.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.window_calls: list[dict[str, Any]] = []
        self.window_returns: list[Any] = []

    def run(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        observations: Optional[jnp.ndarray] = None,
        *,
        return_history: bool = False,
    ) -> Any:
        self.window_calls.append(
            {
                "state": None if state is None else state.copy(deep=True),
                "params": None if params is None else params.copy(deep=True),
                "observations": (
                    None if observations is None else np.asarray(observations)
                ),
                "return_history": return_history,
            }
        )
        result = super().run(
            state=state,
            params=params,
            observations=observations,
            return_history=return_history,
        )
        self.window_returns.append(result)
        return result


def _make_smoother(
    *,
    num_steps: int = 1,
    seed: int = 0,
    recording: bool = False,
    model: Optional[_TrajectoryToyModel] = None,
    **kwargs: Any,
) -> Any:
    """A one-observation smoother over the scalar toy model."""
    cls = _RecordingSmoother if recording else FilterSmoothingESMDA
    return cls(
        num_steps=num_steps,
        rng_key=jax.random.PRNGKey(seed),
        **_smoothing_kwargs(_TrajectoryToyModel() if model is None else model),
        **kwargs,
    )


def _horizon_observations(total_cycles: int) -> jnp.ndarray:
    """``(T, 1)`` batches whose rows are pairwise distinct.

    Distinctness is load-bearing for the slicing test: identical rows would let
    any window's slice match every other window's.
    """
    return jnp.asarray(np.arange(total_cycles, dtype=float)[:, None] + 0.5)


def _knots(num_knots: int, n_e: int, seed: int) -> np.ndarray:
    return np.asarray(jax.random.normal(jax.random.PRNGKey(seed), (num_knots, n_e)))


def _theta(dataset: xarray.Dataset) -> np.ndarray:
    return np.asarray(dataset["theta"].values)


def _u(dataset: xarray.Dataset) -> np.ndarray:
    return np.asarray(dataset["u"].values)


def _initial_state(n_e: int, seed: int) -> xarray.Dataset:
    return _scalar_state(
        np.asarray(jax.random.normal(jax.random.PRNGKey(seed), (n_e,)))
    )


# ---------------------------------------------------------------------------
# (1) One window is exactly Phase 1
# ---------------------------------------------------------------------------


def test_a_single_window_is_bit_identical_to_the_plain_smoother() -> None:
    """``num_windows=1`` must return the Phase 1 result, wrapped (§8a).

    The orchestrator adds scaffolding around ``run()`` — an evolution PRNG key,
    the carry extraction, the assembly — and "one window" is the same result
    only if none of that perturbs the numbers. Exact equality, deliberately: a
    discrepancy at 1e-7 would mean the degenerate case took a different route
    through the RNG, and every multi-window run would inherit that difference
    invisibly.
    """
    n_e, num_cycles, num_steps = 12, 3, 2
    state = _initial_state(n_e, seed=200)
    knots = _knots(num_cycles, n_e, seed=201)
    observations = _horizon_observations(num_cycles)

    plain = _make_smoother(num_steps=num_steps, seed=202).run(
        state=state,
        params=_trajectory_params(knots),
        observations=observations,
    )
    windowed = run_moving_window(
        _make_smoother(num_steps=num_steps, seed=202),
        state=state,
        params=_trajectory_params(knots),
        observations=observations,
        num_windows=1,
    )

    assert isinstance(windowed, MovingWindowResult)
    assert windowed.params is not None and windowed.state is not None
    np.testing.assert_array_equal(_theta(windowed.params), _theta(plain.params))
    np.testing.assert_array_equal(_u(windowed.state), _u(plain.state))
    np.testing.assert_array_equal(
        np.asarray(windowed.params["time"].values),
        np.asarray(plain.params["time"].values),
    )
    # The wrapper still reports one window's worth of bookkeeping.
    assert len(windowed.window_results) == 1
    assert len(windowed.window_diagnostics) == 1
    assert windowed.window_diagnostics[0].window == 0
    assert windowed.window_diagnostics[0].first_cycle == 0
    assert windowed.window_diagnostics[0].last_cycle == num_cycles - 1
    # ... and the assimilation actually moved the trajectory, so the equality
    # above is not two copies of the prior.
    assert not np.allclose(_theta(windowed.params), knots, atol=1e-4)


def test_the_first_window_of_a_multi_window_run_is_also_the_plain_result() -> None:
    """Window 0 is an ordinary Phase 1 window; only what follows is new.

    Sharpens the degenerate-case test above: it makes the *loop* prove it does
    not disturb the first window (by pre-drawing evolution noise from the
    smoother's own key chain, say, or by pre-collecting histories in a way that
    consumes randomness). Everything downstream is conditioned on this window,
    so a perturbation here would be untraceable.
    """
    n_e, num_cycles, num_steps, window_shift = 10, 2, 2, 1
    state = _initial_state(n_e, seed=230)
    knots = _knots(num_cycles, n_e, seed=231)
    observations = _horizon_observations(num_cycles + window_shift)

    plain = _make_smoother(num_steps=num_steps, seed=232).run(
        state=state,
        params=_trajectory_params(knots),
        observations=observations[:num_cycles],
    )
    smoother = _make_smoother(num_steps=num_steps, seed=232, recording=True)
    run_moving_window(
        smoother,
        state=state,
        params=_trajectory_params(knots),
        observations=observations,
        num_windows=2,
        window_shift=window_shift,
    )

    np.testing.assert_array_equal(
        _theta(smoother.window_returns[0].params), _theta(plain.params)
    )
    np.testing.assert_array_equal(_u(smoother.window_returns[0].state), _u(plain.state))


@pytest.mark.parametrize("return_history", [False, True])  # type: ignore[misc]
def test_histories_are_kept_only_when_the_caller_asks(return_history: bool) -> None:
    """Heavy per-cycle histories are internal scaffolding by default (§8a).

    With ``s < L`` the orchestrator *must* collect the final pass's state
    history — the carried state is an intermediate cycle's analysis, which is
    not otherwise reachable — but a long run keeps ``num_windows`` results
    alive, so those histories have to be dropped again unless the caller asked
    for them. Both halves matter: leaking them is a memory leak proportional to
    the horizon, and dropping them when requested loses the diagnostics.
    """
    n_e, num_cycles, num_windows, window_shift = 8, 2, 2, 1
    result = run_moving_window(
        _make_smoother(num_steps=1, seed=203),
        state=_initial_state(n_e, seed=204),
        params=_trajectory_params(_knots(num_cycles, n_e, seed=205)),
        observations=_horizon_observations(num_cycles + window_shift),
        num_windows=num_windows,
        window_shift=window_shift,
        return_history=return_history,
    )

    assert len(result.window_results) == num_windows
    for stored in result.window_results:
        if return_history:
            assert stored.params_history is not None
            assert stored.final_pass.state_history is not None
        else:
            assert stored.params_history is None
            assert stored.final_pass.state_history is None


# ---------------------------------------------------------------------------
# (2) Observation slicing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window_shift", [1, 2])  # type: ignore[misc]
def test_each_window_consumes_its_own_observation_rows(window_shift: int) -> None:
    """Window ``w`` sees exactly rows ``[w*s, w*s + L)`` of the horizon (§8a).

    The whole moving window rests on this alignment: cycle ``k`` of window ``w``
    is absolute cycle ``w*s + k``, so an off-by-one in the slice would assimilate
    every observation one cycle early and still produce a plausible posterior.
    ``s = L`` (here 2) is the non-overlapping regime — the slices must then tile
    the horizon without gaps or repeats.
    """
    n_e, num_cycles, num_windows = 6, 2, 3
    total_cycles = num_cycles + (num_windows - 1) * window_shift
    observations = _horizon_observations(total_cycles)
    smoother = _make_smoother(num_steps=1, seed=206, recording=True)

    run_moving_window(
        smoother,
        state=_initial_state(n_e, seed=207),
        params=_trajectory_params(_knots(num_cycles, n_e, seed=208)),
        observations=observations,
        num_windows=num_windows,
        window_shift=window_shift,
    )

    assert len(smoother.window_calls) == num_windows
    horizon = np.asarray(observations)
    for window, call in enumerate(smoother.window_calls):
        seen = call["observations"]
        assert seen.shape == (num_cycles, 1), (
            f"window {window} was given {seen.shape[0]} observation batches; "
            f"each window is exactly L={num_cycles} cycles long."
        )
        start = window * window_shift
        np.testing.assert_allclose(
            seen, horizon[start : start + num_cycles], rtol=RTOL, atol=ATOL
        )
    # The last window must end on the horizon's last row: T = L + (W-1)*s.
    np.testing.assert_allclose(
        smoother.window_calls[-1]["observations"][-1], horizon[-1], rtol=RTOL
    )


# ---------------------------------------------------------------------------
# (3) State carry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window_shift", [1, 2])  # type: ignore[misc]
def test_the_next_window_starts_from_the_analysis_after_cycle_shift(
    window_shift: int,
) -> None:
    """The carried state is the final pass's analysis after cycle ``s`` (§8a).

    Not the previous window's *end* state (that is the ``s = L`` case only) and
    not a re-forecast: the window advances by ``s`` cycles, so the state must
    advance by ``s`` too. Carrying the end state instead would hand the next
    window an initial condition that has already assimilated observations its
    own cycles have not yet been given.

    ``s = 1`` (maximal overlap) and ``s = L = 2`` (non-overlapping) are the two
    ends of the contract's ``s in [1, L]``.
    """
    n_e, num_cycles, num_windows, num_steps = 10, 2, 2, 1
    total_cycles = num_cycles + (num_windows - 1) * window_shift
    model = _TrajectoryToyModel()
    smoother = _make_smoother(
        num_steps=num_steps, seed=209, recording=True, model=model
    )

    run_moving_window(
        smoother,
        state=_initial_state(n_e, seed=210),
        params=_trajectory_params(_knots(num_cycles, n_e, seed=211)),
        observations=_horizon_observations(total_cycles),
        num_windows=num_windows,
        window_shift=window_shift,
    )

    first = smoother.window_returns[0]
    if window_shift == num_cycles:
        # s = L: the analysis after the last cycle IS the window's end state,
        # so the orchestrator may read it straight off the result.
        expected = _u(first.state)
    else:
        assert first.final_pass.state_history is not None, (
            "with s < L the carried state is an intermediate cycle's analysis, "
            "so the window must be run with history collection on."
        )
        expected = _u(first.final_pass.state_history.isel(cycle=window_shift - 1))

    carried = smoother.window_calls[1]["state"]
    assert "cycle" not in carried.dims  # the cycle dim is dropped, not carried
    assert carried.sizes["ensemble"] == n_e
    np.testing.assert_allclose(_u(carried), expected, rtol=RTOL, atol=ATOL)

    # The carried ensemble is what the next window's FIRST FORECAST actually ran
    # with — read off the forward model's own call log, not just the run()
    # argument, so a smoother that dropped it on the floor would be caught.
    forecasts_per_window = (num_steps + 1) * num_cycles
    np.testing.assert_allclose(
        _u(model.states[forecasts_per_window]), expected, rtol=RTOL, atol=ATOL
    )

    if window_shift < num_cycles:
        # s < L: the carried state is NOT the end state — otherwise this test
        # would also pass for an orchestrator that ignored the shift entirely.
        assert not np.allclose(_u(carried), _u(first.state), atol=1e-4)


# ---------------------------------------------------------------------------
# (4) Parameter carry and the leading edge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window_shift", [1, 2])  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "evolution", [None, IdentityEvolution()], ids=["default", "identity"]
)
def test_identity_evolution_shifts_the_posterior_and_repeats_the_last_knot(
    evolution: Any, window_shift: int
) -> None:
    """The overlapping posterior IS the next prior; the edge is ``evolve``d.

    Paper §8: the window slides, the surviving knots keep their posterior values
    (re-sampling the prior there would throw away everything the last window
    learned), and only the new leading edge comes from the evolution model.
    ``IdentityEvolution`` — also the documented default when ``evolution=None``
    — makes the appended knot exactly the last carried one, which is the
    sharpest available statement of "the edge came from evolving the carried
    ensemble" rather than from a fresh prior draw.

    The knot ``time`` coordinates must continue the incoming grid, since the
    temporal localization and the assembled trajectory both read them.
    """
    n_e, num_cycles, num_windows = 8, 3, 2
    total_cycles = num_cycles + (num_windows - 1) * window_shift
    gamma = np.asarray(0.25 + 0.05 * jax.random.normal(jax.random.PRNGKey(212), (n_e,)))
    smoother = _make_smoother(num_steps=1, seed=213, recording=True)

    run_moving_window(
        smoother,
        state=_initial_state(n_e, seed=214),
        params=_trajectory_params(_knots(num_cycles, n_e, seed=215), static=gamma),
        observations=_horizon_observations(total_cycles),
        num_windows=num_windows,
        window_shift=window_shift,
        evolution=evolution,
    )

    posterior = smoother.window_returns[0].params
    next_prior = smoother.window_calls[1]["params"]

    # A window never changes length: s knots leave, s are appended.
    assert next_prior.sizes["time"] == num_cycles
    assert next_prior.sizes["ensemble"] == n_e

    kept = num_cycles - window_shift
    np.testing.assert_allclose(
        _theta(next_prior)[:kept],
        _theta(posterior)[window_shift:],
        rtol=RTOL,
        atol=ATOL,
    )
    # ... and those carried knots are the POSTERIOR, not the prior the window
    # started from, so the test cannot be satisfied by re-sampling.
    assert not np.allclose(
        _theta(next_prior)[:kept],
        _theta(smoother.window_calls[0]["params"])[window_shift:],
        atol=1e-4,
    )

    last_carried = _theta(posterior)[-1]
    for appended in _theta(next_prior)[kept:]:
        np.testing.assert_allclose(appended, last_carried, rtol=RTOL, atol=ATOL)

    # The knot grid continues at the incoming spacing (1 s here), so knot k of
    # window 1 is absolute cycle k + s.
    np.testing.assert_allclose(
        np.asarray(next_prior["time"].values),
        np.arange(window_shift, window_shift + num_cycles, dtype=float),
        atol=1e-9,
    )

    # Static (no-`time`) variables ride along untouched: they are excluded from
    # the outer update, so they must also be excluded from the shift — and never
    # handed to the evolution model, which would walk them.
    assert "time" not in next_prior["gamma"].dims
    np.testing.assert_allclose(
        np.asarray(next_prior["gamma"].values), gamma, rtol=RTOL, atol=ATOL
    )


def test_random_walk_evolution_appends_a_spread_growing_leading_edge() -> None:
    """The leading edge is a per-member draw whose spread grows by ~``std``.

    The point of an evolution model at the leading edge (§8a's deviation from
    ``ParameterTimeSeries.extrapolate``) is that the incoming knot must carry
    *prior uncertainty* about a cycle no observation has touched yet. One shared
    draw, or a draw around the ensemble mean, would enter the next window with
    the collapsed spread of an already-assimilated knot and the window would
    learn nothing there.

    Large ensemble plus a check on the per-member INCREMENT: ``new - carried``
    is exactly ``std * z``, so its sample std pins ``std`` to a few percent —
    far tighter than comparing two ensemble spreads.
    """
    n_e, num_cycles, std = 512, 2, 1.5
    smoother = _make_smoother(num_steps=1, seed=216, recording=True)

    run_moving_window(
        smoother,
        state=_initial_state(n_e, seed=217),
        params=_trajectory_params(_knots(num_cycles, n_e, seed=218)),
        observations=_horizon_observations(num_cycles + 1),
        num_windows=2,
        window_shift=1,
        evolution=RandomWalkEvolution(std=std),
        rng_key=jax.random.PRNGKey(219),
    )

    posterior = smoother.window_returns[0].params
    next_prior = smoother.window_calls[1]["params"]

    carried_last = _theta(posterior)[-1]
    appended = _theta(next_prior)[-1]
    assert appended.shape == (n_e,)
    # Per-member noise, not one shared perturbation of the whole ensemble.
    assert len(np.unique(appended)) == n_e
    assert not np.allclose(appended, carried_last, atol=1e-4)

    increment = appended - carried_last
    assert float(np.std(increment)) == pytest.approx(std, rel=0.15)
    # Spread grows accordingly: var_new ~ var_carried + std^2.
    assert float(np.std(appended)) == pytest.approx(
        float(np.sqrt(np.var(carried_last) + std**2)), rel=0.15
    )
    assert float(np.std(appended)) > float(np.std(carried_last))

    # Only the leading edge is evolved: the carried knots keep their posterior
    # values exactly (a random walk applied to the whole trajectory would
    # re-inject noise into knots the window already conditioned).
    np.testing.assert_allclose(
        _theta(next_prior)[:-1], _theta(posterior)[1:], rtol=RTOL, atol=ATOL
    )


# ---------------------------------------------------------------------------
# (5) Finalization and assembly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window_shift", [1, 2])  # type: ignore[misc]
def test_the_assembled_trajectory_spans_the_horizon_once(window_shift: int) -> None:
    """``result.params`` is the finalized knots, in order, over all ``T`` cycles.

    Paper §8: a knot is *fixed* when it leaves the window, so the assembled
    trajectory must reproduce, for each early knot, the posterior of the window
    that finalized it — not a later window's view of it (a later window never
    saw those cycles) and not a re-run.  The last window contributes its whole
    posterior, which is the only part still open.
    """
    n_e, num_cycles, num_windows, num_steps = 8, 3, 3, 1
    total_cycles = num_cycles + (num_windows - 1) * window_shift
    gamma = np.asarray(0.25 + 0.05 * jax.random.normal(jax.random.PRNGKey(229), (n_e,)))
    smoother = _make_smoother(num_steps=num_steps, seed=220, recording=True)

    result = run_moving_window(
        smoother,
        state=_initial_state(n_e, seed=221),
        params=_trajectory_params(_knots(num_cycles, n_e, seed=222), static=gamma),
        observations=_horizon_observations(total_cycles),
        num_windows=num_windows,
        window_shift=window_shift,
    )

    assert result.params is not None
    assert result.params.sizes["time"] == total_cycles
    assert result.params.sizes["ensemble"] == n_e
    # One continuous knot grid over the horizon, at the prior's spacing: the
    # concatenation must not restart the clock at each window.
    np.testing.assert_allclose(
        np.asarray(result.params["time"].values),
        np.arange(total_cycles, dtype=float),
        atol=1e-9,
    )

    assembled = _theta(result.params)
    for window in range(num_windows - 1):
        start = window * window_shift
        # Exact: finalization is a slice of an already-computed posterior.
        np.testing.assert_array_equal(
            assembled[start : start + window_shift],
            _theta(smoother.window_returns[window].params)[:window_shift],
        )
    np.testing.assert_array_equal(
        assembled[(num_windows - 1) * window_shift :],
        _theta(smoother.window_returns[-1].params),
    )
    # Consecutive windows really do disagree about the overlap, so "the window
    # that finalized it" is a meaningful attribution and not a distinction
    # without a difference.
    if window_shift < num_cycles:
        assert not np.allclose(
            _theta(smoother.window_returns[0].params)[window_shift:],
            _theta(smoother.window_returns[1].params)[: num_cycles - window_shift],
            atol=1e-4,
        )

    # Static (no-`time`) variables must survive the concatenation as scalars.
    # An `xarray.concat` at its defaults would broadcast `gamma` to a spurious
    # `time` dimension of length T, quietly turning a constant into a fake
    # trajectory in every downstream artifact and figure.
    assert result.params["gamma"].dims == ("ensemble",)
    np.testing.assert_allclose(
        np.asarray(result.params["gamma"].values), gamma, rtol=RTOL, atol=ATOL
    )

    # The returned state is the LAST window's filtered end state (Eq. 16).
    assert result.state is not None
    np.testing.assert_array_equal(
        _u(result.state), _u(smoother.window_returns[-1].state)
    )

    # Per-window bookkeeping: one entry per window, in order, each recording the
    # absolute cycle span it covered and its own outer iterations.
    assert len(result.window_results) == num_windows
    assert len(result.window_diagnostics) == num_windows
    for window, record in enumerate(result.window_diagnostics):
        assert record.window == window
        assert record.first_cycle == window * window_shift
        assert record.last_cycle == window * window_shift + num_cycles - 1
        assert len(record.iteration_diagnostics) == num_steps
    assert result.window_diagnostics[-1].last_cycle == total_cycles - 1


# ---------------------------------------------------------------------------
# (6) Validation
# ---------------------------------------------------------------------------

# The validation cases all share this shape: L = 2 cycles per window, 2 windows,
# shift 1 -> a 3-cycle horizon. ``window_length`` is passed explicitly so L is a
# given rather than something solved out of T, which is what makes a wrong T
# detectable at all.
_VALIDATION_CYCLES = 2


def _validation_call(model: _TrajectoryToyModel, **kwargs: Any) -> Any:
    """Invoke the orchestrator on a toy setup, overridden by ``kwargs``."""
    n_e = 6
    call: dict[str, Any] = {
        "state": _initial_state(n_e, seed=223),
        "params": _trajectory_params(_knots(_VALIDATION_CYCLES, n_e, seed=224)),
        "observations": _horizon_observations(_VALIDATION_CYCLES + 1),
        "num_windows": 2,
        "window_shift": 1,
        "window_length": _VALIDATION_CYCLES,
    }
    call.update(kwargs)
    return run_moving_window(_make_smoother(num_steps=1, seed=225, model=model), **call)


@pytest.mark.parametrize("total_cycles", [2, 4, 5])  # type: ignore[misc]
def test_a_horizon_that_does_not_match_the_window_arithmetic_raises(
    total_cycles: int,
) -> None:
    """``T`` must be exactly ``L + (num_windows - 1) * s`` (§8a, "loudly").

    Too few rows and the last window runs short or silently reuses the tail; too
    many and the horizon's final cycles are never assimilated while the
    artifacts still claim a full-horizon trajectory. Neither is visible in the
    output, so the check has to be up front — before a single forecast is spent.
    """
    model = _TrajectoryToyModel()
    with pytest.raises(ValueError):
        _validation_call(model, observations=_horizon_observations(total_cycles))
    assert model.states == [], "validation must fail before any forecast runs"


@pytest.mark.parametrize("window_shift", [0, -1, 3, 5])  # type: ignore[misc]
def test_a_window_shift_outside_one_to_L_raises(window_shift: int) -> None:
    """``s in [1, L]``: ``s < 1`` never advances, ``s > L`` skips cycles.

    ``L = 2`` here, so 3 and 5 are the "gap in the horizon" side and 0 / -1 the
    "same window forever" side. Each case is given the horizon its own
    arithmetic implies, so the shift itself is the only thing wrong.
    """
    model = _TrajectoryToyModel()
    with pytest.raises(ValueError, match="window_shift"):
        _validation_call(
            model,
            window_shift=window_shift,
            observations=_horizon_observations(
                max(_VALIDATION_CYCLES + window_shift, 1)
            ),
        )
    assert model.states == []


@pytest.mark.parametrize("num_windows", [0, -1])  # type: ignore[misc]
def test_fewer_than_one_window_raises(num_windows: int) -> None:
    """At least one window: zero windows would return an empty trajectory."""
    model = _TrajectoryToyModel()
    with pytest.raises(ValueError, match="num_windows"):
        _validation_call(
            model,
            num_windows=num_windows,
            observations=_horizon_observations(_VALIDATION_CYCLES),
        )
    assert model.states == []


def test_a_window_longer_than_the_trajectory_prior_raises() -> None:
    """Every cycle of a window needs a knot to forecast with.

    The failure mode this guards is subtle: with ``L`` solved out of ``T``, a
    horizon that is too long silently produces a window longer than the prior,
    whose extra cycles would then reuse the last knot (the inner filter clamps)
    and be attributed to a parameter nobody estimated.
    """
    n_e = 6
    model = _TrajectoryToyModel()
    with pytest.raises(ValueError):
        run_moving_window(
            _make_smoother(num_steps=1, seed=226, model=model),
            state=_initial_state(n_e, seed=227),
            params=_trajectory_params(_knots(2, n_e, seed=228)),
            observations=_horizon_observations(6),
            num_windows=2,
            window_shift=1,
        )
    assert model.states == []


# ---------------------------------------------------------------------------
# (7) Config composition and the end-to-end run
# ---------------------------------------------------------------------------


def _window_overrides(
    num_cycles: int,
    num_steps: int,
    num_windows: int,
    window_shift: int,
    extra: Optional[list[str]] = None,
) -> list[str]:
    """The Phase 1 e2e overrides plus the moving-window scalars."""
    return _overrides(
        num_cycles,
        num_steps,
        [
            f"filter_smoothing.num_windows={num_windows}",
            f"filter_smoothing.window_shift={window_shift}",
            *(extra or []),
        ],
    )


def test_moving_window_scalars_default_to_the_phase_1_path(
    compose_test_cfg: Any,
) -> None:
    """Shipped defaults must still be the single-window run (§8a).

    Deliberately composes with NO overrides: Phase 2 is additive, and a default
    of anything other than ``num_windows=1`` would change every existing run of
    this entry point — and its artifact set — without a single Phase 1 test
    noticing.
    """
    cfg = compose_test_cfg([], config_name="run_filter_smoothing")

    assert int(cfg.filter_smoothing.num_windows) == 1
    assert int(cfg.filter_smoothing.window_shift) == 1
    # The default leading-edge model is the no-op one: a single-window run never
    # appends a knot, so anything else would be a knob that silently does
    # nothing until num_windows > 1.
    assert cfg.filter_smoothing.evolution is None


@pytest.mark.parametrize(  # type: ignore[misc]
    "option,expected_target",
    [
        ("none", None),
        (
            "random_walk",
            "data_assimilation.filtering.parameter_evolution.RandomWalkEvolution",
        ),
    ],
)
def test_evolution_group_composes(
    option: str, expected_target: Optional[str], compose_test_cfg: Any
) -> None:
    """Both leading-edge options mount into the composed target.

    ``none`` must resolve to a literal ``null`` rather than an
    ``IdentityEvolution`` node: the orchestrator's own default is the identity,
    and a config that instantiated one anyway would make the two paths differ
    the moment that default changed.
    """
    cfg = compose_test_cfg(
        _window_overrides(2, 2, 2, 1, [f"filter_smoothing/evolution={option}"]),
        config_name="run_filter_smoothing",
    )
    node = cfg.filter_smoothing.evolution

    if expected_target is None:
        assert node is None
    else:
        assert node._target_ == expected_target
        assert float(node.std) > 0.0


def test_composed_random_walk_evolution_constructs(compose_test_cfg: Any) -> None:
    """Instantiate the node, solver-free: compose alone cannot see a bad std.

    ``RandomWalkEvolution`` rejects a negative std at construction, and the
    orchestrator calls ``evolve`` only once per window advance — deep into a
    multi-hour run.
    """
    from hydra.utils import instantiate

    cfg = compose_test_cfg(
        _window_overrides(2, 2, 2, 1, ["filter_smoothing/evolution=random_walk"]),
        config_name="run_filter_smoothing",
    )
    evolution = instantiate(cfg.filter_smoothing.evolution)

    assert isinstance(evolution, RandomWalkEvolution)
    # It is the paper's per-member append: same variables, same dims, a
    # different draw for every member.
    params = xarray.Dataset(
        {"theta": (("ensemble",), jnp.zeros(64))}, coords={"ensemble": np.arange(64)}
    )
    evolved = evolution.evolve(params, jax.random.PRNGKey(0))
    assert set(evolved.data_vars) == {"theta"}
    assert len(np.unique(np.asarray(evolved["theta"].values))) == 64


def test_moving_window_scalars_compose_and_leave_the_smoother_alone(
    compose_test_cfg: Any,
) -> None:
    """``num_windows`` / ``window_shift`` are the SCRIPT's knobs, not the class's.

    ``FilterSmoothingESMDA`` assimilates one window; the loop lives in the
    orchestrator. Passing either scalar into the ``smoother:`` node would make
    every composed config die at instantiation on an unexpected keyword — which
    the compose-only tests could not see.
    """
    from hydra.utils import instantiate

    cfg = compose_test_cfg(
        _window_overrides(2, 2, 3, 2), config_name="run_filter_smoothing"
    )

    assert int(cfg.filter_smoothing.num_windows) == 3
    assert int(cfg.filter_smoothing.window_shift) == 2
    assert "num_windows" not in cfg.filter_smoothing.smoother
    assert "window_shift" not in cfg.filter_smoothing.smoother
    assert "evolution" not in cfg.filter_smoothing.smoother

    smoother = instantiate(
        cfg.filter_smoothing.smoother,
        observation_operator=_stub_observation_operator,
        forward_model=_StubEnsembleModel(),
        C_D=jnp.array([0.1]),
    )
    assert type(smoother).__name__ == "FilterSmoothingESMDA"


@pytest.mark.parametrize(  # type: ignore[misc]
    "override,message",
    [
        ("filter_smoothing.num_windows=0", "num_windows"),
        ("filter_smoothing.window_shift=0", "window_shift"),
        # L = 2 here, so a shift of 3 would leave a cycle of the horizon
        # unassimilated between consecutive windows.
        ("filter_smoothing.window_shift=3", "window_shift"),
    ],
)
def test_run_filter_smoothing_rejects_invalid_window_arithmetic(
    override: str, message: str, compose_test_cfg: Any
) -> None:
    """The script validates the window arithmetic before it simulates anything.

    A multi-window run costs ``num_windows * (num_steps + 1) * L`` segments per
    member plus a ``T``-cycle truth; discovering ``window_shift=0`` after the
    truth simulation would throw away hours.
    """
    from scripts.filter_smoothing.run_filter_smoothing import run

    cfg = compose_test_cfg(
        _window_overrides(2, 2, 2, 1, [override]),
        config_name="run_filter_smoothing",
    )
    with pytest.raises(ValueError, match=message):
        run(cfg)


def test_run_filter_smoothing_moving_window(compose_test_cfg: Any) -> None:
    """Two overlapping windows end to end: the assembled full-horizon posterior.

    The solver-backed counterpart of tests/test_run_filter_smoothing.py's
    single-window smoke test, at the same tiny shape and under the same serial
    e2e conventions: 2 windows x (2 ESMDA iterations + a final pass) x 2 cycles.
    At 2 members nothing about assimilation skill can be read off the result —
    this pins the loop's plumbing: a truth simulated over the full ``T``-cycle
    horizon while the prior spans one window, the per-window diagnostics
    artifact, and the length of the assembled trajectory.
    """
    import xarray as xr

    from scripts.esmda._esmda_common import read_yaml
    from scripts.filter_smoothing.run_filter_smoothing import run

    num_cycles, num_steps, num_windows, window_shift = 2, 2, 2, 1
    total_cycles = num_cycles + (num_windows - 1) * window_shift
    cfg = compose_test_cfg(
        _window_overrides(num_cycles, num_steps, num_windows, window_shift),
        config_name="run_filter_smoothing",
    )
    run(cfg)

    out_dir = pathlib.Path(cfg.paths.results_dir)
    for artifact in (
        "posterior_params.nc",
        "posterior_state.nc",
        "prior_params.nc",
        "true_params.nc",
        "iteration_diagnostics.yaml",
        "cycle_diagnostics.yaml",
        "truth_access.yaml",
        "run_info.yaml",
        # Multi-window only: the single-window path keeps the Phase 1 set.
        "window_diagnostics.yaml",
    ):
        assert (out_dir / artifact).exists(), f"missing artifact: {artifact}"

    prior_params = xr.open_dataset(out_dir / "prior_params.nc")
    posterior_params = xr.open_dataset(out_dir / "posterior_params.nc")
    assert "time" in posterior_params["inflow_angle"].dims
    assert posterior_params.sizes["ensemble"] == 2
    # The prior spans ONE window; the posterior spans the horizon. Asserted as
    # the invariant "one window's knots, plus s per advance" rather than as a
    # literal, so it holds whether or not the sampler emits the trailing knot.
    assert posterior_params.sizes["time"] == (
        prior_params.sizes["time"] + (num_windows - 1) * window_shift
    )
    assert posterior_params.sizes["time"] >= total_cycles
    # One continuous, increasing knot grid: the concatenation must not restart
    # the clock at each window.
    knot_times = np.asarray(posterior_params["time"].values)
    assert np.all(np.diff(knot_times) > 0)

    # §8a: the TRUTH is simulated over the full horizon while the prior covers
    # the first window only, so the two knot grids genuinely differ in length.
    true_params = xr.open_dataset(out_dir / "true_params.nc")
    assert true_params.sizes["time"] > prior_params.sizes["time"]

    posterior_state = xr.open_dataset(out_dir / "posterior_state.nc")
    assert posterior_state.sizes["ensemble"] == 2

    # One record per window; the last window's cycles are the horizon's last.
    window_diagnostics = read_yaml(out_dir / "window_diagnostics.yaml")
    assert len(window_diagnostics) == num_windows

    # The per-iteration / per-cycle diagnostics describe the LAST window, so
    # their counts are unchanged from the single-window run.
    assert len(read_yaml(out_dir / "iteration_diagnostics.yaml")) == num_steps
    assert len(read_yaml(out_dir / "cycle_diagnostics.yaml")) == num_cycles

    configuration = read_yaml(out_dir / "run_info.yaml")["configuration"]
    assert configuration["num_windows"] == num_windows
    assert configuration["window_shift"] == window_shift
    assert configuration["total_cycles"] == total_cycles
    assert configuration["num_cycles"] == num_cycles
    assert configuration["num_steps"] == num_steps
