"""Windowed parameter-trajectory ESMDA with an inner sequential state filter.

The hybrid of ``smoothing/`` and ``filtering/`` described in
``docs/filter_smoothing/parameter_trajectory_esmda_state_filter.pdf``: the
high-dimensional state is **never smoothed** — it is filtered cycle by cycle,
exactly as :class:`~data_assimilation.filtering.base.EnsembleKalmanFilter`
does — while the low-dimensional **parameter trajectory**
``Theta = [theta_0 ... theta_{L-1}]`` over a window of ``L`` cycles is updated
*jointly* by an outer ESMDA loop.

One outer iteration (paper Algorithm 1):

1. reset the state ensemble to the window's initial ensemble and run one inner
   filter pass over the window, forecasting cycle ``k`` with the *restriction of
   the trajectory to segment ``k``* — a time-varying schedule the backend
   interpolates natively, not a constant — and storing each cycle's forecast
   observations ``d_k = H(x_f_k)`` *before* its analysis (Eq. 6–7);
2. stack them into ``D`` of shape ``(L*N_d, N_e)`` and the observation batches
   into ``Y`` of shape ``(L*N_d,)`` (Eq. 9–10);
3. apply ONE tempered stochastic Kalman update to the flattened trajectory
   (Eq. 12) — the same :func:`~data_assimilation.filtering.analysis.\
stochastic_enkf_update` the smoother and the filter use, with the ESMDA
   ``alpha``.

Because ``C_{Theta D}`` spans the whole window, an observation at the end of
the window can revise a knot at its start, while no observation ever revises a
past *state*. That asymmetry is the point of the method. A final inner pass
(§5) makes the returned state consistent with the final trajectory.

**Deviation from the paper, deliberate.** The paper's ``theta_k`` is
piecewise-constant over segment ``k`` (Eq. 6). Here the knot grid is
independent of the cycle grid, and each segment's forecast receives the
trajectory *restricted to that segment* — its endpoints linearly interpolated
between the bracketing knots, every knot strictly inside it carried through —
which the backends already consume (``uvel_time.dat`` for pylbm, the nudging
schedule for pyudales, both interpolated by the Fortran). That matches how the
truth trajectory is generated: the solver interpolates it, so a
piecewise-constant estimate would be fitting a different model of the forcing
than the one that produced the data. It also decouples the trajectory's
resolution (``time.seconds_per_knot``) from the assimilation cycle length.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import xarray
from data_assimilation.augmentation import ParamAugmentation
from data_assimilation.filter_smoothing.temporal_localization import (
    TemporalLocalization,
)
from data_assimilation.filtering.analysis import AnalysisScheme, stochastic_enkf_update
from data_assimilation.filtering.base import EnsembleKalmanFilter, FilterResult
from data_assimilation.inflation import InflationScheme
from data_assimilation.localization.base import BaseLocalization
from data_assimilation.observation_operator import (
    AggregateObservations,
    flatten_observations,
)

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

logger = logging.getLogger(__name__)

#: What :meth:`FilterSmoothingESMDA.run` accepts as observations: one
#: time-resolved ``("time", "obs")`` DataArray per cycle (aggregated and
#: flattened by the estimator itself), or the flat ``(num_cycles, N_d)`` array
#: those reduce to.
ObservationInput = Union[Sequence[xarray.DataArray], np.ndarray, jnp.ndarray]

# Relative slack of every knot-time comparison (is this knot strictly inside the
# segment?). The knot axis is built in float32 by the samplers, so an exact
# comparison against a float64 segment boundary would trip on representation
# alone.
_TIME_RTOL = 1e-6


def _time_tol(scale: float) -> float:
    """Absolute slack for a time comparison at the given scale."""
    return _TIME_RTOL * max(abs(float(scale)), 1.0)


def knot_times(params: xarray.Dataset) -> np.ndarray:
    """The trajectory's knot times, in PHYSICAL SECONDS on its own clock.

    The knot grid is a free choice (``time.seconds_per_knot``), independent of
    the cycle length, so the segment restriction below cannot infer the knots'
    positions from their index — it reads them off the ``time`` coordinate the
    samplers emit (``build_knot_times``). A ``time`` dimension carrying no
    coordinate values is therefore an error rather than something to fill in.
    """
    if "time" not in params.coords:
        raise ValueError(
            "The parameter trajectory has a 'time' dimension but no 'time' "
            "coordinate. Filter smoothing reads the knot times as physical "
            "seconds (the samplers set them from time.seconds_per_knot); "
            "without them a segment cannot be located on the trajectory."
        )
    times = np.asarray(params.coords["time"].values, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError(
            f"The trajectory's 'time' coordinate has shape {times.shape}; a "
            "1-D, non-empty knot axis is required."
        )
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError(
            "The trajectory's knot times are not strictly increasing: "
            f"{np.asarray(times).tolist()}. The segment restriction "
            "interpolates between consecutive knots, which needs an ordered "
            "grid."
        )
    return times


def _interpolate_knots(
    times: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """``np.interp`` of a time-leading array, broadcast over the trailing dims.

    ``values`` is ``(n_knots, ...)`` (typically ``(n_knots, N_e)`` — one
    trajectory per member), ``targets`` a 1-D array of times. Outside the knot
    range the value is CLAMPED to the nearest end knot, exactly as
    :func:`numpy.interp` does: a segment past the last knot holds it, and a
    single-knot trajectory is constant everywhere.
    """
    if times.size == 1:
        return np.repeat(values[:1], targets.size, axis=0)
    upper = np.clip(np.searchsorted(times, targets, side="left"), 1, times.size - 1)
    lower = upper - 1
    span = times[upper] - times[lower]
    weight = np.clip((targets - times[lower]) / span, 0.0, 1.0)
    weight = weight.reshape((targets.size,) + (1,) * (values.ndim - 1))
    return values[lower] * (1.0 - weight) + values[upper] * weight


@dataclass
class IterationDiagnostics:
    """Health indicators of one outer ESMDA iteration.

    All quantities are measured on the *stacked window system* — the
    ``(L*N_d,)`` observation vector and the ``(L*N_d, N_e)`` forecast
    observations the trajectory update actually consumes — so they say whether
    the window as a whole is being fitted, not whether one cycle is:

    * ``obs_rmse``: observation-space RMSE of the ensemble-mean forecast
      observations against the stacked observations, *before* this iteration's
      update. It is the iteration's cost function: it should fall over the
      ESMDA iterations (the next iteration re-runs the window with the updated
      trajectory).
    * ``innovation_chi2``: ``d^T (C_DD + C_D)^{-1} d / (L*N_d)`` with
      ``d = Y - mean(D)``. Same convention as
      :class:`~data_assimilation.filtering.base.CycleDiagnostics` — untempered
      (no ``alpha``), so it keeps meaning "is the ensemble spread consistent
      with the actual errors?" rather than measuring the tempering.
    * ``param_spread_prior`` / ``param_spread_posterior``: mean per-row
      ensemble standard deviation of the trajectory rows before/after the
      update, over the rows the update touches (static parameters are
      excluded, see :class:`FilterSmoothingESMDA`). A collapsing spread across
      iterations is the classic ESMDA over-fitting signature.
    """

    iteration: int
    alpha: float
    obs_rmse: float
    innovation_chi2: float
    param_spread_prior: float
    param_spread_posterior: float
    update_time: Optional[float] = None


@dataclass
class FilterSmoothingResult:
    """Return value of :meth:`FilterSmoothingESMDA.run`.

    ``params`` is the smoothed trajectory ensemble ``Theta^{N_a}`` (Eq. 15) and
    ``state`` the filtered end-of-window state ensemble (Eq. 16) — the latter
    always produced by the *final consistency pass*, never by an iteration's
    inner pass. ``final_pass`` is that pass's full
    :class:`~data_assimilation.filtering.base.FilterResult`, so its per-cycle
    diagnostics (and its histories, when ``return_history``) are available
    unchanged. ``params_history`` holds the per-iteration trajectories
    concatenated along ``esmda_step`` (entry 0 the prior, ``num_steps + 1``
    entries), present only when ``return_history=True``.
    """

    params: xarray.Dataset
    state: Optional[xarray.Dataset]
    iteration_diagnostics: list[IterationDiagnostics]
    final_pass: FilterResult
    params_history: Optional[xarray.Dataset] = None


class _TrajectoryStateFilter(EnsembleKalmanFilter):
    """State-only EnKF whose cycle-``k`` forecast uses SEGMENT ``k`` of a trajectory.

    The only difference from the plain EnKF is
    :meth:`~data_assimilation.filtering.base.BaseFilter._params_for_cycle`:
    the parameter Dataset it carries is a *trajectory* over the window, and each
    cycle forecasts with the piece of it that spans that cycle's segment —
    ``[k*cycle_length, (k+1)*cycle_length]`` on the trajectory's clock — handed
    to the backend as a time-varying schedule on a segment-local axis. The
    backends interpolate such a schedule natively (pylbm writes it to
    ``uvel_time.dat``, pyudales feeds it through nudging), which is the same code
    path dynamic-mode ESMDA exercises, so the knot grid is free to be coarser or
    finer than the cycle.

    ``mode="state"`` (enforced by :class:`FilterSmoothingESMDA`) is what makes
    this legal: the parameter block never enters the inner analysis, so the
    trajectory rides through the pass unmodified — it is the *outer* loop's
    control vector — and ``_check_static_params`` (which rejects time-varying
    parameters) is skipped.

    ``cycle_length`` is the number of seconds one cycle's forecast segment
    spans; without it the cycle index cannot be placed on the trajectory's
    (physical, seconds-valued) knot axis at all.

    The second difference is the observation space. The plain filter assimilates
    every observation frame of a segment serially, one analysis each; this
    hybrid instead assimilates ONE batch per cycle — the paper's ``d_k`` — so
    :meth:`_prepare_pred_obs` aggregates (when an aggregator is configured) and
    flattens the operator's time-resolved output into a single frame. The real
    observations arrive already flat from
    :meth:`FilterSmoothingESMDA.observation_batches`, so both halves of the
    innovation live in that same aggregated space and every inner cycle runs the
    filter's ``T = 1`` path.
    """

    def __init__(
        self,
        *args: Any,
        cycle_length: Optional[float] = None,
        aggregate_observations: Optional[AggregateObservations] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cycle_length: Optional[float] = (
            None if cycle_length is None else float(cycle_length)
        )
        #: Interval aggregator of the PREDICTED observations (the real ones are
        #: aggregated once, by the outer estimator). ``None`` -> the full
        #: time-resolved vector, still as one batch per cycle.
        self.aggregate_observations = aggregate_observations

    def _prepare_pred_obs(self, pred_obs: Any) -> jnp.ndarray:
        """One cycle's predicted observations as a single ``(N_e, 1, N_d)`` frame.

        Overrides the filter's per-frame normalization: a labelled
        ``("ensemble", "time", "obs")`` DataArray is optionally aggregated and
        then flattened time-major — the historical ``[interval0 block,
        interval1 block, ...]`` layout the stacked window system ``D`` is built
        from — and presented to the cycle loop as the one frame it assimilates.
        Anything the operator already returns as an array is left to the base
        implementation.
        """
        if not isinstance(pred_obs, xarray.DataArray):
            return super()._prepare_pred_obs(pred_obs)
        if self.aggregate_observations is not None:
            pred_obs = self.aggregate_observations(pred_obs)
        flat = jnp.asarray(flatten_observations(pred_obs))  # (N_e, N_d)
        return flat[:, None, :]

    def _params_for_cycle(
        self,
        cycle: int,
        params: Optional[xarray.Dataset],
    ) -> Optional[xarray.Dataset]:
        if params is None or "time" not in params.dims:
            return params
        if self.cycle_length is None:
            raise ValueError(
                "The inner filter carries a parameter TRAJECTORY but no "
                "cycle_length, so segment k cannot be located on the knot axis. "
                "Pass cycle_length= to FilterSmoothingESMDA (the run script "
                "ties it to time.simulation_time)."
            )
        cycle_length = self.cycle_length
        times = knot_times(params)

        # Segment k spans [k*dt, (k+1)*dt] on the trajectory's own clock. Its
        # local axis is what the backend sees: the schedule is call-relative
        # (pylbm shifts it onto its continuing nt0 clock itself), so it always
        # runs [0, cycle_length].
        start = cycle * cycle_length
        end = start + cycle_length
        tol = _time_tol(cycle_length)
        # Strictly inside: a knot sitting ON either boundary is already there as
        # the interpolated endpoint, and repeating it would put a duplicate time
        # in the backend's schedule.
        interior = times[(times > start + tol) & (times < end - tol)]
        targets = np.concatenate(([start], interior, [end]))
        local = np.concatenate(([0.0], interior - start, [cycle_length]))

        data_vars: dict[str, xarray.DataArray] = {}
        for name, variable in params.data_vars.items():
            if "time" not in variable.dims:
                # Static parameters pass through unchanged (and keep their own
                # coordinates); they are not part of the trajectory.
                data_vars[str(name)] = variable
                continue
            ordered = variable.transpose("time", ...)
            segment = _interpolate_knots(
                times, np.asarray(ordered.values), targets
            ).astype(variable.dtype, copy=False)
            data_vars[str(name)] = xarray.DataArray(
                segment,
                dims=ordered.dims,
                coords={
                    str(dim): params.coords[dim]
                    for dim in ordered.dims
                    if dim != "time" and dim in params.coords
                },
            ).transpose(*variable.dims)

        return xarray.Dataset(data_vars, attrs=params.attrs).assign_coords(time=local)


class FilterSmoothingESMDA:
    """Outer ESMDA over the parameter trajectory, inner EnKF over the state.

    Composition rather than inheritance from ``_BaseESMDA``: the smoother's
    loop re-forecasts the whole window from an updated initial condition, which
    is precisely what this method replaces with an inner filter pass.

    Args:
        observation_operator: Maps one cycle's forecast segment to that cycle's
            predicted observations — the *same* per-segment operator the
            sequential filter takes (typically a
            ``TemporalObservationOperator``, whose time-resolved output the
            inner filter aggregates and flattens).
        forward_model: Any ensemble forward model; its configured horizon is
            one cycle's forecast segment.
        C_D: Diagonal observation-error covariance of ONE cycle's batch, as a
            1-D variance vector ``(N_d,)`` (a square diagonal matrix is
            accepted). The stacked window covariance is
            ``tile(C_D, L)`` — the paper's block-diagonal ``R_W``.
        cycle_length: Seconds one inner cycle's forecast segment spans. It is
            what places cycle ``k`` on the trajectory's physical knot axis, and
            the unit the temporal localization's coordinates are expressed in.
            ``None`` (default) reads ``forward_model.simulation_time``; if that
            is absent too, a time-varying ``params`` is rejected at
            :meth:`run`.
        num_steps: Number of outer ESMDA iterations ``N_a``.
        alpha: Tempering weight of each iteration. ``None`` (default) means
            ``num_steps``, the equal-weight schedule ``sum_a 1/alpha_a = 1``
            (Eq. 5); a scalar override must satisfy the same identity.
        inner_analysis: Analysis scheme of the inner state filter (default
            :class:`~data_assimilation.filtering.analysis.StochasticEnKFAnalysis`;
            the ETKF/LETKF schemes work unchanged).
        inner_localization: Optional localization of the inner *state*
            analysis (distance/correlation). Not the trajectory update's — see
            ``temporal_localization``.
        inner_inflation: Optional spread maintenance for the inner state
            filter (RTPS, ...).
        temporal_localization: Optional localization of the OUTER trajectory
            update, tapering each knot's update by its time distance to each
            observation (paper Eq. 17). ``None`` -> the global update.
        common_inner_noise: When True (default), every inner pass of a window
            reuses one PRNG key, so the map ``Theta -> D`` is deterministic up
            to the trajectory itself and the ESMDA cross-covariances are not
            diluted by fresh Monte Carlo noise between iterations (§9, common
            random numbers). Set False for independent draws per iteration.
            The *outer* perturbed-observation draws always use fresh subkeys.
        rng_key: PRNG key; defaults to a fresh ``PRNGKey(42)`` per instance.
        aggregate_observations: Optional interval aggregator applied to BOTH
            the real observations (here, once per cycle, in
            :meth:`observation_batches`) and the predicted ones (inside the
            inner filter, which is handed this same object, so it aggregates
            what its own operator produced). ``None`` (default) assimilates the
            full time-resolved observation vector.

    Static (no-``time``) parameters in the trajectory Dataset are carried
    through every forecast unchanged and **excluded from the outer update**:
    they flatten to a single scalar row each, which is masked out of the update
    and restored afterwards. Estimating them jointly with the trajectory is a
    mask away, but it is a different inference problem (one value informed by
    the whole window) and is deliberately not enabled in this phase.
    """

    def __init__(
        self,
        observation_operator: Callable[[xarray.Dataset], Any],
        forward_model: BaseEnsembleForwardModel,
        C_D: jnp.ndarray,
        cycle_length: Optional[float] = None,
        num_steps: int = 4,
        alpha: Optional[float] = None,
        inner_analysis: Optional[AnalysisScheme] = None,
        inner_localization: Optional[BaseLocalization] = None,
        inner_inflation: Optional[InflationScheme] = None,
        temporal_localization: Optional[TemporalLocalization] = None,
        common_inner_noise: bool = True,
        rng_key: Optional[jax.Array] = None,
        aggregate_observations: Optional[AggregateObservations] = None,
    ) -> None:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}.")

        # ES-MDA consistency, same check (and same default) as ``_BaseESMDA``:
        # the tempering coefficients must satisfy ``sum_a 1/alpha_a = 1`` for
        # the N_a updates to equal one Bayesian conditioning (Emerick &
        # Reynolds 2013). A mismatched alpha silently tempers the likelihood by
        # ``num_steps / alpha`` -- a different inference problem.
        effective_alpha = num_steps if alpha is None else alpha
        if abs(num_steps / effective_alpha - 1.0) > 1e-6:
            raise ValueError(
                f"Inconsistent ES-MDA schedule: num_steps={num_steps} and "
                f"alpha={effective_alpha} give sum_a 1/alpha_a = "
                f"{num_steps / effective_alpha:.4g} != 1. Set alpha=num_steps "
                "(the default) or leave alpha unset."
            )

        # The temporal taper is meaningless inside the state filter: its rows
        # are grid cells, and the plumbing there fills ``row_coords`` with
        # PHYSICAL coordinates, whose first component is x -- so a
        # TemporalLocalization would silently taper by |x - t|.
        if isinstance(inner_localization, TemporalLocalization):
            raise ValueError(
                "TemporalLocalization localizes the outer trajectory update, "
                "not the inner state analysis (whose rows are grid cells with "
                "physical coordinates). Pass it as temporal_localization= and "
                "use a distance/correlation strategy for inner_localization."
            )

        # The cycle length is the trajectory clock's unit: the forward model
        # already knows how long one of its runs is, so an explicit argument is
        # only needed when the two differ (or when the model does not expose
        # it). Resolved here, but a missing value is only fatal for a
        # time-varying params Dataset — which run() is what sees.
        resolved_cycle_length = (
            cycle_length
            if cycle_length is not None
            else getattr(forward_model, "simulation_time", None)
        )
        if resolved_cycle_length is not None and float(resolved_cycle_length) <= 0.0:
            raise ValueError(
                f"cycle_length must be > 0, got {float(resolved_cycle_length)}: it "
                "is the number of seconds one cycle's forecast segment spans."
            )

        self.observation_operator = observation_operator
        self.forward_model = forward_model
        #: Seconds one cycle's forecast segment spans (``None`` when unknown).
        self.cycle_length: Optional[float] = (
            None if resolved_cycle_length is None else float(resolved_cycle_length)
        )
        self.num_steps = num_steps
        self.alpha = num_steps if alpha is None else alpha
        self.temporal_localization = temporal_localization
        self.common_inner_noise = common_inner_noise
        #: Interval aggregator of the observation space (``None`` -> the full
        #: time-resolved vector). Shared with the inner filter below.
        self.aggregate_observations = aggregate_observations

        # Default the PRNG key here (not in the signature): a default argument
        # would be evaluated at import time -- initializing the JAX backend as
        # a side effect and sharing one key across every instance built without
        # an explicit one.
        self.rng_key = jax.random.PRNGKey(42) if rng_key is None else rng_key

        # The inner pass is a stock filter: cycle loop, augmentation,
        # inflation, localization plumbing, failure substitution, on-disk
        # cycle_{k}/ management and per-cycle diagnostics all come for free,
        # and every analysis-scheme/localization validation happens in its
        # constructor rather than inside the first iteration.
        self._inner = _TrajectoryStateFilter(
            observation_operator=observation_operator,
            forward_model=forward_model,
            C_D=C_D,
            analysis=inner_analysis,
            mode="state",
            localization=inner_localization,
            inflation=inner_inflation,
            cycle_length=self.cycle_length,
            # The PREDICTED observations are aggregated where they are produced
            # (the inner filter's _prepare_pred_obs override, which is why the
            # aggregator lives on _TrajectoryStateFilter rather than on the
            # plain filter -- that one assimilates every frame serially and
            # aggregates nothing); the real ones are aggregated once by this
            # class before the batches are handed down already flat -- see
            # observation_batches.
            aggregate_observations=aggregate_observations,
        )
        # The whole method rests on this: ``D`` must be the forecast
        # observations stored before each analysis (§9, first bullet).
        self._inner.collect_pred_obs = True
        #: One cycle's observation-error variances, after the filter's own
        #: validation/1-D reduction of ``C_D``.
        self.C_D_diag = self._inner.C_D_diag

    @property
    def inner_filter(self) -> _TrajectoryStateFilter:
        """The inner state filter, for callers that configure it further.

        The on-disk knobs (``prune_disk_cycles`` / ``keep_first_disk_cycle``)
        and the per-cycle diagnostics live on it; a run script sets them here
        the same way it sets them on a standalone filter.
        """
        return self._inner

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self, observations: Any) -> jnp.ndarray:
        """One cycle's observations as the flat vector the update consumes.

        A plain array is returned as it is: pre-flattened input is the caller's
        responsibility (and is what the inner filter receives, see
        :meth:`observation_batches`). A time-resolved ``("time", "obs")``
        DataArray goes through the aggregator — the same one the predicted
        observations go through — and is flattened time-major.
        """
        if not isinstance(observations, xarray.DataArray):
            return jnp.asarray(observations)
        if self.aggregate_observations is not None:
            observations = self.aggregate_observations(observations)
        return jnp.asarray(flatten_observations(observations))

    def observation_batches(self, observations: Any) -> jnp.ndarray:
        """``(num_cycles, N_d)`` flat batches from either accepted input form.

        Takes an :data:`ObservationInput` — typed loosely so the guards below
        can reject what a caller might plausibly pass instead.

        The real observations are aggregated HERE, once per cycle, and the
        inner filter is then handed the resulting flat array — which its own
        ``_get_observations`` passes straight through. So a cycle's
        observations are aggregated exactly once no matter how many inner
        passes the outer loop runs, while the inner filter still aggregates the
        predicted observations *it* produces (a different array every pass).

        Public because the moving-window orchestrator normalizes the horizon's
        batches once, up front, before slicing windows out of them.
        """
        if isinstance(observations, xarray.DataArray):
            raise ValueError(
                "observations is a single DataArray; filter smoothing consumes "
                "one batch PER CYCLE. Pass a list/tuple of per-cycle "
                "('time', 'obs') DataArrays, or the flat (num_cycles, N_d) "
                "array they reduce to."
            )
        if isinstance(observations, (list, tuple)):
            if not observations:
                raise ValueError("observations must contain at least one cycle.")
            batches = jnp.stack(
                [self._get_observations(cycle) for cycle in observations]
            )
        else:
            batches = jnp.asarray(observations)
        if batches.ndim != 2:
            raise ValueError(
                "observations must have shape (num_cycles, N_d) — one batch "
                f"per cycle — got shape {batches.shape}."
            )
        return batches

    # ------------------------------------------------------------------
    # The outer loop
    # ------------------------------------------------------------------

    def run(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        observations: Optional[ObservationInput] = None,
        *,
        return_history: bool = False,
    ) -> FilterSmoothingResult:
        """Assimilate one window: ``num_steps`` iterations plus a final pass.

        Args:
            state: The window's initial state ensemble. EVERY iteration starts
                from this exact ensemble (paper §4.4: the state trajectories of
                previous passes are auxiliary latent variables and are
                discarded). ``None`` is a legal cold start — each iteration
                then cold-starts identically.
            params: The prior trajectory ensemble: a Dataset with ``time``
                (knots, in seconds) and ``ensemble`` dims, plus optional static
                (no-``time``) variables. The knot grid is free — coarser or
                finer than the cycle length, and of any length from one knot up
                — because each cycle forecasts with the trajectory restricted
                to its segment rather than with one knot. Knots outside the
                window's span (the trailing knot a windowed prior sampler
                emits) ride along in the update through prior temporal
                correlations only.
            observations: One batch per cycle, exactly as the sequential filter
                consumes them: either a list/tuple of time-resolved
                ``("time", "obs")`` DataArrays (aggregated by
                ``aggregate_observations`` and flattened here) or the flat
                ``(num_cycles, N_d)`` array they reduce to.
            return_history: Collect the per-iteration trajectories into
                ``params_history`` and pass through to the final inner pass's
                own histories.

        Returns:
            A :class:`FilterSmoothingResult`.
        """
        if params is None:
            raise ValueError(
                "params must be provided: the parameter trajectory ensemble is "
                "this method's control vector."
            )
        if observations is None:
            raise ValueError("observations must be provided.")
        # Aggregated (if configured) and flattened once, at entry: everything
        # below — the inner passes included — sees flat per-cycle batches.
        obs_batches = self.observation_batches(observations)
        num_cycles, n_d = (int(obs_batches.shape[0]), int(obs_batches.shape[1]))
        if n_d != int(self.C_D_diag.shape[0]):
            raise ValueError(
                f"Observation batches have N_d={n_d} but C_D has "
                f"{int(self.C_D_diag.shape[0])} variances."
            )
        if "time" not in params.dims:
            raise ValueError(
                "params has no 'time' dimension: filter smoothing estimates a "
                "parameter TRAJECTORY over the window. Sample a dynamic "
                "(time-varying) prior, or use the sequential filter for static "
                "parameters."
            )
        if self.cycle_length is None:
            raise ValueError(
                "params is a time-varying trajectory but the smoother has no "
                "cycle_length, so cycle k cannot be placed on the knot axis "
                "(the knots are physical seconds, the cycles are segments of "
                "cycle_length seconds). Pass cycle_length= explicitly, or a "
                "forward model exposing simulation_time."
            )
        cycle_length = self.cycle_length
        num_knots = int(params.sizes["time"])
        if num_knots < 1:
            raise ValueError("The trajectory prior has no knots to estimate.")
        # Read (and validate) the knot times once: the row layout places them on
        # the cycle clock, and the inner filter restricts them per segment.
        times = knot_times(params)

        # Cycle-major stacking of the window system (Eq. 9-10): observation j
        # of cycle k is row ``k * N_d + j`` of Y, of D, and of C_D_stacked.
        Y = obs_batches.reshape(num_cycles * n_d)
        C_D_stacked = jnp.tile(self.C_D_diag, num_cycles)

        # num_time_points comes from the sampled prior, never from a config
        # literal (same pattern as run_esmda.py's smoother override).
        augmentation = ParamAugmentation(num_time_points=num_knots)
        update_mask, row_times = self._trajectory_row_layout(
            params, times, cycle_length
        )
        row_coords, obs_coords, group_ids, localize_mask = self._localization_plumbing(
            params, augmentation, update_mask, row_times, num_cycles, n_d
        )

        initial_state = state
        # One inner key per window: with common_inner_noise every pass reuses
        # it, so identical trajectories produce bit-identical D.
        self.rng_key, inner_base_key = jax.random.split(self.rng_key)

        iteration_diagnostics: list[IterationDiagnostics] = []
        params_history: list[xarray.Dataset] = [params] if return_history else []

        for iteration in range(self.num_steps):
            self._inner.rng_key = self._inner_key(inner_base_key, iteration)
            # Reset to the SAME initial ensemble every iteration; the pass's
            # own analyzed states are discarded (only its forecast
            # observations survive).
            self._inner.run(
                state=initial_state, params=params, observations=obs_batches
            )
            D = self._stacked_pred_obs(num_cycles, n_d)

            update_started = time.perf_counter()
            flat_params = augmentation.flatten(params)
            theta = ParamAugmentation.to_array(flat_params)

            self.rng_key, subkey = jax.random.split(self.rng_key)
            updated = stochastic_enkf_update(
                augmented=theta,
                pred_obs=D,
                obs=Y,
                C_D_diag=C_D_stacked,
                rng_key=subkey,
                alpha=self.alpha,
                localization=self.temporal_localization,
                group_ids=group_ids,
                localize_mask=localize_mask,
                row_coords=row_coords,
                obs_coords=obs_coords,
            )
            # Static parameters ride along through the forecasts but are not
            # estimated in this phase: restore their prior rows unchanged.
            updated = jnp.where(update_mask[:, None], updated, theta)

            params = augmentation.unflatten(
                ParamAugmentation.from_array(updated, flat_params), params
            )
            diagnostics = self._iteration_diagnostics(
                iteration=iteration,
                D=D,
                Y=Y,
                C_D_stacked=C_D_stacked,
                theta_prior=theta,
                theta_posterior=updated,
                update_mask=update_mask,
            )
            diagnostics.update_time = time.perf_counter() - update_started
            iteration_diagnostics.append(diagnostics)

            if return_history:
                params_history.append(params)

            logger.info(
                "Filter-smoothing ESMDA iteration %d completed (window obs "
                "RMSE %.4g)",
                iteration,
                diagnostics.obs_rmse,
            )

        # Final consistency pass (§5): the states produced during the last
        # iteration were forecast with the PRE-update trajectory, so the
        # returned state must come from one more pass run with Theta^{N_a}.
        self._inner.rng_key = self._inner_key(inner_base_key, self.num_steps)
        final_pass = self._inner.run(
            state=initial_state,
            params=params,
            observations=obs_batches,
            return_history=return_history,
        )

        return FilterSmoothingResult(
            params=params,
            state=final_pass.state,
            iteration_diagnostics=iteration_diagnostics,
            final_pass=final_pass,
            params_history=(
                xarray.concat(params_history, dim="esmda_step", join="override")
                if params_history
                else None
            ),
        )

    def _inner_key(self, base_key: jax.Array, iteration: int) -> jax.Array:
        """The inner pass's PRNG key for one outer iteration.

        Common random numbers (§9, last bullet): with ``common_inner_noise``
        every pass of the window — including the final consistency pass —
        reuses one key, so the inner stochastic analysis contributes the same
        realization each time and the differences in ``D`` between iterations
        are caused by the trajectory alone. Otherwise each pass folds its index
        into the key for an independent draw.
        """
        if self.common_inner_noise:
            return base_key
        return jax.random.fold_in(base_key, iteration)

    def _stacked_pred_obs(self, num_cycles: int, n_d: int) -> jnp.ndarray:
        """Stack the inner pass's recorded forecast observations, ``(L*N_d, N_e)``."""
        history = self._inner.pred_obs_history
        if len(history) != num_cycles:
            raise RuntimeError(
                f"The inner filter recorded {len(history)} forecast-observation "
                f"blocks for {num_cycles} cycles. This means collect_pred_obs "
                "was cleared on the inner filter."
            )
        D = jnp.concatenate([jnp.asarray(block) for block in history], axis=0)
        if D.shape[0] != num_cycles * n_d:
            raise ValueError(
                f"Stacked forecast observations have {D.shape[0]} rows, "
                f"expected num_cycles * N_d = {num_cycles * n_d}. The "
                "observation operator's output length must match the "
                "observation batches."
            )
        return D

    # ------------------------------------------------------------------
    # Row layout of the flattened trajectory
    # ------------------------------------------------------------------

    def _trajectory_row_layout(
        self,
        params: xarray.Dataset,
        knot_seconds: np.ndarray,
        cycle_length: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """``(update_mask, row_times)`` for the flattened trajectory rows.

        Mirrors :meth:`~data_assimilation.augmentation.ParamAugmentation.\
flatten`'s ordering exactly — for each variable in Dataset order, one row per
        knot if it has a ``time`` dimension, otherwise one — and labels each
        row with

        * whether the outer update may move it (static parameters: no, see the
          class docstring), and
        * the knot's time, in units of CYCLES: ``t_j = knot_seconds[j] /
          cycle_length``, which is FRACTIONAL for a knot grid that is not the
          cycle grid (a knot half a cycle in sits at 0.5). With one knot per
          cycle it is exactly the knot index, as it always was. Static rows are
          masked out of the localized update, so their (unused) time is 0.
        """
        mask: list[bool] = []
        times: list[float] = []
        cycle_coords = [float(t) / cycle_length for t in knot_seconds]
        for name in params.data_vars:
            if "time" in params[name].dims:
                mask.extend([True] * len(cycle_coords))
                times.extend(cycle_coords)
            else:
                mask.append(False)
                times.append(0.0)
        if not any(mask):
            raise ValueError(
                "None of the parameters vary in time, so there is no "
                "trajectory to smooth. Use the sequential filter "
                "(mode='parameter'/'joint') for static parameters."
            )
        return jnp.asarray(mask, dtype=bool), jnp.asarray(times, dtype=float)

    def _localization_plumbing(
        self,
        params: xarray.Dataset,
        augmentation: ParamAugmentation,
        update_mask: jnp.ndarray,
        row_times: jnp.ndarray,
        num_cycles: int,
        n_d: int,
    ) -> tuple[
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
    ]:
        """(row_coords, obs_coords, group_ids, localize_mask) for the update.

        All ``None`` without a temporal localization (the global update ignores
        them). With one, the coordinates carry TIME in component 0 — knot ``j``
        at its own ``knot_time / cycle_length`` and observation batch ``k`` at
        ``t = k + 1`` (segment end, the time its state is observed), both in
        units of cycles, so ``temporal_radius`` is a number of cycles whatever
        the knot spacing is. The remaining two components are zero: this is a
        1-D geometry reusing the ``(N, 3)`` coordinate contract of the spatial
        strategies.

        Consequence of that convention, worth knowing when choosing a radius:
        the knot at the START of segment ``k`` sits one full cycle before the
        observation of that segment, so a ``temporal_radius`` of 1 cycle keeps
        only that batch (at maximum taper) and anything below 1 excludes every
        observation from the knots on the cycle boundaries, leaving them
        untouched. A useful radius spans several cycles. (Placing the
        observations at the segment MIDPOINT instead is a one-line change here
        if the resulting asymmetry ever proves to matter.)

        ``localize_mask`` excludes the static rows (they take the global update
        and are restored afterwards anyway), and ``group_ids`` is supplied only
        when the strategy asks for the block analysis.
        """
        if self.temporal_localization is None:
            return None, None, None, None

        zeros = jnp.zeros_like(row_times)
        row_coords = jnp.stack([row_times, zeros, zeros], axis=1)
        # Cycle-major, matching the stacking of D and Y: the N_d observations
        # of cycle k all share that segment's end time.
        obs_times = jnp.repeat(
            jnp.arange(1, num_cycles + 1, dtype=float), n_d
        )  # (L*N_d,)
        obs_coords = jnp.stack(
            [obs_times, jnp.zeros_like(obs_times), jnp.zeros_like(obs_times)], axis=1
        )

        group_ids = (
            augmentation.group_ids(params)
            if self.temporal_localization.block_grouping
            else None
        )
        return row_coords, obs_coords, group_ids, update_mask

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _iteration_diagnostics(
        self,
        iteration: int,
        D: jnp.ndarray,
        Y: jnp.ndarray,
        C_D_stacked: jnp.ndarray,
        theta_prior: jnp.ndarray,
        theta_posterior: jnp.ndarray,
        update_mask: jnp.ndarray,
    ) -> IterationDiagnostics:
        """Window-level innovation statistics and trajectory spreads.

        See :class:`IterationDiagnostics` for what each field means. ``D`` is
        the raw forecast-observation stack, so the chi2 spread term reflects
        what the inner passes produced.
        """
        n_stacked = int(Y.shape[0])
        N_e = int(D.shape[1])

        innovation = Y - jnp.mean(D, axis=1)
        pred_obs_dev = D - jnp.mean(D, axis=1, keepdims=True)
        C_DD = jnp.dot(pred_obs_dev, pred_obs_dev.T) / (N_e - 1)
        S = C_DD + jnp.diag(C_D_stacked)
        chi2 = float(
            innovation
            @ jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(S), innovation)
            / n_stacked
        )

        # Boolean row selection host-side: these are the (few) trajectory rows,
        # and the mask is a plain layout fact, not a traced quantity.
        updated_rows = np.asarray(update_mask)

        def _spread(theta: jnp.ndarray) -> float:
            return float(jnp.mean(jnp.std(theta[updated_rows], axis=1, ddof=1)))

        return IterationDiagnostics(
            iteration=iteration,
            alpha=float(self.alpha),
            obs_rmse=float(jnp.sqrt(jnp.mean(innovation**2))),
            innovation_chi2=chi2,
            param_spread_prior=_spread(theta_prior),
            param_spread_posterior=_spread(theta_posterior),
        )
