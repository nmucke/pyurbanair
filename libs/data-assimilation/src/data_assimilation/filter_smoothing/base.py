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
   filter pass over the window, forecasting cycle ``k`` with knot ``k`` of the
   current trajectory and storing each cycle's forecast observations
   ``d_k = H(x_f_k)`` *before* its analysis (Eq. 6–7);
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
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

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

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

logger = logging.getLogger(__name__)


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
    """State-only EnKF whose cycle-``k`` forecast uses knot ``k`` of a trajectory.

    The only difference from the plain EnKF is
    :meth:`~data_assimilation.filtering.base.BaseFilter._params_for_cycle`:
    the parameter Dataset it carries is a *trajectory* over the window, and
    each cycle forecasts with its own knot as plain scalar ``(ensemble,)``
    parameters, so no backend ever sees a ``time`` dimension. Piecewise-constant
    ``theta_k`` over segment ``k`` is the paper's discrete dynamics (Eq. 6).

    ``mode="state"`` (enforced by :class:`FilterSmoothingESMDA`) is what makes
    this legal: the parameter block never enters the inner analysis, so the
    trajectory rides through the pass unmodified — it is the *outer* loop's
    control vector — and ``_check_static_params`` (which rejects time-varying
    parameters) is skipped.
    """

    def _params_for_cycle(
        self,
        cycle: int,
        params: Optional[xarray.Dataset],
    ) -> Optional[xarray.Dataset]:
        if params is None or "time" not in params.dims:
            return params
        # Clamp at the final knot: a prior sampled over the window's horizon
        # emits L+1 knots for L cycles (the trailing knot is the next window's
        # leading edge), but a caller that supplies exactly L knots must not
        # index past the end on the last cycle.
        knot = min(cycle, int(params.sizes["time"]) - 1)
        # ``drop=True`` removes the scalar ``time`` coordinate; variables
        # without a ``time`` dimension (static parameters) pass through.
        return params.isel(time=knot, drop=True)


class FilterSmoothingESMDA:
    """Outer ESMDA over the parameter trajectory, inner EnKF over the state.

    Composition rather than inheritance from ``_BaseESMDA``: the smoother's
    loop re-forecasts the whole window from an updated initial condition, which
    is precisely what this method replaces with an inner filter pass.

    Args:
        observation_operator: Maps one cycle's forecast segment to the flat
            predicted-observation vector — the *same* per-segment operator the
            sequential filter takes (typically a
            ``TemporalObservationOperator`` with one interval per segment).
        forward_model: Any ensemble forward model; its configured horizon is
            one cycle's forecast segment.
        C_D: Diagonal observation-error covariance of ONE cycle's batch, as a
            1-D variance vector ``(N_d,)`` (a square diagonal matrix is
            accepted). The stacked window covariance is
            ``tile(C_D, L)`` — the paper's block-diagonal ``R_W``.
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
        num_steps: int = 4,
        alpha: Optional[float] = None,
        inner_analysis: Optional[AnalysisScheme] = None,
        inner_localization: Optional[BaseLocalization] = None,
        inner_inflation: Optional[InflationScheme] = None,
        temporal_localization: Optional[TemporalLocalization] = None,
        common_inner_noise: bool = True,
        rng_key: Optional[jax.Array] = None,
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

        self.observation_operator = observation_operator
        self.forward_model = forward_model
        self.num_steps = num_steps
        self.alpha = num_steps if alpha is None else alpha
        self.temporal_localization = temporal_localization
        self.common_inner_noise = common_inner_noise

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
    # The outer loop
    # ------------------------------------------------------------------

    def run(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        observations: Optional[jnp.ndarray] = None,
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
                (knots) and ``ensemble`` dims, plus optional static
                (no-``time``) variables. Knot ``k`` is the parameter value of
                cycle ``k``; the knot count may exceed the cycle count by the
                trailing knot a windowed prior sampler emits (it rides along in
                the update through prior temporal correlations only).
            observations: Array of shape ``(num_cycles, N_d)`` — one batch per
                cycle, exactly as the sequential filter consumes them.
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
        obs_batches = jnp.asarray(observations)
        if obs_batches.ndim != 2:
            raise ValueError(
                "observations must have shape (num_cycles, N_d) — one batch "
                f"per cycle — got shape {obs_batches.shape}."
            )
        num_cycles, n_d = (int(obs_batches.shape[0]), int(obs_batches.shape[1]))
        if n_d != int(self.C_D_diag.shape[0]):
            raise ValueError(
                f"Observation batches have N_d={n_d} but C_D has "
                f"{int(self.C_D_diag.shape[0])} variances."
            )
        if "time" not in params.dims:
            raise ValueError(
                "params has no 'time' dimension: filter smoothing estimates a "
                "parameter TRAJECTORY over the window (one knot per cycle). "
                "Sample a dynamic (time-varying) prior, or use the sequential "
                "filter for static parameters."
            )
        num_knots = int(params.sizes["time"])
        if num_knots < num_cycles:
            raise ValueError(
                f"The trajectory prior has {num_knots} knots but the window "
                f"has {num_cycles} cycles. Cycles beyond the last knot would "
                "silently reuse it; sample the prior over the full window "
                "(seconds_per_knot = one cycle's simulation_time)."
            )

        # Cycle-major stacking of the window system (Eq. 9-10): observation j
        # of cycle k is row ``k * N_d + j`` of Y, of D, and of C_D_stacked.
        Y = obs_batches.reshape(num_cycles * n_d)
        C_D_stacked = jnp.tile(self.C_D_diag, num_cycles)

        # num_time_points comes from the sampled prior, never from a config
        # literal (same pattern as run_esmda.py's smoother override).
        augmentation = ParamAugmentation(num_time_points=num_knots)
        update_mask, row_times = self._trajectory_row_layout(params, num_knots)
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
        num_knots: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """``(update_mask, row_times)`` for the flattened trajectory rows.

        Mirrors :meth:`~data_assimilation.augmentation.ParamAugmentation.\
flatten`'s ordering exactly — for each variable in Dataset order, ``num_knots``
        rows if it has a ``time`` dimension, otherwise one — and labels each
        row with

        * whether the outer update may move it (static parameters: no, see the
          class docstring), and
        * the knot's time, in units of CYCLES: knot ``j`` sits at the start of
          segment ``j``, so ``t_j = j``. Static rows are masked out of the
          localized update, so their (unused) time is 0.
        """
        mask: list[bool] = []
        times: list[float] = []
        for name in params.data_vars:
            if "time" in params[name].dims:
                mask.extend([True] * num_knots)
                times.extend(float(knot) for knot in range(num_knots))
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
        at ``t_j = j`` (segment start) and observation batch ``k`` at
        ``t = k + 1`` (segment end, the time its state is observed), both in
        units of cycles, so ``temporal_radius`` is a number of cycles. The
        remaining two components are zero: this is a 1-D geometry reusing the
        ``(N, 3)`` coordinate contract of the spatial strategies.

        Consequence of that convention, worth knowing when choosing a radius:
        the knot that DROVE segment ``k`` sits one full cycle before the
        observation of that segment, so a ``temporal_radius`` of 1 cycle keeps
        only each knot's own observation batch (at maximum taper) and anything
        below 1 excludes every observation, leaving the trajectory untouched. A
        useful radius spans several cycles. (Placing the observations at the
        segment MIDPOINT instead is a one-line change here if the resulting
        asymmetry ever proves to matter.)

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
