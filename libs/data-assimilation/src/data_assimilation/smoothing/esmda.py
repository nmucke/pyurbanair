import logging
import os
import pathlib
import shutil
from abc import abstractmethod
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import xarray
from data_assimilation.augmentation import ParamAugmentation, StateAugmentation
from data_assimilation.filtering.analysis import stochastic_enkf_update
from data_assimilation.io import load_dataset as _load_dataset
from data_assimilation.localization.base import BaseLocalization
from data_assimilation.observation_operator import (
    AggregateObservations,
    ObservationOperator,
    sensor_observation_coords,
)
from data_assimilation.reduction import OnlineStateReduction
from data_assimilation.smoothing.base import BaseSmoothing, Observations

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

logger = logging.getLogger(__name__)


def _block_grouping_enabled(localization: Optional[BaseLocalization]) -> bool:
    """True when a localization is set and requests joint block updates."""
    return localization is not None and localization.block_grouping


class _BaseESMDA(BaseSmoothing):
    """Shared ESMDA logic for parameter-only and joint state-parameter variants.

    ``aggregate_observations`` (forwarded to :class:`~data_assimilation.\
smoothing.base.BaseSmoothing`) is applied to the real observations and to
    every ``H(x)``, so ``C_D`` must be sized for the *aggregated* observation
    vector.
    """

    #: Whether this smoother can supply physical row coordinates to a
    #: coordinate-based localization (distance strategy).  Only the state-bearing
    #: variants can; the parameter-only variants override nothing and stay
    #: ``False`` so an incompatible pairing is rejected at construction rather
    #: than deep inside the first Kalman update.
    _supplies_row_coordinates: bool = False

    def __init__(
        self,
        observation_operator: ObservationOperator,
        forward_model: BaseEnsembleForwardModel,
        C_D: jnp.ndarray,
        num_steps: int = 3,
        alpha: Optional[float] = None,
        rng_key: Optional[jax.Array] = None,
        localization: Optional[BaseLocalization] = None,
        aggregate_observations: Optional[AggregateObservations] = None,
    ) -> None:
        super().__init__(
            observation_operator,
            forward_model,
            aggregate_observations=aggregate_observations,
        )

        # ``C_D_sqrt = sqrt(C_D)`` (element-wise) and ``jnp.diag(C_D)`` in the
        # localized update are only correct for a diagonal covariance. Validate
        # it here so a full (non-diagonal) C_D fails loudly instead of silently
        # producing a wrong perturbation/inflation.
        C_D = jnp.asarray(C_D)
        if C_D.ndim != 2 or C_D.shape[0] != C_D.shape[1]:
            raise ValueError(
                f"C_D must be a square (N_d, N_d) matrix, got shape {C_D.shape}."
            )
        off_diagonal = C_D - jnp.diag(jnp.diag(C_D))
        if not bool(jnp.all(off_diagonal == 0.0)):
            raise ValueError(
                "C_D must be diagonal: the element-wise C_D_sqrt and the "
                "diagonal used by the localized update assume a diagonal "
                "observation-error covariance. Pass sigma**2 on the diagonal."
            )
        # Strictly positive variances keep ``C_DD + alpha * C_D`` positive
        # definite (C_DD is only rank <= N_e - 1). A zero/negative variance -- a
        # config typo or a "perfect" synthetic sensor -- makes the analysis
        # system singular in the directions outside the ensemble span, and JAX's
        # solve returns NaN without raising.
        if not bool(jnp.all(jnp.diag(C_D) > 0.0)):
            raise ValueError(
                "C_D must have strictly positive diagonal variances; a zero or "
                "negative observation-error variance makes the analysis system "
                "singular (NaN-poisoning the ensemble). Check obs_error_std."
            )

        # ES-MDA consistency: the tempering coefficients must satisfy
        # ``sum_k 1/alpha_k = 1`` for the multiple updates to equal one Bayesian
        # conditioning (Emerick & Reynolds 2013). With a single scalar alpha
        # applied every step that means ``num_steps / alpha == 1``. A mismatched
        # alpha silently tempers the likelihood by ``num_steps / alpha`` -- a
        # different inference problem -- so reject it here (the default
        # ``alpha = num_steps`` is always consistent). The final-time trajectory
        # smoothing passes alpha=1 through ``_compute_kalman_update`` directly,
        # not via ``self.alpha``, so it is unaffected by this check.
        effective_alpha = num_steps if alpha is None else alpha
        if abs(num_steps / effective_alpha - 1.0) > 1e-6:
            raise ValueError(
                f"Inconsistent ES-MDA schedule: num_steps={num_steps} and "
                f"alpha={effective_alpha} give sum_k 1/alpha_k = "
                f"{num_steps / effective_alpha:.4g} != 1. Set alpha=num_steps "
                "(the default) or leave alpha unset."
            )

        # Reject a coordinate-based localization on a smoother that cannot supply
        # row coordinates (parameter-only variants). Deferring this to the first
        # Kalman update produces an opaque error deep in the analysis loop.
        if (
            localization is not None
            and localization.requires_coordinates
            and not self._supplies_row_coordinates
        ):
            raise ValueError(
                f"{type(localization).__name__} requires physical row "
                "coordinates, which this smoother cannot supply. Distance-based "
                "localization only applies to a state-bearing smoother "
                "(esmda/smoother=state, state_and_parameter, or "
                "state_and_dynamic)."
            )

        self.alpha = num_steps if alpha is None else alpha
        self.C_D = C_D
        self.C_D_sqrt = jnp.sqrt(self.C_D)
        # Default the PRNG key here (not in the signature): a default argument
        # would be evaluated at import time -- initializing the JAX backend as a
        # side effect and sharing one key across every smoother built without an
        # explicit one.
        self.rng_key = jax.random.PRNGKey(42) if rng_key is None else rng_key
        self.num_steps = num_steps
        # Optional localization strategy. When None, the global (unlocalized)
        # Kalman update is used. When set, each augmented-state row is updated
        # via a local analysis driven by the strategy's inflation factors.
        self.localization = localization

        # On-disk peak-storage control, set by the caller (e.g. run_esmda.py)
        # after construction. When ``prune_disk_steps`` is True and the forward
        # model saves on disk, each ESMDA step's per-member forecast directory is
        # deleted as soon as its Kalman update is computed -- the warm-start IC
        # for the next forecast is already carried in memory, and only the prior
        # (step 0) and posterior (final step) forecasts are re-read by the caller.
        # ``keep_prior_disk_step`` retains step 0 for the caller to assemble; set
        # it False when the prior state is not being saved. Together these cap the
        # on-disk peak at ~2x ensemble_size regardless of ``num_steps`` (instead of
        # num_steps+1 step directories coexisting until the window ends).
        self.prune_disk_steps = False
        self.keep_prior_disk_step = True

        # Observation-space diagnostics, set by the caller (e.g. run_esmda.py)
        # after construction, same attribute-plumbing pattern as
        # ``prune_disk_steps``. When ``collect_obs_diagnostics`` is True each
        # ``_one_step`` records the predicted observations it materialized, and
        # ``_analysis`` appends one final entry for the posterior forecast --
        # ``num_steps + 1`` entries per window, entry 0 the prior. Off by
        # default: nothing is collected, and no extra observation operator
        # evaluation happens, unless a caller asks for it.
        self.collect_obs_diagnostics = False
        self.pred_obs_history: list[np.ndarray] = []

        if self.forward_model.save_on_disk:
            self.base_results_dir = self.forward_model.results_dir
            for i in range(num_steps + 1):
                step_dir = self.base_results_dir / f"step_{i}"
                os.makedirs(step_dir, exist_ok=True)
                for state_file in step_dir.glob("state_*.nc"):
                    state_file.unlink(missing_ok=True)

    def _set_step_results_dir(self, step: int) -> None:
        """Point the forward model's results directory at the given step."""
        if self.forward_model.save_on_disk:
            self.forward_model.set_results_dir(self.base_results_dir / f"step_{step}")

    def _prune_step_results_dir(self, step: int) -> None:
        """Delete one finished ESMDA step's on-disk forecast directory.

        Called once a step's Kalman update is computed, so its per-member forecast
        files are no longer needed (the next forecast warm-starts from the in-memory
        analyzed IC, and the caller re-reads only the prior/posterior steps). No-op
        unless on-disk pruning is enabled; ``ignore_errors`` so an already-removed
        or never-written directory is harmless.
        """
        if self.forward_model.save_on_disk and self.prune_disk_steps:
            shutil.rmtree(self.base_results_dir / f"step_{step}", ignore_errors=True)

    def _record_pred_obs(self, pred_obs: jnp.ndarray) -> None:
        """Record one iteration's predicted observations ``(N_d, N_e)``.

        No-op unless ``collect_obs_diagnostics`` is set. ``np.asarray`` copies
        the block off the JAX device, so a whole window's history never pins
        accelerator memory (it is KB-scale on the host).
        """
        if self.collect_obs_diagnostics:
            self.pred_obs_history.append(np.asarray(pred_obs))

    def _results_dir_or_none(self) -> Optional[pathlib.Path]:
        """The forward model's results dir on the disk path, ``None`` in memory.

        What every ``_observation_step`` call site passes: in on-disk save mode
        ``_forecast_step`` returns ``None`` and the operator reads the per-member
        files, in memory mode the state is passed directly and the dir is unused.
        """
        return (
            self.forward_model.results_dir if self.forward_model.save_on_disk else None
        )

    def get_state(self, ensemble_member: int, step: int) -> xarray.Dataset:
        """Get the state for one ensemble member at a given step."""
        return _load_dataset(
            self.base_results_dir / f"step_{step}" / f"state_{ensemble_member}.nc"
        )

    def _compute_kalman_update(
        self,
        augmented: jnp.ndarray,
        pred_obs: jnp.ndarray,
        obs: jnp.ndarray,
        N_e: int,
        group_ids: Optional[jnp.ndarray] = None,
        localize_mask: Optional[jnp.ndarray] = None,
        row_coords: Optional[jnp.ndarray] = None,
        obs_coords: Optional[jnp.ndarray] = None,
        alpha: Optional[float] = None,
    ) -> jnp.ndarray:
        """Compute the ESMDA Kalman update for an augmented state vector.

        Args:
            augmented: Array of shape (N_aug, N_e) — the augmented state
                (parameters only, or state + parameters).
            pred_obs: Array of shape (N_d, N_e) — predicted observations.
            obs: Array of shape (N_d,) — true observations.
            N_e: Ensemble size.
            group_ids: Optional block id per augmented row, shape (N_aug,),
                used only by a localized update with block grouping enabled.
                Rows sharing an id are updated jointly. ``None`` -> per-row.
            localize_mask: Optional boolean array, shape (N_aug,), forwarded to
                a localized update. Rows where ``False`` get the exact global
                update; ``True`` rows get the localized update. Ignored on the
                global (``localization is None``) path.
            row_coords: Optional augmented-row coordinates, shape (N_aug, 3),
                and ``obs_coords``: observation coordinates, shape (N_d, 3),
                forwarded to a distance-based localized update. Ignored on the
                global path and by coordinate-free strategies.
            alpha: Optional override of the ESMDA inflation coefficient for
                this single update (used by the un-tempered final trajectory
                smoothing step). ``None`` -> ``self.alpha``.

        Returns:
            Updated augmented array of the same shape.
        """
        # The ESMDA per-step update IS the stochastic EnKF analysis with a
        # tempered alpha; the single shared implementation lives in
        # ``filtering/analysis.py`` (1-D variance-vector C_D contract). ``N_e``
        # is kept in the signature for callers but derived from the arrays.
        alpha = self.alpha if alpha is None else alpha
        self.rng_key, subkey = jax.random.split(self.rng_key)
        return stochastic_enkf_update(
            augmented=augmented,
            pred_obs=pred_obs,
            obs=obs,
            C_D_diag=jnp.diag(self.C_D),
            rng_key=subkey,
            alpha=alpha,
            localization=self.localization,
            group_ids=group_ids,
            localize_mask=localize_mask,
            row_coords=row_coords,
            obs_coords=obs_coords,
        )

    def _observation_coords(self, n_d: int) -> jnp.ndarray:
        """Physical (x, y, z) coordinate of each of the ``n_d`` observations.

        Delegates to :func:`~data_assimilation.observation_operator.\
sensor_observation_coords` (shared with the filtering package); see its
        docstring for the sensor-tiling layout. Requires coordinate-based
        observations (``obs_x``/``obs_y``/``obs_z``).
        """
        return jnp.asarray(
            sensor_observation_coords(self.observation_operator, n_d)
        )  # (n_d, 3)

    def _final_time_smoothing_step(
        self,
        state: Optional[xarray.Dataset],
        observations: jnp.ndarray,
    ) -> Optional[xarray.Dataset]:
        """Optional post-loop smoothing of the full window trajectory.

        No-op in the base class; overridden by the state-bearing variants when
        ``final_time_smoothing`` is enabled.
        """
        return state

    @abstractmethod
    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
        group_ids: Optional[jnp.ndarray] = None,
    ) -> tuple[Optional[xarray.Dataset], xarray.Dataset]:
        """Perform one ESMDA assimilation step.

        ``group_ids`` is an optional block id per augmented row used by a
        localized update with block grouping; a subclass that flattens a
        structured parameter passes the ids it built while flattening.

        Returns:
            (updated_state_or_None, updated_params)
        """
        raise NotImplementedError

    def _analysis(
        self,
        params: xarray.Dataset,
        observations: Observations,
        state: Optional[xarray.Dataset] = None,
        return_params_history: bool = False,
        return_state_history: bool = False,
    ) -> xarray.Dataset | tuple[xarray.Dataset, xarray.Dataset]:
        """Perform the ESMDA analysis loop.

        ``observations`` is the window's time-resolved observation DataArray
        (or an already flattened vector); it is converted ONCE here, and
        everything below works on the flat ``(N_d,)`` array.

        Iterated joint estimation: the forecast at iteration ``i`` starts from
        the *current* initial-condition estimate. For the state-bearing variants
        (``StateAndParameterESMDA`` / ``StateAndTimeVaryingParameterESMDA``) the
        Kalman-updated initial condition from ``_one_step`` is fed forward, so it
        is actually used by the next forecast (and hence the next Kalman update)
        and propagates into the posterior forecast and the next window. For the
        parameter-only variants ``_one_step`` returns no state (``None``), so
        ``initial_state`` keeps the caller's pinned value and the loop reduces to
        the original parameter-only behavior.

        Note: feeding the analyzed IC forward warm-starts the next forecast from
        it (skipping a fresh spin-up after iteration 0). This is intentional — it
        is what makes the state estimate matter — but it means the analyzed field
        is integrated directly, so it must be a usable warm-start state for the
        forward model (as the cross-window carry-over already assumes).
        """
        if return_state_history and self.forward_model.save_on_disk:
            raise ValueError(
                "return_state_history is not supported in on-disk save mode: "
                "the per-step states live in the step_{i}/ directories "
                "(see get_state). Use an in-memory forward model "
                "(results_dir=None) to collect the state history."
            )

        # Aggregate + flatten the real observations once (a plain array passes
        # through). The predicted observations take the same path inside
        # ``_observation_step``, so the two always live in the same space.
        obs = self._get_observations(observations)

        initial_state = state

        # One history per call, so a multi-window caller reading it after each
        # window gets that window's entries alone. Rebound rather than cleared:
        # the caller may still hold the previous window's list.
        self.pred_obs_history = []

        params_history: list[xarray.Dataset] = [params] if return_params_history else []
        state_history: list[xarray.Dataset] = []

        for i in range(self.num_steps):
            self._set_step_results_dir(i)

            state = self._forecast_step(state=initial_state, params=params)
            params = self.forward_model.apply_failure_substitutions_to_params(params)

            if return_state_history:
                state_history.append(state)

            updated_state, params = self._one_step(
                params=params,
                obs=obs,
                state=state,
            )

            # Feed the Kalman-updated initial condition forward (state-bearing
            # variants only; param-only variants return ``None`` and keep the
            # caller's pinned IC).
            if updated_state is not None:
                initial_state = updated_state
            # Repair any diverged members in the IC used by the next forecast
            # (clones a donor's known-good field into each failed slot). No-op on
            # a cold start (``initial_state is None``) or when nothing failed.
            initial_state = self.forward_model.apply_failure_substitutions_to_state(
                initial_state
            )

            if return_params_history:
                params_history.append(params)

            # Free this step's on-disk forecast now that its update is done. The
            # next forecast warm-starts from the in-memory analyzed IC, and only
            # the prior (step 0, kept when ``keep_prior_disk_step``) and the
            # posterior (final step, written after the loop) are re-read by the
            # caller -- so every intermediate step is safe to drop here. No-op
            # unless on-disk pruning is enabled.
            if not (i == 0 and self.keep_prior_disk_step):
                self._prune_step_results_dir(i)

            logger.info("ESMDA step %d completed", i)

        # Final forecast from the analyzed initial condition + updated params.
        self._set_step_results_dir(self.num_steps)
        state = self._forecast_step(state=initial_state, params=params)

        # Close the observation-space history with the POSTERIOR forecast: the
        # in-loop appends only cover the num_steps prior/intermediate forecasts,
        # and the posterior is the one the run is judged on. Recorded here --
        # before ``_final_time_smoothing_step``, which is a second Kalman update
        # of the trajectory rather than a forecast (see its docstring), so its
        # own predicted observations are a different object and stay out.
        if self.collect_obs_diagnostics:
            self._record_pred_obs(
                jnp.asarray(
                    self._observation_step(
                        state=state, results_dir=self._results_dir_or_none()
                    )
                ).T
            )

        # Optional post-loop trajectory smoothing (state-bearing variants with
        # final_time_smoothing enabled; no-op otherwise). Reuses the final
        # forecast — no extra forward solve — and never touches the params.
        state = self._final_time_smoothing_step(state, obs)

        if return_state_history:
            state_history.append(state)

        # Build return values
        result_params = (
            xarray.concat(params_history, dim="esmda_step", join="override")
            if return_params_history
            else params
        )

        if self.forward_model.save_on_disk:
            return result_params

        if return_state_history:
            result_state = xarray.concat(
                state_history, dim="esmda_step", join="override"
            )
            return result_params, result_state

        return result_params, state


class ParameterESMDA(_BaseESMDA):
    """Parameter-only ESMDA smoothing."""

    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
        group_ids: Optional[jnp.ndarray] = None,
    ) -> tuple[Optional[xarray.Dataset], xarray.Dataset]:
        obs = jnp.asarray(obs)
        param_names = list(params.data_vars.keys())
        N_e = params.sizes["ensemble"]

        pred_obs = self._observation_step(
            state=state, results_dir=self._results_dir_or_none()
        )
        pred_obs = jnp.asarray(pred_obs).T

        if pred_obs.ndim != 2 or pred_obs.shape[1] != N_e:
            raise ValueError(
                "Predicted observations shape does not match ensemble size. "
                f"Expected second dimension {N_e}, got {pred_obs.shape}. "
                "This usually indicates stale or unexpected files in the ESMDA "
                "results step directory."
            )
        self._record_pred_obs(pred_obs)

        params_array = jnp.array([params[name].values for name in param_names])

        # Block grouping (paper sec. 3b): co-locate the augmented rows that
        # share a block. A subclass that flattens a structured parameter (e.g.
        # time knots) passes the block ids it built while flattening; the plain
        # static case leaves each parameter in its own block.
        if _block_grouping_enabled(self.localization):
            if group_ids is None:
                group_ids = jnp.arange(len(param_names), dtype=int)
        else:
            group_ids = None
        params_updated = self._compute_kalman_update(
            params_array, pred_obs, obs, N_e, group_ids=group_ids
        )

        updated_data_vars = {
            name: ("ensemble", params_updated[i, :])
            for i, name in enumerate(param_names)
        }

        return None, xarray.Dataset(data_vars=updated_data_vars, coords=params.coords)


class TimeVaryingParameterESMDA(ParameterESMDA):
    """Parameter-only ESMDA for time-varying inflow parameters.

    Each time point of each parameter is treated as an independent scalar
    parameter during the Kalman update.  The flattening groups all time
    points of one parameter together (e.g. ``inflow_angle_0``, …,
    ``inflow_angle_{N_t-1}``, ``velocity_magnitude_0``, …) so that
    different physical quantities are never mixed.
    """

    def __init__(
        self,
        observation_operator: ObservationOperator,
        forward_model: BaseEnsembleForwardModel,
        C_D: jnp.ndarray,
        num_time_points: int,
        num_steps: int = 3,
        alpha: Optional[float] = None,
        rng_key: Optional[jax.Array] = None,
        pin_initial_time_point: bool = False,
        localization: Optional[BaseLocalization] = None,
        aggregate_observations: Optional[AggregateObservations] = None,
    ) -> None:
        super().__init__(
            observation_operator=observation_operator,
            forward_model=forward_model,
            C_D=C_D,
            num_steps=num_steps,
            alpha=alpha,
            rng_key=rng_key,
            localization=localization,
            aggregate_observations=aggregate_observations,
        )
        self.num_time_points = num_time_points
        # When True, ``t=0`` of every time-varying parameter is excluded
        # from the Kalman-updated augmented state and reinserted unchanged
        # during unflatten — per ensemble member. Useful for preserving
        # cross-window continuity in rollout ESMDA.
        self.pin_initial_time_point = pin_initial_time_point

    @property
    def _param_augmentation(self) -> ParamAugmentation:
        """The shared flatten/unflatten transform for time-varying params.

        Built on the fly from ``num_time_points`` / ``pin_initial_time_point``
        (cheap, stateless) so it always reflects the instance's current
        attributes. See :class:`~data_assimilation.augmentation.\
ParamAugmentation` for the flattening semantics.
        """
        return ParamAugmentation(
            num_time_points=self.num_time_points,
            pin_initial_time_point=self.pin_initial_time_point,
        )

    def _check_num_time_points(self, params: xarray.Dataset) -> None:
        """Validate the data's ``time`` size against the configured knot count."""
        self._param_augmentation.check_num_time_points(params)

    def _time_varying_group_ids(self, params: xarray.Dataset) -> jnp.ndarray:
        """Block id per flattened param row, grouping knots of one parameter."""
        return self._param_augmentation.group_ids(params)

    def _flatten_time_varying_params(self, params: xarray.Dataset) -> xarray.Dataset:
        """Flatten ``(time, ensemble)`` params to scalar ``(ensemble,)`` vars."""
        return self._param_augmentation.flatten(params)

    def _unflatten_params(
        self,
        flat_params: xarray.Dataset,
        original_params: xarray.Dataset,
    ) -> xarray.Dataset:
        """Reverse :meth:`_flatten_time_varying_params`."""
        return self._param_augmentation.unflatten(flat_params, original_params)

    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
        group_ids: Optional[jnp.ndarray] = None,
    ) -> tuple[Optional[xarray.Dataset], xarray.Dataset]:
        flat_params = self._flatten_time_varying_params(params)
        # Group ids built from the true name->knot mapping (grouping this
        # parameter's time knots), passed down so the block update never has to
        # re-parse the flattened names.
        knot_group_ids = self._time_varying_group_ids(params)
        _, updated_flat = super()._one_step(
            flat_params, obs, state, group_ids=knot_group_ids
        )
        updated_params = self._unflatten_params(updated_flat, params)
        return None, updated_params


class StateAndParameterESMDA(_BaseESMDA):
    """Joint state and parameter ESMDA smoothing.

    Optionally performs the state part of the Kalman update in a reduced
    SVD/KL basis fitted ONLINE to the current forecast ensemble
    (``state_reduction``, see :class:`~data_assimilation.reduction.\
OnlineStateReduction` and ``docs/reduced_state_da.md``), and an optional
    post-loop Kalman smoothing of the full window trajectory
    (``final_time_smoothing``). Both default to off, which reproduces the
    full-space behavior exactly.
    """

    # State-bearing smoothers know each state row's physical grid coordinate, so
    # they can pair with a coordinate-based (distance) localization.
    _supplies_row_coordinates: bool = True

    #: Shared, stateless state flatten/unflatten transform (class attribute so
    #: it is available on instances built without ``__init__``, as some unit
    #: tests do via ``__new__``).
    _state_augmentation: StateAugmentation = StateAugmentation()

    def __init__(
        self,
        *args: Any,
        state_reduction: Optional[OnlineStateReduction] = None,
        final_time_smoothing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if state_reduction is not None and self.localization is not None:
            raise ValueError(
                "state_reduction is incompatible with (state) localization: "
                "the reduced coefficients are nonlocal, so neither distance- "
                "nor correlation-based state localization applies to them. "
                "Set localization=null or disable the state reduction."
            )
        if final_time_smoothing and state_reduction is None:
            raise ValueError(
                "final_time_smoothing requires state_reduction: the joint "
                "all-time-steps state update is only feasible in the reduced "
                "SVD basis."
            )
        if final_time_smoothing and self.forward_model.save_on_disk:
            raise ValueError(
                "final_time_smoothing is not supported in on-disk save mode: "
                "the analysis returns no state there, so the smoothed "
                "trajectory would be discarded. Use an in-memory forward "
                "model (results_dir=None)."
            )
        self.state_reduction = state_reduction
        self.final_time_smoothing = final_time_smoothing

    def _get_states(
        self,
        state: Optional[xarray.Dataset] = None,
        results_dir: Optional[pathlib.Path] = None,
    ) -> xarray.Dataset:
        """Get ensemble states, selecting the first timestep.

        Both branches must select the same frame: the augmented Kalman vector
        holds the window's initial condition, and ``_analysis`` feeds the
        analyzed result forward as the next forecast's warm start. Reading a
        different frame on disk would re-assimilate the window's observations
        against a state from a different time.
        """
        if state is not None:
            return state.isel(time=0)
        if results_dir is not None:
            state_files = self._get_sorted_state_files(pathlib.Path(results_dir))
            if not state_files:
                raise FileNotFoundError(
                    f"No state_*.nc files found in results directory: {results_dir}"
                )
            states = [_load_dataset(f).isel(time=0) for f in state_files]
            return xarray.concat(states, dim="ensemble", join="override")
        raise ValueError("Either state or results_dir must be provided.")

    def _flatten_state(self, state: xarray.Dataset) -> jnp.ndarray:
        """Flatten ensemble state to (degrees_of_freedom, ensemble_size)."""
        return self._state_augmentation.flatten(state)

    def _get_window_states(
        self,
        state: Optional[xarray.Dataset] = None,
        results_dir: Optional[pathlib.Path] = None,
    ) -> xarray.Dataset:
        """Get the full-window ensemble states (all time frames).

        Used by the ``window_snapshots`` basis source; mirrors
        :meth:`_get_states` without the ``time=0`` selection.
        """
        if state is not None:
            return state
        if results_dir is not None:
            state_files = self._get_sorted_state_files(pathlib.Path(results_dir))
            if not state_files:
                raise FileNotFoundError(
                    f"No state_*.nc files found in results directory: {results_dir}"
                )
            states = [_load_dataset(f) for f in state_files]
            return xarray.concat(states, dim="ensemble", join="override")
        raise ValueError("Either state or results_dir must be provided.")

    def _flatten_window_snapshots(self, state: xarray.Dataset) -> jnp.ndarray:
        """Flatten every (member, time frame) into a snapshot column.

        Returns an array of shape (degrees_of_freedom, N_e * N_t) whose row
        ordering matches :meth:`_flatten_state` exactly (sorted variables,
        spatial dims flattened in C-order), so a column is directly comparable
        to a flattened ``time=0`` state vector. Time frames are thinned by the
        reduction's ``snapshot_stride``.
        """
        assert self.state_reduction is not None  # only called for window_snapshots
        return self._state_augmentation.flatten_snapshots(
            state, self.state_reduction.snapshot_stride
        )

    def _basis_snapshots(
        self,
        state: Optional[xarray.Dataset],
        results_dir: Optional[pathlib.Path],
    ) -> Optional[jnp.ndarray]:
        """Snapshot matrix for the online basis, or ``None`` for the IC source.

        ``None`` makes :meth:`_augmented_state_update` fit the basis on the
        flattened ``time=0`` ensemble itself (the ``initial_condition``
        source); the ``window_snapshots`` source assembles every output frame
        of every member.
        """
        if (
            self.state_reduction is None
            or self.state_reduction.basis_source != "window_snapshots"
        ):
            return None
        window_states = self._get_window_states(state=state, results_dir=results_dir)
        return self._flatten_window_snapshots(window_states)

    def _state_group_ids(self, state: xarray.Dataset) -> jnp.ndarray:
        """Block id per flattened state row, grouping co-located cells.

        Variables sharing the same physical coordinate grid share blocks (for
        example pylbm's collocated ``u/v/w``). Variables on distinct staggered
        grids receive distinct block ids even when their array shapes match.
        """
        return self._state_augmentation.group_ids(state)

    def _unflatten_state(
        self,
        states_array: jnp.ndarray,
        state_template: xarray.Dataset,
    ) -> xarray.Dataset:
        """Unflatten a (degrees_of_freedom, ensemble_size) array back to xarray."""
        return self._state_augmentation.unflatten(states_array, state_template)

    def _state_row_coords(self, state: xarray.Dataset) -> jnp.ndarray:
        """Physical ``(x, y, z)`` coordinate of each flattened state row.

        Mirrors :meth:`_flatten_state`'s ordering exactly: for each variable
        (sorted), the non-ensemble dims are flattened in C-order.  The grid
        coordinate along each axis is read from the dim's coordinate values; a
        dim's axis (x/y/z) is taken from the leading character of its name
        (``x``/``xt``/``xm``/``xu`` -> x, ``y``/``yt``/``ym``/``yv`` -> y,
        ``z``/``zt``/``zm`` -> z), which holds for every supported solver grid.
        Used only by distance-based localization.
        """
        return self._state_augmentation.row_coords(state)

    def _augmented_state_update(
        self,
        states_array: xarray.Dataset,
        flat_params: xarray.Dataset,
        pred_obs: jnp.ndarray,
        obs: jnp.ndarray,
        N_e: int,
        snapshots_flat: Optional[jnp.ndarray] = None,
        param_group_ids: Optional[jnp.ndarray] = None,
    ) -> tuple[xarray.Dataset, xarray.Dataset]:
        """Build ``[state | params]``, apply the localized Kalman update, split.

        ``flat_params`` is a Dataset of scalar ``(ensemble,)`` variables (static
        params, or time-varying params already flattened to ``{name}_{t}``).
        State rows are localized by either strategy. Correlation localization
        also applies to parameter rows; distance localization leaves parameters
        on the global update because scalar/global parameters have no physical
        coordinates.
        Returns ``(updated_state, updated_flat_params)``.

        With ``state_reduction`` set, the state rows are replaced by reduced
        SVD/KL coefficients of a basis fitted online to ``snapshots_flat``
        (``None`` -> the flattened ``time=0`` ensemble itself), and the Kalman
        increment is decoded back onto each member's full state. Localization
        is ``None`` on this path (enforced at construction).
        """
        param_names = list(flat_params.data_vars.keys())
        # ``_unflatten_state`` reads only dims/sizes/coords/dtype from its
        # template, so ``states_array`` itself serves -- no need to allocate a
        # full ensemble-sized copy of empties (a real device allocation).
        state_template = states_array

        states_flat = self._flatten_state(states_array)

        params_array = jnp.array([flat_params[name].values for name in param_names])

        # Reduced path: Kalman-update SVD/KL coefficients instead of the raw
        # state rows, then decode the increment onto each member's full state
        # (preserving the projection residual of the window_snapshots source).
        if self.state_reduction is not None:
            self.state_reduction.fit(
                states_flat if snapshots_flat is None else snapshots_flat
            )
            xi = self.state_reduction.encode(states_flat)
            N_s = xi.shape[0]
            augmented = jnp.concatenate([xi, params_array], axis=0)
            augmented = self._compute_kalman_update(augmented, pred_obs, obs, N_e)
            states_updated_flat = states_flat + self.state_reduction.decode_increment(
                augmented[:N_s, :] - xi
            )
            updated_states = self._unflatten_state(states_updated_flat, state_template)
            updated_flat = xarray.Dataset(
                {
                    name: ("ensemble", augmented[N_s + i, :])
                    for i, name in enumerate(param_names)
                },
                coords={"ensemble": flat_params.coords["ensemble"]},
            )
            return updated_states, updated_flat

        N_s = states_flat.shape[0]
        augmented = jnp.concatenate([states_flat, params_array], axis=0)

        # Localization is a no-op on the global path. Correlation localization
        # can act on parameter rows; distance localization masks them out so they
        # receive the exact global update.
        localize_mask = None
        group_ids = None
        row_coords = None
        obs_coords = None
        if self.localization is not None:
            localize_params = self.localization.localizes_parameters
            localize_mask = jnp.concatenate(
                [
                    jnp.ones(N_s, dtype=bool),
                    jnp.full((len(param_names),), localize_params, dtype=bool),
                ]
            )
            # Block grouping: co-locate state rows by their physical grid and
            # group structured parameter rows when the caller supplies their
            # mapping (e.g. all time knots of one dynamic parameter).
            if _block_grouping_enabled(self.localization):
                state_groups = self._state_group_ids(states_array)
                offset = int(state_groups.max()) + 1
                if param_group_ids is None:
                    param_group_ids = jnp.arange(len(param_names), dtype=int)
                param_groups = offset + param_group_ids
                group_ids = jnp.concatenate([state_groups, param_groups])
            # Distance-based localization additionally needs grid/sensor coords.
            if self.localization.requires_coordinates:
                state_coords = self._state_row_coords(states_array)  # (N_s, 3)
                param_coords = jnp.zeros((len(param_names), 3))  # masked out
                row_coords = jnp.concatenate([state_coords, param_coords], axis=0)
                obs_coords = self._observation_coords(obs.shape[0])

        augmented = self._compute_kalman_update(
            augmented,
            pred_obs,
            obs,
            N_e,
            group_ids=group_ids,
            localize_mask=localize_mask,
            row_coords=row_coords,
            obs_coords=obs_coords,
        )

        updated_states = self._unflatten_state(augmented[:N_s, :], state_template)
        updated_flat = xarray.Dataset(
            {
                name: ("ensemble", augmented[N_s + i, :])
                for i, name in enumerate(param_names)
            },
            coords={"ensemble": flat_params.coords["ensemble"]},
        )
        return updated_states, updated_flat

    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
        group_ids: Optional[jnp.ndarray] = None,
    ) -> tuple[xarray.Dataset, xarray.Dataset]:
        # ``group_ids`` is unused: the state-bearing variant builds its own block
        # ids (state cells + per-parameter) inside ``_augmented_state_update``.
        obs = jnp.asarray(obs)
        results_dir = self._results_dir_or_none()

        pred_obs = self._observation_step(state=state, results_dir=results_dir)
        pred_obs = jnp.asarray(pred_obs).T
        self._record_pred_obs(pred_obs)

        states_array = self._get_states(state=state, results_dir=results_dir)
        N_e = params.sizes["ensemble"]

        # Static params are already scalar (ensemble,) vars: no flatten needed.
        updated_states, updated_flat = self._augmented_state_update(
            states_array,
            params,
            pred_obs,
            obs,
            N_e,
            snapshots_flat=self._basis_snapshots(state, results_dir),
        )
        return updated_states, xarray.Dataset(
            data_vars={name: updated_flat[name] for name in updated_flat.data_vars},
            coords=params.coords,
        )

    def _final_time_smoothing_step(
        self,
        state: Optional[xarray.Dataset],
        observations: jnp.ndarray,
    ) -> Optional[xarray.Dataset]:
        """One un-tempered Kalman update of the FULL window trajectory.

        Optional post-loop step (``final_time_smoothing=True``): the state at
        every time step of the window is updated jointly in the reduced SVD
        basis, reusing the final posterior forecast (no extra forward solve).
        Parameters are not part of the augmented vector here — they are frozen.

        .. warning::
            This applies the observations a **second** time. The MDA loop already
            consumed the full likelihood (``sum_k 1/alpha_k = 1``), and the
            trajectory being smoothed here was forecast from the analyzed IC and
            parameters — so it is *already conditioned* on these observations
            (through the dynamics). Adding one more full-weight (``alpha=1``)
            update conditions on the same data twice: in the linear-Gaussian
            limit the total likelihood weight is 2, so the returned trajectory
            ensemble is **overconfident** — its spread underestimates the
            posterior spread and its mean is pulled too hard toward the data.

            Use this only as a point (RMSE) estimate of the trajectory; the
            resulting ensemble spread is NOT a posterior spread and must not be
            used for uncertainty quantification. The rigorous alternative is to
            carry the reduced trajectory coefficients in the augmented vector
            through the MDA schedule so the trajectory is conditioned exactly
            once (see docs/temp/da_review_math.md §1.1 / §3.10).
        """
        if not self.final_time_smoothing or state is None:
            return state

        logger.warning(
            "final_time_smoothing applies the observations a second time: the "
            "returned trajectory ensemble is overconfident. Use its mean as a "
            "point estimate only, not its spread for uncertainty quantification "
            "(see _final_time_smoothing_step docstring)."
        )

        obs = jnp.asarray(observations)
        pred_obs = jnp.asarray(self._observation_step(state=state)).T
        N_e = state.sizes["ensemble"]

        # _flatten_state/_unflatten_state are agnostic to the time dim: the
        # full (time, space) trajectory of each member becomes one column.
        # final_time_smoothing requires state_reduction (enforced at construction).
        assert self.state_reduction is not None
        traj_flat = self._flatten_state(state)
        self.state_reduction.fit(traj_flat)
        xi = self.state_reduction.encode(traj_flat)
        xi_updated = self._compute_kalman_update(xi, pred_obs, obs, N_e, alpha=1.0)
        traj_updated = traj_flat + self.state_reduction.decode_increment(
            xi_updated - xi
        )
        return self._unflatten_state(traj_updated, state)


class StateESMDA(StateAndParameterESMDA):
    """State-only ESMDA smoothing with fixed forward-model parameters.

    The window's initial-condition state is the complete augmented vector.
    ``params`` are still supplied to every forecast, but are returned unchanged
    after every analysis step. Correlation- and distance-based localization both
    act on the state rows; distance localization therefore remains fully valid
    in this mode.
    """

    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
        group_ids: Optional[jnp.ndarray] = None,
    ) -> tuple[xarray.Dataset, xarray.Dataset]:
        obs = jnp.asarray(obs)
        results_dir = self._results_dir_or_none()

        pred_obs = self._observation_step(state=state, results_dir=results_dir)
        pred_obs = jnp.asarray(pred_obs).T
        self._record_pred_obs(pred_obs)

        states_array = self._get_states(state=state, results_dir=results_dir)
        states_flat = self._flatten_state(states_array)
        N_e = states_flat.shape[1]
        snapshots_flat = self._basis_snapshots(state, results_dir)

        if self.state_reduction is not None:
            self.state_reduction.fit(
                states_flat if snapshots_flat is None else snapshots_flat
            )
            xi = self.state_reduction.encode(states_flat)
            xi_updated = self._compute_kalman_update(xi, pred_obs, obs, N_e)
            updated_flat = states_flat + self.state_reduction.decode_increment(
                xi_updated - xi
            )
        else:
            state_group_ids = None
            row_coords = None
            obs_coords = None
            if self.localization is not None:
                if _block_grouping_enabled(self.localization):
                    state_group_ids = self._state_group_ids(states_array)
                if self.localization.requires_coordinates:
                    row_coords = self._state_row_coords(states_array)
                    obs_coords = self._observation_coords(obs.shape[0])
            updated_flat = self._compute_kalman_update(
                states_flat,
                pred_obs,
                obs,
                N_e,
                group_ids=state_group_ids,
                row_coords=row_coords,
                obs_coords=obs_coords,
            )

        return self._unflatten_state(updated_flat, states_array), params


class StateAndTimeVaryingParameterESMDA(
    StateAndParameterESMDA, TimeVaryingParameterESMDA
):
    """Joint state and time-varying-parameter ESMDA smoothing.

    Combines :class:`StateAndParameterESMDA` (the window's ``time=0`` initial-
    condition state is flattened into the augmented vector) with
    :class:`TimeVaryingParameterESMDA` (each time-varying parameter is flattened
    into per-time-knot scalars ``{name}_{t}``, respecting
    ``pin_initial_time_point``).

    Correlation localization applies to both state and parameter rows. Physical-
    distance localization applies only to state rows, leaving parameter rows on
    the global update. The augmented update is built by the shared
    :meth:`StateAndParameterESMDA._augmented_state_update`; grid-block grouping
    co-locates state rows and groups time knots belonging to one parameter.

    The constructor chains through the MRO: ``StateAndParameterESMDA.__init__``
    consumes the state-specific keywords (``state_reduction``,
    ``final_time_smoothing``) and forwards the rest to
    :class:`TimeVaryingParameterESMDA`, so the effective signature is
    ``(observation_operator, forward_model, C_D, num_time_points,
    num_steps=3, alpha=None, rng_key=..., pin_initial_time_point=False,
    localization=None, state_reduction=None, final_time_smoothing=False)``.
    """

    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
        group_ids: Optional[jnp.ndarray] = None,
    ) -> tuple[xarray.Dataset, xarray.Dataset]:
        # ``group_ids`` is unused: block ids are built inside
        # ``_augmented_state_update`` (state cells + per-parameter blocks).
        obs = jnp.asarray(obs)
        results_dir = self._results_dir_or_none()

        pred_obs = self._observation_step(state=state, results_dir=results_dir)
        pred_obs = jnp.asarray(pred_obs).T
        self._record_pred_obs(pred_obs)

        # Time=0 state ensemble + time-varying params flattened to per-knot
        # scalars (respecting pin_initial_time_point). The shared localized
        # update handles the augmented [state | params] vector.
        states_array = self._get_states(state=state, results_dir=results_dir)
        flat_params = self._flatten_time_varying_params(params)
        N_e = flat_params.sizes["ensemble"]

        updated_states, updated_flat = self._augmented_state_update(
            states_array,
            flat_params,
            pred_obs,
            obs,
            N_e,
            snapshots_flat=self._basis_snapshots(state, results_dir),
            param_group_ids=self._time_varying_group_ids(params),
        )
        updated_params = self._unflatten_params(updated_flat, params)
        return updated_states, updated_params
