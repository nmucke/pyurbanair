"""Sequential filter cycle loop (BaseFilter) and the composed EnKF.

A filter's unit of work is a **cycle**: forecast the ensemble to the next
observation time, apply one full-weight analysis to the state/parameters at
that time, continue from the analyzed state. This inverts the ESMDA
smoother's loop (which re-forecasts the same window ``num_steps`` times with
tempered weight); the analysis math itself is shared with the smoother (see
:mod:`data_assimilation.filtering.analysis`).

Observation-time semantics: one cycle assimilates one observation batch. The
observation operator is applied to the whole forecast segment, so with the
config-default ``TemporalObservationOperator(mode="intervals")`` and one
interval per segment the batch is the segment's interval mean — an
observation *of the segment*, assimilated into the end-of-segment state.
This is an observation-operator choice (H and y agree by construction), not
an approximation error.
"""

import logging
import os
import pathlib
import shutil
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import xarray
from data_assimilation.augmentation import ParamAugmentation, StateAugmentation
from data_assimilation.filtering.analysis import (
    AnalysisScheme,
    StochasticEnKFAnalysis,
    validate_variances,
)
from data_assimilation.filtering.parameter_evolution import ParameterEvolution
from data_assimilation.inflation import InflationScheme
from data_assimilation.io import get_sorted_state_files, load_dataset
from data_assimilation.localization.base import BaseLocalization
from data_assimilation.observation_operator import sensor_observation_coords

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

logger = logging.getLogger(__name__)

FilterMode = Literal["state", "parameter", "joint"]


@dataclass
class CycleDiagnostics:
    """Per-cycle filter health indicators.

    These make "the filter is diverging / overconfident" visible at cycle k
    instead of at the end of the run:

    * ``innovation_chi2``: innovation consistency
      ``d^T (C_DD + C_D)^{-1} d / N_d`` with ``d = y - H(x_f)`` (ensemble
      mean). Values persistently >> 1 mean the ensemble is overconfident
      (spread too small for the actual errors); << 1 means overdispersion.
      ``C_DD`` is built from the *raw* forecast anomalies, before any prior
      inflation — so the statistic keeps answering "does the forecast need
      (more) inflation?" rather than measuring the spread the inflation chose.
    * ``obs_prior_rmse`` / ``obs_posterior_rmse``: observation-space RMSE of
      the ensemble-mean predicted observations before/after the analysis.
    * spreads: mean per-row ensemble standard deviation of each augmented
      block, before and after the analysis (after any inflation).
    """

    cycle: int
    obs_prior_rmse: float
    obs_posterior_rmse: float
    innovation_chi2: float
    state_spread_prior: Optional[float] = None
    state_spread_posterior: Optional[float] = None
    param_spread_prior: Optional[float] = None
    param_spread_posterior: Optional[float] = None


@dataclass
class FilterResult:
    """Return value of :meth:`BaseFilter.run` (no return-type polymorphism).

    ``state`` is the analyzed end-of-run state (final frame; the warm start
    for any continuation) and ``params`` the final analyzed/evolved
    parameters. Histories are ``cycle``-concatenated Datasets, present only
    when ``return_history=True`` (``params_history`` additionally holds the
    prior as its first entry).
    """

    params: Optional[xarray.Dataset]
    state: Optional[xarray.Dataset]
    diagnostics: list[CycleDiagnostics]
    params_history: Optional[xarray.Dataset] = None
    state_history: Optional[xarray.Dataset] = None


class BaseFilter:
    """Sequential ensemble filter: the cycle loop around an analysis scheme.

    The filter owns time management (one forecast per observation batch),
    augmentation (which blocks enter the analysis, per ``mode``), inflation,
    parameter evolution, failure substitution, and diagnostics; the analysis
    scheme is a pure function of arrays (see :class:`AnalysisScheme`).

    Args:
        observation_operator: Maps a forecast segment to the flat predicted-
            observation vector (typically a ``TemporalObservationOperator``).
        forward_model: Any ensemble forward model; its configured horizon is
            the cycle's forecast segment (set it to the observation interval).
        C_D: Diagonal observation-error covariance as a 1-D variance vector
            (the honest contract for the diagonal assumption). A square
            diagonal matrix is accepted and reduced to its diagonal.
        analysis: The update math applied once, at full weight, per cycle.
        mode: Which blocks the analysis updates. ``"state"``: the flattened
            end-of-segment state (params, if any, are carried unmodified);
            ``"parameter"``: the flattened params only (analyzed params apply
            from the next cycle); ``"joint"``: ``[state | params]``.
            Correlation localization applies to both joint blocks; physical-
            distance localization applies only to state rows.
        localization: Optional localization strategy, reused from the
            smoother unchanged. Distance-based strategies need state rows.
        inflation: Optional spread maintenance applied to the augmented
            anomalies (prior hook also applied to the predicted-observation
            anomalies, keeping the gain consistent with the inflated
            ensemble).
        parameter_evolution: Parameter forecast model applied after each
            analysis; required (or ``inflation``) for the parameter-updating
            modes (``"parameter"``/``"joint"``), whose parameter block
            otherwise collapses silently.
        rng_key: PRNG key; defaults to a fresh ``PRNGKey(42)`` per instance.

    On-disk mode mirrors the smoother: each cycle's forecast is written to
    ``cycle_{k}/`` under the forward model's results dir. Setting
    ``prune_disk_cycles = True`` after construction deletes each finished
    cycle's directory once its analysis is computed (the warm start is
    carried in memory); ``keep_first_disk_cycle`` retains ``cycle_0``, and
    the final cycle is always kept.
    """

    def __init__(
        self,
        observation_operator: Callable[[xarray.Dataset], Any],
        forward_model: BaseEnsembleForwardModel,
        C_D: jnp.ndarray,
        analysis: AnalysisScheme,
        mode: FilterMode = "joint",
        localization: Optional[BaseLocalization] = None,
        inflation: Optional[InflationScheme] = None,
        parameter_evolution: Optional[ParameterEvolution] = None,
        rng_key: Optional[jax.Array] = None,
    ) -> None:
        if mode not in ("state", "parameter", "joint"):
            raise ValueError(
                f"mode must be 'state', 'parameter' or 'joint', got {mode!r}."
            )
        self.observation_operator = observation_operator
        self.forward_model = forward_model
        self.analysis = analysis
        self.mode: FilterMode = mode

        C_D = jnp.asarray(C_D)
        if C_D.ndim == 2:
            # Convenience: accept the smoother's (N_d, N_d) diagonal matrix
            # (e.g. from create_C_D) and reduce it to the honest 1-D contract.
            if C_D.shape[0] != C_D.shape[1]:
                raise ValueError(
                    "C_D must be a 1-D variance vector or a square diagonal "
                    f"matrix, got shape {C_D.shape}."
                )
            if not bool(jnp.all(C_D - jnp.diag(jnp.diag(C_D)) == 0.0)):
                raise ValueError(
                    "C_D must be a 1-D variance vector or a square diagonal "
                    "matrix; the given matrix has off-diagonal entries."
                )
            C_D = jnp.diag(C_D)
        self.C_D_diag = validate_variances(C_D)

        if (
            localization is not None
            and localization.requires_coordinates
            and mode == "parameter"
        ):
            raise ValueError(
                f"{type(localization).__name__} requires physical row "
                "coordinates, which a parameter-only filter cannot supply. "
                "Distance-based localization needs mode='state' or 'joint'."
            )
        self.localization = localization
        self.inflation = inflation

        if mode == "state" and parameter_evolution is not None:
            raise ValueError(
                "parameter_evolution has no effect in mode='state' (parameters "
                "are carried unmodified through cycles). Remove it or use "
                "mode='parameter'/'joint'."
            )
        # Refuse silently-collapsing configurations: an un-evolved,
        # un-inflated parameter ensemble loses spread every analysis and stops
        # learning after a few cycles. Joint mode is just as exposed — the
        # forecast regenerates state spread every cycle but never the
        # parameter block's.
        if (
            mode in ("parameter", "joint")
            and parameter_evolution is None
            and inflation is None
        ):
            raise ValueError(
                f"mode={mode!r} needs spread maintenance: without "
                "parameter_evolution (e.g. RandomWalkEvolution) or inflation "
                "(e.g. RTPS) the parameter ensemble collapses after a few "
                "cycles and the filter stops learning."
            )
        self.parameter_evolution = parameter_evolution

        # Default the PRNG key here (not in the signature): a default argument
        # would be evaluated at import time -- initializing the JAX backend as
        # a side effect and sharing one key across every filter built without
        # an explicit one.
        self.rng_key = jax.random.PRNGKey(42) if rng_key is None else rng_key

        # Phase 1: static scalar parameters only (time-varying parameter
        # filtering — evolving knots with the prior's AR model — is Phase 2).
        self._param_augmentation = ParamAugmentation()
        self._state_augmentation = StateAugmentation()

        # On-disk peak-storage control, set by the caller after construction
        # (mirrors the smoother's prune_disk_steps / keep_prior_disk_step).
        self.prune_disk_cycles = False
        self.keep_first_disk_cycle = True
        if self.forward_model.save_on_disk:
            self.base_results_dir = self.forward_model.results_dir

    # ------------------------------------------------------------------
    # Forecast / observation plumbing
    # ------------------------------------------------------------------

    def _forecast_step(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
    ) -> Optional[xarray.Dataset]:
        """Run the ensemble over one cycle's segment (None in on-disk mode)."""
        return self.forward_model.run_ensemble(state=state, params=params)

    def _observation_step(
        self,
        state: Optional[xarray.Dataset] = None,
        results_dir: Optional[pathlib.Path] = None,
    ) -> jnp.ndarray:
        """Predicted observations for the current forecast, shape (N_e, N_d)."""
        if state is not None:
            return jnp.asarray(self.observation_operator(state))
        if results_dir is not None:
            file_list = get_sorted_state_files(pathlib.Path(results_dir))
            if not file_list:
                raise FileNotFoundError(
                    f"No state_*.nc files found in results directory: {results_dir}"
                )
            return jnp.stack(
                [
                    jnp.asarray(self.observation_operator(load_dataset(f)))
                    for f in file_list
                ],
                axis=0,
            )
        raise ValueError("Either state or results_dir must be provided.")

    def _get_final_states(
        self,
        state: Optional[xarray.Dataset] = None,
        results_dir: Optional[pathlib.Path] = None,
    ) -> xarray.Dataset:
        """The end-of-segment ensemble state (the frame the analysis updates).

        The filter analyzes the state *at the observation time* — the final
        frame of the forecast segment — and that analyzed frame warm-starts
        the next cycle. (Contrast with the smoother, which analyzes the
        window's initial condition.)
        """
        if state is not None:
            return state.isel(time=-1) if "time" in state.dims else state
        if results_dir is not None:
            state_files = get_sorted_state_files(pathlib.Path(results_dir))
            if not state_files:
                raise FileNotFoundError(
                    f"No state_*.nc files found in results directory: {results_dir}"
                )
            members = []
            for f in state_files:
                member = load_dataset(f)
                members.append(
                    member.isel(time=-1) if "time" in member.dims else member
                )
            return xarray.concat(members, dim="ensemble", join="override")
        raise ValueError("Either state or results_dir must be provided.")

    def _set_cycle_results_dir(self, cycle: int) -> None:
        """Point the forward model's results directory at the given cycle."""
        if self.forward_model.save_on_disk:
            cycle_dir = self.base_results_dir / f"cycle_{cycle}"
            os.makedirs(cycle_dir, exist_ok=True)
            for state_file in cycle_dir.glob("state_*.nc"):
                state_file.unlink(missing_ok=True)
            self.forward_model.set_results_dir(cycle_dir)

    def _prune_cycle_results_dir(self, cycle: int, num_cycles: int) -> None:
        """Delete one finished cycle's on-disk forecast directory.

        No-op unless pruning is enabled; the final cycle (the posterior
        forecast) is always kept, and ``cycle_0`` is kept while
        ``keep_first_disk_cycle`` is True.
        """
        if not (self.forward_model.save_on_disk and self.prune_disk_cycles):
            return
        if cycle == num_cycles - 1:
            return
        if cycle == 0 and self.keep_first_disk_cycle:
            return
        shutil.rmtree(self.base_results_dir / f"cycle_{cycle}", ignore_errors=True)

    def get_state(self, ensemble_member: int, cycle: int) -> xarray.Dataset:
        """Re-open one member's forecast for a given cycle (on-disk mode)."""
        return load_dataset(
            self.base_results_dir / f"cycle_{cycle}" / f"state_{ensemble_member}.nc"
        )

    # ------------------------------------------------------------------
    # The cycle loop
    # ------------------------------------------------------------------

    def run(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        observations: Optional[jnp.ndarray] = None,
        *,
        return_history: bool = False,
    ) -> FilterResult:
        """Run the filter over a sequence of observation batches.

        Args:
            state: Optional warm-start ensemble state for the first cycle
                (``None`` -> the forward model's cold start).
            params: Ensemble parameters; required for ``mode="parameter"`` /
                ``"joint"``, optional (carried through unmodified) for
                ``mode="state"``.
            observations: Array of shape ``(num_cycles, N_d)`` — one
                observation batch per cycle, each consumed exactly once (a
                filter has no MDA schedule).
            return_history: Collect per-cycle analyzed params/state into
                ``cycle``-concatenated history Datasets.

        Returns:
            A :class:`FilterResult`.
        """
        if observations is None:
            raise ValueError("observations must be provided.")
        obs_batches = jnp.asarray(observations)
        if obs_batches.ndim != 2:
            raise ValueError(
                "observations must have shape (num_cycles, N_d) — one batch "
                f"per cycle — got shape {obs_batches.shape}. For a single "
                "cycle pass a (1, N_d) array."
            )
        if obs_batches.shape[1] != self.C_D_diag.shape[0]:
            raise ValueError(
                f"Observation batches have N_d={obs_batches.shape[1]} but C_D "
                f"has {self.C_D_diag.shape[0]} variances."
            )
        if self.mode in ("parameter", "joint"):
            if params is None:
                raise ValueError(f"mode={self.mode!r} requires params.")
            self._check_static_params(params)

        num_cycles = int(obs_batches.shape[0])
        analysis_state = state
        diagnostics: list[CycleDiagnostics] = []
        params_history: list[xarray.Dataset] = (
            [params] if (return_history and params is not None) else []
        )
        state_history: list[xarray.Dataset] = []

        for cycle in range(num_cycles):
            self._set_cycle_results_dir(cycle)

            forecast = self._forecast_step(state=analysis_state, params=params)
            if params is not None:
                params = self.forward_model.apply_failure_substitutions_to_params(
                    params
                )
            results_dir = (
                self.forward_model.results_dir
                if self.forward_model.save_on_disk
                else None
            )

            pred_obs = self._observation_step(
                state=forecast, results_dir=results_dir
            ).T  # (N_d, N_e)
            final_state = self._get_final_states(
                state=forecast, results_dir=results_dir
            )

            analysis_state, params, cycle_diag = self._analysis_cycle(
                cycle, final_state, params, pred_obs, obs_batches[cycle]
            )

            # Repair any diverged members in the warm start for the next
            # forecast (clones a donor's known-good field into failed slots).
            analysis_state = self.forward_model.apply_failure_substitutions_to_state(
                analysis_state
            )

            diagnostics.append(cycle_diag)
            if return_history:
                assert analysis_state is not None  # always set by _analysis_cycle
                state_history.append(analysis_state)
                if params is not None:
                    params_history.append(params)

            self._prune_cycle_results_dir(cycle, num_cycles)
            logger.info("Filter cycle %d completed", cycle)

        return FilterResult(
            params=params,
            state=analysis_state,
            diagnostics=diagnostics,
            params_history=(
                xarray.concat(params_history, dim="cycle", join="override")
                if params_history
                else None
            ),
            state_history=(
                xarray.concat(state_history, dim="cycle", join="override")
                if state_history
                else None
            ),
        )

    def _check_static_params(self, params: xarray.Dataset) -> None:
        """Phase 1 supports scalar (ensemble,) parameters only."""
        time_vars = [n for n in params.data_vars if "time" in params[n].dims]
        if time_vars:
            raise NotImplementedError(
                f"Time-varying parameters {time_vars} are not supported by the "
                "filter yet: filtering estimates the parameter value *now*, "
                "evolved between cycles by a parameter evolution model (see "
                "docs/temp/da_filtering_module_plan.md §4.4). Use the "
                "TimeVaryingParameterESMDA smoother, or reduce the parameters "
                "to static scalars."
            )

    # ------------------------------------------------------------------
    # One analysis
    # ------------------------------------------------------------------

    def _analysis_cycle(
        self,
        cycle: int,
        final_state: xarray.Dataset,
        params: Optional[xarray.Dataset],
        pred_obs: jnp.ndarray,
        obs: jnp.ndarray,
    ) -> tuple[xarray.Dataset, Optional[xarray.Dataset], CycleDiagnostics]:
        """Build the augmented vector, analyze, split back, evolve, diagnose."""
        obs = jnp.asarray(obs)
        N_d = obs.shape[0]
        if pred_obs.shape[0] != N_d:
            raise ValueError(
                f"Predicted observations have N_d={pred_obs.shape[0]} but the "
                f"observation batch has N_d={N_d}. This usually indicates a "
                "mismatch between the observation operator and the supplied "
                "observations, or stale files in the cycle results directory."
            )

        blocks: list[jnp.ndarray] = []
        n_state = 0
        n_param = 0
        flat_params: Optional[xarray.Dataset] = None
        if self.mode in ("state", "joint"):
            states_flat = self._state_augmentation.flatten(final_state)
            n_state = states_flat.shape[0]
            blocks.append(states_flat)
        if self.mode in ("parameter", "joint"):
            assert params is not None  # validated in run()
            flat_params = self._param_augmentation.flatten(params)
            params_array = ParamAugmentation.to_array(flat_params)
            n_param = params_array.shape[0]
            blocks.append(params_array)
        augmented = jnp.concatenate(blocks, axis=0)

        # Prior (multiplicative-style) inflation. The predicted-observation
        # anomalies are inflated with the same scheme so the Kalman gain is
        # consistent with the inflated ensemble (equivalent to re-applying a
        # linear H to the inflated members).
        aug_mean = jnp.mean(augmented, axis=1, keepdims=True)
        dev_prior = augmented - aug_mean
        pred_obs_forecast = pred_obs  # raw (pre-inflation), for the chi2 diagnostic
        if self.inflation is not None:
            dev_prior = self.inflation.inflate_prior(dev_prior)
            augmented = aug_mean + dev_prior
            pred_obs_mean = jnp.mean(pred_obs, axis=1, keepdims=True)
            pred_obs = pred_obs_mean + self.inflation.inflate_prior(
                pred_obs - pred_obs_mean
            )

        # Append the predicted observations as diagnostic rows: their update
        # is the exact posterior in observation space (they ride along without
        # influencing any other row), giving obs_posterior_rmse without a
        # second forecast. They are masked out of localization and stripped
        # before the split below.
        augmented_ext = jnp.concatenate([augmented, pred_obs], axis=0)
        group_ids, localize_mask, row_coords, obs_coords = self._localization_plumbing(
            final_state, n_state, n_param, N_d
        )

        self.rng_key, subkey = jax.random.split(self.rng_key)
        updated_ext = self.analysis(
            augmented_ext,
            pred_obs,
            obs,
            self.C_D_diag,
            subkey,
            localization=self.localization,
            group_ids=group_ids,
            localize_mask=localize_mask,
            row_coords=row_coords,
            obs_coords=obs_coords,
        )
        updated = updated_ext[: n_state + n_param]
        pred_obs_post = updated_ext[n_state + n_param :]

        # Posterior (relaxation-style) inflation, e.g. RTPS/RTPP.
        post_mean = jnp.mean(updated, axis=1, keepdims=True)
        dev_post = updated - post_mean
        if self.inflation is not None:
            dev_post = self.inflation.inflate_posterior(dev_prior, dev_post)
            updated = post_mean + dev_post

        cycle_diag = self._cycle_diagnostics(
            cycle,
            obs,
            pred_obs_forecast,
            pred_obs_post,
            dev_prior,
            dev_post,
            n_state,
            n_param,
        )

        # Split the augmented vector back per mode.
        if self.mode in ("state", "joint"):
            analysis_state = self._state_augmentation.unflatten(
                updated[:n_state], final_state
            )
        else:
            # Parameter-only: the (unanalyzed) forecast final frame is still
            # the next cycle's warm start.
            analysis_state = final_state

        if self.mode in ("parameter", "joint"):
            assert params is not None and flat_params is not None
            updated_flat = ParamAugmentation.from_array(updated[n_state:], flat_params)
            params = xarray.Dataset(
                data_vars={name: updated_flat[name] for name in updated_flat.data_vars},
                coords=params.coords,
            )
            if self.parameter_evolution is not None:
                self.rng_key, evolve_key = jax.random.split(self.rng_key)
                params = self.parameter_evolution.evolve(params, evolve_key)

        return analysis_state, params, cycle_diag

    def _localization_plumbing(
        self,
        final_state: xarray.Dataset,
        n_state: int,
        n_param: int,
        n_d: int,
    ) -> tuple[
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
        Optional[jnp.ndarray],
    ]:
        """(group_ids, localize_mask, row_coords, obs_coords) for the update.

        Mirrors the smoother's conventions: state rows are localized; parameter
        rows are localized when the strategy supports non-spatial rows
        (correlation), but receive the exact global update for physical-distance
        localization. The appended diagnostic predicted-observation rows are
        always masked to the global update.
        """
        if self.localization is None:
            return None, None, None, None

        mask_blocks = []
        if n_state:
            mask_blocks.append(jnp.ones(n_state, dtype=bool))
        if n_param:
            mask_blocks.append(
                jnp.full((n_param,), self.localization.localizes_parameters, dtype=bool)
            )
        mask_blocks.append(jnp.zeros(n_d, dtype=bool))
        localize_mask = jnp.concatenate(mask_blocks)

        group_ids = None
        if self.localization.block_grouping:
            id_blocks = []
            offset = 0
            if n_state:
                state_groups = self._state_augmentation.group_ids(final_state)
                id_blocks.append(state_groups)
                offset = int(state_groups.max()) + 1
            if n_param:
                id_blocks.append(offset + jnp.arange(n_param, dtype=int))
                offset += n_param
            id_blocks.append(offset + jnp.arange(n_d, dtype=int))
            group_ids = jnp.concatenate(id_blocks)

        row_coords = None
        obs_coords = None
        if self.localization.requires_coordinates:
            # mode='parameter' is rejected at construction, so state rows exist.
            state_coords = self._state_augmentation.row_coords(final_state)
            row_coords = jnp.concatenate(
                [state_coords, jnp.zeros((n_param + n_d, 3))], axis=0
            )  # non-state rows are masked out above
            obs_coords = jnp.asarray(
                sensor_observation_coords(self.observation_operator, n_d)
            )

        return group_ids, localize_mask, row_coords, obs_coords

    def _cycle_diagnostics(
        self,
        cycle: int,
        obs: jnp.ndarray,
        pred_obs: jnp.ndarray,
        pred_obs_post: jnp.ndarray,
        dev_prior: jnp.ndarray,
        dev_post: jnp.ndarray,
        n_state: int,
        n_param: int,
    ) -> CycleDiagnostics:
        """Innovation statistics and block spreads for one cycle.

        ``pred_obs`` must be the raw forecast (pre-inflation) so the chi2
        spread term reflects what the model produced, not what the inflation
        chose; the block spreads intentionally use the (possibly inflated)
        analysis anomalies.
        """
        N_d = obs.shape[0]
        N_e = pred_obs.shape[1]

        innovation = obs - jnp.mean(pred_obs, axis=1)
        pred_obs_dev = pred_obs - jnp.mean(pred_obs, axis=1, keepdims=True)
        C_DD = jnp.dot(pred_obs_dev, pred_obs_dev.T) / (N_e - 1)
        S = C_DD + jnp.diag(self.C_D_diag)
        chi2 = float(
            innovation
            @ jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(S), innovation)
            / N_d
        )

        def _block_spread(dev: jnp.ndarray, start: int, size: int) -> Optional[float]:
            if size == 0:
                return None
            return float(jnp.mean(jnp.std(dev[start : start + size], axis=1, ddof=1)))

        return CycleDiagnostics(
            cycle=cycle,
            obs_prior_rmse=float(jnp.sqrt(jnp.mean(innovation**2))),
            obs_posterior_rmse=float(
                jnp.sqrt(jnp.mean((obs - jnp.mean(pred_obs_post, axis=1)) ** 2))
            ),
            innovation_chi2=chi2,
            state_spread_prior=_block_spread(dev_prior, 0, n_state),
            state_spread_posterior=_block_spread(dev_post, 0, n_state),
            param_spread_prior=_block_spread(dev_prior, n_state, n_param),
            param_spread_posterior=_block_spread(dev_post, n_state, n_param),
        )


class EnsembleKalmanFilter(BaseFilter):
    """The user-facing EnKF: :class:`BaseFilter` composed with the stochastic
    (perturbed-observation) analysis. Pass ``analysis=...`` to swap in another
    scheme (ETKF/LETKF once available)."""

    def __init__(
        self,
        observation_operator: Callable[[xarray.Dataset], Any],
        forward_model: BaseEnsembleForwardModel,
        C_D: jnp.ndarray,
        analysis: Optional[AnalysisScheme] = None,
        mode: FilterMode = "joint",
        localization: Optional[BaseLocalization] = None,
        inflation: Optional[InflationScheme] = None,
        parameter_evolution: Optional[ParameterEvolution] = None,
        rng_key: Optional[jax.Array] = None,
    ) -> None:
        super().__init__(
            observation_operator=observation_operator,
            forward_model=forward_model,
            C_D=C_D,
            analysis=StochasticEnKFAnalysis() if analysis is None else analysis,
            mode=mode,
            localization=localization,
            inflation=inflation,
            parameter_evolution=parameter_evolution,
            rng_key=rng_key,
        )
