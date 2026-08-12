"""Sequential filter cycle loop (BaseFilter) and the composed EnKF.

A filter's unit of work is a **cycle**: forecast the ensemble to the next
observation time, assimilate that segment's observations into the
state/parameters at that time, continue from the analyzed state. This inverts
the ESMDA smoother's loop (which re-forecasts the same window ``num_steps``
times with tempered weight); the analysis math itself is shared with the
smoother (see :mod:`data_assimilation.filtering.analysis`).

Observation-time semantics: the observation operator is applied to the whole
forecast segment, so one cycle carries ``T`` time-resolved observation frames
— and the filter assimilates them ONE AT A TIME, in time order, within the
cycle: a serial (asynchronous) EnKF. Each frame's ``(num_sensors x
num_states)`` vector gets its own full-weight analysis of the end-of-segment
augmented state through the ensemble cross-covariances, and the remaining
frames' predicted observations are appended as ride-along rows, so every
analysis sees what the earlier frames already did. ``T = 1`` — one frame per
cycle, and every flat ``(num_cycles, N_d)`` input — is the degenerate case:
exactly the one full-weight analysis per cycle the filter has always applied.

The sequential filter does **not** aggregate observations: interval
aggregation (``AggregateObservations``) is a smoother-side choice, kept by the
ESMDA smoothers and by ``FilterSmoothingESMDA`` (whose inner filter aggregates
in its own ``_prepare_pred_obs`` override). Here every frame the operator
produced is assimilated, so H and y agree frame by frame by construction.
"""

import logging
import os
import pathlib
import shutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import xarray
from data_assimilation.augmentation import ParamAugmentation, StateAugmentation
from data_assimilation.filtering.analysis import (
    LOCALIZATION_POLICIES,
    AnalysisScheme,
    StochasticEnKFAnalysis,
    validate_variances,
)
from data_assimilation.filtering.parameter_evolution import ParameterEvolution
from data_assimilation.inflation import InflationScheme
from data_assimilation.io import get_sorted_state_files, load_dataset
from data_assimilation.localization.base import BaseLocalization
from data_assimilation.observation_operator import sensor_observation_coords
from data_assimilation.reduction import OnlineStateReduction

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

    The ``reduction_*``, ``transform_*`` and ``local_*`` groups are additive
    and nullable: a field is ``None`` on every path where the quantity does not
    exist, so the schema of ``cycle_diagnostics.yaml`` is the same for every
    configuration and a reader never has to branch on the analysis scheme. The
    stochastic analysis leaves both transform groups ``None``; a global
    ensemble transform fills ``transform_*``; a localized one fills
    ``local_*``. They are read back off the analysis object by name (see
    :meth:`BaseFilter._record_transform_diagnostics`), so this module stays
    independent of the ensemble-transform implementation.

    The ``local_*`` distributions are summarized to scalars on purpose: the
    underlying per-row/per-block arrays have one entry per state row
    (``N_s ~ 2e5`` in production) and this record is written every cycle. The
    full arrays remain on the analysis object for interactive inspection.
    """

    cycle: int
    obs_prior_rmse: float
    obs_posterior_rmse: float
    innovation_chi2: float
    state_spread_prior: Optional[float] = None
    state_spread_posterior: Optional[float] = None
    param_spread_prior: Optional[float] = None
    param_spread_posterior: Optional[float] = None
    # Provenance for obs_posterior_rmse: "exact" when the appended
    # predicted-observation rows really are H applied to the analyzed state;
    # otherwise the name of the approximation they represent.
    obs_posterior_rmse_kind: str = "exact"
    analysis_time: Optional[float] = None
    reduction_rank: Optional[int] = None
    reduction_available_rank: Optional[int] = None
    reduction_retained_energy: Optional[float] = None
    reduction_projection_residual: Optional[float] = None
    reduction_increment_norm: Optional[float] = None
    reduction_discarded_increment_fraction: Optional[float] = None
    reduction_basis_time: Optional[float] = None
    reduction_basis_updated: Optional[bool] = None
    reduction_condition_number: Optional[float] = None
    reduction_spectrum_max: Optional[float] = None
    reduction_subspace_drift: Optional[float] = None
    # One global ensemble transform (ETKF). retained == available means the
    # scientific truncation removed nothing; transform_discarded_spectrum_max
    # is then 0.0 (nothing was discarded), NOT None (which means "no global
    # transform ran").
    transform_available_rank: Optional[int] = None
    transform_retained_rank: Optional[int] = None
    transform_retained_energy: Optional[float] = None
    transform_discarded_spectrum_max: Optional[float] = None
    # One transform per distinct local observation selection (LETKF). Counts
    # are per cycle; the rank and active-observation summaries are over the
    # ACTIVE blocks only (an inactive block computes no transform and leaves
    # its rows untouched, so it would otherwise drag every minimum to zero).
    local_num_blocks: Optional[int] = None
    local_num_active_blocks: Optional[int] = None
    local_num_updated_rows: Optional[int] = None
    local_active_obs_min: Optional[int] = None
    local_active_obs_median: Optional[float] = None
    local_active_obs_max: Optional[int] = None
    local_retained_rank_min: Optional[int] = None
    local_retained_rank_mean: Optional[float] = None
    local_retained_rank_max: Optional[int] = None
    local_available_rank_max: Optional[int] = None
    # The local counterparts of transform_retained_energy /
    # transform_discarded_spectrum_max. Same convention: 1.0 and 0.0 mean the
    # truncation ran and discarded nothing, None means no local transform ran.
    local_retained_energy_min: Optional[float] = None
    local_retained_energy_mean: Optional[float] = None
    local_discarded_spectrum_max: Optional[float] = None
    local_chunk_size: Optional[int] = None


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

    The filter owns time management (one forecast per cycle), augmentation
    (which blocks enter the analysis, per ``mode``), inflation, parameter
    evolution, failure substitution, and diagnostics; the analysis scheme is a
    pure function of arrays (see :class:`AnalysisScheme`). Within a cycle the
    segment's observation frames are assimilated serially, one full-weight
    analysis each (see :meth:`_analysis_cycle`).

    Args:
        observation_operator: Maps a forecast segment to its predicted
            observations. A ``TemporalObservationOperator`` returns them
            time-resolved as a ``("ensemble", "time", "obs")`` DataArray;
            anything returning a plain ``(N_e, N_obs)`` array is treated as a
            single frame.
        forward_model: Any ensemble forward model; its configured horizon is
            the cycle's forecast segment.
        C_D: Diagonal observation-error covariance of ONE observation frame,
            as a 1-D variance vector (the honest contract for the diagonal
            assumption). A square diagonal matrix is accepted and reduced to
            its diagonal.
        analysis: The update math applied once, at full weight, per
            observation frame.
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
        state_reduction: Optional current or streaming SVD/POD representation
            for the physical state block. Supported only by unlocalized
            ``"state"`` and ``"joint"`` analyses.
        rng_key: PRNG key; defaults to a fresh ``PRNGKey(42)`` per instance.

    On-disk mode mirrors the smoother: each cycle's forecast is written to
    ``cycle_{k}/`` under the forward model's results dir. Setting
    ``prune_disk_cycles = True`` after construction deletes each finished
    cycle's directory once its analysis is computed (the warm start is
    carried in memory); ``keep_first_disk_cycle`` retains ``cycle_0``, and
    the final cycle is always kept.
    """

    #: Opt-in recording of the per-cycle predicted observations, same
    #: attribute-plumbing pattern as the smoother's ``collect_obs_diagnostics``:
    #: set it on the instance after construction. When True, :meth:`run`
    #: rebinds :attr:`pred_obs_history` and appends each cycle's RAW
    #: (pre-inflation) forecast observations *before* the analysis, and
    #: :attr:`pred_obs_post_history` with the matching POSTERIOR rows. Off by
    #: default — nothing is recorded and the cycle loop is unchanged. Used by
    #: the ``filter_smoothing`` outer ESMDA loop, which needs exactly the
    #: forecast observations ``d_k = H(x_f_k)`` stored before ``y_k`` is
    #: assimilated, and by ``run_filtering.py``, which pairs the two histories
    #: into the ESMDA-schema ``window_{w}_pred_obs.nc`` (step 0 = prior,
    #: step 1 = posterior).
    collect_pred_obs: bool = False

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
        state_reduction: Optional[OnlineStateReduction] = None,
    ) -> None:
        if mode not in ("state", "parameter", "joint"):
            raise ValueError(
                f"mode must be 'state', 'parameter' or 'joint', got {mode!r}."
            )
        self.observation_operator = observation_operator
        self.forward_model = forward_model
        self.analysis = analysis
        self.mode: FilterMode = mode

        # One FRAME's error covariance, not one cycle's: the serial sweep hands
        # the analysis one frame's (num_sensors x num_states) vector at a time,
        # so C_D is sized by the observation operator's per-frame output and
        # does NOT grow with the number of frames in a segment. (It used to be
        # the covariance of the whole aggregated batch, which for a one-interval
        # aggregation was the same vector — hence the unchanged configs.)
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

        # Declared analysis capability (see AnalysisScheme.localization_policy):
        # reject the combination here so a mislabelled config fails before the
        # first forecast rather than silently running a global update under a
        # localized name. The policy VALUE is validated too — it is a plain
        # class attribute, so a typo would otherwise match no branch and
        # disable the check entirely.
        #
        # Deliberately ahead of the state_reduction checks below, which is a
        # choice of *which message* a config that trips both gets, not a
        # correctness property: LETKF + state_reduction + localization=None
        # reports the 'required' policy error. That is the fault the user must
        # fix either way — dropping the reduction alone would still leave a
        # LETKF without a localization — whereas the reduction message would
        # send them to the wrong knob first.
        policy = analysis.localization_policy
        if policy not in LOCALIZATION_POLICIES:
            raise ValueError(
                f"{type(analysis).__name__}.localization_policy={policy!r} is not "
                f"one of {LOCALIZATION_POLICIES}."
            )
        if policy == "forbidden" and localization is not None:
            raise ValueError(
                f"{type(analysis).__name__} declares localization_policy="
                "'forbidden': it applies one global update and would ignore "
                f"{type(localization).__name__}. Use a localized analysis "
                "scheme (e.g. LETKF) or remove the localization strategy."
            )
        if policy == "required" and localization is None:
            raise ValueError(
                f"{type(analysis).__name__} declares localization_policy="
                "'required' and has no meaning without one. Configure a "
                "localization strategy or use the global analysis scheme."
            )
        self.localization = localization
        self.inflation = inflation

        if state_reduction is not None and mode == "parameter":
            raise ValueError(
                "state_reduction is not applicable to mode='parameter': there "
                "is no state block in the analysis. Remove state_reduction or "
                "use mode='state'/'joint'."
            )
        if state_reduction is not None and (
            state_reduction.basis_source != "initial_condition"
            or state_reduction.snapshot_stride != 1
        ):
            raise ValueError(
                "basis_source/snapshot_stride select which snapshots the ESMDA "
                "smoother fits; the filter always fits the current cycle's final "
                "forecast frame, so setting them here would silently do nothing."
            )
        if state_reduction is not None and localization is not None:
            raise ValueError(
                "state_reduction is incompatible with localization in the "
                "sequential filter: global POD coefficients have no physical "
                "coordinates. Remove localization or state_reduction."
            )
        self.state_reduction = state_reduction

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

        # Per-cycle forecast observations, filled only when the caller sets
        # ``collect_pred_obs`` (see the class attribute). The ``_post`` twin
        # holds the same cycles' ride-along POSTERIOR rows, entry for entry and
        # shape for shape.
        self.pred_obs_history: list[np.ndarray] = []
        self.pred_obs_post_history: list[np.ndarray] = []

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

    def _params_for_cycle(
        self,
        cycle: int,
        params: Optional[xarray.Dataset],
    ) -> Optional[xarray.Dataset]:
        """The parameters cycle ``cycle``'s forecast is run with.

        Extension point, identity by default: every cycle forecasts with the
        same parameter ensemble. A subclass whose parameters are a *trajectory*
        over the window (see :class:`~data_assimilation.filter_smoothing.base.\
_TrajectoryStateFilter`) overrides this to hand the forward model the piece of
        that trajectory spanning the cycle's segment — a time-varying schedule
        on a segment-local clock, which the backends already interpolate.
        """
        return params

    def _record_pred_obs(self, pred_obs: jnp.ndarray) -> None:
        """Record one cycle's raw forecast observations ``(T*N_obs, N_e)``.

        The cycle's frames stacked frame-major, exactly as the serial sweep
        appends them. For the ``T = 1`` cycles the ``filter_smoothing`` outer
        loop runs this is the ``(N_d, N_e)`` block it has always consumed.

        No-op unless ``collect_pred_obs`` is set. ``np.asarray`` copies the
        block off the JAX device, so a whole window's history never pins
        accelerator memory (it is KB-scale on the host); mirrors the smoother's
        ``_record_pred_obs``.
        """
        if self.collect_pred_obs:
            self.pred_obs_history.append(np.asarray(pred_obs))

    def _record_pred_obs_post(self, pred_obs_post: jnp.ndarray) -> None:
        """Record one cycle's POSTERIOR observation rows ``(T*N_obs, N_e)``.

        The ride-along rows :meth:`_analysis_cycle` already computes — the
        exact posterior in observation space on the unlocalized, unreduced path
        (see :meth:`_cycle_diagnostics` for what the other paths mean) — in the
        same frame-major layout, and therefore row for row alignable with the
        matching :attr:`pred_obs_history` entry. Gated on the same
        ``collect_pred_obs`` flag and appended once per cycle, so the two
        histories always have equal length.

        Recorded here rather than derived downstream because nothing outside
        the analysis can reproduce these rows without a second forecast; this
        is what makes the ESMDA-schema ``pred_obs`` artifact's step-(-1) entry
        a real posterior mismatch rather than an approximation.
        """
        if self.collect_pred_obs:
            self.pred_obs_post_history.append(np.asarray(pred_obs_post))

    def _prepare_pred_obs(self, pred_obs: Any) -> jnp.ndarray:
        """Normalize the operator's output to frames, ``(N_e, T, N_obs)``.

        A ``TemporalObservationOperator`` returns a labelled ``("ensemble",
        "time", "obs")`` DataArray, one entry per output frame of the segment;
        a bare operator (or a toy one) returns a plain ``(N_e, N_obs)`` array,
        which *is* one frame and is treated as ``T = 1``.

        The hook exists so a subclass can put the predicted observations into a
        different observation space before the analysis sees them — that is
        exactly what :class:`~data_assimilation.filter_smoothing.base.\
_TrajectoryStateFilter` does, aggregating them into the single flat frame the
        hybrid's outer ESMDA assimilates.
        """
        if isinstance(pred_obs, xarray.DataArray):
            return jnp.asarray(pred_obs.transpose("ensemble", "time", "obs").values)
        frames = jnp.asarray(pred_obs)
        if frames.ndim == 2:
            return frames[:, None, :]
        if frames.ndim != 3:
            raise ValueError(
                "Predicted observations must be (N_e, N_obs) for a single "
                f"frame or (N_e, T, N_obs) for a segment, got shape "
                f"{frames.shape}."
            )
        return frames

    def _observation_step(
        self,
        state: Optional[xarray.Dataset] = None,
        results_dir: Optional[pathlib.Path] = None,
    ) -> jnp.ndarray:
        """Predicted observations for the current forecast, ``(N_e, T, N_obs)``."""
        if state is not None:
            return self._prepare_pred_obs(self.observation_operator(state))
        if results_dir is not None:
            file_list = get_sorted_state_files(pathlib.Path(results_dir))
            if not file_list:
                raise FileNotFoundError(
                    f"No state_*.nc files found in results directory: {results_dir}"
                )
            members = [self.observation_operator(load_dataset(f)) for f in file_list]
            if isinstance(members[0], xarray.DataArray):
                # Convert ONCE over the stacked ensemble, so a subclass hook
                # that carries per-call state (the hybrid's interval-count
                # consistency check) sees one call per cycle, as on the
                # in-memory path, rather than one per member.
                # ``join="override"``: per-member time coords are not
                # bit-identical, and the default outer join would silently
                # union the axes and NaN-pad the observations (same convention
                # as ``_get_final_states`` below).
                return self._prepare_pred_obs(
                    xarray.concat(members, dim="ensemble", join="override").transpose(
                        "ensemble", "time", "obs"
                    )
                )
            return self._prepare_pred_obs(
                jnp.stack([jnp.asarray(member) for member in members], axis=0)
            )
        raise ValueError("Either state or results_dir must be provided.")

    def _cycle_observations(self, observations: Any) -> jnp.ndarray:
        """One cycle's real observations as frames, ``(T, N_obs)``.

        A labelled ``("time", "obs")`` DataArray keeps its frames; a plain
        ``(N_obs,)`` vector is the single frame of a legacy flat batch. No
        aggregation happens here (or anywhere else in this class) — every frame
        is assimilated, in time order.
        """
        if isinstance(observations, xarray.DataArray):
            return jnp.asarray(observations.transpose("time", "obs").values)
        frames = jnp.asarray(observations)
        if frames.ndim == 1:
            return frames[None, :]
        if frames.ndim != 2:
            raise ValueError(
                "Each cycle's observations must be (N_obs,) for a single frame "
                f'or a ("time", "obs") DataArray, got shape {frames.shape}.'
            )
        return frames

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
        observations: Optional[Union[jnp.ndarray, Sequence[xarray.DataArray]]] = None,
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
            observations: One segment of observations per cycle, each frame
                consumed exactly once (a filter has no MDA schedule). Either a
                list/tuple of per-cycle DataArrays with dims ``("time",
                "obs")`` and a cycle-local seconds time coordinate — whose ``T``
                frames are assimilated serially, in time order — or a flat
                ``(num_cycles, N_obs)`` array, one frame per cycle.
            return_history: Collect per-cycle analyzed params/state into
                ``cycle``-concatenated history Datasets.

        Returns:
            A :class:`FilterResult`.
        """
        if observations is None:
            raise ValueError("observations must be provided.")
        if isinstance(observations, (list, tuple)):
            if not observations:
                raise ValueError("observations must contain at least one cycle.")
            # One (T, N_obs) block per cycle. Cycles whose frame counts differ
            # fail loudly right here, at the stack: the analysis is sized per
            # frame, but the cycle loop assumes one common T.
            obs_batches = jnp.stack(
                [self._cycle_observations(o) for o in observations], axis=0
            )
        else:
            obs_batches = jnp.asarray(observations)
        if obs_batches.ndim == 2:
            # The legacy flat input: one already-flat batch per cycle IS one
            # frame per cycle, so it takes the T == 1 path — bit-identical to
            # the single analysis this filter applied before the sweep existed.
            obs_batches = obs_batches[:, None, :]
        if obs_batches.ndim != 3:
            raise ValueError(
                "observations must have shape (num_cycles, N_obs) — one frame "
                "per cycle — or (num_cycles, T, N_obs), got shape "
                f"{obs_batches.shape}. For a single cycle pass a (1, N_obs) "
                'array, or a one-element list of per-cycle ("time", "obs") '
                "DataArrays."
            )
        if obs_batches.shape[2] != self.C_D_diag.shape[0]:
            raise ValueError(
                f"Observation frames have N_obs={obs_batches.shape[2]} but C_D "
                f"has {self.C_D_diag.shape[0]} variances. C_D is the error "
                "covariance of ONE frame (sensors x observed states); the "
                "frames of a segment are assimilated one at a time."
            )
        if self.mode in ("parameter", "joint"):
            if params is None:
                raise ValueError(f"mode={self.mode!r} requires params.")
            self._check_static_params(params)

        num_cycles = int(obs_batches.shape[0])
        analysis_state = state
        # One history per call, so a caller that runs the same filter over
        # several passes (the filter_smoothing outer loop) reads that pass's
        # entries alone. Rebound rather than cleared: the caller may still hold
        # the previous pass's list.
        if self.collect_pred_obs:
            self.pred_obs_history = []
            self.pred_obs_post_history = []
        diagnostics: list[CycleDiagnostics] = []
        params_history: list[xarray.Dataset] = (
            [params] if (return_history and params is not None) else []
        )
        state_history: list[xarray.Dataset] = []

        for cycle in range(num_cycles):
            self._set_cycle_results_dir(cycle)

            forecast = self._forecast_step(
                state=analysis_state, params=self._params_for_cycle(cycle, params)
            )
            if params is not None:
                params = self.forward_model.apply_failure_substitutions_to_params(
                    params
                )
            results_dir = (
                self.forward_model.results_dir
                if self.forward_model.save_on_disk
                else None
            )

            # (N_e, T, N_obs) -> the segment's frames stacked frame-major,
            # (T*N_obs, N_e): frame t occupies rows [t*N_obs, (t+1)*N_obs),
            # which is the layout the serial sweep and the appended ride-along
            # rows both index by. T == 1 makes this exactly the old ``.T``.
            pred_obs_frames = self._observation_step(
                state=forecast, results_dir=results_dir
            )
            pred_obs = jnp.transpose(pred_obs_frames, (1, 2, 0)).reshape(
                -1, pred_obs_frames.shape[0]
            )
            final_state = self._get_final_states(
                state=forecast, results_dir=results_dir
            )

            # Record here, not inside the analysis: these are the forecast
            # observations before ANY inflation touches them (prior inflation
            # is applied inside ``_analysis_cycle``) and before ``y_cycle`` is
            # assimilated.
            self._record_pred_obs(pred_obs)

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
        """Build the augmented vector, sweep the frames, split back, evolve, diagnose.

        The cycle's ``T`` observation frames are assimilated SERIALLY (see
        :meth:`_assimilate_frames`): one full-weight analysis each, in time
        order, all of them updating the same end-of-segment augmented state
        through the ensemble cross-covariances. Everything around the sweep
        happens once per cycle — the basis fit, prior and posterior inflation,
        the localization plumbing, the parameter evolution — because they are
        properties of the cycle's forecast, not of an individual observation.

        A configured ``state_reduction`` changes only the *analysis
        representation* of the state block: the rows handed to the analysis are
        POD coefficients, and the coefficient increment is decoded back onto
        the physical prior members. Everything else — prior/posterior
        inflation, the parameter block, the appended observation-space
        diagnostic rows and the state spreads — stays in physical space. Every
        inflation scheme rescales rows independently, so inflating the physical
        augmented array reproduces the unreduced semantics exactly; relaxing
        coefficient rows would not, because RTPS/RTPP do not commute with a
        basis rotation.

        Args:
            pred_obs: The segment's predicted observations, ``(T*N_obs, N_e)``,
                frame-major.
            obs: The segment's real observations, ``(T, N_obs)``.
        """
        analysis_started = time.perf_counter()
        obs = jnp.asarray(obs)
        num_frames, N_obs = int(obs.shape[0]), int(obs.shape[1])
        N_d = num_frames * N_obs
        if pred_obs.shape[0] != N_d:
            raise ValueError(
                f"Predicted observations have {pred_obs.shape[0]} rows but the "
                f"observation segment has T={num_frames} frames of N_obs="
                f"{N_obs} ({N_d} rows). This usually indicates a mismatch "
                "between the observation operator and the supplied "
                "observations (a different number of output frames per "
                "segment), or stale files in the cycle results directory."
            )

        blocks: list[jnp.ndarray] = []
        n_state = 0
        n_param = 0
        flat_params: Optional[xarray.Dataset] = None
        states_forecast: Optional[jnp.ndarray] = None
        if self.mode in ("state", "joint"):
            states_forecast = self._state_augmentation.flatten(final_state)
            n_state = states_forecast.shape[0]
            blocks.append(states_forecast)
        if self.mode in ("parameter", "joint"):
            assert params is not None  # validated in run()
            flat_params = self._param_augmentation.flatten(params)
            params_array = ParamAugmentation.to_array(flat_params)
            n_param = params_array.shape[0]
            blocks.append(params_array)
        augmented = jnp.concatenate(blocks, axis=0)

        # Fit the basis to the RAW forecast anomalies: before inflation, and
        # before this cycle's observations influence anything. Validation runs
        # first and the fit itself is transactional, so one diverged member
        # cannot poison a streaming basis for the rest of the run.
        basis_seconds: Optional[float] = None
        if self.state_reduction is not None:
            assert states_forecast is not None  # mode='parameter' rejected above
            basis_started = time.perf_counter()
            self.state_reduction.fit(
                states_forecast,
                row_scales=self.state_reduction.resolve_row_scales(final_state),
            )
            basis_seconds = time.perf_counter() - basis_started

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

        # Encode the (inflated, physical) state block. ``encode`` subtracts the
        # forecast mean the basis was fitted around, which inflation preserves.
        analysis_rows = augmented
        coefficients_prior: Optional[jnp.ndarray] = None
        if self.state_reduction is not None:
            coefficients_prior = self.state_reduction.encode(augmented[:n_state])
            analysis_rows = jnp.concatenate(
                [coefficients_prior, augmented[n_state:]], axis=0
            )
        n_analysis = analysis_rows.shape[0]

        # Append the predicted observations as diagnostic rows: their update
        # is the exact posterior in observation space (they ride along without
        # influencing any other row), giving obs_posterior_rmse without a
        # second forecast. They are masked out of localization and stripped
        # before the split below. With T > 1 frames they are ALSO how the sweep
        # works: frame t's own rows are the predicted observations of its
        # analysis, and the other frames' rows are carried along the update, so
        # each frame is assimilated against what the earlier ones already did.
        augmented_ext = jnp.concatenate([analysis_rows, pred_obs], axis=0)
        # Localization and state reduction are rejected together at
        # construction, so these row descriptors always describe physical rows.
        # Sized with the PER-FRAME observation count (one analysis sees one
        # frame) but with all T*N_obs appended rows, whose mask/coords entries
        # are the same for every frame — the sensors do not move in time.
        group_ids, localize_mask, row_coords, obs_coords = self._localization_plumbing(
            final_state, n_state, n_param, N_obs, N_d
        )

        # One split per cycle, as before. With a single frame the subkey is used
        # DIRECTLY, so the classic one-analysis-per-cycle path draws exactly the
        # perturbations it always did; only a genuine sweep (T > 1) splits it
        # further, giving each frame its own independent draw.
        self.rng_key, subkey = jax.random.split(self.rng_key)
        frame_keys = (
            [subkey] if num_frames == 1 else list(jax.random.split(subkey, num_frames))
        )
        updated_ext = self._assimilate_frames(
            augmented_ext,
            n_analysis,
            obs,
            frame_keys,
            localization=self.localization,
            group_ids=group_ids,
            localize_mask=localize_mask,
            row_coords=row_coords,
            obs_coords=obs_coords,
        )
        updated = updated_ext[:n_analysis]
        pred_obs_post = updated_ext[n_analysis:]
        # The posterior twin of the ``_record_pred_obs`` call in ``run``: same
        # gate, same layout, one entry per cycle. No-op unless the caller opted
        # in, so the default path is unchanged.
        self._record_pred_obs_post(pred_obs_post)

        # Decode ONLY the coefficient increment onto the physical prior, so
        # every member keeps its own projection residual and a zero-gain
        # analysis leaves the full state numerically unchanged.
        state_increment: Optional[jnp.ndarray] = None
        if self.state_reduction is not None:
            assert coefficients_prior is not None
            n_coeff = coefficients_prior.shape[0]
            state_increment = self.state_reduction.decode_increment(
                updated[:n_coeff] - coefficients_prior
            )
            updated = jnp.concatenate(
                [augmented[:n_state] + state_increment, updated[n_coeff:]], axis=0
            )

        # Posterior (relaxation-style) inflation, e.g. RTPS/RTPP.
        post_mean = jnp.mean(updated, axis=1, keepdims=True)
        dev_post = updated - post_mean
        if self.inflation is not None:
            dev_post = self.inflation.inflate_posterior(dev_prior, dev_post)
            updated = post_mean + dev_post

        # The observation-space diagnostics are over the cycle's FRAMES: the
        # stacked (T*N_obs,) observation vector against the stacked forecast /
        # posterior rows, which for T == 1 is exactly today's per-cycle
        # statistic.
        cycle_diag = self._cycle_diagnostics(
            cycle,
            obs.reshape(-1),
            pred_obs_forecast,
            pred_obs_post,
            dev_prior,
            dev_post,
            n_state,
            n_param,
            num_frames,
        )
        # Read before _record_reduction_diagnostics below, which calls the
        # analysis a SECOND time (on the fit coordinates) and overwrites the
        # scheme's last-call attributes. Every shipped scheme recomputes the
        # same transform on that second call (same observations), so no
        # production configuration can tell the two orderings apart; reading
        # first is what ties the recorded values to the call that produced this
        # cycle's posterior rather than to an implementation detail of the
        # diagnostic. Pinned by test_filtering.py::test_transform_diagnostics_
        # come_from_the_posterior_producing_call, which makes the two calls
        # publish different transforms so the ordering is enforced rather than
        # merely asserted here.
        #
        # A scheme publishes the transform of its LAST call, so with T > 1 the
        # recorded transform_* / local_* values describe the sweep's final
        # frame (T == 1: the cycle's only analysis, unchanged).
        self._record_transform_diagnostics(cycle_diag)

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

        cycle_diag.analysis_time = time.perf_counter() - analysis_started
        if self.state_reduction is not None:
            assert states_forecast is not None and state_increment is not None
            self._record_reduction_diagnostics(
                cycle_diag,
                states_forecast=states_forecast,
                state_increment=state_increment,
                pred_obs=pred_obs,
                obs=obs,
                frame_keys=frame_keys,
                basis_seconds=basis_seconds,
            )

        return analysis_state, params, cycle_diag

    def _assimilate_frames(
        self,
        rows: jnp.ndarray,
        obs_offset: int,
        obs: jnp.ndarray,
        frame_keys: Sequence[jax.Array],
        **plumbing: Any,
    ) -> jnp.ndarray:
        """The serial sweep: one full-weight analysis per observation frame.

        ``rows`` is ``[analysis rows | predicted-observation frames]``, with
        frame ``t``'s ``N_obs`` predicted-observation rows at
        ``[obs_offset + t*N_obs, obs_offset + (t+1)*N_obs)``. Frame ``t``'s
        analysis uses those rows *as they currently are* — already updated by
        frames ``0..t-1`` — as its predicted observations, and updates every row
        of the array, so the state, the parameters and all the frames'
        observation-space rows advance together. That is what makes the sweep a
        serial (asynchronous) EnKF rather than T independent updates, and what
        keeps the appended rows a posterior observation-space diagnostic for
        every frame, not just the last one.

        The same helper serves the reduction's discarded-increment diagnostic
        (see :meth:`_record_reduction_diagnostics`), which must reproduce this
        sweep exactly — same frames, same order, same keys — for its difference
        to measure the truncation and nothing else.
        """
        num_frames, n_obs = int(obs.shape[0]), int(obs.shape[1])
        for frame in range(num_frames):
            start = obs_offset + frame * n_obs
            rows = self.analysis(
                rows,
                rows[start : start + n_obs],
                obs[frame],
                self.C_D_diag,
                frame_keys[frame],
                **plumbing,
            )
        return rows

    def _record_reduction_diagnostics(
        self,
        diag: CycleDiagnostics,
        *,
        states_forecast: jnp.ndarray,
        state_increment: jnp.ndarray,
        pred_obs: jnp.ndarray,
        obs: jnp.ndarray,
        frame_keys: Sequence[jax.Array],
        basis_seconds: Optional[float],
    ) -> None:
        """Fill one cycle's reduction fields (all ``None`` on the full path).

        The discarded-increment fraction never re-runs a full-space analysis:
        the fit already produced the forecast anomalies' coordinates in the
        complete (untruncated) basis, ``anomalies = U_full @ C``, and the EnKF
        increment is ``anomalies @ W`` for an ensemble-space weight matrix
        ``W``. Applying the analysis to ``C`` itself — a tiny ``(k, N_e)``
        array whose rows are already centred — yields ``C @ W``, so rows
        ``[rank:]`` are exactly the part of the update the truncation drops.
        The fraction is measured in the reduction's own (scaled) norm and is
        unaffected by prior inflation, which rescales both parts alike.

        With a multi-frame cycle the weight matrix is the sweep's composition,
        so the coordinates go through the SAME :meth:`_assimilate_frames` call
        — with the frames appended as ride-along rows, so they evolve exactly
        as they did in the real sweep, and with the same per-frame keys. The
        appended rows are stripped again before the norms are taken.
        """
        reduction = self.state_reduction
        assert reduction is not None
        diag.reduction_rank = reduction.rank
        diag.reduction_available_rank = reduction.available_rank
        diag.reduction_retained_energy = reduction.retained_energy_fraction
        diag.reduction_projection_residual = reduction.projection_residual(
            states_forecast
        )
        diag.reduction_increment_norm = float(jnp.linalg.norm(state_increment))
        diag.reduction_basis_time = basis_seconds
        diag.reduction_basis_updated = reduction.basis_updated
        condition = reduction.condition_number
        diag.reduction_condition_number = (
            None if condition is None else float(condition)
        )
        diag.reduction_spectrum_max = reduction.spectrum_max
        drift = reduction.subspace_drift
        diag.reduction_subspace_drift = None if drift is None else float(drift)

        coordinates = reduction.fit_coordinates
        if coordinates is None:
            return
        n_coordinates = coordinates.shape[0]
        swept = self._assimilate_frames(
            jnp.concatenate([coordinates, pred_obs], axis=0),
            n_coordinates,
            obs,
            frame_keys,
        )
        increment_coordinates = swept[:n_coordinates] - coordinates
        full_norm = float(jnp.linalg.norm(increment_coordinates))
        diag.reduction_discarded_increment_fraction = (
            float(jnp.linalg.norm(increment_coordinates[reduction.rank :]) / full_norm)
            if full_norm > 0.0
            else 0.0
        )

    def _record_transform_diagnostics(self, diag: CycleDiagnostics) -> None:
        """Fill one cycle's ensemble-transform fields (all ``None`` otherwise).

        Deterministic analyses publish what they actually computed — the
        observation-space rank they had available, the rank they kept, and (for
        a localized transform) how many local blocks that took — as attributes
        of the last call. Without this hook those numbers exist only inside the
        scheme: they are the resource-gate quantities of
        ``docs/plans/filtering_state_reduction_and_transforms.md`` §6, and the
        cost and meaning of a localized analysis are otherwise invisible from
        its output.

        Read by name rather than by type: any scheme that does not publish
        these attributes (the stochastic analysis, and any future scheme)
        leaves both groups ``None``, and this module never imports the
        ensemble-transform module.
        """
        transform = getattr(self.analysis, "last_transform", None)
        if transform is not None:
            available = transform.available_rank
            diag.transform_available_rank = (
                None if available is None else int(available)
            )
            diag.transform_retained_rank = int(transform.retained_rank)
            energy = transform.retained_energy
            diag.transform_retained_energy = None if energy is None else float(energy)
            discarded = jnp.asarray(transform.discarded_spectrum)
            # 0.0 = the truncation discarded nothing; None (the default) = no
            # global transform ran at all.
            diag.transform_discarded_spectrum_max = (
                float(jnp.max(discarded)) if discarded.size else 0.0
            )

        local = getattr(self.analysis, "last_diagnostics", None)
        if local is None:
            return
        diag.local_num_blocks = int(local.num_blocks)
        diag.local_num_active_blocks = int(local.num_active_blocks)
        diag.local_num_updated_rows = int(local.num_updated_rows)
        diag.local_chunk_size = int(local.chunk_size)

        # Kept in NumPy: these arrays are per block (one per state row in the
        # worst case), and summarizing them host-side avoids a device round trip
        # every cycle for numbers nobody differentiates through.
        counts = np.asarray(local.active_observation_counts)
        active_counts = counts[counts > 0]
        if active_counts.size:
            diag.local_active_obs_min = int(active_counts.min())
            diag.local_active_obs_median = float(np.median(active_counts))
            diag.local_active_obs_max = int(active_counts.max())
        retained = np.asarray(local.retained_ranks)
        if retained.size:
            diag.local_retained_rank_min = int(retained.min())
            diag.local_retained_rank_mean = float(retained.mean())
            diag.local_retained_rank_max = int(retained.max())
        available_ranks = np.asarray(local.available_ranks)
        if available_ranks.size:
            diag.local_available_rank_max = int(available_ranks.max())
        energies = np.asarray(local.retained_energies)
        if energies.size:
            diag.local_retained_energy_min = float(energies.min())
            diag.local_retained_energy_mean = float(energies.mean())
        block_discarded = np.asarray(local.discarded_spectrum_maxima)
        if block_discarded.size:
            diag.local_discarded_spectrum_max = float(block_discarded.max())

    def _localization_plumbing(
        self,
        final_state: xarray.Dataset,
        n_state: int,
        n_param: int,
        n_obs: int,
        n_appended: int,
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

        Two observation counts, because one analysis sees one frame while the
        appended block holds all of them: ``n_obs`` is the PER-FRAME count that
        sizes ``obs_coords`` (every frame observes the same sensors, so one
        frame's coordinates serve them all), ``n_appended = T * n_obs`` the
        number of ride-along rows the row-wise descriptors must cover. They are
        equal on the ``T = 1`` path.
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
        mask_blocks.append(jnp.zeros(n_appended, dtype=bool))
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
            id_blocks.append(offset + jnp.arange(n_appended, dtype=int))
            group_ids = jnp.concatenate(id_blocks)

        row_coords = None
        obs_coords = None
        if self.localization.requires_coordinates:
            # mode='parameter' is rejected at construction, so state rows exist.
            state_coords = self._state_augmentation.row_coords(final_state)
            row_coords = jnp.concatenate(
                [state_coords, jnp.zeros((n_param + n_appended, 3))], axis=0
            )  # non-state rows are masked out above
            obs_coords = jnp.asarray(
                sensor_observation_coords(self.observation_operator, n_obs)
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
        num_frames: int = 1,
    ) -> CycleDiagnostics:
        """Innovation statistics and block spreads for one cycle.

        Every observation-space quantity is over the cycle's ``num_frames``
        frames stacked together — ``obs`` is the ``(T*N_obs,)`` vector, and
        ``pred_obs`` / ``pred_obs_post`` the matching ``(T*N_obs, N_e)`` rows —
        so ``obs_prior_rmse``, ``obs_posterior_rmse`` and ``innovation_chi2``
        answer "how well did this cycle fit the observations it assimilated?".
        ``C_D`` is per frame, so the stacked system's error covariance is
        ``tile(C_D_diag, T)``, the block-diagonal repetition of it.

        ``pred_obs`` must be the raw forecast (pre-inflation) so the chi2
        spread term reflects what the model produced, not what the inflation
        chose; the block spreads intentionally use the (possibly inflated)
        analysis anomalies.

        The appended predicted-observation rows take a global ride-along
        update, so ``obs_posterior_rmse`` is only ``H`` applied to the analyzed
        state on the unlocalized, unreduced path — and that is unchanged by the
        serial sweep, whose every frame updates all of those rows too. Under a
        state reduction it is computed from the FULL-space observation-row
        update and is therefore insensitive to truncation (bit-identical to the
        unreduced filter); under localization the state rows are localized while
        these rows are not. ``obs_posterior_rmse_kind`` labels which case
        produced the value so it is never silently compared across
        configurations.
        """
        N_d = obs.shape[0]
        N_e = pred_obs.shape[1]

        innovation = obs - jnp.mean(pred_obs, axis=1)
        pred_obs_dev = pred_obs - jnp.mean(pred_obs, axis=1, keepdims=True)
        C_DD = jnp.dot(pred_obs_dev, pred_obs_dev.T) / (N_e - 1)
        S = C_DD + jnp.diag(jnp.tile(self.C_D_diag, num_frames))
        chi2 = float(
            innovation
            @ jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(S), innovation)
            / N_d
        )

        def _block_spread(dev: jnp.ndarray, start: int, size: int) -> Optional[float]:
            if size == 0:
                return None
            return float(jnp.mean(jnp.std(dev[start : start + size], axis=1, ddof=1)))

        if self.state_reduction is not None:
            obs_posterior_kind = "unreduced_ride_along"
        elif self.localization is not None:
            obs_posterior_kind = "unlocalized_ride_along"
        else:
            obs_posterior_kind = "exact"

        return CycleDiagnostics(
            cycle=cycle,
            obs_posterior_rmse_kind=obs_posterior_kind,
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
        state_reduction: Optional[OnlineStateReduction] = None,
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
            state_reduction=state_reduction,
            rng_key=rng_key,
        )
