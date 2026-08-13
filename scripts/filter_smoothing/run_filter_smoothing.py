"""Run the filter-smoothing HYBRID: an ESMDA parameter MDA finished by a filter.

The third assimilation entry point, sitting between ``scripts/esmda/run_esmda.py``
and ``scripts/filtering/run_filtering.py`` and deliberately built out of both.
Per assimilation window:

  1. the SMOOTHER runs ESMDA's normal multiple-data-assimilation loop over the
     whole window, estimating parameters only — static scalars
     (``ParameterESMDA``) or an AR(2) knot trajectory
     (``TimeVaryingParameterESMDA``) — but with the seam
     ``final_forecast=False``, so the posterior forward pass is SKIPPED;
  2. the FILTER (an ``EnsembleKalmanFilter``, ``mode="state"`` or ``"joint"``)
     then forecasts the window observation to observation using those
     parameters and applies one full-weight analysis per cycle, producing the
     window's state — and, in joint mode, a parameter CORRECTION carried on top
     of the ESMDA schedule.

Both halves are driven through ``data_assimilation.FilterSmoothing``, which owns
the algorithm; this script owns the experiment: geometry, truth, the two
observation products, the two assimilation model stacks, and every artifact.

Declarative axes (see conf/run_filter_smoothing.yaml):

  * ``esmda/smoother=static|dynamic``
        the MDA parameter loop. The state-bearing smoothers are rejected: the
        filter owns the state here.
  * ``params@prior_params=static|dynamic``
        pair with the smoother above.
  * ``filtering.mode=state|joint``
        whether the filter also corrects the parameters. ``parameter`` is not a
        hybrid mode (no state update means the filter would only re-derive what
        the smoother just estimated).
  * ``filtering/analysis|localization|state_reduction|inflation|evolution``
        the filter's machinery, reused unchanged from run_filtering.yaml.
  * ``filter_smoothing.num_assimilation_windows=W``
        the horizon, in windows.

and the truth source, mirroring both siblings:

  * ``run.truth_dir=null``    simulate the truth inline (default).
  * ``run.truth_dir=<path>``  load a state.nc/params.nc artifact.

GEOMETRY AND THE TWO OBSERVATION PRODUCTS. One window is
``time.simulation_time`` seconds — the same unit ``esmda.num_assimilation_windows``
and ``filtering.num_assimilation_windows`` count. Inside it the two halves see
the SAME per-cycle observations, built once, in global cycle order, before the
window loop:

  * the FILTER consumes them as they are: one cycle is
    ``filtering.assimilate_every_n_step`` observation intervals
    (``time.output_frequency`` s each), of which only the LAST frame — the
    analysis time — is assimilated, exactly as in run_filtering.py (the
    intermediate frames are still simulated and still written). A window
    therefore holds ``simulation_time / (output_frequency * every_n)`` cycles,
    DERIVED and validated to divide evenly, never configured.
  * the SMOOTHER consumes their concatenation over the window, aggregated by its
    own ``AggregateObservations`` into ``esmda.interval_seconds`` bins. That
    aggregation happens INSIDE the smoother, as it does in run_esmda.py; nothing
    in the filter phase aggregates anything.

  Under a stride the thinning applies to BOTH halves identically — the real
  batches keep only each cycle's analysis frame, and both DA instances get the
  same ``_StridedObservationOperator`` wrapper so their predicted observations
  are subset row for row — which is what keeps the run at exactly ONE
  observation product (``esmda.interval_seconds`` must then be coarse enough
  that every aggregation bin still contains a strided frame, or the smoother's
  aggregator raises on the empty bin).

THE OBSERVATION CLOCK IS WINDOW-LOCAL AND NOMINAL. Each cycle's batch is
ASSIGNED the time coordinate ``(local_cycle + 1) * cycle_seconds`` — the
nominal END of its forecast segment on the window clock — rather than keeping
the truth's own frame times. Two reasons, both load-bearing:

  * both backends rebase their post-spin-up frames to START at t = 0 (pylbm
    ``assign_coords(time=arange(N)*output_frequency)``, pyudales
    ``time - time[0]``), so the raw coordinate of a window's first frame is
    0.0 (window 0) or off the nominal grid by the truth's accumulated cadence
    drift (later windows) — both of which ``segment_bounds`` rightly rejects.
    Semantically, truth frame ``l`` IS the frame the cycle-``l`` analysis
    consumes at the END of its forecast segment; the nominal clock states that
    convention explicitly instead of hoping the raw coordinates imply it.
  * the filter's forward model integrates EXACTLY ``cycle_seconds`` (=
    ``output_frequency`` x the stride) per cycle, so nominal bounds keep the
    trajectory restriction (``params_for_segment``) in lockstep with what the
    model actually runs — truth-cadence jitter must not stretch the parameter
    schedule.

The nominal clock is the axis run_esmda.py's per-window trajectory prior
already lives on (knots on ``[0, simulation_time]``), which is exactly the
alignment the hybrid needs. It is inert to the smoother's aggregator (which
bins relative to its first frame) and to the filter itself (which consumes
batches positionally). PHYSICAL frame times are kept separately in
``cycle_times`` and used for the artifacts.

ARTIFACTS. Written in BOTH downstream schemas from one run, exactly as
run_filtering.py does, so neither pipeline needs a hybrid-specific reader:

  * the ESMDA per-window schema —
    ``windows/window_{w}_{prior,posterior}_params.nc`` (the posterior IS the MDA
    posterior ``result.esmda_params``), ``window_{w}_posterior_state.nc`` (the
    filter's analyzed frames), ``window_{w}_{obs,pred_obs}.nc`` and the assembled
    ``prior_params.nc`` / ``posterior_params.nc`` / ``posterior_state_mean.nc``;
  * the filtering-native artifacts with GLOBAL cycle indices —
    ``posterior_state.nc``, ``cycle_diagnostics.yaml``, and (under
    ``run.save_history``) ``params_history.nc`` / ``state_history.nc``;
  * and two hybrid-specific ones: ``windows/window_{w}_filter_params.nc`` (joint
    mode: the filter's corrected parameters, which the ESMDA-schema posterior
    deliberately does NOT contain) and ``windows/window_{w}_esmda_pred_obs.nc``
    (the MDA iterations' predicted observations — ``num_steps`` entries with NO
    posterior entry, see ``_ESMDA_PRED_OBS_SEMANTICS``).

There is no ``window_{w}_prior_state.nc``: as in a pure filtering run there is
no window-long prior rollout to save (the MDA's own iterate forecasts are
pruned), and ``run_info.yaml``'s ``configuration.save_prior_state: false``
records the absence so the shared stages skip the prior halves.

Examples::

    python scripts/filter_smoothing/run_filter_smoothing.py \
        esmda/smoother=dynamic params@prior_params=dynamic filtering.mode=joint
    python scripts/filter_smoothing/run_filter_smoothing.py \
        esmda/smoother=static params@prior_params=static \
        params@truth_params=static_truth filtering.mode=joint
    python scripts/filter_smoothing/run_filter_smoothing.py \
        esmda/smoother=dynamic params@prior_params=dynamic \
        filtering.mode=state filter_smoothing.num_assimilation_windows=4
"""

import dataclasses
import pathlib
import sys
import time
from typing import Any, Sequence

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import xarray
from data_assimilation import FilterSmoothing
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise; must precede `import jax`)
from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    create_aggregate_observations,
    create_observation_operator,
    filter_parameter_config,
)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# The two sibling entry points are the source of every artifact writer used
# here; the hybrid's whole point is that its outputs are byte-compatible with
# theirs, so REUSING the writers is the only way to keep that true as they
# evolve. Both modules are import-safe (module level does imports and constant
# definitions only), and this is the same cross-package import
# scripts/filtering/run_filtering.py already makes into scripts/esmda.
from scripts.esmda._esmda_common import (  # noqa: E402
    open_truth,
    truth_x_min,
    write_yaml,
)
from scripts.esmda.run_esmda import (  # noqa: E402
    _OBS_DIM,
    _OBS_ORDERING,
    _flatten_obs,
    _obs_index_coords,
    _save_assembled_outputs,
)
from scripts.filtering.run_filtering import (  # noqa: E402
    _ESMDA_STEP_SEMANTICS,
    _collect_window_cycle_dirs,
    _flat_obs_vector,
    _moment_sampling,
    _save_window_obs_diagnostics,
    _save_window_params,
    _save_window_state,
    _stack_cycle_pred_obs,
    _StridedObservationOperator,
    _window_staging_dir,
)

# ---------------------------------------------------------------------------
# Semantics recorded in the artifacts
# ---------------------------------------------------------------------------

# The two entries of the FILTERING-schema window_{w}_pred_obs.nc are
# run_filtering.py's `_ESMDA_STEP_SEMANTICS`, imported above: the file is
# produced by the same writer from the same per-cycle lists, because the
# hybrid's filter phase is an ordinary filter run.

# The entries of the hybrid-only window_{w}_esmda_pred_obs.nc. Spelled out at
# length because it is precisely where the hybrid departs from run_esmda.py's
# artifact contract, and a reader that assumed the ESMDA schema would silently
# score an MDA iterate as if it were the posterior.
_ESMDA_PRED_OBS_SEMANTICS = (
    "esmda_step i = the predicted observations H(x) of MDA iteration i, i.e. "
    "the forecast the i-th tempered update was computed FROM. There are "
    "exactly esmda.num_steps entries and the LAST one is NOT a posterior "
    "forecast: the hybrid calls the smoother with final_forecast=False (the "
    "FILTER produces the window's posterior state), so run_esmda.py's "
    "'entry -1 = posterior forecast' convention does NOT hold here. That is "
    "also why this file is not named window_{w}_pred_obs.nc, which the shared "
    "diagnostics read and which holds the filter's per-cycle prior/posterior "
    "rows on the raw (unaggregated) observation axis of window_{w}_obs.nc. "
    "This file's own axis is the smoother's AGGREGATED one "
    "(esmda.interval_seconds bins), so its obs/obs_clean/obs_error_std "
    "variables are carried alongside rather than being read from that file."
)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _nominal_window_clock(
    observations: Any, local_cycle: int, cycle_seconds: float
) -> Any:
    """Place one cycle's batch on the NOMINAL window clock.

    The batch's ``T`` frames are assigned times tiling the cycle's forecast
    segment, ending exactly at its nominal end ``(local_cycle + 1) *
    cycle_seconds`` (with the v1-pinned one frame per cycle, that single value).
    The truth's own frame times are deliberately NOT used: both backends rebase
    their frames to start at t = 0 and carry cadence jitter, so the raw
    coordinates neither mark the segment ENDS the analyses consume nor stay on
    the grid the filter's forward model (which integrates exactly
    ``cycle_seconds`` per cycle) and the window-local trajectory prior live on.
    See the module docstring; physical frame times live in ``cycle_times``.
    """
    num_frames = int(observations.sizes["time"])
    frame_times = float(local_cycle) * cycle_seconds + (
        np.arange(1, num_frames + 1, dtype=float) * (cycle_seconds / num_frames)
    )
    return observations.assign_coords(time=frame_times)


def _save_window_esmda_pred_obs(
    windows_dir: pathlib.Path,
    window: int,
    pred_obs_history: Sequence[np.ndarray],
    obs: np.ndarray,
    obs_clean: np.ndarray,
    obs_error_std: np.ndarray,
    obs_op: Any,
) -> None:
    """Write the MDA iterations' observation-space arrays for one window.

    The smoother half's counterpart to ``_save_window_obs_diagnostics``, on the
    smoother's own AGGREGATED observation axis and under its own file name. See
    :data:`_ESMDA_PRED_OBS_SEMANTICS` for why this is not
    ``window_{w}_pred_obs.nc``; the semantics are copied into the file's attrs
    so the distinction travels with the artifact.

    A smoother that recorded nothing (an empty history) writes no file, matching
    ``run_esmda.py``'s behaviour rather than emitting an empty ``esmda_step``
    axis downstream readers would have to special-case.
    """
    if not len(pred_obs_history):
        return
    stacked = np.stack([np.asarray(p, dtype=float) for p in pred_obs_history])
    n_d = int(stacked.shape[1])
    coords = _obs_index_coords(obs_op, n_d)

    ds = xarray.Dataset(
        data_vars={
            "pred_obs": (("esmda_step", _OBS_DIM, "ensemble"), stacked),
            "obs": (_OBS_DIM, np.asarray(obs, dtype=float).ravel()),
            "obs_clean": (_OBS_DIM, np.asarray(obs_clean, dtype=float).ravel()),
            "obs_error_std": (
                _OBS_DIM,
                np.asarray(obs_error_std, dtype=float).ravel(),
            ),
        },
        coords={
            "esmda_step": np.arange(stacked.shape[0]),
            _OBS_DIM: np.arange(n_d),
            **coords,
        },
    )
    ds.attrs["ordering"] = _OBS_ORDERING
    ds.attrs["esmda_step"] = _ESMDA_PRED_OBS_SEMANTICS
    ds["pred_obs"].attrs["long_name"] = "MDA-iteration predicted observations"
    ds["obs"].attrs["long_name"] = "assimilated observation (truth + noise)"
    ds["obs_clean"].attrs["long_name"] = "noise-free truth projection"
    ds["obs_error_std"].attrs["long_name"] = "sqrt(diag(C_D)), un-inflated"
    ds.to_netcdf(windows_dir / f"window_{window}_esmda_pred_obs.nc")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    # --- Geometry (all validated BEFORE any solver is started) ----------------
    num_windows = int(cfg.filter_smoothing.num_assimilation_windows)
    if num_windows < 1:
        raise ValueError(
            "filter_smoothing.num_assimilation_windows must be >= 1, got "
            f"{num_windows}."
        )
    sim_time = float(cfg.time.simulation_time)  # ONE window
    output_frequency = float(cfg.time.output_frequency)
    if output_frequency <= 0.0:
        raise ValueError(
            "time.output_frequency must be > 0: it is the observation cadence "
            "the filter's forecast segments are built from."
        )
    # The filter's analysis stride, exactly as in run_filtering.py: a cycle
    # spans `every_n` observation intervals, all of them simulated and written,
    # and only the LAST frame — the analysis time — is assimilated. The stride
    # thins BOTH halves' observations here (see the obs loop below), so the run
    # still has exactly one observation product.
    every_n = int(cfg.filtering.get("assimilate_every_n_step", 1))
    if every_n < 1:
        raise ValueError(
            "filtering.assimilate_every_n_step must be >= 1 (1 = assimilate "
            f"every observation frame), got {every_n}."
        )
    cycle_seconds = every_n * output_frequency  # ONE cycle
    # The filter's cycles must tile the window exactly: the smoother assimilates
    # the whole window and the filter the same frames one at a time, so a
    # partial cycle at the boundary would leave the two halves scoring different
    # observation sets. Checked here, at config time, like run_filtering.py's
    # stride check.
    cycles_per_window_nominal = round(sim_time / cycle_seconds)
    if cycles_per_window_nominal < 1 or abs(
        sim_time - cycles_per_window_nominal * cycle_seconds
    ) > 1e-9 * max(sim_time, cycle_seconds):
        raise ValueError(
            f"One cycle is time.output_frequency x "
            f"filtering.assimilate_every_n_step = {cycle_seconds:g}s, which "
            f"does not divide time.simulation_time={sim_time:g}s, so the "
            "filter's cycles would not tile the assimilation window. Adjust "
            "time.simulation_time, time.output_frequency or the stride."
        )
    ensemble_size = int(cfg.ensemble.ensemble_size)
    final_time = sim_time * num_windows
    domain_x_min = float(cfg.domain.bounds[0][0])
    rng_key = jax.random.PRNGKey(cfg.filter_smoothing.seed)

    # --- Mode validation ------------------------------------------------------
    # FilterSmoothing's constructor is the authoritative check for both of
    # these, but it is only reachable after the truth has been simulated (its
    # C_D comes from the truth's observations), so mirror them here — a typo in
    # a mode name must not cost a full truth rollout first.
    filter_mode = str(cfg.filtering.mode)
    if filter_mode not in ("state", "joint"):
        raise ValueError(
            f"filtering.mode={filter_mode!r} is not a hybrid mode. Use 'state' "
            "(the ESMDA parameters ride through the filter unmodified) or "
            "'joint' (the filter additionally corrects them on the ESMDA "
            "schedule). 'parameter' would only re-estimate what the smoother "
            "just estimated, with no state update at all — use "
            "scripts/filtering/run_filtering.py for a pure parameter filter."
        )
    smoother_target = str(cfg.esmda.smoother._target_).rsplit(".", 1)[-1]
    if "State" in smoother_target:
        raise ValueError(
            f"esmda/smoother selects {smoother_target}, a state-bearing "
            "smoother, which the hybrid rejects: the FILTER owns the state "
            "here, so a smoother that also estimates it would fight the filter "
            "for the same increment. Use esmda/smoother=static or "
            "esmda/smoother=dynamic (parameter-only), or "
            "scripts/esmda/run_esmda.py for a state-bearing smoother."
        )

    # A time-varying PRIOR is what makes the ESMDA posterior a knot TRAJECTORY,
    # which in turn is what makes the filter phase run one pass per forecast
    # segment instead of one pass per window. A time-varying TRUTH is an
    # independent choice (it only sets the truth sampler's horizon), so unlike
    # run_esmda.py — which pairs the two and reads the flag off the truth — the
    # two flags are kept apart here.
    is_dynamic_prior = "seconds_per_knot" in list(cfg.prior_params.keys())
    is_dynamic_truth = "seconds_per_knot" in list(cfg.truth_params.keys())

    # Select which parameters are estimated (the same contract as both
    # siblings): null -> every parameter the sampler configs define; a list ->
    # that subset, applied to prior AND truth.
    selected = cfg.get("params_to_estimate", None)
    selected = list(selected) if selected is not None else None
    truth_params_cfg = filter_parameter_config(cfg.truth_params, selected)
    prior_params_cfg = filter_parameter_config(cfg.prior_params, selected)
    if selected is not None:
        print(f"Estimating parameters: {selected}")

    # --- Output and windows dir -----------------------------------------------
    # The run lands in the configured results dir (not a timestamped Hydra one),
    # exactly as both siblings do, so the downstream metric/figure stages locate
    # it from the config alone.
    out_dir = pathlib.Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=out_dir / "config.yaml")

    # --- Truth (simulated inline, or loaded from disk) ------------------------
    if cfg.run.truth_dir is None:
        true_forward_model = instantiate(
            cfg.truth_model.forward_model,
            results_dir=None,
            simulation_time=final_time,
        )
        # A dynamic truth must span the FULL horizon so it drifts across every
        # window (the same override run_esmda.py and run_filtering.py apply); a
        # static truth ignores the horizon entirely.
        if is_dynamic_truth:
            truth_sampler = instantiate(truth_params_cfg, simulation_time=final_time)
        else:
            truth_sampler = instantiate(truth_params_cfg)
        true_params = truth_sampler.sample(1)

        instantiate(cfg.truth_model.prepare, forward_model=true_forward_model)
        clean_outputs(model_name=cfg.truth_model.name, forward_model=true_forward_model)
        true_state = true_forward_model(params=true_params.isel(ensemble=0))

        true_state_path = out_dir / "true_state.nc"
        true_state.to_netcdf(true_state_path)
        n_total = int(true_state.sizes["time"])
        x_offset = domain_x_min - truth_x_min(true_state)
        start_idx = 0
        t_offset = 0.0
        del true_state
    else:
        truth_dir = pathlib.Path(cfg.run.truth_dir)
        true_params = xarray.load_dataset(truth_dir / "params.nc")

        # Optionally begin the horizon partway into a pre-simulated truth (skip
        # a spin-up); frames are rebased so that time becomes t=0.
        start_time = float(cfg.run.truth_start_time or 0.0)
        if "time" in true_params.dims:
            if start_time:
                true_params = true_params.sel(time=true_params.time >= start_time)
                true_params = true_params.assign_coords(
                    time=true_params.time - start_time
                )
            true_params = true_params.sel(time=true_params.time < final_time)

        true_state_path = truth_dir / "state.nc"
        with xarray.open_dataset(true_state_path) as _truth_meta:
            true_times = np.asarray(_truth_meta["time"].values, dtype=float)
            x_offset = domain_x_min - truth_x_min(_truth_meta)
        start_idx = int((true_times < start_time).sum())
        t_offset = start_time
        n_total = int(((true_times[start_idx:] - t_offset) < final_time).sum())

    if x_offset:
        print(
            f"Shifting truth x by {x_offset:+g} to align with domain "
            f"x_min={domain_x_min:g}"
        )

    # Truth frames per window (contiguous half-open blocks, as run_esmda.py
    # slices them). One cycle is one frame here, so the two counts coincide —
    # the stride that separates them in run_filtering.py is pinned to 1.
    if n_total < num_windows:
        raise ValueError(
            f"The truth provides {n_total} frame(s) within the {final_time:g}s "
            f"horizon, fewer than filter_smoothing.num_assimilation_windows="
            f"{num_windows} (each window needs at least one observation "
            "frame). Increase time.simulation_time, reduce "
            "time.output_frequency or num_assimilation_windows, or point "
            "run.truth_dir at a longer truth."
        )
    cycles_per_window = n_total // num_windows // every_n
    frames_per_window = cycles_per_window * every_n
    n_dropped = n_total - frames_per_window * num_windows
    if n_dropped:
        print(
            f"Truth frames ({n_total}) do not divide evenly into {num_windows} "
            f"window(s) of {cycles_per_window} cycle(s) x {every_n} frame(s); "
            f"the trailing {n_dropped} frame(s) are not assimilated."
        )
    n_total = frames_per_window * num_windows
    num_cycles = cycles_per_window * num_windows
    if cycles_per_window != cycles_per_window_nominal:
        # A hard error, not a warning: the batches are placed on the NOMINAL
        # window clock and the filter's forward model integrates exactly
        # `cycle_seconds` per cycle, so a truth whose cadence disagrees would
        # be silently assimilated at the wrong times (and the trajectory
        # restricted to the wrong segments) rather than merely sampled oddly.
        raise ValueError(
            f"The truth provides {cycles_per_window} cycle(s) of "
            f"{every_n} frame(s) per {sim_time:g}s window, but "
            f"time.output_frequency={output_frequency:g}s x "
            f"filtering.assimilate_every_n_step={every_n} implies "
            f"{cycles_per_window_nominal}. The hybrid requires the truth's "
            "output cadence to match time.output_frequency; regenerate the "
            "truth or point run.truth_dir at one with the matching cadence."
        )
    print(
        f"{num_windows} window(s) of {sim_time:g}s = {cycles_per_window} "
        f"cycle(s) of {cycle_seconds:g}s each ({num_cycles} cycles over "
        f"{final_time:g}s, assimilating every {every_n} observation frame(s))."
    )
    true_params.to_netcdf(out_dir / "true_params.nc")

    # --- TWO assimilation model stacks ----------------------------------------
    # The forecast horizon is fixed at INSTANTIATE time, and the two halves need
    # different ones: the smoother re-forecasts the whole window every MDA
    # iteration, while the filter forecasts one cycle per analysis. One stack
    # each, therefore — the `simulation_time` override on the filter's is the
    # same lever run_filtering.py pulls (`run_filtering.py:610-624`), and it
    # leaves the model's `output_frequency` alone so a cycle emits exactly one
    # frame.
    #
    # Each stack gets its OWN `temp_dir` subtree. The backends nest their
    # experiment dir AND the per-member `ensemble_experiments/{NNN}` copies
    # under `temp_dir`, so two same-config stacks sharing the default would
    # clobber each other: the second constructor re-copies the case files over
    # the first's (for pyudales that overwrites RUN/runtime with the filter's
    # one-cycle horizon, so every ESMDA window forecast would come up short and
    # fail the frame-count check), and the shared member dirs would mix the
    # two halves' warm-start carry/restart files — a filter cycle would warm-
    # start its subgrid state from whatever MDA window rollout last wrote
    # there. Distinct subtrees make the two stacks as independent as the two
    # scripts they were lifted from. (The truth model keeps the base
    # `experiment_dir`, as in the siblings — it finishes before either stack
    # is built.)
    assim_results_dir = (
        pathlib.Path(cfg.run.results_dir) if cfg.run.results_dir is not None else None
    )
    stack_temp_root = pathlib.Path(cfg.paths.experiment_dir)
    esmda_temp_dir = stack_temp_root / "filter_smoothing_esmda"
    filter_temp_dir = stack_temp_root / "filter_smoothing_filter"
    esmda_temp_dir.mkdir(parents=True, exist_ok=True)
    filter_temp_dir.mkdir(parents=True, exist_ok=True)
    esmda_assim_model = instantiate(
        cfg.assim_model.forward_model,
        results_dir=assim_results_dir,
        temp_dir=esmda_temp_dir,
    )
    instantiate(cfg.assim_model.prepare, forward_model=esmda_assim_model)
    filter_assim_model = instantiate(
        cfg.assim_model.forward_model,
        results_dir=assim_results_dir,
        temp_dir=filter_temp_dir,
        simulation_time=cycle_seconds,
    )
    instantiate(cfg.assim_model.prepare, forward_model=filter_assim_model)

    # On-disk ensemble forecasts get SEPARATE roots per half: the smoother
    # writes `step_{i}/` directories and the filter `cycle_{k}/` ones, and
    # letting the two layouts cohabit would have each half's pruning delete
    # directories it does not own.
    save_on_disk = bool(cfg.run.get("ensemble_save_on_disk", False))
    esmda_states_dir = out_dir / "_ensemble_states" / "esmda" if save_on_disk else None
    filter_states_dir = (
        out_dir / "_ensemble_states" / "filter" if save_on_disk else None
    )
    esmda_ensemble_model = instantiate(
        cfg.assim_model.ensemble_model,
        forward_model=esmda_assim_model,
        results_dir=esmda_states_dir,
    )
    filter_ensemble_model = instantiate(
        cfg.assim_model.ensemble_model,
        forward_model=filter_assim_model,
        results_dir=filter_states_dir,
    )

    # --- Prior parameter ensemble ---------------------------------------------
    # NOT written here: the run's `prior_params.nc` is the ASSEMBLED per-window
    # prior (this ensemble is window 0's), written with the posterior at the end.
    prior_sampler = instantiate(prior_params_cfg)
    prior_params = prior_sampler.sample(ensemble_size)

    # --- Observation operators and the per-cycle observations ------------------
    truth_obs_op = create_observation_operator(cfg.obs, cfg.truth_model.solver_name)
    assim_obs_op = create_observation_operator(cfg.obs, cfg.assim_model.solver_name)
    # The predicted-observation twin of the truth-side stride, for BOTH halves
    # (run_filtering.py's wrapper): the filter's cycle forecast emits `every_n`
    # frames and the smoother's window forecast every frame of the window,
    # while the real observations below keep only the strided subset — so each
    # half's H(x) must be subset the same way, row for row. Wrapped only when
    # the stride bites, so an unstrided run hands both halves the very same
    # operator object it always did.
    if every_n > 1:
        assim_obs_op = _StridedObservationOperator(assim_obs_op, every_n)
    # Interval aggregation for the SMOOTHER ONLY. ONE instance, shared between
    # the C_D sizing below and the smoother, so its interval-count consistency
    # check spans the truth and the forecasts (run_esmda.py's contract).
    aggregate_obs = create_aggregate_observations(cfg.esmda)

    obs_error_std = float(cfg.filter_smoothing.obs_error_std)
    # Every cycle's observation is built and perturbed HERE, over the whole
    # horizon and in global cycle order, before the window loop — the same
    # windowing-inert draw sequence run_filtering.py uses. The arrays are
    # KB-scale (sensors x states per cycle); the truth stays lazily on disk.
    observations: list[Any] = []
    observations_clean: list[Any] = []
    cycle_times: list[float] = []  # PHYSICAL (global) frame times, for the artifacts
    truth_view = open_truth(true_state_path, n_total, x_offset, start_idx, t_offset)
    for cycle in range(num_cycles):
        window_of_cycle = cycle // cycles_per_window
        # The cycle's truth BLOCK is its `every_n` output frames (half-open, so
        # no frame is covered twice); the frames it ASSIMILATES are the strided
        # subset ending on the block's last frame — the analysis time
        # (run_filtering.py's stride slicing, verbatim). At every_n = 1 the
        # block is one frame and the stride is the identity. The SMOOTHER later
        # assimilates the concatenation of exactly these strided batches, so
        # the stride thins both halves' observations identically and the run
        # keeps one observation product.
        cycle_block = truth_view.isel(
            time=slice(cycle * every_n, (cycle + 1) * every_n)
        )
        cycle_truth = cycle_block.isel(time=slice(every_n - 1, None, every_n))
        cycle_times.append(float(np.asarray(cycle_truth["time"].values).ravel()[0]))
        cycle_obs_clean = truth_obs_op(cycle_truth)
        if not isinstance(cycle_obs_clean, xarray.DataArray):
            raise ValueError(
                "The hybrid requires LABELLED observations: it derives the "
                "filter's forecast-segment boundaries from the batches' `time` "
                "coordinates and hands their concatenation to the smoother's "
                "time-binning aggregator. Set obs.temporal_mode=full on the "
                "case config (a bare spatial operator returns an unlabelled "
                "flat vector)."
            )
        rng_key, subkey = jax.random.split(rng_key)
        cycle_obs = cycle_obs_clean + obs_error_std * np.asarray(
            jax.random.normal(subkey, cycle_obs_clean.shape)
        )
        # Onto the nominal window clock (see the module docstring): batch l of
        # a window ends at (l + 1) * cycle_seconds, so every window's batches
        # run over (0, sim_time] and the forecast segments tile [0, sim_time]
        # exactly, independent of the truth's own (0-based, jittered) frame
        # coordinates.
        local_cycle = cycle - window_of_cycle * cycles_per_window
        observations.append(
            _nominal_window_clock(cycle_obs, local_cycle, cycle_seconds)
        )
        observations_clean.append(
            _nominal_window_clock(cycle_obs_clean, local_cycle, cycle_seconds)
        )
    truth_view.close()

    # --- Two observation-error covariances ------------------------------------
    # They live in different spaces and are deliberately built separately (the
    # hybrid never converts one into the other).
    #
    # The FILTER's is the covariance of ONE observation frame (sensors x
    # observed states), which is what one analysis consumes; the library
    # validates it per frame.
    n_d_frame = int(observations[0].sizes["obs"])
    C_D_diag = (obs_error_std**2) * jnp.ones(n_d_frame)
    # The SMOOTHER's is the covariance of a whole window's AGGREGATED and
    # flattened observation vector — sized off window 0, whose length every
    # window shares. Aggregating here also primes the shared aggregator's
    # interval-count check with the truth's interval count, exactly as
    # run_esmda.py does before instantiating the smoother.
    # `join="override"` (as the hybrid itself uses on this concat): the batches
    # share one `obs` axis by construction, and the default outer join would
    # silently NaN-pad if a backend ever wrote it with differing last bits.
    first_window_obs = xarray.concat(
        observations[:cycles_per_window], dim="time", join="override"
    )
    smoother_obs_flat = _flatten_obs(first_window_obs, aggregate_obs)
    n_d_window = int(np.shape(smoother_obs_flat)[0])
    C_D = jnp.diag((obs_error_std**2) * jnp.ones(n_d_window))

    # --- The two DA instances, and the hybrid around them ----------------------
    rng_key, esmda_key = jax.random.split(rng_key)
    smoother_overrides: dict[str, Any] = {}
    if "num_time_points" in cfg.esmda.smoother:
        # The time-varying smoother flattens each knot into its own
        # augmented-state scalar, so `num_time_points` must equal the sampled
        # prior's knot count (run_esmda.py derives it the same way).
        smoother_overrides["num_time_points"] = int(prior_params.sizes["time"])
    smoother = instantiate(
        cfg.esmda.smoother,
        observation_operator=assim_obs_op,
        forward_model=esmda_ensemble_model,
        C_D=C_D,
        rng_key=esmda_key,
        aggregate_observations=aggregate_obs,
        **smoother_overrides,
    )
    # Cap on-disk peak storage: drop each MDA step's forecast as soon as its
    # update is computed. Unlike run_esmda.py, step 0 is dropped too — the
    # hybrid saves no window prior STATE (the filter, not the MDA, produces this
    # window's state artifact), so nothing downstream reads it.
    smoother.prune_disk_steps = True
    smoother.keep_prior_disk_step = False
    # Always on: window_{w}_esmda_pred_obs.nc is the only record of what the MDA
    # loop actually saw, it is KB-scale, and (unlike run_esmda.py's) it costs no
    # extra observation-operator evaluation, because there is no posterior
    # forecast to project.
    smoother.collect_obs_diagnostics = True

    rng_key, filter_key = jax.random.split(rng_key)
    filter_overrides: dict[str, Any] = {}
    if filter_mode == "state":
        # The config ships an evolution/inflation pair for the parameter-
        # updating mode; in state mode the parameters are fixed by the smoother,
        # so do not pass an otherwise-default evolution into the constructor
        # (run_filtering.py:702-711 makes the same exception).
        filter_overrides["parameter_evolution"] = None
    enkf = instantiate(
        cfg.filtering.filter,
        observation_operator=assim_obs_op,
        forward_model=filter_ensemble_model,
        C_D=C_D_diag,
        rng_key=filter_key,
        **filter_overrides,
    )
    # The per-window observation-space arrays are what the shared normalized
    # data-mismatch diagnostic reads, so they are always produced; the hybrid
    # accumulates these lists across the window's per-segment filter calls.
    enkf.collect_pred_obs = True

    hybrid = FilterSmoothing(smoother=smoother, filter=enkf)

    save_history = bool(cfg.run.get("save_history", True))

    # --- Window loop -----------------------------------------------------------
    # Unlike a pure filtering run's, a window boundary here is NOT purely
    # computational: the MDA restarts from a fresh prior (extrapolated from this
    # window's posterior for a dynamic trajectory) and the joint correction is
    # reset. Only the STATE is carried unchanged across it.
    state_input: Any = None  # cold start: window 0's first cycle pays the spin-up
    diagnostic_rows: list[dict] = []
    params_history_pieces: list[xarray.Dataset] = []
    applied_params_pieces: list[xarray.Dataset] = []
    state_history_pieces: list[xarray.Dataset] = []
    esmda_params_pieces: list[xarray.Dataset] = []
    window_seconds: list[float] = []
    result: Any = None
    hybrid_start = time.perf_counter()
    for window in range(num_windows):
        window_start = time.perf_counter()
        first_cycle = window * cycles_per_window
        window_slice = slice(first_cycle, first_cycle + cycles_per_window)
        window_times = cycle_times[window_slice]

        # On-disk mode: give this window its own cycle-directory root, so its
        # cycle_0 does not delete the previous window's. The hybrid may point
        # the filter at a staging subdir BELOW this one while it renumbers its
        # per-segment calls; it restores this root afterwards, leaving
        # cycle_0..cycle_{K-1} here for the global collection below.
        staging_dir = _window_staging_dir(filter_states_dir, window)
        if staging_dir is not None:
            staging_dir.mkdir(parents=True, exist_ok=True)
            enkf.base_results_dir = staging_dir

        # Pin the t=0 knot from window 1 onward so the MDA update preserves the
        # cross-window continuity the extrapolation established at the boundary
        # (run_esmda.py:864-865). Window 0's t=0 is a cold-start draw, so the
        # smoother is free to fit it. Only the time-varying smoother has the flag.
        if hasattr(smoother, "pin_initial_time_point"):
            smoother.pin_initial_time_point = window > 0

        # The ensemble this window STARTS from: the sampled prior for window 0,
        # the previous window's carried/extrapolated posterior afterwards. A
        # dynamic prior is a trajectory with its own window-local `time` dim; a
        # static one gets the scalar physical coord that makes the assembled
        # file read in seconds (run_filtering.py's `_save_window_params`).
        prior_path = windows_dir / f"window_{window}_prior_params.nc"
        if is_dynamic_prior:
            prior_params.to_netcdf(prior_path)
        else:
            _save_window_params(prior_params, prior_path, window * sim_time)

        result = hybrid.run(
            state=state_input,
            params=prior_params,
            # This window's per-cycle batches, already on the window clock.
            observations=observations[window_slice],
            # Unconditional: the window's analyzed frames ARE
            # window_{w}_posterior_state.nc, whose footprint is one ESMDA window
            # rollout's. `run.save_history` decides only whether the
            # horizon-long histories are accumulated and written.
            return_history=True,
        )

        # The hybrid numbers cycles WITHIN a window; every filtering-native
        # artifact is indexed by the global cycle, so renumber before saving
        # (run_filtering.py:771-775).
        for local_cycle, diagnostics in enumerate(result.diagnostics):
            diagnostics.cycle = first_cycle + local_cycle
            diagnostic_rows.append(dataclasses.asdict(diagnostics))

        # --- The ESMDA-schema per-window artifacts ---
        assert result.state_history is not None  # return_history=True above
        _save_window_state(
            result.state_history,
            windows_dir / f"window_{window}_posterior_state.nc",
            window_times,
        )
        # The window's "posterior params" in the ESMDA schema is the MDA
        # posterior, so the shared metric/figure stages score exactly what the
        # smoother estimated. In joint mode the filter's corrected parameters
        # are a DIFFERENT quantity (and carry a per-cycle history), so they get
        # their own file rather than overwriting this one.
        posterior_path = windows_dir / f"window_{window}_posterior_params.nc"
        if is_dynamic_prior:
            result.esmda_params.to_netcdf(posterior_path)
        else:
            _save_window_params(
                result.esmda_params, posterior_path, (window + 1) * sim_time
            )
        if result.params is not None:
            _save_window_params(
                result.params,
                windows_dir / f"window_{window}_filter_params.nc",
                (window + 1) * sim_time,
            )

        # --- Observation space, both halves ---
        # The filtering schema, from the hybrid's accumulated per-cycle lists.
        # `obs_error_std` is TILED to the window's full K*N_obs length: C_D_diag
        # is the per-FRAME covariance and the diagnostic broadcasts sigma
        # against the whole window's vector.
        _save_window_obs_diagnostics(
            windows_dir,
            window,
            _flat_obs_vector(observations[window_slice]),
            _flat_obs_vector(observations_clean[window_slice]),
            np.tile(np.sqrt(np.asarray(C_D_diag)), cycles_per_window),
            _stack_cycle_pred_obs(hybrid.pred_obs_history),
            _stack_cycle_pred_obs(hybrid.pred_obs_post_history),
            truth_obs_op,
        )
        # The smoother half, on its aggregated axis and under its own name.
        window_obs = xarray.concat(
            observations[window_slice], dim="time", join="override"
        )
        window_obs_clean = xarray.concat(
            observations_clean[window_slice], dim="time", join="override"
        )
        _save_window_esmda_pred_obs(
            windows_dir,
            window,
            smoother.pred_obs_history,
            _flatten_obs(window_obs, aggregate_obs),
            _flatten_obs(window_obs_clean, aggregate_obs),
            np.sqrt(np.diag(np.asarray(C_D))),
            truth_obs_op,
        )

        if save_history:
            state_history_pieces.append(result.state_history)
            # NOTE the deliberate difference from run_filtering.py:806-814,
            # which has to drop each window's repeated leading entry:
            # FilterSmoothing's `params_history` holds ONE ENTRY PER CYCLE (the
            # analyzed values, no prepended prior), so `params_history`,
            # `applied_params_history`, `state_history` and `diagnostics` all
            # index the same cycles and the pieces concatenate as they are.
            if result.params_history is not None:
                params_history_pieces.append(result.params_history)
            if result.applied_params_history is not None:
                applied_params_pieces.append(result.applied_params_history)
            if result.esmda_params_history is not None:
                esmda_params_pieces.append(
                    result.esmda_params_history.expand_dims(window=[window])
                )

        # On-disk mode: move this window's cycle_{0..K-1} dirs onto the global
        # cycle index, emptying the staging root for the next window.
        _collect_window_cycle_dirs(filter_states_dir, window, cycles_per_window)

        # --- Carry into the next window ---
        # The STATE carries unchanged (the filter's analyzed end-of-window
        # frame). The PARAMETERS restart the MDA from a fresh prior. The joint
        # correction deliberately does NOT carry: FilterSmoothing re-initializes
        # it to zero on every run() call, so each window's filter parameters
        # start from that window's own ESMDA schedule. This is the v1 default —
        # window_{w}_filter_params.nc preserves the information, so carrying the
        # correction across windows stays a one-line follow-up.
        state_input = result.state
        if window < num_windows - 1:
            if is_dynamic_prior:
                # Reuse the prior sampler's own per-window knot grid (spaced by
                # `seconds_per_knot`, incl. any extrapolated endpoint), shifted
                # to the next window, then rebased back onto [0, sim_time] —
                # the window-local clock everything in a window shares
                # (run_esmda.py:984-996).
                knot_times = jnp.asarray(prior_sampler.time_coords)
                prediction_times = knot_times + sim_time
                rng_key, subkey = jax.random.split(rng_key)
                extrapolated = prior_sampler.extrapolate(
                    result.esmda_params, prediction_times, subkey
                )
                prior_params = extrapolated.assign_coords(time=np.asarray(knot_times))
            else:
                prior_params = result.esmda_params

        window_seconds.append(time.perf_counter() - window_start)

    hybrid_seconds = time.perf_counter() - hybrid_start

    # --- Outputs ---------------------------------------------------------------
    # The filtering-native artifacts, accumulated over the WHOLE horizon with
    # global cycle indices, so the filtering metric/figure stages read this
    # directory exactly as they read a pure filtering run's.
    if result.state is not None:
        result.state.to_netcdf(out_dir / "posterior_state.nc")
    if params_history_pieces:
        xarray.concat(params_history_pieces, dim="cycle", join="override").to_netcdf(
            out_dir / "params_history.nc"
        )
    if applied_params_pieces:
        # Joint mode on a DYNAMIC trajectory only: the parameters each forecast
        # segment was actually run with, `e_k + c_k` — the ESMDA schedule
        # evaluated at the segment midpoint plus the filter's carried
        # correction. Distinct from params_history (the ANALYZED values) and
        # the only artifact from which the correction itself is recoverable
        # per cycle.
        xarray.concat(applied_params_pieces, dim="cycle", join="override").to_netcdf(
            out_dir / "applied_params_history.nc"
        )
    if state_history_pieces:
        xarray.concat(state_history_pieces, dim="cycle", join="override").to_netcdf(
            out_dir / "state_history.nc"
        )
    if esmda_params_pieces:
        # The MDA iterates, one `window` entry per window and `esmda_step`
        # within it. Hybrid-only: run_esmda.py's per-window
        # window_{w}_params_steps.nc is the closest equivalent.
        xarray.concat(esmda_params_pieces, dim="window", join="override").to_netcdf(
            out_dir / "esmda_params_history.nc"
        )
    write_yaml(diagnostic_rows, out_dir / "cycle_diagnostics.yaml")

    # Truth-access parameters (slicing/offsets + BOTH geometries) so the
    # downstream stages reconstruct the same lazy view of the on-disk truth. The
    # ESMDA keys keep their ESMDA meanings (`sim_time` is WINDOW seconds,
    # `n_per_window` its frame count); the filtering stages read the cycle
    # geometry from `cycle_seconds` / `n_per_cycle` / `num_cycles`. Identical in
    # shape to run_filtering.py's, since the two runs' artifacts are.
    write_yaml(
        {
            "true_state_path": str(true_state_path),
            "x_offset": float(x_offset),
            "start_idx": int(start_idx),
            "t_offset": float(t_offset),
            "n_total": int(n_total),
            "n_per_window": int(cycles_per_window),
            "num_windows": int(num_windows),
            "sim_time": float(sim_time),
            # The truth frames one cycle covers, of which it assimilates the
            # last (run_filtering.py's key, same meaning).
            "n_per_cycle": int(every_n),
            "num_cycles": int(num_cycles),
            "cycle_seconds": float(cycle_seconds),
            # The assembled parameter files carry a physical (seconds) time axis
            # either way — a dynamic trajectory through its rebased knot dim, a
            # static posterior through the scalar per-window coord — so the
            # parameter plots' window edges land at multiples of sim_time.
            "is_dynamic": True,
            "truth_solver_name": str(cfg.truth_model.solver_name),
            "assim_solver_name": str(cfg.assim_model.solver_name),
            # One analyzed frame per cycle. Unstrided they tile the window;
            # under a stride they are a sparse sample of it, flagged exactly as
            # run_filtering.py flags its own (the prose comes from the same
            # helper, so the two runs' figure labels cannot drift apart).
            "moment_sampling": _moment_sampling(every_n),
            "moment_sampling_is_sparse": every_n > 1,
        },
        out_dir / "truth_access.yaml",
    )

    write_yaml(
        {
            "configuration": {
                # BOTH halves are recorded: neither alone identifies the run.
                "hybrid": type(hybrid).__name__,
                "smoother": type(smoother).__name__,
                "filter": type(enkf).__name__,
                "mode": filter_mode,
                "joint_state_and_parameter": False,
                "time_varying_parameters": bool(is_dynamic_prior),
                "num_assimilation_windows": int(num_windows),
                "cycles_per_window": int(cycles_per_window),
                "num_cycles": int(num_cycles),
                "ensemble_size": int(ensemble_size),
                "simulation_time_per_window": float(sim_time),
                "seconds_per_cycle": float(cycle_seconds),
                # The filter's analysis stride; recorded so a hybrid run dir
                # reads like a filtering one to the shared stages.
                "assimilate_every_n_step": int(every_n),
                "final_time": float(final_time),
                "observation_error_std": obs_error_std,
                "num_esmda_steps": int(cfg.esmda.num_steps),
                # The smoother's observation vector is aggregated, the filter's
                # is not; both lengths are recorded because the two per-window
                # observation-space files live on those two different axes.
                "num_observations_per_frame": int(n_d_frame),
                "num_observations_per_window_aggregated": int(n_d_window),
                "observation_interval_seconds": (
                    float(cfg.esmda.interval_seconds)
                    if cfg.esmda.interval_seconds is not None
                    else None
                ),
                "observation_aggregation_mode": (
                    str(cfg.esmda.aggregation_mode)
                    if cfg.esmda.interval_seconds is not None
                    else None
                ),
                # The gate the shared observation-space diagnostic reads before
                # it opens windows/window_*_{obs,pred_obs}.nc.
                "save_obs_diagnostics": True,
                # No window-long prior rollout exists: the MDA's iterate
                # forecasts are pruned and the filter interleaves its analyses
                # with the forecast, so windows/window_{w}_prior_state.nc is
                # deliberately absent and the prior halves of the state
                # diagnostics are skipped.
                "save_prior_state": False,
                "esmda_step_semantics": _ESMDA_STEP_SEMANTICS,
                "esmda_pred_obs_semantics": _ESMDA_PRED_OBS_SEMANTICS,
                "seed": int(cfg.filter_smoothing.seed),
                "truth_model": str(cfg.truth_model.name),
                "assimilation_model": str(cfg.assim_model.name),
                "truth_source": "disk" if cfg.run.truth_dir is not None else "inline",
                "truth_dir": (
                    str(cfg.run.truth_dir) if cfg.run.truth_dir is not None else None
                ),
                "num_truth_frames": int(n_total),
                # The filter's composed class stays EnsembleKalmanFilter for
                # every update flavor (the analysis is injected, not
                # subclassed), so the resolved analysis subtree is the only
                # record of which update math ran, and the localization beside
                # it is its coupled pair (localization_policy).
                "analysis": OmegaConf.to_container(
                    cfg.filtering.analysis, resolve=True
                ),
                "localization": (
                    OmegaConf.to_container(cfg.filtering.localization, resolve=True)
                    if cfg.filtering.localization is not None
                    else None
                ),
                "state_reduction": (
                    OmegaConf.to_container(cfg.filtering.state_reduction, resolve=True)
                    if cfg.filtering.state_reduction is not None
                    else None
                ),
                "state_reduction_resolved_variable_scales": (
                    enkf.state_reduction.resolved_variable_scales
                    if enkf.state_reduction is not None
                    else None
                ),
                # The smoother's own localization, a separate axis from the
                # filter's above.
                "esmda_localization": (
                    OmegaConf.to_container(cfg.esmda.localization, resolve=True)
                    if cfg.esmda.localization is not None
                    else None
                ),
            },
            "timing": {
                "hybrid_total_seconds": float(hybrid_seconds),
                "mean_window_seconds": (
                    float(np.mean(window_seconds)) if window_seconds else None
                ),
                "mean_cycle_seconds": float(hybrid_seconds / max(num_cycles, 1)),
                "per_window_seconds": [float(s) for s in window_seconds],
            },
        },
        out_dir / "run_info.yaml",
    )

    # Assemble the ESMDA-schema, downstream-facing outputs from the per-window
    # files. `is_dynamic` selects the REBASE: a dynamic trajectory's window-local
    # knot axis is shifted onto the global one, while the static case's scalar
    # per-window coords are concatenated as they are (already physical). On the
    # dynamic path the rebase also touches the window STATE files, whose frames
    # already carry physical truth times — `(t - t[0]) + w*sim_time` is then a
    # no-op for an on-grid truth and, for a jittered one, snaps each window's
    # start back onto the nominal grid (removing accumulated cadence drift),
    # which is benign for every downstream reader.
    _save_assembled_outputs(
        out_dir, windows_dir, num_windows, sim_time, is_dynamic=is_dynamic_prior
    )
    print(f"Saved outputs in {out_dir}")


@hydra.main(  # type: ignore[misc]
    version_base=None, config_path="../../conf", config_name="run_filter_smoothing"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
