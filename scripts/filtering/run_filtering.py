"""Run sequential ensemble filtering (EnKF): state / parameter / joint modes.

The filtering counterpart of ``scripts/esmda/run_esmda.py``: instead of ESMDA's
multiple tempered updates per assimilation window, the ensemble Kalman filter
forecasts the ensemble from one observation to the next and applies ONE
full-weight analysis at each, warm-starting the next cycle from the analyzed
end-of-segment state (see ``docs/data_assimilation.md`` and
``libs/data-assimilation/src/data_assimilation/filtering/``).

This is the first stage of a three-script single-run pipeline (see
``scripts/run_filtering_pipeline.sh``), mirroring the ESMDA pipeline:

  1. scripts/filtering/run_filtering.py         (THIS) -- runs the filter and
                                       saves every raw artifact (the per-window
                                       ESMDA-schema files under windows/, the
                                       assembled posterior/prior params and
                                       posterior_state_mean.nc, the
                                       posterior/history states, the truth
                                       state/params, cycle_diagnostics.yaml,
                                       truth_access.yaml and run_info.yaml).
  2. scripts/filtering/compute_filtering_metrics.py -- reads those, writes
                                       run_summary.yaml.
  3. scripts/filtering/make_filtering_figures.py -- reads those, draws figures.

The metric and figure stages reuse the ESMDA pipeline's truth-access and
sensor-series helpers via ``scripts/filtering/_filtering_common.py``.

Mode and machinery are declarative axes (see conf/run_filtering.yaml):

  * ``filtering.mode=state|parameter|joint``
        which augmented blocks the analysis updates.
  * ``filtering/analysis=stochastic|etkf|etkf_tsvd|letkf|letkf_tsvd``
        the update math: the perturbed-observation stochastic EnKF, or a
        deterministic ensemble transform, globally (``etkf*``, which REQUIRES
        ``filtering/localization=none``) or per block (``letkf*``, which
        REQUIRES a non-null localization). The ``*_tsvd`` variants additionally
        truncate weak directions of the whitened observation anomalies.
  * ``filtering/localization=none|correlation|distance``
        reuses the smoother's localization strategies unchanged.
  * ``filtering/state_reduction=none|svd_current|svd_streaming``
        optionally projects only the state-analysis increment onto an SVD/POD
        basis; reduced analyses require global (``none``) localization.
  * ``filtering/inflation=none|multiplicative|rtps|rtpp``
        ensemble spread maintenance.
  * ``filtering/evolution=none|random_walk``
        the parameters' forecast model between cycles (required — or an
        inflation — for the parameter-updating modes ``parameter``/``joint``).

and the truth source mirrors run_esmda.py:

  * ``run.truth_dir=null``    simulate the truth inline (default).
  * ``run.truth_dir=<path>``  load a state.nc/params.nc artifact written by
                              run_forward_model.py.

The run is chunked into ``filtering.num_assimilation_windows`` windows of
``time.simulation_time`` seconds each — the same unit ESMDA's
``esmda.num_assimilation_windows`` counts, so the two entry points are
configured identically and their runs compare like for like. Inside a window
the filter forecasts OBSERVATION TO OBSERVATION: one cycle is one observation
interval (``time.output_frequency`` seconds) ending in one full-weight
analysis of that frame, so a window holds
``simulation_time / output_frequency`` cycles (derived, never configured) and
the horizon holds ``num_assimilation_windows`` times that. Nothing is
aggregated (see ``docs/data_assimilation.md`` §8).

``filtering.assimilate_every_n_step = n`` (default 1) thins the ANALYSIS
cadence without touching the model's output cadence: a cycle becomes
``n * output_frequency`` seconds, its forecast segment still emits one frame
per ``output_frequency`` (they are on disk, in the per-cycle
``_ensemble_states/cycle_{k}/`` files, in full), and only the segment's LAST
frame — the analysis time — is assimilated. The skipped frames simply never
see a Kalman step, so the ensemble runs ``n`` intervals between analyses. The
IN-MEMORY histories (``state_history.nc``, the per-window state files) hold
the analysis-time frames only, at the ``n * output_frequency`` cadence; the
full-resolution frames live in the on-disk cycle segments. ``n`` must divide
``simulation_time / output_frequency`` so the cycles still tile a window.

A window is a COMPUTATIONAL boundary only: it bounds RAM and peak disk (one
``run()`` call, one set of per-window artifacts) and changes nothing
mathematically. The analyzed state and parameters are carried into the next
window exactly as ESMDA carries its own (``state_input = posterior state``),
the filter's PRNG stream continues across ``run()`` calls (``BaseFilter``
mutates ``rng_key`` in place and never resets it), and the per-cycle
observation noise is drawn once for the whole horizon before the window loop —
so the same horizon run as 1 window or as W windows is identical, which
``tests/test_filtering.py`` pins.

Artifacts are written in BOTH schemas from one run: the ESMDA per-window
schema (``windows/window_{w}_{prior,posterior}_params.nc``,
``window_{w}_posterior_state.nc``, ``window_{w}_{obs,pred_obs}.nc``, the
assembled ``prior_params.nc`` / ``posterior_params.nc`` /
``posterior_state_mean.nc``), so ``scripts/esmda/compute_esmda_metrics.py``
and ``make_esmda_figures.py`` run on a filtering run directory; and the
filtering-native artifacts (``cycle_diagnostics.yaml``, ``params_history.nc``,
``state_history.nc``, the optional ``_ensemble_states/cycle_{k}/`` tree),
accumulated over the whole horizon with GLOBAL cycle indices, for the
filtering-specific stages.

The PRIOR must be a static scalar sampler (the filter estimates the parameter
value *now*, re-tracked each cycle; a time-varying/AR(2) posterior stays with
the ESMDA smoothers) — pair with ``params@prior_params=static``. The TRUTH may
be dynamic (``params@truth_params=dynamic_truth``): the filter then tracks a
drifting truth, its scalar estimate following the truth's per-cycle value. A
dynamic truth is sampled over the full horizon so it drifts across every cycle.

Examples::

    python scripts/filtering/run_filtering.py filtering.mode=joint \
        filtering.num_assimilation_windows=4
    python scripts/filtering/run_filtering.py filtering.mode=parameter \
        filtering/evolution=random_walk filtering/inflation=none
    python scripts/filtering/run_filtering.py filtering.mode=state \
        filtering/localization=correlation
    python scripts/filtering/run_filtering.py filtering.mode=state \
        filtering/localization=none filtering/state_reduction=svd_current
    python scripts/filtering/run_filtering.py filtering.mode=state \
        filtering/analysis=etkf filtering/localization=none
    python scripts/filtering/run_filtering.py filtering.mode=state \
        filtering/analysis=letkf filtering/localization=distance
"""

import dataclasses
import pathlib
import shutil
import sys
import time
from typing import Any, Optional, Sequence

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise; must precede `import jax`)
from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    create_observation_operator,
    filter_parameter_config,
)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.esmda._esmda_common import open_truth, truth_x_min, write_yaml

# The ESMDA-schema writers are reused rather than reimplemented, the same
# cross-package import the filtering metric stage already does
# (scripts/filtering/compute_filtering_metrics.py). ``_obs_index_coords`` reads
# the observation operator to decode the flat observation layout, and that
# layout is identical here: with one cycle per observation frame the flat index
# ``j = cycle*(n_states*n_sensors) + state*n_sensors + sensor`` is exactly what
# ``_OBS_ORDERING`` describes, with its "interval" axis reading as the cycle
# index within the window.
from scripts.esmda.run_esmda import (  # noqa: E402
    _OBS_DIM,
    _OBS_ORDERING,
    _obs_index_coords,
    _save_assembled_outputs,
)

# ---------------------------------------------------------------------------
# Per-window artifact writers (the ESMDA schema, see
# scratchpad `filtering_esmda_artifact_contract.md`)
# ---------------------------------------------------------------------------

# Recorded in truth_access.yaml and stamped onto eval_fields.nc by the metric
# stage: what the per-window state files actually hold. An ESMDA window file is
# a free posterior rollout; a filter's is the analyzed series, so the moments
# taken over it carry the analysis increments. NOT sparse — the analyzed frames
# tile the window one per output frame.
_MOMENT_SAMPLING = (
    "the analyzed end-of-cycle state at every observation frame of each window "
    "(one analysis per frame), so the moments carry the analysis increments"
)


def _moment_sampling(every_n: int) -> str:
    """The ``moment_sampling`` prose for a stride of ``every_n``.

    A strided run analyzes one frame in every ``every_n``, so the analyzed
    series no longer tiles the window and the moments taken over it are a
    strided SAMPLE rather than a within-window time average (the caller flags
    that as ``moment_sampling_is_sparse``).
    """
    if every_n == 1:
        return _MOMENT_SAMPLING
    return (
        "the analyzed end-of-cycle state at every "
        f"{every_n}th observation frame of each window (one analysis per cycle "
        f"of {every_n} frames), so the moments carry the analysis increments "
        "and are a strided sample of the window, not a full time average"
    )


# The filter has no ESMDA iteration axis; the two ``esmda_step`` entries of
# window_{w}_pred_obs.nc are one analysis' before/after, pooled over the
# window's cycles.
_ESMDA_STEP_SEMANTICS = (
    "0 = the per-cycle prior forecast H(x_f) stacked over the window's cycles "
    "in cycle order; 1 = the matching ride-along posterior rows"
)


class _StridedObservationOperator:
    """The assimilation operator restricted to the frames a cycle assimilates.

    With ``filtering.assimilate_every_n_step = n > 1`` a cycle's forecast
    segment emits ``n`` output frames but only the LAST of them — the analysis
    time — is assimilated, so the predicted observations handed to the filter
    must be subset the same way the truth side is
    (``isel(time=slice(n - 1, None, n))``), row for row. The STRIDE form rather
    than ``isel(time=-1)`` is deliberate: it stays correct if a cycle ever
    spans several analysis frames.

    Used ONLY when ``n > 1``. At ``n = 1`` the operator is passed to the filter
    unchanged, so a default run is bit-identical to one built before this knob
    existed.

    A bare (non-temporal) ``ObservationOperator`` already reduces a segment to
    its last frame and returns an unlabelled flat vector, so there is nothing
    to stride and its output is returned untouched. Attribute access is
    delegated, which keeps the wrapper transparent to the operator
    introspection the library does (``sensor_observation_coords`` unwraps one
    ``observation_operator`` level for distance localization).
    """

    def __init__(self, operator: Any, stride: int) -> None:
        self._operator = operator
        self._stride = int(stride)

    def __call__(self, state: xarray.Dataset) -> Any:
        obs = self._operator(state)
        if isinstance(obs, xarray.DataArray) and "time" in obs.dims:
            # The stride only ends on the segment's LAST frame — the frame
            # ``BaseFilter._get_final_states`` analyzes — while the frame count
            # is a whole number of cycles. A segment carrying anything extra
            # (stale files in the cycle directory, a backend that writes its
            # cold-start spin-up into cycle 0) would otherwise be silently
            # subset to the wrong frame and still line up with the truth side
            # row for row. Unstrided, the library's own row check raises on
            # exactly this; keep it loud here too.
            n_frames = int(obs.sizes["time"])
            if n_frames % self._stride:
                raise ValueError(
                    f"The forecast segment produced {n_frames} observation "
                    f"frame(s), which filtering.assimilate_every_n_step="
                    f"{self._stride} does not divide, so the analysis frame "
                    "would not be the segment's last. Check the assimilation "
                    "model's output cadence and the cycle results directory "
                    "for stale files."
                )
            return obs.isel(time=slice(self._stride - 1, None, self._stride))
        return obs

    def __getattr__(self, name: str) -> Any:
        # Private names are never delegated: `self._operator` itself is looked
        # up through here on a half-initialised instance (unpickling), and
        # delegating it would recurse forever.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._operator, name)


def _save_window_params(
    params: xarray.Dataset, path: pathlib.Path, time_seconds: float
) -> None:
    """One window's parameter ensemble with a scalar PHYSICAL ``time`` coord.

    The coordinate is what makes the assembled ``prior_params.nc`` /
    ``posterior_params.nc`` carry seconds rather than a window index: a plain
    concat of these scalar-coord pieces along ``time`` yields the physical
    axis. Without it a *drifting* truth would be interpolated at
    t = 0, 1, 2, ... seconds when the parameter metrics align the two.
    """
    params.assign_coords(time=float(time_seconds)).to_netcdf(path)


def _save_window_state(
    state_history: xarray.Dataset, path: pathlib.Path, frame_times: Sequence[float]
) -> None:
    """One window's analyzed states as an ESMDA-schema window state file.

    ``state_history`` is the window ``run()``'s ``cycle``-concatenated analyzed
    frames — one per cycle, i.e. one per observation frame, occupying exactly
    the time slots an ESMDA window's posterior rollout occupies. The stale
    scalar ``time`` coordinate the concat carried over from the first cycle is
    dropped and replaced by the window's own truth frame times, so the file
    reads on a physical axis like every other window state file.
    """
    ds = state_history
    if "time" in ds.coords:
        ds = ds.drop_vars("time")
    ds = ds.rename({"cycle": "time"}).assign_coords(
        time=np.asarray(frame_times, dtype=float)
    )
    if "ensemble" in ds.dims:
        # Ensemble-first, matching the ESMDA window files the downstream
        # streaming readers were written against.
        ds = ds.transpose("ensemble", "time", ...)
    ds.to_netcdf(path)


def _flat_obs_vector(cycle_observations: Sequence[Any]) -> np.ndarray:
    """One window's observations as a single flat vector, cycle-major.

    Each entry is one cycle's single observation frame — a ``("time", "obs")``
    DataArray with ``T = 1`` from the temporal operator, or a plain flat vector
    from a bare one. Stacking them in cycle order gives
    ``row = cycle*N_obs + state*n_sensors + sensor``, which is byte-for-byte
    the layout ``_obs_index_coords`` decodes and the layout the filter's
    per-cycle ``pred_obs`` blocks stack into.
    """
    pieces = []
    for obs in cycle_observations:
        values = (
            obs.transpose("time", "obs").values
            if isinstance(obs, xarray.DataArray)
            else np.asarray(obs)
        )
        pieces.append(np.asarray(values, dtype=float).ravel())
    return np.concatenate(pieces)


def _stack_cycle_pred_obs(history: Sequence[np.ndarray]) -> np.ndarray:
    """The window's per-cycle ``(N_obs, N_e)`` blocks as one ``(N_d, N_e)``."""
    return np.concatenate([np.asarray(p, dtype=float) for p in history], axis=0)


def _save_window_obs_diagnostics(
    windows_dir: pathlib.Path,
    window: int,
    obs: np.ndarray,
    obs_clean: np.ndarray,
    obs_error_std: np.ndarray,
    pred_obs: Optional[np.ndarray],
    pred_obs_post: Optional[np.ndarray],
    obs_op: Any,
) -> None:
    """Write one window's observation-space arrays in the ESMDA schema.

    The filtering twin of ``run_esmda.py``'s ``_save_obs_diagnostics``, minus
    the per-iteration ``params_steps`` file (a filter has no iteration axis, and
    nothing reads it) and with an ``esmda_step`` attr that says what the two
    entries mean here. The dimension is ``obs_index``, not ``obs``: a variable
    whose name equals its dimension is promoted to an index coordinate on the
    netCDF round trip.
    """
    n_d = int(np.size(obs))
    coords = _obs_index_coords(obs_op, n_d)

    obs_ds = xarray.Dataset(
        data_vars={
            "obs": (_OBS_DIM, np.asarray(obs, dtype=float).ravel()),
            "obs_clean": (_OBS_DIM, np.asarray(obs_clean, dtype=float).ravel()),
            "obs_error_std": (
                _OBS_DIM,
                np.asarray(obs_error_std, dtype=float).ravel(),
            ),
        },
        coords={_OBS_DIM: np.arange(n_d), **coords},
    )
    obs_ds.attrs["ordering"] = _OBS_ORDERING
    obs_ds["obs"].attrs["long_name"] = "assimilated observation (truth + noise)"
    obs_ds["obs_clean"].attrs["long_name"] = "noise-free truth projection"
    obs_ds["obs_error_std"].attrs["long_name"] = "sqrt(diag(C_D)), un-inflated"
    obs_ds.to_netcdf(windows_dir / f"window_{window}_obs.nc")

    if pred_obs is None or pred_obs_post is None:
        return
    stacked = np.stack([pred_obs, pred_obs_post])
    pred_ds = xarray.Dataset(
        data_vars={"pred_obs": (("esmda_step", _OBS_DIM, "ensemble"), stacked)},
        coords={
            "esmda_step": np.arange(stacked.shape[0]),
            _OBS_DIM: np.arange(stacked.shape[1]),
            **coords,
        },
    )
    pred_ds.attrs["ordering"] = _OBS_ORDERING
    pred_ds.attrs["esmda_step"] = (
        "iteration index; 0 = prior forecast, -1 = posterior forecast. "
        f"For a sequential filter: {_ESMDA_STEP_SEMANTICS}."
    )
    pred_ds.to_netcdf(windows_dir / f"window_{window}_pred_obs.nc")


def _window_staging_dir(
    states_dir: Optional[pathlib.Path], window: int
) -> Optional[pathlib.Path]:
    """Where one window's on-disk forecasts are written before renumbering.

    ``BaseFilter._set_cycle_results_dir`` indexes cycles within ONE ``run()``
    call — and empties the directory it is pointed at — so every window would
    write, and first delete, ``cycle_0 .. cycle_{K-1}``. Each window therefore
    gets its own staging root, collected onto the run's GLOBAL cycle index by
    :func:`_collect_window_cycle_dirs` when it finishes; the downstream
    ``forecast`` state source (``scripts/filtering/_filtering_common.py``)
    wants exactly one directory per global cycle.
    """
    if states_dir is None:
        return None
    return states_dir / f"_window_{window}"


def _collect_window_cycle_dirs(
    states_dir: Optional[pathlib.Path], window: int, cycles_per_window: int
) -> None:
    """Move one finished window's cycle dirs onto the run's GLOBAL index."""
    staging = _window_staging_dir(states_dir, window)
    if states_dir is None or staging is None or not staging.is_dir():
        return
    offset = window * cycles_per_window
    for cycle in range(cycles_per_window):
        source = staging / f"cycle_{cycle}"
        if not source.is_dir():
            continue
        target = states_dir / f"cycle_{cycle + offset}"
        # A stale directory from an earlier run into the same output dir would
        # make ``rename`` fail (POSIX refuses to replace a non-empty dir).
        shutil.rmtree(target, ignore_errors=True)
        source.rename(target)
    shutil.rmtree(staging, ignore_errors=True)


def run(cfg: DictConfig) -> None:
    num_windows = int(cfg.filtering.num_assimilation_windows)
    if num_windows < 1:
        raise ValueError(
            "filtering.num_assimilation_windows must be >= 1, got " f"{num_windows}."
        )
    sim_time = float(cfg.time.simulation_time)  # ONE window
    output_frequency = float(cfg.time.output_frequency)  # the OBSERVATION cadence
    if output_frequency <= 0.0:
        raise ValueError(
            "time.output_frequency must be > 0: it is the observation cadence, "
            "and (times filtering.assimilate_every_n_step) the filter's "
            "forecast segment (one cycle)."
        )
    # How many observation frames one analysis skips over. 1 (the default) is
    # "assimilate every frame"; n > 1 leaves the model's OUTPUT cadence alone
    # and only thins the analyses, so a cycle spans n output frames and
    # assimilates the last of them.
    every_n = int(cfg.filtering.get("assimilate_every_n_step", 1))
    if every_n < 1:
        raise ValueError(
            "filtering.assimilate_every_n_step must be >= 1 (1 = assimilate "
            f"every observation frame), got {every_n}."
        )
    # Fail at CONFIG time, before any solver runs: a stride that does not
    # divide the window's frame count would leave a partial cycle at the window
    # boundary, whose frames would be silently dropped.
    frames_per_window_nominal = round(sim_time / output_frequency)
    if frames_per_window_nominal % every_n:
        raise ValueError(
            f"filtering.assimilate_every_n_step={every_n} does not divide the "
            f"{frames_per_window_nominal} observation frame(s) a window holds "
            f"(time.simulation_time={sim_time:g}s / "
            f"time.output_frequency={output_frequency:g}s). The strided cycles "
            "must tile the window exactly; adjust time.simulation_time, "
            "time.output_frequency or filtering.assimilate_every_n_step."
        )
    cycle_seconds = every_n * output_frequency  # ONE cycle
    ensemble_size = int(cfg.ensemble.ensemble_size)
    final_time = sim_time * num_windows
    domain_x_min = float(cfg.domain.bounds[0][0])
    rng_key = jax.random.PRNGKey(cfg.filtering.seed)

    # The filter estimates a static scalar parameter that it TRACKS across
    # cycles (re-analysed each cycle, evolved between cycles): a time-varying
    # (AR(2)) PRIOR belongs to the dynamic ESMDA smoothers. A time-varying TRUTH
    # is supported, and is the point of the mixed default -- the filter then
    # tracks the drifting truth with its scalar estimate.
    if "seconds_per_knot" in list(cfg.prior_params.keys()):
        raise ValueError(
            "prior_params is a time-varying (dynamic) sampler, which the filter "
            "does not support: it estimates a scalar parameter value it tracks "
            "per cycle. Use params@prior_params=static (a time-varying TRUTH -- "
            "params@truth_params=dynamic_truth -- IS supported, so the filter "
            "can track a drifting truth), or the ESMDA smoothers "
            "(scripts/esmda/run_esmda.py) for a time-varying posterior."
        )
    is_dynamic_truth = "seconds_per_knot" in list(cfg.truth_params.keys())

    # Select which parameters the filter estimates (same contract as
    # run_esmda.py `params_to_estimate`): null -> every parameter the sampler
    # configs define; a list -> that subset, applied to prior AND truth.
    selected = cfg.get("params_to_estimate", None)
    selected = list(selected) if selected is not None else None
    truth_params_cfg = filter_parameter_config(cfg.truth_params, selected)
    prior_params_cfg = filter_parameter_config(cfg.prior_params, selected)
    if selected is not None:
        print(f"Estimating parameters: {selected}")

    # --- Output and windows dir -----------------------------------------------
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
        # A dynamic (time-varying) truth must span the FULL horizon so its
        # parameter drifts across every cycle, not just the first: sample its
        # knots over final_time (mirrors run_esmda.py's is_dynamic branch, which
        # samples over sim_time * num_windows). A static truth ignores it.
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

        # Optionally begin the horizon partway into a pre-simulated truth
        # (skip a spin-up); frames are rebased so that time becomes t=0.
        start_time = float(cfg.run.truth_start_time or 0.0)
        true_state_path = truth_dir / "state.nc"
        with xarray.open_dataset(true_state_path) as _truth_meta:
            true_times = np.asarray(_truth_meta["time"].values, dtype=float)
            x_offset = domain_x_min - truth_x_min(_truth_meta)
        start_idx = int((true_times < start_time).sum())
        t_offset = start_time
        n_total = int(((true_times[start_idx:] - t_offset) < final_time).sum())

    if x_offset:
        print(
            f"Shifting truth x by {x_offset:+g} to align with domain x_min={domain_x_min:g}"
        )

    # Truth frames per WINDOW (contiguous, half-open blocks, exactly as
    # run_esmda.py slices them), and from those the cycles per window: a cycle
    # covers `every_n` frames and assimilates the last of them, so the two
    # counts differ by the stride. Guard the degenerate slicings explicitly —
    # fewer than `every_n` frames per window would leave a window with nothing
    # to assimilate, and a remainder would be dropped silently.
    if n_total < num_windows * every_n:
        raise ValueError(
            f"The truth provides {n_total} frame(s) within the {final_time:g}s "
            f"horizon, fewer than filtering.num_assimilation_windows="
            f"{num_windows} x filtering.assimilate_every_n_step={every_n} "
            "(each window needs at least one full cycle of observation "
            "frames). Increase time.simulation_time, reduce "
            "time.output_frequency, filtering.num_assimilation_windows or "
            "filtering.assimilate_every_n_step, or point run.truth_dir at a "
            "longer truth."
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
    # Everything downstream (truth_access.yaml, the assembled state file) is
    # written against the frames the run actually covers.
    n_total = frames_per_window * num_windows
    num_cycles = cycles_per_window * num_windows
    expected_per_window = round(sim_time / cycle_seconds)
    if cycles_per_window != expected_per_window:
        print(
            f"The truth provides {cycles_per_window} cycle(s) per "
            f"{sim_time:g}s window, but a cycle of {cycle_seconds:g}s "
            f"(time.output_frequency x filtering.assimilate_every_n_step) "
            f"implies {expected_per_window}. The assimilation model forecasts "
            "one cycle per analysis, so the cycles will not tile the window at "
            "the truth's cadence."
        )
    print(
        f"{num_windows} window(s) of {sim_time:g}s = {cycles_per_window} "
        f"cycle(s) of {cycle_seconds:g}s each ({num_cycles} cycles over "
        f"{final_time:g}s, assimilating every {every_n} observation frame(s))."
    )
    true_params.to_netcdf(out_dir / "true_params.nc")

    # --- Assimilation ensemble model ------------------------------------------
    assim_results_dir = (
        pathlib.Path(cfg.run.results_dir) if cfg.run.results_dir is not None else None
    )
    # The assimilation ensemble's horizon is ONE CYCLE, i.e.
    # `assimilate_every_n_step` observation intervals — not
    # `time.simulation_time`, which the model configs interpolate and which
    # keeps its window meaning everywhere else (the truth model above is built
    # over the full horizon from the same key). Overriding the constructor
    # argument here is the same lever run_esmda.py pulls for the truth model,
    # and it leaves the model's `output_frequency` alone, so a cycle emits one
    # frame per observation interval — all of them written out, only the last
    # one (the analysis time) assimilated.
    #
    # The SPIN-UP is untouched by this and still happens exactly once, at the
    # cold start: both backends zero `spinup_time` for a warm start
    # (pylbm forward_model.run_single, pyudales ForwardModel._window_runtime /
    # run_single), and every cycle after the very first is warm-started from the
    # previous cycle's analyzed state — within a window by BaseFilter's cycle
    # loop, across windows by the `state_input` carry below.
    assim_model = instantiate(
        cfg.assim_model.forward_model,
        results_dir=assim_results_dir,
        simulation_time=cycle_seconds,
    )
    instantiate(cfg.assim_model.prepare, forward_model=assim_model)

    # Optional on-disk ensemble forecasts (one NetCDF per member per cycle,
    # under cycle_{k}/ dirs) — keeps a large ensemble field off host RAM. The
    # analyzed end-of-cycle state is carried in memory either way.
    ensemble_states_dir = (
        out_dir / "_ensemble_states"
        if bool(cfg.run.get("ensemble_save_on_disk", False))
        else None
    )
    ensemble_model = instantiate(
        cfg.assim_model.ensemble_model,
        forward_model=assim_model,
        results_dir=ensemble_states_dir,
    )

    # --- Prior parameter ensemble ---------------------------------------------
    # NOT written here: the run's `prior_params.nc` is the ASSEMBLED per-window
    # prior (this ensemble is window 0's), written with the posterior at the end.
    prior_sampler = instantiate(prior_params_cfg)
    prior_params = prior_sampler.sample(ensemble_size)

    # --- Observation operators and per-cycle observations ----------------------
    truth_obs_op = create_observation_operator(cfg.obs, cfg.truth_model.solver_name)
    assim_obs_op = create_observation_operator(cfg.obs, cfg.assim_model.solver_name)
    # The predicted-observation twin of the truth-side stride: a cycle's segment
    # emits `every_n` frames, and the filter must only see the one its analysis
    # assimilates. Wrapped ONLY when the stride bites, so an unstrided run hands
    # the filter the very same operator object it always did.
    filter_obs_op: Any = assim_obs_op
    if every_n > 1:
        filter_obs_op = _StridedObservationOperator(assim_obs_op, every_n)

    obs_error_std = float(cfg.filtering.obs_error_std)
    # Every cycle's observation is built and perturbed HERE, over the whole
    # horizon and in global cycle order, before the window loop: the draw
    # sequence then depends only on the cycle index, never on how the horizon is
    # chunked into windows, which is what makes windowing mathematically inert.
    # The arrays are KB-scale (sensors x states per cycle), so holding the
    # horizon's worth costs nothing; the truth itself stays lazily on disk.
    observations: list[Any] = []
    observations_clean: list[Any] = []
    cycle_times: list[float] = []
    truth_view = open_truth(true_state_path, n_total, x_offset, start_idx, t_offset)
    for cycle in range(num_cycles):
        # The cycle's truth BLOCK is its `every_n` output frames (half-open, so
        # no frame is covered twice); the frames it ASSIMILATES are the strided
        # subset ending on the block's last frame — the analysis time. At
        # every_n = 1 that block is one frame and the stride is the identity, so
        # this is exactly the every-frame filter.
        cycle_block = truth_view.isel(
            time=slice(cycle * every_n, (cycle + 1) * every_n)
        )
        cycle_truth = cycle_block.isel(time=slice(every_n - 1, None, every_n))
        cycle_times.append(float(np.asarray(cycle_truth["time"].values).ravel()[0]))
        cycle_obs_clean = truth_obs_op(cycle_truth)
        rng_key, subkey = jax.random.split(rng_key)
        cycle_obs = cycle_obs_clean + obs_error_std * np.asarray(
            jax.random.normal(subkey, cycle_obs_clean.shape)
        )
        observations.append(cycle_obs)
        observations_clean.append(cycle_obs_clean)
    truth_view.close()

    # Size C_D from the vector ONE analysis consumes: a single observation
    # frame (sensors x observed states). That is a cycle's whole observation
    # vector here (one ASSIMILATED frame per cycle, whatever the stride), and
    # the library validates C_D per frame either way, so this stays the
    # per-frame contract.
    first_obs = observations[0]
    if isinstance(first_obs, xarray.DataArray):
        n_d = int(first_obs.sizes["obs"])
    else:
        # obs.temporal_mode null: the bare spatial operator already returns the
        # flat observation vector of the cycle's single frame.
        n_d = int(np.asarray(first_obs).size)
    C_D_diag = (obs_error_std**2) * jnp.ones(n_d)

    # --- Filter ----------------------------------------------------------------
    rng_key, filter_key = jax.random.split(rng_key)
    filter_overrides: dict[str, Any] = {}
    if cfg.filtering.mode == "state":
        # The default config selects a random-walk evolution for the parameter-
        # updating modes. A plain `filtering.mode=state` override must remain a
        # valid independent axis: parameters are fixed in this mode, so do not
        # pass the otherwise-default evolution into the constructor.
        filter_overrides["parameter_evolution"] = None
    enkf = instantiate(
        cfg.filtering.filter,
        observation_operator=filter_obs_op,
        forward_model=ensemble_model,
        C_D=C_D_diag,
        rng_key=filter_key,
        **filter_overrides,
    )
    # The per-window observation-space arrays (window_{w}_{obs,pred_obs}.nc) are
    # KB-scale and are what the shared normalized-data-mismatch diagnostic reads,
    # so they are always produced. This records the per-cycle prior AND (via the
    # ride-along rows the analysis already computes) posterior forecast
    # observations; no extra forward solve.
    enkf.collect_pred_obs = True

    save_history = bool(cfg.run.get("save_history", True))

    # --- Window loop -----------------------------------------------------------
    # One run() call per window; state, params and the filter's PRNG stream are
    # carried across the boundary, so the cycle chain is one continuous filter
    # run and the chunking is pure book-keeping (see the module docstring).
    state_input: Any = None  # cold start: cycle 0 pays the model's spin-up
    params: Any = prior_params
    diagnostic_rows: list[dict] = []
    params_history_pieces: list[xarray.Dataset] = []
    state_history_pieces: list[xarray.Dataset] = []
    window_seconds: list[float] = []
    filter_start = time.perf_counter()
    for window in range(num_windows):
        window_start = time.perf_counter()
        first_cycle = window * cycles_per_window
        window_slice = slice(first_cycle, first_cycle + cycles_per_window)
        window_times = cycle_times[window_slice]

        # On-disk mode: give this window its own cycle-directory root, so its
        # cycle_0 does not delete the previous window's.
        staging_dir = _window_staging_dir(ensemble_states_dir, window)
        if staging_dir is not None:
            staging_dir.mkdir(parents=True, exist_ok=True)
            enkf.base_results_dir = staging_dir

        # The ensemble this window STARTS from: the sampled prior for window 0,
        # the previous window's carried posterior afterwards.
        _save_window_params(
            params,
            windows_dir / f"window_{window}_prior_params.nc",
            window * sim_time,
        )

        # ``return_history=True`` regardless of ``run.save_history``: the
        # window's analyzed frames ARE window_{w}_posterior_state.nc, whose
        # footprint is one ESMDA window rollout's. ``save_history`` still decides
        # whether the horizon-long histories are accumulated and written.
        result = enkf.run(
            state=state_input,
            params=params,
            observations=observations[window_slice],
            return_history=True,
        )

        # BaseFilter numbers cycles WITHIN a run() call; the filtering pipeline's
        # artifacts are indexed by the global cycle, so renumber before saving.
        for local_cycle, diagnostics in enumerate(result.diagnostics):
            diagnostics.cycle = first_cycle + local_cycle
            diagnostic_rows.append(dataclasses.asdict(diagnostics))

        assert result.state_history is not None  # return_history=True above
        _save_window_state(
            result.state_history,
            windows_dir / f"window_{window}_posterior_state.nc",
            window_times,
        )
        if result.params is not None:
            _save_window_params(
                result.params,
                windows_dir / f"window_{window}_posterior_params.nc",
                (window + 1) * sim_time,
            )

        # Observation space. ``obs_error_std`` is TILED to the window's full
        # K*N_obs length: C_D is the per-FRAME covariance, and the diagnostic
        # broadcasts sigma against the whole window's vector.
        _save_window_obs_diagnostics(
            windows_dir,
            window,
            _flat_obs_vector(observations[window_slice]),
            _flat_obs_vector(observations_clean[window_slice]),
            np.tile(np.sqrt(np.asarray(C_D_diag)), cycles_per_window),
            _stack_cycle_pred_obs(enkf.pred_obs_history),
            _stack_cycle_pred_obs(enkf.pred_obs_post_history),
            truth_obs_op,
        )

        if save_history:
            state_history_pieces.append(result.state_history)
            if result.params_history is not None:
                # Each call prepends the params it was handed, so every window
                # after the first repeats the previous window's last entry. Drop
                # it, keeping the accumulated history's "entry 0 is the prior,
                # entry i>=1 is cycle i-1's posterior" convention true.
                piece = result.params_history
                if window > 0:
                    piece = piece.isel(cycle=slice(1, None))
                params_history_pieces.append(piece)

        # On-disk mode: move this window's cycle_{0..K-1} dirs onto the global
        # cycle index, emptying the staging root for the next window.
        _collect_window_cycle_dirs(ensemble_states_dir, window, cycles_per_window)

        state_input = result.state
        params = result.params
        window_seconds.append(time.perf_counter() - window_start)

    filter_seconds = time.perf_counter() - filter_start

    # --- Outputs ----------------------------------------------------------------
    # The filtering-native artifacts, accumulated over the WHOLE horizon (global
    # cycle indices) so the filtering metric/figure stages read them exactly as
    # before: the analyzed end-of-run state and parameters, the per-cycle
    # histories when requested, and the per-cycle diagnostics always.
    if result.state is not None:
        result.state.to_netcdf(out_dir / "posterior_state.nc")
    if params_history_pieces:
        xarray.concat(params_history_pieces, dim="cycle", join="override").to_netcdf(
            out_dir / "params_history.nc"
        )
    if state_history_pieces:
        xarray.concat(state_history_pieces, dim="cycle", join="override").to_netcdf(
            out_dir / "state_history.nc"
        )
    write_yaml(diagnostic_rows, out_dir / "cycle_diagnostics.yaml")

    # Persist the truth-access parameters (slicing/offsets + the window AND
    # cycle geometry) so the downstream metric/figure scripts reconstruct the
    # exact same lazy view of the on-disk truth without re-deriving the horizon
    # logic. The ESMDA keys keep their ESMDA meanings (`sim_time` is WINDOW
    # seconds, `n_per_window` its frame count) so the shared ESMDA stages read
    # this directory unchanged; the filtering stages get the cycle geometry
    # under its own keys (`cycle_seconds`, `n_per_cycle`, `num_cycles`).
    write_yaml(
        {
            "true_state_path": str(true_state_path),
            "x_offset": float(x_offset),
            "start_idx": int(start_idx),
            "t_offset": float(t_offset),
            "n_total": int(n_total),
            # TRUTH frames per window (the contiguous block the shared stages
            # slice), which is the cycle count times the stride — the two
            # coincide only at assimilate_every_n_step=1.
            "n_per_window": int(frames_per_window),
            "num_windows": int(num_windows),
            "sim_time": float(sim_time),
            # A cycle covers `assimilate_every_n_step` truth frames and analyzes
            # the LAST of them, which is exactly the frame
            # `end_of_cycle_indices` pairs with cycle c: (c+1)*n_per_cycle - 1.
            "n_per_cycle": int(every_n),
            "num_cycles": int(num_cycles),
            "cycle_seconds": float(cycle_seconds),
            # The assembled parameter files carry a physical (seconds) time axis
            # — the per-window pieces each hold a scalar `time` — so the
            # parameter plots' window edges land at multiples of sim_time.
            "is_dynamic": True,
            "truth_solver_name": str(cfg.truth_model.solver_name),
            "assim_solver_name": str(cfg.assim_model.solver_name),
            # What the per-window state files hold, stamped onto the mean-field
            # artifacts by the metric stage: analyzed frames, not a free
            # rollout. Unstrided they tile the window one per frame (NOT
            # sparse); a stride makes them a strided sample of it.
            "moment_sampling": _moment_sampling(every_n),
            "moment_sampling_is_sparse": every_n > 1,
        },
        out_dir / "truth_access.yaml",
    )

    write_yaml(
        {
            "configuration": {
                "filter": type(enkf).__name__,
                "mode": str(cfg.filtering.mode),
                "num_assimilation_windows": int(num_windows),
                "cycles_per_window": int(cycles_per_window),
                "num_cycles": int(num_cycles),
                "ensemble_size": int(ensemble_size),
                "simulation_time_per_window": float(sim_time),
                "seconds_per_cycle": float(cycle_seconds),
                # The analysis stride: a cycle spans this many observation
                # frames (output_frequency each) and assimilates the last one.
                "assimilate_every_n_step": int(every_n),
                "final_time": float(final_time),
                "observation_error_std": obs_error_std,
                # The gate the shared observation-space diagnostic reads before
                # it opens windows/window_*_{obs,pred_obs}.nc.
                "save_obs_diagnostics": True,
                # A filter has no window-long prior rollout: within a window the
                # analyses are interleaved with the forecast, so
                # windows/window_{w}_prior_state.nc is deliberately absent and
                # the prior halves of the state diagnostics are skipped.
                "save_prior_state": False,
                "esmda_step_semantics": _ESMDA_STEP_SEMANTICS,
                "seed": int(cfg.filtering.seed),
                "truth_model": str(cfg.truth_model.name),
                "assimilation_model": str(cfg.assim_model.name),
                "truth_source": "disk" if cfg.run.truth_dir is not None else "inline",
                "truth_dir": (
                    str(cfg.run.truth_dir) if cfg.run.truth_dir is not None else None
                ),
                "num_truth_frames": int(n_total),
                # `filter` above is the composed filter class, which stays
                # `EnsembleKalmanFilter` for every update flavor (the analysis
                # is injected, not a subclass). The resolved analysis subtree is
                # therefore the ONLY record of which update math ran, and of its
                # nested observation-TSVD settings. Never null: the
                # filtering/analysis group always sets a `_target_`.
                "analysis": OmegaConf.to_container(
                    cfg.filtering.analysis, resolve=True
                ),
                # The analysis and the localization are a coupled pair (each
                # analysis declares a localization_policy of optional /
                # forbidden / required), and the per-block LETKF diagnostics in
                # cycle_diagnostics.yaml are only interpretable against the
                # strategy and radius that produced them. Recorded here for the
                # same reason as `analysis`, in the same fully resolved form;
                # `null` is a real value (the global update).
                "localization": (
                    OmegaConf.to_container(cfg.filtering.localization, resolve=True)
                    if cfg.filtering.localization is not None
                    else None
                ),
                # Preserve the fully resolved Hydra subtree (including the
                # implementation target and variable scales) so benchmark
                # records never need to infer which reduction was run.
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
            },
            "timing": {
                "filter_total_seconds": float(filter_seconds),
                "mean_cycle_seconds": float(filter_seconds / max(num_cycles, 1)),
                "per_window_seconds": [float(s) for s in window_seconds],
            },
        },
        out_dir / "run_info.yaml",
    )

    # Assemble the ESMDA-schema, downstream-facing outputs from the per-window
    # files: prior_params.nc / posterior_params.nc (concatenated on the physical
    # `time` axis the scalar per-window coords carry -- NOT rebased, which only
    # applies to pieces with a time DIM) and posterior_state_mean.nc (the
    # ensemble mean plus vel_mean/vel_std, reduced one window at a time so the
    # ensemble is never held across windows).
    _save_assembled_outputs(
        out_dir, windows_dir, num_windows, sim_time, is_dynamic=False
    )
    print(f"Saved outputs in {out_dir}")


@hydra.main(  # type: ignore[misc]
    version_base=None, config_path="../../conf", config_name="run_filtering"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
