"""Run ESMDA: parameter-only / state-only / joint state+parameter, static / time-varying
parameters, single-window / multi-window rollout, with truth simulated inline
or loaded from disk.

This is the first stage of a three-script single-run pipeline (see
``scripts/run_esmda_pipeline.sh``):

  1. scripts/esmda/run_esmda.py            (THIS) -- runs the assimilation and saves
                                       every raw artifact: the per-window
                                       posterior (and, unless
                                       run.save_prior_state=false, prior) states
                                       + params, the
                                       assembled posterior_params.nc /
                                       prior_params.nc / posterior_state_mean.nc,
                                       the truth state/params, truth_access.yaml
                                       and run_info.yaml. Under
                                       esmda.save_obs_diagnostics (default true)
                                       it also writes the per-window
                                       observation-space arrays
                                       windows/window_{w}_{obs,pred_obs,
                                       params_steps}.nc.
  2. scripts/esmda/compute_esmda_metrics.py -- reads those, computes the parameter /
                                       state / sensor metrics and writes
                                       run_summary.yaml.
  3. scripts/esmda/make_esmda_figures.py   -- reads those, draws the figures.

The metric and figure stages share their truth-access / sensor-series logic via
``scripts/esmda/_esmda_common.py``.

This single script replaces the former
run_{parameter,state_and_parameter,rollout,time_varying_parameter,
time_varying_parameters_rollout}_esmda.py family. Two declarative axes (plus the
window count) select the mode (see conf/run_esmda.yaml):

  * ``esmda/smoother=static|state|state_and_parameter|dynamic|state_and_dynamic``
        which augmented state the Kalman update acts on:
          - ``static``             parameter-only, static scalar parameters.
          - ``state``              state-only, with static parameters fixed.
          - ``state_and_parameter`` joint time=0 state + static parameters.
          - ``dynamic``            parameter-only, time-varying (AR(2)) params.
          - ``state_and_dynamic``  joint time=0 state + time-varying params
                                   (StateAndTimeVaryingParameterESMDA): both
                                   ``include_state`` and ``is_dynamic`` are True.
  * ``params@prior_params=static|dynamic``
        static scalar parameters vs a time-varying (AR(2)) prior. Pair
        static/state/state_and_parameter with ``static``; dynamic/state_and_dynamic
        with ``dynamic``.
  * ``esmda.num_assimilation_windows=1|N``
        a single assimilation window vs an N-window rollout.

and the truth source:

  * ``run.truth_dir=null``    simulate the truth inline (default).
  * ``run.truth_dir=<path>``  load a state.nc/params.nc artifact written
                              by run_forward_model.py run.time_varying=true.

The mode is driven entirely by the smoother type and the params mounts: the
window loop is generic over ``include_state`` (``isinstance(esmda,
StateAndParameterESMDA)`` — True for both joint variants via inheritance) and
``is_dynamic`` (``seconds_per_knot`` present in ``truth_params`` — True for both
dynamic variants), so no per-combination branching is needed.

Truth (states + parameters) for every window is generated up front, before any
assimilation runs. The window loop then consumes the precomputed truth.

Examples::

    python scripts/esmda/run_esmda.py esmda/smoother=static \
        params@prior_params=static params@truth_params=static_truth
    python scripts/esmda/run_esmda.py esmda/smoother=state_and_parameter \
        params@prior_params=static esmda.num_assimilation_windows=3
    python scripts/esmda/run_esmda.py esmda/smoother=state \
        params@prior_params=static esmda/localization=distance
    python scripts/esmda/run_esmda.py esmda/smoother=dynamic \
        params@prior_params=dynamic params@truth_params=dynamic_truth \
        esmda.num_assimilation_windows=3
    python scripts/esmda/run_esmda.py esmda/smoother=state_and_dynamic \
        params@prior_params=dynamic params@truth_params=dynamic_truth \
        esmda.num_assimilation_windows=3
"""

# mypy: ignore-errors
# Legacy untyped entry point, waived on the same terms as
# ``scripts/esmda/_esmda_common.py``: it predates the strict mypy config and has
# never satisfied it (~15 errors, a couple of them latent rather than cosmetic —
# see the None/int divisions in the state-summary helpers). It is type-checked
# transitively whenever a test importing it is committed, which several already
# do (tests/test_run_esmda.py, tests/test_localization.py,
# tests/test_run_probe_series.py). Waived wholesale rather than annotated
# piecemeal; drop this when the file is typed, and fix those divisions then.

import pathlib
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import hydra
import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import xarray
from data_assimilation.observation_operator import flatten_observations
from data_assimilation.smoothing.esmda import StateAndParameterESMDA
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise; must precede `import jax`)
from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    create_aggregate_observations,
    create_observation_operator,
    filter_parameter_config,
)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.esmda._esmda_common import open_truth, truth_x_min, write_yaml

# ---------------------------------------------------------------------------
# Small helpers (in the style of run_forward_model.py)
# ---------------------------------------------------------------------------


def _concat_windows(paths, sim_time, rebase, transform=None, open_fn=None):
    """Concatenate per-window NetCDF files along ``time``.

    Files are opened one at a time and (optionally) reduced by ``transform``
    before being appended, so a memory-heavy per-window reduction (e.g. taking
    the ensemble mean of a state field) keeps only the reduced result instead of
    holding every window's full data in memory at once.

    ``open_fn`` opens (and may already reduce) one window file; it defaults to
    loading the whole file into memory, which is fine for the small parameter
    files but must NOT be used on the tens-of-GB ensemble state files -- pass
    ``_streaming_state_summary`` there so the ensemble is never materialised.

    ``rebase`` (used for the time-varying case) shifts each window's local time
    onto a single monotonic global axis (window ``w`` starts at ``w*sim_time``);
    the static case stacks the windows as-is, matching the old rollout script.
    """
    if open_fn is None:
        open_fn = lambda path: xarray.open_dataset(path).load()
    pieces = []
    for w, path in enumerate(paths):
        ds = open_fn(path)
        if transform is not None:
            ds = transform(ds)
        if rebase and "time" in ds.dims:
            t = np.asarray(ds["time"].values, dtype=float)
            ds = ds.assign_coords(time=(t - t[0]) + w * sim_time)
        pieces.append(ds)
    if len(pieces) == 1:
        return pieces[0]
    return xarray.concat(pieces, dim="time", join="override")


# ---------------------------------------------------------------------------
# Disk-backed ensemble reassembly (used when the assimilation ensemble ran
# ``save_on_disk``: each member/ESMDA-step forecast is its own NetCDF on disk
# instead of one in-memory ensemble Dataset). The full ensemble field is tens of
# GB at the high-resolution sweep points, so it must never be concatenated in
# RAM -- not here and not by xarray.concat. These helpers stream the per-member
# files (no dask required) so peak memory stays at one member (~hundreds of MB).
# ---------------------------------------------------------------------------


def _member_state_files(step_dir, ensemble_size, sim_name="state"):
    """Per-member state file paths for an ESMDA step directory, in member order."""
    return [step_dir / f"{sim_name}_{m}.nc" for m in range(ensemble_size)]


def _stream_concat_members(member_files, out_path):
    """Concatenate per-member state files along a new leading ``ensemble`` dim.

    On-disk equivalent of ``xarray.concat(members, dim="ensemble")``, written one
    member at a time so the full ensemble is never materialised. Every data
    variable (one whose name is not also a dimension, i.e. not a coordinate) gains
    a leading ``ensemble`` axis; dimensions, coordinate variables and attributes
    are copied verbatim from the first member. The result opens identically to the
    old in-memory window state file for the downstream metric/plot scripts.
    """
    member_files = list(member_files)
    n = len(member_files)
    with netCDF4.Dataset(member_files[0]) as ref:
        dim_names = set(ref.dimensions.keys())
        data_vars = [name for name in ref.variables if name not in dim_names]
        with netCDF4.Dataset(out_path, "w") as out:
            out.setncatts({k: ref.getncattr(k) for k in ref.ncattrs()})
            out.createDimension("ensemble", n)
            for name, dim in ref.dimensions.items():
                out.createDimension(name, len(dim))
            # Coordinate variables (name == a dim) are identical across members,
            # so copy them straight through with no ensemble axis.
            for name in dim_names:
                if name not in ref.variables:
                    continue
                var = ref.variables[name]
                cvar = out.createVariable(name, var.dtype, var.dimensions)
                cvar.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                cvar[...] = var[...]
            # Data variables get the leading ensemble axis; created empty here and
            # filled member by member below.
            for name in data_vars:
                var = ref.variables[name]
                dvar = out.createVariable(
                    name, var.dtype, ("ensemble", *var.dimensions)
                )
                dvar.setncatts({k: var.getncattr(k) for k in var.ncattrs()})

            for m, f in enumerate(member_files):
                with netCDF4.Dataset(f) as src:
                    for name in data_vars:
                        out.variables[name][m, ...] = src.variables[name][...]


def _last_frame_ensemble(member_files):
    """Ensemble state at the final time step, read from per-member files.

    Only the last time frame of each member is loaded, so this stays small (~one
    frame x ensemble) even when each member's full series is many GB. Returns the
    same thing as the in-memory path's ``posterior_state.isel(time=-1)`` -- the
    warm-start initial condition for the next assimilation window.
    """
    frames = [xarray.open_dataset(f).isel(time=-1).load() for f in member_files]
    return xarray.concat(frames, dim="ensemble", join="override")


# Reducing one window's ensemble is the bulk of ``_save_assembled_outputs`` and
# is read-bandwidth-bound (the per-window file is uncompressed and tens of GB).
# Split the ensemble axis across a few reader threads, each with its own netCDF
# handle accumulating partial sums -- independent read-only HDF5 handles read
# concurrently, overlapping I/O wait. The win plateaus at ~2 workers on this
# DRAM-bandwidth-bound hardware (see the ensemble-scaling memory note), so keep
# the count small; each worker holds one member, so peak memory stays bounded.
_SUMMARY_WORKERS = 2


def _streaming_state_summary(path):
    """Ensemble-reduce one window state file without materialising the ensemble.

    Returns the ensemble mean of every reduced data variable plus
    ``vel_mean``/``vel_std`` (the ensemble mean and population std of the
    velocity magnitude), streaming over the leading ``ensemble`` axis a member at
    a time so the tens-of-GB window is never held in full -- materialising it
    exhausts RAM/swap and drops the reduction onto a single thread under heavy
    paging. Peak memory is ``_SUMMARY_WORKERS`` members plus a few ensemble-free
    float64 accumulators (a couple of GB).

    Two cost cuts over a naive "mean every variable, then recompute |vel|" pass:

    * Each member's ``u``/``v``/``w`` is read **once** and reused for both its own
      mean and the ``|vel|`` accumulators, instead of read again for the velocity.
    * A stored per-member ``vel_magnitude`` (== ``|vel|``) is **not** reduced: its
      mean would equal ``vel_mean``, which every downstream consumer
      (``velmag_field`` prefers ``vel_mean``; ``compute_esmda_metrics`` rebuilds
      |U| from ``u``/``v``/``w``) uses instead. Dropping it avoids reading the
      single largest (float64) variable off disk for an unused result.

    Means use a running sum; ``vel_std`` uses the running sum and sum-of-squares
    of ``|vel|`` (``std = sqrt(<x^2> - <x>^2)``, ddof=0, matching the old
    ``DataArray.std(dim="ensemble")``). The result opens identically to the old
    in-memory summary for ``_concat_windows`` and the downstream scripts (minus
    the now-redundant ``vel_magnitude`` mean).
    """
    with netCDF4.Dataset(path) as ds:
        dim_names = set(ds.dimensions.keys())
        if "ensemble" not in dim_names:  # already reduced -- nothing to stream
            return xarray.open_dataset(path).load()
        data_vars = [name for name in ds.variables if name not in dim_names]
        n_ens = len(ds.dimensions["ensemble"])
        has_vel = all(v in ds.variables for v in ("u", "v", "w"))
        # ``vel_magnitude`` mean is redundant with ``vel_mean`` (see docstring);
        # skip it so the largest variable is never read.
        reduce_vars = [n for n in data_vars if not (has_vel and n == "vel_magnitude")]

        # One worker: own handle, accumulate sums (and |vel| moments) over the
        # members assigned to it. Returns ensemble-free arrays, cheap to combine.
        def _accumulate(members):
            with netCDF4.Dataset(path) as wds:

                def _member(var, m):
                    idx = tuple(
                        m if d == "ensemble" else slice(None) for d in var.dimensions
                    )
                    return np.asarray(var[idx], dtype=np.float64)  # owns its buffer

                sums = {}  # data var -> running ensemble sum (ensemble axis dropped)
                vel_s1 = vel_s2 = None  # running sum / sum-of-squares of |vel|
                for m in members:
                    buf = {}
                    for name in reduce_vars:
                        chunk = _member(wds.variables[name], m)
                        buf[name] = chunk
                        if name in sums:
                            sums[name] += chunk
                        else:
                            sums[name] = chunk
                    if has_vel:
                        u, v, w = buf["u"], buf["v"], buf["w"]
                        vmag = np.sqrt(u * u + v * v + w * w)
                        if vel_s1 is None:
                            vel_s1, vel_s2 = vmag, vmag * vmag
                        else:
                            vel_s1 += vmag
                            vel_s2 += vmag * vmag
                return sums, vel_s1, vel_s2

        n_workers = max(1, min(_SUMMARY_WORKERS, n_ens))
        if n_workers == 1:
            partials = [_accumulate(range(n_ens))]
        else:
            # Round-robin members across workers for an even split.
            groups = [list(range(i, n_ens, n_workers)) for i in range(n_workers)]
            with ThreadPoolExecutor(n_workers) as ex:
                partials = list(ex.map(_accumulate, groups))

        sums = {}  # combined ensemble sums
        vel_s1 = vel_s2 = None
        for psums, ps1, ps2 in partials:
            for name, arr in psums.items():
                sums[name] = arr if name not in sums else sums[name] + arr
            if ps1 is not None:
                vel_s1 = ps1 if vel_s1 is None else vel_s1 + ps1
                vel_s2 = ps2 if vel_s2 is None else vel_s2 + ps2

        # Coordinate variables (name == a dim) are identical across members; copy
        # them through with no ensemble axis, like ``_stream_concat_members``.
        coords = {}
        for name in dim_names:
            if name == "ensemble" or name not in ds.variables:
                continue
            var = ds.variables[name]
            coords[name] = (
                var.dimensions,
                np.asarray(var[...]),
                {k: var.getncattr(k) for k in var.ncattrs()},
            )

        data = {}
        for name in reduce_vars:
            var = ds.variables[name]
            out_dims = tuple(d for d in var.dimensions if d != "ensemble")
            mean = (sums[name] / n_ens).astype(var.dtype)
            data[name] = (out_dims, mean, {k: var.getncattr(k) for k in var.ncattrs()})
        if has_vel:
            vel_dims = tuple(d for d in ds.variables["u"].dimensions if d != "ensemble")
            vel_dtype = ds.variables["u"].dtype
            mean = vel_s1 / n_ens
            std = np.sqrt(np.maximum(vel_s2 / n_ens - mean * mean, 0.0))
            data["vel_mean"] = (vel_dims, mean.astype(vel_dtype))
            data["vel_std"] = (vel_dims, std.astype(vel_dtype))

        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}

    return xarray.Dataset(data, coords=coords, attrs=attrs)


# ---------------------------------------------------------------------------
# Observation-space diagnostics (phase 2)
# ---------------------------------------------------------------------------

# The flattened observation vector is built by nested concatenation in
# ``data_assimilation.observation_operator``: ``flatten_observations`` is
# time-major, and within each time entry ``ObservationOperator`` concatenates per
# state component, each block holding all sensors. So the sensor index is the
# fastest axis and the time entry (an aggregation interval when
# ``obs.interval_seconds`` is set, otherwise an output frame) the slowest:
#
#     j = interval * (n_states * n_sensors) + state * n_sensors + sensor
#
# ``_obs_index_coords`` decodes that into per-observation labels so the saved
# arrays are interpretable without re-deriving the layout downstream. Kept here
# rather than in ``evaluation`` because it reads the operator object, which is
# data-assimilation machinery the leaf library must not import (see the phase-2
# plan and the master plan's invariant 5).
#
# The dimension is ``obs_index``, not ``obs``: the observation variable is called
# ``obs``, and a variable whose name equals its dimension is silently promoted to
# an index coordinate on the netCDF round-trip -- so a file written with an
# ``obs`` dimension reads back with ``obs`` as a coordinate rather than the data
# variable the docs describe.
_OBS_DIM = "obs_index"
_OBS_ORDERING = (
    "obs_index j = interval*(n_states*n_sensors) + state*n_sensors + sensor; "
    "the sensor index is the fastest-varying axis. Without obs.interval_seconds "
    "the slowest axis is the output frame rather than an aggregation interval, "
    "and obs_interval indexes frames."
)


def _obs_index_coords(obs_op, n_d):
    """Per-observation ``sensor`` / ``state`` / ``interval`` labels, or ``{}``.

    ``obs_op`` is the observation operator (a ``TemporalObservationOperator`` or
    a bare ``ObservationOperator``); ``n_d`` the length of the flat observation
    vector. Returns coordinate arrays keyed on the ``obs_index`` dimension.
    Degrades to an empty mapping -- the flat index alone -- when the operator
    does not expose the counts or when ``n_d`` is not a whole number of (state,
    sensor) blocks, so an unusual operator costs metadata rather than the whole
    artifact.

    ``obs_interval`` counts aggregation intervals when ``obs.interval_seconds``
    is set and output *frames* when it is not; the arithmetic is identical
    either way, and the file attrs say so.
    """
    base = getattr(obs_op, "observation_operator", obs_op)
    n_sensors = int(getattr(base, "num_sensors", 0))
    states = list(getattr(base, "obs_states", []) or [])
    if n_sensors <= 0 or not states:
        return {}

    block = n_sensors * len(states)
    if int(n_d) % block != 0:
        return {}

    j = np.arange(int(n_d))
    return {
        "obs_sensor": (_OBS_DIM, (j % n_sensors).astype(int)),
        "obs_state": (
            _OBS_DIM,
            np.asarray(states, dtype=object)[(j // n_sensors) % len(states)],
        ),
        "obs_interval": (_OBS_DIM, (j // block).astype(int)),
    }


def _flatten_obs(observations, aggregate_obs):
    """Aggregate + flatten one block of observations into the DA's flat vector.

    The script-side twin of ``BaseSmoothing._get_observations``: the smoother
    is handed the raw time-resolved DataArray (and the SAME aggregator
    instance, so its interval-count consistency check spans both), but ``C_D``
    and the persisted per-window observation files are sized/written in the
    flat aggregated space the assimilation actually works in. A bare
    (non-temporal) observation operator returns plain arrays, which pass
    through untouched.
    """
    if not isinstance(observations, xarray.DataArray):
        return np.asarray(observations)
    if aggregate_obs is not None:
        observations = aggregate_obs(observations)
    return flatten_observations(observations)


def _save_obs_diagnostics(
    windows_dir,
    window,
    obs,
    obs_clean,
    obs_error_std,
    pred_obs_history,
    result_params,
    obs_op,
):
    """Write one window's observation-space arrays (WP2.1).

    Three small files under ``windows/``: the assimilated observations with the
    noise-free truth projection and the per-observation error std, the predicted
    observations at every ESMDA iteration, and the per-iteration parameter
    ensembles. All KB-scale.

    ``pred_obs_history`` is the smoother's ``(N_d, N_e)`` block per iteration --
    ``num_steps + 1`` of them, entry 0 the prior forecast and entry -1 the
    posterior forecast. A history of a different length (a smoother variant that
    does not record every iteration) is still written as-is; the ``esmda_step``
    coordinate is positional, so the diagnostic downstream reads whatever is
    there rather than assuming a count.
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

    if pred_obs_history:
        stacked = np.stack([np.asarray(p, dtype=float) for p in pred_obs_history])
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
            "iteration index; 0 = prior forecast, -1 = posterior forecast. With "
            "esmda.final_time_smoothing=true the last entry is PRE-smoothing."
        )
        pred_ds.to_netcdf(windows_dir / f"window_{window}_pred_obs.nc")

    # The per-iteration parameter ensembles, before the caller's
    # ``.isel(esmda_step=-1)`` discards all but the posterior. Kept for debugging;
    # no diagnostic in this phase reads it.
    result_params.to_netcdf(windows_dir / f"window_{window}_params_steps.nc")


# ---------------------------------------------------------------------------
# Assembled rollout outputs (the small, downstream-facing artifacts)
# ---------------------------------------------------------------------------


def _save_assembled_outputs(out_dir, windows_dir, num_windows, sim_time, is_dynamic):
    """Assemble the per-window files in ``windows_dir`` into the run's outputs.

    Writes ``posterior_params.nc`` / ``prior_params.nc`` (full ensemble, small
    enough to hold in memory) and ``posterior_state_mean.nc`` (the ensemble mean
    state plus the ensemble mean/std of the velocity magnitude). State files are
    reduced one window at a time before concatenating, so the full ensemble is
    never held across windows.

    Metrics (``run_summary.yaml``) and figures are produced downstream by
    ``scripts/esmda/compute_esmda_metrics.py`` and ``scripts/esmda/make_esmda_figures.py``
    from these artifacts plus ``truth_access.yaml``/``run_info.yaml``.
    """
    state_paths = [
        windows_dir / f"window_{w}_posterior_state.nc" for w in range(num_windows)
    ]
    posterior_param_paths = [
        windows_dir / f"window_{w}_posterior_params.nc" for w in range(num_windows)
    ]
    prior_param_paths = [
        windows_dir / f"window_{w}_prior_params.nc" for w in range(num_windows)
    ]

    # Parameters: full ensemble in memory.
    posterior_params = _concat_windows(
        posterior_param_paths, sim_time, rebase=is_dynamic
    )
    prior_params = _concat_windows(prior_param_paths, sim_time, rebase=is_dynamic)
    posterior_params.to_netcdf(out_dir / "posterior_params.nc")
    prior_params.to_netcdf(out_dir / "prior_params.nc")

    # States: reduce each window's ensemble in a streaming, member-at-a-time pass
    # (``_streaming_state_summary``) so the tens-of-GB ensemble is never loaded
    # into memory -- the old ``open_dataset(path).load()`` materialised the whole
    # window, exhausting RAM/swap and stalling on a single thread. Keep the mean
    # state (u/v/w) plus the ensemble mean/std of the velocity magnitude
    # (``vel_mean``/``vel_std``).
    posterior_state = _concat_windows(
        state_paths, sim_time, rebase=is_dynamic, open_fn=_streaming_state_summary
    )
    posterior_state.to_netcdf(out_dir / "posterior_state_mean.nc")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    num_windows = int(cfg.esmda.num_assimilation_windows)
    sim_time = float(cfg.time.simulation_time)
    ensemble_size = int(cfg.ensemble.ensemble_size)
    is_dynamic = "seconds_per_knot" in list(cfg.truth_params.keys())
    final_time = sim_time * num_windows
    domain_x_min = float(cfg.domain.bounds[0][0])
    rng_key = jax.random.PRNGKey(cfg.esmda.seed)

    # Select which parameters ESMDA estimates (conf/run_esmda.yaml
    # `params_to_estimate`): null -> every parameter the sampler configs define;
    # a list -> that subset. The same filter is applied to the prior and the
    # truth samplers, so excluding a parameter reproduces the run as if the knob
    # did not exist on either side (docs/esmda_model_error_parameters.md §4).
    selected = cfg.get("params_to_estimate", None)
    selected = list(selected) if selected is not None else None
    truth_params_cfg = filter_parameter_config(cfg.truth_params, selected)
    prior_params_cfg = filter_parameter_config(cfg.prior_params, selected)
    if selected is not None:
        print(f"Estimating parameters: {selected}")

    # --- Output and windows dir ---------------------------------------------------------
    # The run lands in the configured results dir (conf/run_esmda.yaml's
    # ``paths.results_dir``), so the downstream metric/figure scripts -- and the
    # pipeline shell -- locate it from the config alone, without a timestamped
    # Hydra output dir.
    out_dir = pathlib.Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = out_dir / "windows"

    # Persist the raw composed Hydra config used to launch this run (interpolations
    # intact), so the run is fully reproducible from the output folder alone.
    OmegaConf.save(config=cfg, f=out_dir / "config.yaml")

    # --- True parameter sampler and state -----------------------------------------------------------
    # ``true_state_path`` points at the truth on disk; ``n_total`` is the number
    # of truth frames inside the assimilation horizon [0, final_time). The state
    # itself is opened lazily everywhere downstream (see ``_open_truth``) so a
    # large truth is never held in memory in full.
    if cfg.run.truth_dir is None:
        if num_windows > 0:
            true_forward_model = instantiate(
                cfg.truth_model.forward_model,
                results_dir=None,
                simulation_time=sim_time * num_windows,
            )
            if is_dynamic:
                # Sample the truth over the FULL horizon at the same
                # `seconds_per_knot` spacing as the assimilation prior, so its
                # knots share x-locations with the per-window assim curves
                # (they coincide at every multiple of the spacing, and on the
                # window boundaries when sim_time is a multiple of it).
                truth_sampler = instantiate(
                    truth_params_cfg, simulation_time=sim_time * num_windows
                )
            else:
                truth_sampler = instantiate(truth_params_cfg)
        else:
            true_forward_model = instantiate(
                cfg.truth_model.forward_model,
                results_dir=None,
                simulation_time=sim_time,
            )
            truth_sampler = instantiate(truth_params_cfg)

        true_params = truth_sampler.sample(1)

        instantiate(cfg.truth_model.prepare, forward_model=true_forward_model)
        clean_outputs(model_name=cfg.truth_model.name, forward_model=true_forward_model)
        true_state = true_forward_model(params=true_params.isel(ensemble=0))

        # Persist the simulated truth so the window loop and final plots can
        # re-open it lazily (one window / one z-plane at a time) instead of
        # holding the whole rollout in memory.
        true_state_path = out_dir / "true_state.nc"
        true_state.to_netcdf(true_state_path)
        n_total = int(true_state.sizes["time"])
        # Inline truth is already simulated on the domain grid, starting at t=0.
        x_offset = domain_x_min - truth_x_min(true_state)
        start_idx = 0
        t_offset = 0.0
        del true_state

    else:
        truth_dir = pathlib.Path(cfg.run.truth_dir)
        true_params = xarray.load_dataset(truth_dir / "params.nc")

        # Optionally begin the assimilation horizon partway into a pre-simulated
        # truth (e.g. skip a spin-up): keep the truth from ``start_time`` onward
        # and shift its time axis so ``start_time`` becomes t=0. Both the state
        # and the parameters are sliced and rebased by the same offset.
        start_time = float(cfg.run.truth_start_time or 0.0)

        if "time" in true_params.dims:
            if start_time:
                true_params = true_params.sel(time=true_params.time >= start_time)
                true_params = true_params.assign_coords(
                    time=true_params.time - start_time
                )
            true_params = true_params.sel(time=true_params.time < final_time)

        # The truth state can be very large (tens of GB). Reference the existing
        # artifact directly -- never load or copy the whole file -- and only read
        # the frames inside the assimilation horizon [start_time, start_time +
        # final_time).
        true_state_path = truth_dir / "state.nc"
        with xarray.open_dataset(true_state_path) as _truth_meta:
            true_times = np.asarray(_truth_meta["time"].values, dtype=float)
            # Shift the truth's x onto the domain frame: a truth saved in its own
            # coordinates (e.g. x in [0, 100]) is offset so its upstream edge
            # lines up with the domain's (e.g. x_min = -20). Other axes (y, z)
            # already share the domain origin.
            x_offset = domain_x_min - truth_x_min(_truth_meta)
        # Drop the frames before ``start_time`` and rebase the kept frames onto a
        # [0, final_time) axis (t_offset = start_time), so frame-index slicing in
        # the window loop counts from the chosen start.
        start_idx = int((true_times < start_time).sum())
        t_offset = start_time
        n_total = int(((true_times[start_idx:] - t_offset) < final_time).sum())

    if x_offset:
        print(
            f"Shifting truth x by {x_offset:+g} to align with domain x_min={domain_x_min:g}"
        )

    # Number of truth frames per assimilation window (contiguous, half-open).
    n_per_window = n_total // max(num_windows, 1)

    # Save the (small) truth parameters for the final plots.
    true_params.to_netcdf(out_dir / "true_params.nc")

    # --- Assimilation ensemble model -------------
    assim_results_dir = (
        pathlib.Path(cfg.run.results_dir) if cfg.run.results_dir is not None else None
    )
    assim_model = instantiate(
        cfg.assim_model.forward_model, results_dir=assim_results_dir
    )
    instantiate(cfg.assim_model.prepare, forward_model=assim_model)

    # When ``run.ensemble_save_on_disk`` is set, give the ensemble model a results
    # directory so each member/ESMDA-step forecast is written to its own NetCDF
    # rather than concatenated into one in-memory ensemble Dataset. This keeps the
    # full (tens-of-GB) ensemble field off host RAM -- the per-window prior and
    # posterior states are reassembled below by streaming the per-member files.
    # The directory lives under the run's output dir (so it shares its disk and is
    # cleaned up at the end); only sequential single-process backends use it.
    ensemble_states_dir = (
        out_dir / "_ensemble_states"
        if bool(cfg.run.get("ensemble_save_on_disk", False))
        else None
    )

    # Whether to persist the per-window prior ensemble state (the large artifact;
    # the posterior state and the prior/posterior params are always saved). When
    # false, the downstream prior-vs-posterior state diagnostics simply skip the
    # absent files (see conf/run_esmda.yaml `run.save_prior_state`).
    save_prior_state = bool(cfg.run.get("save_prior_state", True))
    ensemble_model = instantiate(
        cfg.assim_model.ensemble_model,
        forward_model=assim_model,
        results_dir=ensemble_states_dir,
    )

    # --- Prior parameter sampler -----------------------------------------------------------
    prior_sampler = instantiate(prior_params_cfg)
    prior_params = prior_sampler.sample(ensemble_size)

    # --- Optional training-data warm start --------------------------------------------
    # When the neural surrogate is the assimilation model and its
    # forward_model.spinup_source is "training_data", seed the FIRST assimilation
    # window from pre-computed training trajectories instead of a CFD spin-up:
    #   * each member starts from the LAST frame of a training sample, streamed
    #     one frame at a time to per-member files on disk (the full ensemble
    #     never sits in RAM), handed to ESMDA as the window-0 initial state; and
    #   * its sampled prior inflow params are anchored to that sample's final
    #     inflow value (the AR(2) draw's shape is kept, only its level is pinned),
    #     so the params ESMDA estimates match the warm-start state's forcing.
    # The (now known) t=0 is pinned in the window loop via pin_initial_from_spinup.
    fm_cfg = cfg.assim_model.forward_model
    initial_state_dir: pathlib.Path | None = None
    pin_initial_from_spinup = False
    if fm_cfg.get("spinup_source", None) == "training_data":
        from neural_surrogates.training_spinup import (  # type: ignore[import-untyped]
            anchor_prior_params,
            list_split_samples,
            resolve_training_root,
            write_initial_state_files,
        )

        spinup_cfg = cfg.assim_model.get("training_data_spinup", {}) or {}
        root = resolve_training_root(
            fm_cfg.get("model_dir", None), spinup_cfg.get("root", None)
        )
        split = str(spinup_cfg.get("split", "train"))
        jitter = float(spinup_cfg.get("initial_param_jitter_scale", 0.0))
        state_files, param_files = list_split_samples(root, split)

        initial_state_dir = write_initial_state_files(
            state_files, ensemble_size, out_dir / "_initial_states"
        )
        prior_params = anchor_prior_params(
            prior_params, param_files, ensemble_size, jitter
        )
        pin_initial_from_spinup = True
        print(
            f"Training-data warm start: seeded {ensemble_size} members from the "
            f"'{split}' split of {root}"
        )

    # --- Observation operator -----------------------------------------------------------
    truth_obs_op = create_observation_operator(cfg.obs, cfg.truth_model.solver_name)
    assim_obs_op = create_observation_operator(cfg.obs, cfg.assim_model.solver_name)
    # Interval aggregation (or None for full-resolution assimilation). ONE
    # instance, shared between the C_D sizing below and the smoother, so its
    # interval-count consistency check spans the truth and the forecasts.
    aggregate_obs = create_aggregate_observations(cfg.obs)

    # --- Observation error covariance ---------
    # Truth frames sit on a uniform grid over [0, sim_time*num_windows); each
    # window owns exactly `n_per_window` of them. Size C_D from the first such
    # block -- aggregated and flattened exactly as the assimilation will see it
    # -- so it matches every window's observation vector (and the per-window
    # count the assimilation model emits). Opened lazily and sliced, so only the
    # first window's frames are read.
    truth_first_window = open_truth(
        true_state_path, n_total, x_offset, start_idx, t_offset
    ).isel(time=slice(0, n_per_window))
    obs = _flatten_obs(truth_obs_op(truth_first_window), aggregate_obs)
    truth_first_window.close()
    C_D = jnp.diag((cfg.esmda.obs_error_std**2) * jnp.ones(obs.shape[0]))

    # --- Smoother -----------------------------------------------------------
    # The time-varying smoothers flatten each knot into its own augmented-state
    # scalar, so `num_time_points` must equal the sampled prior's knot count
    # (set by `seconds_per_knot`, incl. any extrapolated endpoint). Derive it
    # here rather than from a config constant.
    rng_key, esmda_key = jax.random.split(rng_key)
    smoother_overrides: dict = {}
    if "num_time_points" in cfg.esmda.smoother:
        smoother_overrides["num_time_points"] = int(prior_params.sizes["time"])
    esmda = instantiate(
        cfg.esmda.smoother,
        observation_operator=assim_obs_op,
        forward_model=ensemble_model,
        C_D=C_D,
        rng_key=esmda_key,
        aggregate_observations=aggregate_obs,
        **smoother_overrides,
    )
    include_state = isinstance(esmda, StateAndParameterESMDA)

    # Cap on-disk peak storage: have the smoother delete each ESMDA step's
    # forecast as soon as its update is computed, keeping only the prior (step 0,
    # when ``save_prior_state``) and the posterior (final step) for reassembly
    # below. Without this, all num_steps+1 step directories coexist until the
    # window ends; with it the peak is ~2x ensemble_size. No effect on the
    # in-memory path (the in-memory analog is ``return_state_history`` below).
    esmda.prune_disk_steps = True
    esmda.keep_prior_disk_step = save_prior_state

    # Observation-space diagnostics (WP2.1): have the smoother record the
    # predicted observations it materializes at each ESMDA iteration, so the
    # window loop can persist them alongside the observations themselves.
    # ``.get`` so a config predating the key still composes.
    save_obs_diagnostics = bool(cfg.esmda.get("save_obs_diagnostics", False))
    esmda.collect_obs_diagnostics = save_obs_diagnostics

    # --- Run ESMDA -----------------------------------------------------------
    # Time the assimilation. ``window_seconds`` is each window's full wall-clock
    # cost (observation extraction + Kalman solve + I/O + extrapolation);
    # ``solve_seconds`` isolates the ESMDA Kalman solve itself.
    # When a training-data warm start was prepared, window 0 starts from the
    # per-member initial-state files (a directory ESMDA reads one member at a
    # time); otherwise window 0 cold-starts (``state_input is None``).
    state_input = initial_state_dir
    window_seconds: list[float] = []
    solve_seconds: list[float] = []
    esmda_start = time.perf_counter()
    for window in tqdm(range(num_windows)):
        window_start = time.perf_counter()
        windows_dir.mkdir(parents=True, exist_ok=True)
        prior_params.to_netcdf(windows_dir / f"window_{window}_prior_params.nc")

        # Pin the t=0 knot from window 1 onward so the Kalman update preserves
        # the cross-window continuity that the GP extrapolation established at
        # each window boundary. Window 0's prior t=0 is normally just a
        # cold-start GP draw (over a spun-up flow), so ESMDA is free to fit it.
        # The exception is the training_data warm start: there window 0's t=0 is
        # *known* — the provided initial state was generated under that inflow,
        # and the prior was anchored to it above — so ESMDA must pin it too,
        # leaving the known initial inflow fixed while it fits the rest of the
        # window. Only the time-varying smoother carries this flag.
        if hasattr(esmda, "pin_initial_time_point"):
            esmda.pin_initial_time_point = window > 0 or pin_initial_from_spinup

        # Get observations in window and add noise. Select the w-th contiguous
        # block of frames (half-open) rather than an inclusive time-slice: the
        # frame at the next window's start (t=(window+1)*sim_time) must NOT be
        # double-counted, or interior windows would be one frame longer than the
        # assimilation model emits and the observation vector would misalign.
        window_true_state = open_truth(
            true_state_path, n_total, x_offset, start_idx, t_offset
        ).isel(time=slice(window * n_per_window, (window + 1) * n_per_window))
        window_obs_clean = truth_obs_op(window_true_state)
        window_true_state.close()
        # Perturb every RAW frame with obs_error_std and hand the smoother the
        # time-resolved observations (coords intact -- the aggregator bins on
        # the time coordinate). Under interval-mean aggregation the aggregated
        # noise is then milder than C_D says, which is deliberate (the
        # assimilation stays mildly conservative); with no aggregation the two
        # agree exactly.
        rng_key, subkey = jax.random.split(rng_key)
        window_obs = window_obs_clean + float(cfg.esmda.obs_error_std) * np.asarray(
            jax.random.normal(subkey, window_obs_clean.shape)
        )

        # Sample posterior. ``return_state_history=True`` makes the smoother also
        # return the per-iteration forecast states (esmda_step=0 is the PRIOR
        # forecast, esmda_step=-1 the POSTERIOR). We only need the full history to
        # extract the prior, so request it solely on the in-memory path AND only
        # when the prior state is being saved; otherwise ask for the posterior
        # forecast alone (return_state_history=False) so the smoother never
        # accumulates the num_steps+1 ensemble states in RAM -- the in-memory
        # analog of the on-disk step pruning. In on-disk save mode the per-step
        # states live under the step_{i}/ dirs (the smoother refuses a state
        # history there); the disk path reassembles the prior/posterior states
        # below by streaming those files.
        in_memory = ensemble_states_dir is None
        want_state_history = in_memory and save_prior_state
        solve_start = time.perf_counter()
        output = esmda(
            state=state_input,
            params=prior_params,
            observations=window_obs,
            return_params_history=True,
            return_state_history=want_state_history,
        )
        solve_seconds.append(time.perf_counter() - solve_start)

        # In the in-memory save mode the smoother returns ``(params_history,
        # state)`` -- ``state`` is the full per-step history when requested, else
        # just the posterior forecast. In the disk save mode it returns
        # ``params_history`` alone (the forecasts live as per-member files under
        # the ESMDA step dirs instead).
        if in_memory:
            result_params, state_obj = output
        else:
            result_params, state_obj = output, None

        # Observation-space arrays for this window, written before the
        # ``esmda_step`` axis is collapsed away below (WP2.1). Off by default in
        # a config without the key, in which case the artifact set is unchanged.
        if save_obs_diagnostics:
            # Persisted in the FLAT aggregated space -- the space ``pred_obs``
            # lives in and the metric/figure stages read.
            _save_obs_diagnostics(
                windows_dir,
                window,
                _flatten_obs(window_obs, aggregate_obs),
                _flatten_obs(window_obs_clean, aggregate_obs),
                np.sqrt(np.diag(np.asarray(C_D))),
                esmda.pred_obs_history,
                result_params,
                truth_obs_op,
            )

        posterior_params = result_params.isel(esmda_step=-1)
        posterior_params.to_netcdf(windows_dir / f"window_{window}_posterior_params.nc")

        if in_memory:
            # In-memory path: the ensemble forecasts are already in RAM. With the
            # history requested, slice the prior (step 0) and posterior (step -1)
            # out of it; without it, ``state_obj`` is already the posterior alone.
            if want_state_history:
                posterior_state = state_obj.isel(esmda_step=-1)
                prior_state = state_obj.isel(esmda_step=0)
                posterior_state.to_netcdf(
                    windows_dir / f"window_{window}_posterior_state.nc"
                )
                prior_state.to_netcdf(windows_dir / f"window_{window}_prior_state.nc")
                del prior_state
            else:
                posterior_state = state_obj
                posterior_state.to_netcdf(
                    windows_dir / f"window_{window}_posterior_state.nc"
                )
            state_input = posterior_state.isel(time=-1)
            del state_obj, posterior_state
        else:
            # Disk-backed path: stream the per-member forecast files into the
            # per-window prior/posterior state files (step 0 = prior forecast,
            # step num_steps = posterior forecast) without holding the full
            # ensemble in RAM, then warm-start the next window from the
            # posterior's final frame. The (tens-of-GB-per-step) per-member files
            # are dropped once assembled, so scratch does not grow across windows.
            base = esmda.base_results_dir
            if save_prior_state:
                _stream_concat_members(
                    _member_state_files(base / "step_0", ensemble_size),
                    windows_dir / f"window_{window}_prior_state.nc",
                )
            posterior_files = _member_state_files(
                base / f"step_{esmda.num_steps}", ensemble_size
            )
            _stream_concat_members(
                posterior_files,
                windows_dir / f"window_{window}_posterior_state.nc",
            )
            state_input = _last_frame_ensemble(posterior_files)
            for step in range(esmda.num_steps + 1):
                shutil.rmtree(base / f"step_{step}", ignore_errors=True)

        # Next window's prior: extrapolate the posterior
        if window < num_windows - 1:
            if is_dynamic:
                # Reuse the prior sampler's own per-window knot grid (spaced by
                # `seconds_per_knot`, incl. any extrapolated endpoint), shifted
                # to the next window, then rebase back to [0, sim_time].
                window_times = jnp.asarray(prior_sampler.time_coords)
                prediction_times = window_times + sim_time
                rng_key, subkey = jax.random.split(rng_key)
                extrapolated = prior_sampler.extrapolate(
                    posterior_params, prediction_times, subkey
                )
                prior_params = extrapolated.assign_coords(time=np.asarray(window_times))
            else:
                prior_params = posterior_params

        del output, result_params
        window_seconds.append(time.perf_counter() - window_start)

    esmda_seconds = time.perf_counter() - esmda_start

    # Persist the truth-access parameters (slicing/offsets) so the downstream
    # metric/figure scripts reconstruct the exact same lazy view of the on-disk
    # truth without re-deriving the horizon logic.
    write_yaml(
        {
            "true_state_path": str(true_state_path),
            "x_offset": float(x_offset),
            "start_idx": int(start_idx),
            "t_offset": float(t_offset),
            "n_total": int(n_total),
            "n_per_window": int(n_per_window),
            "num_windows": int(num_windows),
            "sim_time": float(sim_time),
            "is_dynamic": bool(is_dynamic),
            "truth_solver_name": str(cfg.truth_model.solver_name),
            "assim_solver_name": str(cfg.assim_model.solver_name),
        },
        out_dir / "truth_access.yaml",
    )

    # Run metadata + timing persisted as ``run_info.yaml``. The downstream
    # ``scripts/esmda/compute_esmda_metrics.py`` seeds ``run_summary.yaml`` from it and
    # augments it with the parameter / state / sensor estimation metrics.
    run_info = {
        "configuration": {
            "smoother": type(esmda).__name__,
            "joint_state_and_parameter": bool(include_state),
            "time_varying_parameters": bool(is_dynamic),
            "num_assimilation_windows": int(num_windows),
            "ensemble_size": int(ensemble_size),
            "simulation_time_per_window": float(sim_time),
            "final_time": float(final_time),
            "observation_error_std": float(cfg.esmda.obs_error_std),
            "num_esmda_steps": int(cfg.esmda.num_steps),
            "save_obs_diagnostics": bool(save_obs_diagnostics),
            "seed": int(cfg.esmda.seed),
            "truth_model": str(cfg.truth_model.name),
            "assimilation_model": str(cfg.assim_model.name),
            "truth_source": "disk" if cfg.run.truth_dir is not None else "inline",
            "truth_dir": (
                str(cfg.run.truth_dir) if cfg.run.truth_dir is not None else None
            ),
            "num_truth_frames": int(n_total),
        },
        "timing": {
            "esmda_total_seconds": float(esmda_seconds),
            "esmda_solve_seconds": float(sum(solve_seconds)),
            "mean_window_seconds": (
                float(np.mean(window_seconds)) if window_seconds else None
            ),
            "per_window_seconds": [float(s) for s in window_seconds],
            "per_window_solve_seconds": [float(s) for s in solve_seconds],
        },
    }

    write_yaml(run_info, out_dir / "run_info.yaml")

    # Assemble the small, downstream-facing outputs from the per-window files.
    _save_assembled_outputs(out_dir, windows_dir, num_windows, sim_time, is_dynamic)
    print(f"Saved outputs in {out_dir}")


@hydra.main(version_base=None, config_path="../../conf", config_name="run_esmda")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
