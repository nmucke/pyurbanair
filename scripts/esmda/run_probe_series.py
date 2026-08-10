"""High-rate probe time series for ONE window of a finished ESMDA run (WP3.2a).

Welch spectra need on the order of 1e3 samples per sensor; the assimilation
output cadence (``time.output_frequency``, 2-10 s) gives tens of frames per
window, ~30x too coarse. The fields the assimilation wrote are all there is --
the pylbm wrapper runs the compiled Fortran binary as a single
``subprocess.run`` and only ever sees the snapshot files after it exits -- so the
only route to a high-rate series that leaves the pinned Fortran (and the
assimilation pipeline) untouched is a **dedicated re-run**: replay one window's
truth and posterior members at ~1 s cadence, reduce every snapshot to the probe
points, and delete the snapshots again.

**The snapshots are transient but not streamed.** One member's whole window is on
disk before Python reduces any of it, because Python is blocked in
``subprocess.run`` until the solver exits; the spec's "extract as it accumulates"
is not achievable without touching the Fortran. So peak disk is
``num_parallel_processes`` x one member's ``(window + lead-in) / cadence``
snapshots -- tens of GB per member at Barcelona resolution. Size the scratch
filesystem for that, not for the (megabyte) probe files.

This script *adds* artifacts to a finished run directory and changes none. It
reads ``truth_access.yaml``, ``run_info.yaml``, the window's ``*_params.nc`` and
the window's ``*_state.nc`` (one member-frame at a time, never the whole ensemble
file), runs in **its own scratch** (``paths.experiment_dir`` defaults to
``.temp_probes``, and both models are constructed under the
``_PROBE_EXPERIMENT`` experiment name so the assimilation run's ``runcase``
experiment dir, restart files and build stamp are untouched even if the two roots
are pointed at each other), and writes

  * ``truth_probes.nc``                        -- ``u, v, w (time, sensor)``
  * ``windows/window_{w}_probes.nc``            -- ``u, v, w (ensemble, time, sensor)``
  * ``windows/window_{w}_probes_prior.nc``      -- same, prior members, opt-in

Every file carries ``time`` in seconds from the window start (the trimmed
lead-in excluded), the probe locations as ``sensor_x/y/z`` plus a
``sensor_set`` label (``assimilation``/``validation``) on the ``sensor`` dim, and
the attrs ``output_frequency`` (the cadence actually achieved), ``window_index``,
``member_source`` and ``spinup_time``. ``evaluation.turbulence`` (WP3.2b) reads
them for the spectrum + LSD and figure S4, and no-ops when they are absent.
``truth_probes.nc`` is deliberately unversioned (the plan fixes the name) and is
overwritten by every invocation, so all three files carry ``window_index`` and a
consumer must check that the truth's agrees with the members' before comparing
them -- re-probing a second window leaves the first window's member files behind.

Three things to know before reading the numbers:

  * **pylbm only.** The re-run drives the LBM's warm-start/restart path
    directly; a non-pylbm assimilation model is refused. A truth that cannot be
    re-run (another solver, a different grid, no ``rho`` to build a restart
    from) falls back to the truth's *stored* cadence, and says so in the output
    attrs -- the comparison then has to truncate to the common resolved band.
  * **Same window, not the same wall clock.** Each series re-runs a window of
    the same LENGTH, from the window's own initial field, under the window's own
    parameters, after a discarded lead-in. It is not bit-identical to the
    assimilation forecast -- the restart carries the macroscopic field
    essentially exactly, but every re-run starts from a cleared ``restart/``
    (see the note in ``run``: a restart left there is another run's field, and
    on a repeat invocation it is the *truth's*), so its ghost cells and
    non-equilibrium stress are reconstructed as equilibrium rather than carried
    from this member's own history, and the inflow settings and velocity scale
    are re-applied for this run -- and it is offset from the forecast in
    absolute time. Spectra and window statistics are what this is for;
    instantaneous phase agreement is not. That reconstruction is the same on
    both sides of the comparison and is what ``probes.spinup_time`` trims.
  * **Cadence.** ``output_frequency`` is what the solver *achieved*
    (``iout * dt``), not what was asked for: ``dt = C_l/C_u`` is quantised by the
    member's own ``velocity_magnitude``, so members differ by ~1 % and the
    request is almost never hit exactly. A re-run's ``time`` axis is therefore
    synthesised as ``arange(n) * achieved`` (the snapshots are, by then, verified
    to sit on the solver's regular ``iout`` grid); the stored-cadence fallback
    instead keeps the truth's own timestamps, rebased to the window start.

Usage (compose with the SAME overrides the ESMDA run used)::

    python scripts/esmda/run_probe_series.py \
        case=barcelona model@truth_model=pylbm model@assim_model=pylbm \
        probes.run_dir=<esmda output dir> probes.window_index=-1
"""

import collections
import logging
import multiprocessing as mp
import pathlib
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional, Sequence

import hydra
import numpy as np
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise; must precede `import jax`)
from pyurbanair.utils.cpu_pinning import (
    build_cpu_queue,
    cpu_pinning_disabled,
    pin_worker_initializer,
)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from evaluation.turbulence import (
    SPECTRAL_BAND_DECADE_BINS,
    minimum_spectral_samples,
    spectral_band_bins,
)

from scripts.esmda._esmda_common import (
    _sensor_component_timeseries,
    build_sensor_sets,
    load_run_config,
    open_truth,
    read_yaml,
)

logger = logging.getLogger(__name__)

# The only backend whose warm-start path this script drives (see the module
# docstring): the re-run writes an LBM restart file from a stored frame.
_SOLVER = "pylbm"

# Experiment name both re-run models are constructed under, so their experiment
# dir (`<experiment_dir>/experiment/<name>/`, with its own output/ and restart/)
# is never the assimilation run's `runcase`, whatever `paths.experiment_dir` says.
# It is compiled into the binary (mod_dimensions + the geometry case), so a probe
# build tree is stamped for `probe_runcase`: `model.compile=false` works for a
# REPEAT probe run, not for reusing an assimilation run's binary.
_PROBE_EXPERIMENT = "probe_runcase"

# Relative spread of the members' achieved cadences that is still worth one
# shared (fs, nperseg) downstream. dt is quantised by C_u = int(15*|U|), so a
# per-member spread of well under a percent is normal and harmless.
_CADENCE_SPREAD_TOLERANCE = 0.01

# Iteration in an LBM snapshot name (`out_0000_F000304.nc`), width-agnostic.
_SNAPSHOT_ITERATION = re.compile(r"_F(\d+)$")

# Config keys that must agree between this invocation and the run being
# re-probed. The grid, the geometry and the window length decide whether the
# restart can even be written and whether the re-run covers the same window; the
# sensor points decide where the probes land, and probing points the run's own
# sensor metrics never looked at is a silent mismatch rather than an error.
_GUARDED_KEYS = (
    "domain.nx",
    "domain.ny",
    "domain.nz",
    "domain.bounds",
    "geometry.stl_path",
    "time.simulation_time",
    "assim_model.solver_name",
    "truth_model.solver_name",
    "obs.mode",
    "obs.x_points",
    "obs.y_points",
    "obs.z_points",
    "obs.validation_x_points",
    "obs.validation_y_points",
    "obs.validation_z_points",
)

# Keys of ``truth_access.yaml`` this script cannot work without. An older run dir
# missing any of them is a graceful no-op, not a KeyError (invariant 3).
_REQUIRED_TRUTH_ACCESS = (
    "true_state_path",
    "num_windows",
    "sim_time",
    "n_per_window",
    "truth_solver_name",
)


# ---------------------------------------------------------------------------
# Probe points and output schema
# ---------------------------------------------------------------------------


def _probe_points(
    sensor_sets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Concatenate the sensor sets into one probe list plus its set labels.

    The probes default to the union of the assimilation and validation sets, so
    the spectrum is available at held-out locations too. The sets are kept as a
    ``sensor_set`` label rather than as separate files: every consumer of the
    probe series slices by it, and one file per member source is already three.
    """
    xs, ys, zs, labels = [], [], [], []
    for name, (obs_x, obs_y, obs_z) in sensor_sets.items():
        xs.append(np.asarray(obs_x, dtype=float))
        ys.append(np.asarray(obs_y, dtype=float))
        zs.append(np.asarray(obs_z, dtype=float))
        labels.append(np.full(len(xs[-1]), name))
    points = (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs))
    return points, np.concatenate(labels)


def _probe_dataset(
    values: np.ndarray,
    points: tuple[np.ndarray, np.ndarray, np.ndarray],
    sensor_set_labels: np.ndarray,
    times: np.ndarray,
    *,
    window_index: int,
    member_source: str,
    output_frequency: float,
    spinup_time: float,
    member_indices: Optional[Sequence[int]] = None,
    cadence_fallback_reason: Optional[str] = None,
    output_frequency_spread: Optional[float] = None,
    initial_state_source: Optional[str] = None,
) -> xarray.Dataset:
    """Wrap probe values in the WP3.2 output schema.

    ``values`` is ``(time, component, sensor)`` for the truth or
    ``(ensemble, time, component, sensor)`` for a member ensemble; the component
    axis is split into the ``u``/``v``/``w`` data variables.

    Beyond the contract's ``output_frequency`` / ``window_index`` /
    ``member_source`` / ``spinup_time``, three attrs are additive and optional:
    ``cadence_fallback`` (0/1) with ``cadence_fallback_reason``,
    ``output_frequency_spread`` (max-min of the members' achieved cadences, so a
    reader can see how well one shared ``fs`` fits them) and
    ``initial_state_source`` (``"prior"``/``"posterior"`` -- which window state
    file seeded the re-run; see :func:`_member_probes`).
    """
    values = np.asarray(values, dtype=np.float32)
    has_ensemble = values.ndim == 4
    dims = ("ensemble", "time", "sensor") if has_ensemble else ("time", "sensor")
    obs_x, obs_y, obs_z = points

    coords: dict[str, Any] = {
        "time": ("time", np.asarray(times, dtype=float)),
        "sensor": ("sensor", np.arange(obs_x.size)),
        "sensor_x": ("sensor", obs_x),
        "sensor_y": ("sensor", obs_y),
        "sensor_z": ("sensor", obs_z),
        "sensor_set": ("sensor", np.asarray(sensor_set_labels)),
    }
    if member_indices is not None:
        # The run's own member indices, so a member dropped by a failed re-run
        # stays identifiable (the ensemble axis is compacted, not padded).
        coords["member"] = ("ensemble", np.asarray(member_indices, dtype=int))

    attrs: dict[str, Any] = {
        "output_frequency": float(output_frequency),
        "window_index": int(window_index),
        "member_source": str(member_source),
        "spinup_time": float(spinup_time),
        # 0/1 rather than a bool: NetCDF attributes have no boolean type.
        "cadence_fallback": int(cadence_fallback_reason is not None),
    }
    if cadence_fallback_reason is not None:
        attrs["cadence_fallback_reason"] = str(cadence_fallback_reason)
    if output_frequency_spread is not None:
        attrs["output_frequency_spread"] = float(output_frequency_spread)
    if initial_state_source is not None:
        attrs["initial_state_source"] = str(initial_state_source)

    return xarray.Dataset(
        {
            name: (dims, values[..., index, :])
            for index, name in enumerate(("u", "v", "w"))
        },
        coords=coords,
        attrs=attrs,
    )


# ---------------------------------------------------------------------------
# One high-rate solver run, reduced to probes as its snapshots are read
# ---------------------------------------------------------------------------


def _snapshot_iteration(path: pathlib.Path) -> Optional[int]:
    """The absolute LBM timestep encoded in a snapshot filename, if any."""
    match = _SNAPSHOT_ITERATION.search(path.stem)
    return int(match.group(1)) if match else None


def _window_snapshots(
    files: Sequence[pathlib.Path],
    iout: int,
    n_window: int,
    *,
    expected_total: int | None = None,
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """Split the run's snapshots into ``(keep, drop)``: the trailing window, evenly spaced.

    ``(keep)`` is the last ``n_window`` snapshots that sit on the solver's regular
    ``iout`` grid, which is **not** simply the last ``n_window`` files.
    ``main.F90`` dumps on ``mod(it,iout)==0 .or. it==nt1``, and a warm start makes
    ``nt1 = nt0 + N*iout`` a non-multiple of ``iout`` whenever ``nt0 % iout != 0``
    -- the normal case here, since the probe cadence is deliberately not the
    cadence that produced ``nt0``. That extra final frame sits ``nt0 % iout``
    timesteps after its predecessor (0.013 s where the axis says 0.5 s), so
    keeping it would put a near-duplicate at the end of a series whose whole
    purpose is a spectrum.

    Raises ``RuntimeError`` when the surviving grid is short or has a hole in it:
    a solver that stopped early (a Fortran ``stop`` exits 0, so nothing else
    reports it) must not yield a series whose synthetic uniform time axis is a
    lie about what is in it.
    """
    on_grid: list[tuple[int, pathlib.Path]] = []
    off_grid: list[pathlib.Path] = []
    for path in files:
        iteration = _snapshot_iteration(path)
        if iteration is not None and iteration % iout == 0:
            on_grid.append((iteration, path))
        else:
            off_grid.append(path)
    on_grid.sort()

    if len(on_grid) < n_window:
        raise RuntimeError(
            f"The probe run left {len(on_grid)} snapshots on the {iout}-timestep "
            f"output grid, fewer than the {n_window} the window needs (the solver "
            "stopped early -- a Fortran `stop` exits 0, so nothing else reports "
            "it)."
        )
    # A lower bound is not enough. The run is `lead-in + window` frames long, so a
    # solver that died anywhere inside the LEAD-IN still leaves >= n_window on-grid
    # snapshots, and the trailing slice below would silently return a window
    # shifted back into the discarded lead-in -- evenly spaced, so the hole check
    # cannot catch it either. The exact count is known, so require it.
    if expected_total is not None and len(on_grid) != expected_total:
        raise RuntimeError(
            f"The probe run left {len(on_grid)} snapshots on the {iout}-timestep "
            f"output grid, not the {expected_total} the lead-in plus window should "
            "produce; the solver stopped early (a Fortran `stop` exits 0), and the "
            "trailing window would silently start inside the discarded lead-in."
        )

    kept = on_grid[len(on_grid) - n_window :]
    steps = np.diff([iteration for iteration, _ in kept])
    if steps.size and not np.all(steps == iout):
        raise RuntimeError(
            f"The kept probe snapshots are not evenly spaced: iteration steps "
            f"{sorted(set(int(s) for s in steps))} against an output interval of "
            f"{iout} timesteps. A gap means missing snapshots, and the series' "
            "uniform time axis would misrepresent them."
        )
    return [path for _, path in kept], off_grid + [
        path for _, path in on_grid[: len(on_grid) - n_window]
    ]


def _probe_window(
    forward_model: Any,
    points: tuple[np.ndarray, np.ndarray, np.ndarray],
    state: Optional[xarray.Dataset] = None,
    params: Optional[xarray.Dataset] = None,
) -> tuple[np.ndarray, float]:
    """Run one window at the model's (probe) cadence.

    Returns ``((time, component, sensor) probes, achieved cadence in seconds)``.
    The cadence is ``iout * dt``, which is what the solver actually wrote: ``dt =
    C_l/C_u`` is quantised by this member's ``velocity_magnitude``, so the
    requested ``output_frequency`` is almost never hit exactly.

    This mirrors ``pylbm.ForwardModel.run_single``'s launch sequence -- warm-start
    C_u first, then the restart, then the inflow settings, then a clean output dir
    (the C_u ordering is load-bearing, see docs/pylbm.md §9) -- and replaces only
    its collection step: instead of loading every snapshot into one Dataset (tens
    of GB at a 1 s cadence) each file is reduced to the probe points and unlinked.
    It also deliberately does NOT suppress the spin-up on a warm start the way
    ``run_single`` does: the lead-in here trims the restart's own transient.

    Runs as-is inside a worker process, so it must stay a module-level function.
    """
    # Local imports: this script is pylbm-only, and importing pylbm runs the LBM
    # submodule bootstrap -- a run dir that no-ops early should not trigger it.
    from pylbm.utils.state_utils import scale_velocity_to_physical
    from pylbm.utils.warm_start_utils import remove_old_restart_files

    obs_x, obs_y, obs_z = points
    fm = forward_model
    if not fm.enable_netcdf:
        # Same guard as ``run_single``: without NETCDF the solver writes
        # ``output/tec_*.plt`` instead, which carries no probe-readable fields and
        # which ``_clean_output`` (``out_*.nc``) would never delete -- a GB-scale
        # leak behind a confusing "no snapshots" error.
        raise RuntimeError(
            "Probe re-runs need NETCDF output, but this model was built with "
            "enable_netcdf=False (it would write output/tec_*.plt instead)."
        )

    try:
        if state is not None:
            # C_u must be fixed for THIS run before the restart is written, or the
            # carried field is scaled to lattice units with the wrong velocity
            # scale (docs/pylbm.md, "Warm-start C_u sequencing").
            fm._set_scaling_factors(params)
            fm._prepare_warmstart(state)
        fm._set_scaling_factors(params)
        if params is not None:
            fm._apply_inflow_settings(params)
        fm._clean_output()
        fm.run()

        # Keep the trailing window on the solver's regular output grid; everything
        # before it is the discarded lead-in, and the extra ``it==nt1`` dump is not
        # on that grid at all (see ``_window_snapshots``).
        iout = int(fm.output_frequency_timesteps)
        achieved_cadence = iout * float(fm.seconds_per_timestep)
        keep, drop = _window_snapshots(
            fm._get_output_files_for_current_run(),
            iout,
            int(round(fm.simulation_time / fm.output_frequency)),
            expected_total=int(round(fm.simulation_time / fm.output_frequency))
            + int(fm._spinup_outputs),
        )
        for path in drop:
            path.unlink(missing_ok=True)
        frames: list[np.ndarray] = []
        for path in keep:
            with xarray.open_dataset(path, engine="netcdf4") as snapshot:
                # The raw LBM snapshot carries no coordinates and lattice-unit
                # velocities; ``run_single`` attaches both after collection.
                frame = snapshot.assign(x=fm.x_grid, y=fm.y_grid, z=fm.z_grid)
                frame = scale_velocity_to_physical(frame, scale=fm.C_u)
                probes = _sensor_component_timeseries(
                    frame, obs_x, obs_y, obs_z, _SOLVER
                )
                frames.append(
                    np.asarray(
                        probes.transpose("component", "sensor").values,
                        dtype=np.float32,
                    )
                )
            path.unlink(missing_ok=True)
    finally:
        # The transient full-field cost must not survive the run, including when
        # the solver or the extraction died partway through it.
        fm._clean_output()
        # Leaves exactly one restart, at the kept iteration: a re-run's field
        # left behind in a shared scratch dir is what a later warm start would
        # silently pick up.
        remove_old_restart_files(fm.dirs)

    if not frames:
        raise RuntimeError(
            f"No snapshots collected from {fm.dirs.output_dir} for the probe window."
        )
    return np.stack(frames, axis=0), achieved_cadence


def _member_probe_window(
    forward_model: Any,
    points: tuple[np.ndarray, np.ndarray, np.ndarray],
    state_path: pathlib.Path,
    member: int,
    params: xarray.Dataset,
) -> tuple[np.ndarray, float]:
    """Worker entry point: read this member's initial frame, then probe the window.

    The frame is read inside the worker rather than handed to it, so the parent
    never holds (or pickles) one frame per member -- at Barcelona resolution that
    is tens of MB each.
    """
    return _probe_window(
        forward_model, points, _window_initial_state(state_path, member), params
    )


def _run_members(
    models: Sequence[Any],
    member_indices: Sequence[int],
    state_path: pathlib.Path,
    params: Sequence[xarray.Dataset],
    points: tuple[np.ndarray, np.ndarray, np.ndarray],
    num_parallel_processes: int,
    num_cpus_per_process: int,
) -> tuple[list[np.ndarray], list[float], list[int]]:
    """Per member, in member order: ``(series, achieved cadences, member indices)``.

    Parallel because the re-run costs one full window of forward solves: at
    Barcelona scale a serial sweep over the ensemble is the window's wall clock
    times the member count. The executor mirrors
    ``BaseEnsembleForwardModel._run_parallel`` (forkserver + CPU pinning, since
    the parent has imported JAX and the members are CPU-bound solver
    subprocesses); each worker returns only its probe array, not a field.

    **Any** failure costs its member, not the run. It is deliberately not just
    ``CalledProcessError``: the dominant warm-start failure mode here is a Fortran
    ``stop`` that exits 0 (docs/pylbm.md §9), which surfaces as a
    ``FileNotFoundError`` or as ``_window_snapshots``' short/uneven-grid
    ``RuntimeError`` -- and one such member must not throw away every other
    member's solve, nor the truth solve that preceded them. A systematic failure
    still fails loudly, because :func:`_member_probes` raises when nothing
    survived.
    """
    series: list[np.ndarray] = []
    cadences: list[float] = []
    kept: list[int] = []

    def _record(member: int, result: tuple[np.ndarray, float]) -> None:
        values, cadence = result
        series.append(values)
        cadences.append(cadence)
        kept.append(member)

    if num_parallel_processes <= 1:
        for member, fm, member_params in zip(member_indices, models, params):
            try:
                _record(
                    member,
                    _member_probe_window(fm, points, state_path, member, member_params),
                )
            except Exception:
                logger.warning(
                    "Probe re-run of member %d failed; dropping it",
                    member,
                    exc_info=True,
                )
        return series, cadences, kept

    ctx = mp.get_context("forkserver")
    executor_kwargs: dict[str, Any] = {
        "max_workers": num_parallel_processes,
        "mp_context": ctx,
    }
    if not cpu_pinning_disabled():
        executor_kwargs.update(
            initializer=pin_worker_initializer,
            initargs=(
                build_cpu_queue(
                    ctx=ctx,
                    num_workers=num_parallel_processes,
                    cpus_per_worker=max(1, num_cpus_per_process),
                ),
            ),
        )
    with ProcessPoolExecutor(**executor_kwargs) as executor:
        futures = [
            executor.submit(
                _member_probe_window, fm, points, state_path, member, member_params
            )
            for member, fm, member_params in zip(member_indices, models, params)
        ]
        for member, future in zip(member_indices, futures):
            try:
                _record(member, future.result())
            except Exception:
                logger.warning(
                    "Probe re-run of member %d failed; dropping it",
                    member,
                    exc_info=True,
                )
    return series, cadences, kept


def _stack_members(
    series: Sequence[np.ndarray],
    cadences: Sequence[float],
    members: Sequence[int],
) -> tuple[np.ndarray, list[float], list[int]]:
    """Stack the members that agree on frame count; drop (loudly) the ones that don't.

    A member whose solver stopped early returns a short series -- today's real run
    had one member exit 0 with 3 of 48 frames -- and stacking those together is
    either a bare ``ValueError`` or, worse, a ragged comparison. The odd ones out
    are dropped rather than padded: the ``member`` coordinate already records which
    of the run's members survived, and a NaN-padded member would poison the Welch
    average it feeds.
    """
    lengths = [int(values.shape[0]) for values in series]
    # The LONGEST length, not the most common one: `Counter.most_common` breaks
    # ties by first-encountered, so two short members ahead of two full ones would
    # drop the full ones. `_window_snapshots` returns exactly the window's frame
    # count or raises, so a short series means that member's solver died -- and the
    # full-length group is the correct one however many of each there are.
    modal = max(lengths)
    dropped = [member for member, n in zip(members, lengths) if n != modal]
    if dropped:
        logger.warning(
            "Dropping probe members %s: %s frames against the modal %d (their "
            "solver stopped early)",
            dropped,
            [n for n in lengths if n != modal],
            modal,
        )
    keep = [index for index, n in enumerate(lengths) if n == modal]
    return (
        np.stack([series[index] for index in keep], axis=0),
        [cadences[index] for index in keep],
        [members[index] for index in keep],
    )


# ---------------------------------------------------------------------------
# Run-directory access (window artifacts + the lazy truth view)
# ---------------------------------------------------------------------------


def _selected(cfg: DictConfig, key: str) -> Any:
    """A config value as a plain Python object (lists included), for comparison."""
    node = OmegaConf.select(cfg, key)
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)
    return node


def _check_config_matches_run(cfg: DictConfig, run_dir: pathlib.Path) -> None:
    """Refuse to re-probe a run this invocation was not composed for.

    The probe run needs its own forward models, so it composes the config afresh
    rather than reading the run's; that only reproduces the run when the caller
    passes the same overrides. Compare the keys a mismatch would silently
    corrupt (the grid the restart is written onto, the window length, the
    solvers) against the config ``run_esmda.py`` saved.
    """
    if not (run_dir / "config.yaml").exists():
        logger.warning(
            "No config.yaml in %s -- cannot verify this invocation matches the run.",
            run_dir,
        )
        return

    saved = load_run_config(run_dir)
    mismatches = []
    for key in _GUARDED_KEYS:
        here = _selected(cfg, key)
        there = _selected(saved, key)
        if here != there:
            mismatches.append(f"{key}: {there!r} in the run, {here!r} here")
    if mismatches:
        raise ValueError(
            "This invocation does not match the ESMDA run in "
            f"{run_dir}:\n  " + "\n  ".join(mismatches) + "\nCompose the probe run "
            "with the same overrides the ESMDA run used."
        )


def _state_bearing_smoother(cfg: DictConfig, run_dir: pathlib.Path) -> bool:
    """Did the run's smoother update the STATE as well as the parameters?

    Read from the run's own ``run_info.yaml``
    (``configuration.joint_state_and_parameter``, written from
    ``isinstance(esmda, StateAndParameterESMDA)``) rather than re-deriving it, with
    the composed smoother target as the fallback for a run dir that predates that
    key. It decides whether a prior probe re-run may be seeded from the posterior
    state (see :func:`_member_probes`).
    """
    recorded = read_yaml(run_dir / "run_info.yaml").get("configuration", {})
    if "joint_state_and_parameter" in recorded:
        return bool(recorded["joint_state_and_parameter"])
    target = str(_selected(cfg, "esmda.smoother._target_") or "")
    return target.rsplit(".", 1)[-1].startswith("StateAnd")


def _truth_window(truth_access: dict, window: int) -> xarray.Dataset:
    """The chosen window's frames of the truth, opened lazily (never loaded).

    Every optional slicing key is read with ``.get``: an older ``truth_access.yaml``
    that predates one of them degrades to "no offset" rather than raising
    (invariant 3). The keys the script genuinely cannot work without are checked
    once, up front, in :func:`run`.
    """
    n_per_window = int(truth_access["n_per_window"])
    truth = open_truth(
        truth_access["true_state_path"],
        truth_access.get("n_total"),
        truth_access.get("x_offset", 0.0),
        truth_access.get("start_idx", 0),
        truth_access.get("t_offset", 0.0),
    )
    return truth.isel(time=slice(window * n_per_window, (window + 1) * n_per_window))


def _window_initial_state(
    state_path: pathlib.Path, member: Optional[int] = None
) -> xarray.Dataset:
    """The first frame of a window state file (its initial field), one member only.

    A full-ensemble window state file runs to gigabytes, so it is opened lazily
    and only this one frame of one member is materialised. The first *stored*
    frame is one assimilation output interval into the window rather than exactly
    at its start; that offset is immaterial for a re-run whose lead-in is
    discarded anyway (see the module docstring).
    """
    with xarray.open_dataset(state_path) as ds:
        frame = ds.isel(time=0)
        if member is not None and "ensemble" in frame.dims:
            frame = frame.isel(ensemble=member)
        return frame.load()


def _window_params(params_path: pathlib.Path, member: int) -> xarray.Dataset:
    """One member's parameters for the window (already window-local in time)."""
    with xarray.open_dataset(params_path) as ds:
        return ds.isel(ensemble=member).load()


def _truth_window_params(
    run_dir: pathlib.Path, window: int, sim_time: float
) -> xarray.Dataset:
    """The truth parameters over the window, rebased to window-local time.

    ``true_params.nc`` covers the whole assimilation horizon on a global clock
    (and carries a length-1 ensemble axis); the window slice is inclusive of both
    boundary knots so the re-run's inflow schedule spans the window.
    """
    with xarray.open_dataset(run_dir / "true_params.nc") as ds:
        params = ds.load()
    if "ensemble" in params.dims:
        params = params.isel(ensemble=0)
    if "time" in params.dims:
        start = window * sim_time
        params = params.sel(time=slice(start, start + sim_time))
        params = params.assign_coords(time=params["time"] - start)
    return params


# ---------------------------------------------------------------------------
# Truth probes: high-rate re-run, or the truth's stored cadence
# ---------------------------------------------------------------------------


def _rerun_blocker(
    cfg: DictConfig,
    truth_access: dict,
    run_dir: pathlib.Path,
    truth_start: xarray.Dataset,
) -> Optional[str]:
    """Why the truth cannot be re-run at the probe cadence, or ``None``.

    A truth that was pre-simulated by another solver, on another grid, or without
    the density field the LBM restart is built from, is not re-runnable; the
    caller then falls back to its stored cadence and records the reason.
    """
    solver = str(truth_access.get("truth_solver_name", ""))
    if solver != _SOLVER:
        return f"the truth was produced by {solver!r}, not {_SOLVER!r}"
    if not (run_dir / "true_params.nc").exists():
        return "the run dir has no true_params.nc to re-run the truth with"
    if "rho" not in truth_start.data_vars:
        return "the stored truth has no 'rho' to build an LBM restart from"
    grid = tuple(int(cfg.domain[axis]) for axis in ("nx", "ny", "nz"))
    truth_grid = tuple(int(truth_start.sizes[axis]) for axis in ("x", "y", "z"))
    if truth_grid != grid:
        return (
            f"the stored truth is on grid {truth_grid}, not the assimilation "
            f"grid {grid}"
        )
    return None


def _stored_truth_probes(
    truth_access: dict,
    window: int,
    points: tuple[np.ndarray, np.ndarray, np.ndarray],
    sensor_set_labels: np.ndarray,
    reason: str,
) -> xarray.Dataset:
    """Truth probes at the truth's OWN stored cadence (the no-re-run fallback).

    The frames are reduced one at a time -- a window of a multi-GB truth is never
    held in memory -- and the achieved cadence is recorded in the attrs, because
    the downstream comparison has to truncate truth and members to the band both
    resolve.

    Unlike the re-run path, this keeps the truth's **own** timestamps (rebased so
    the window starts at 0) rather than synthesising a uniform axis: they are what
    the truth actually has, and a non-uniform stored cadence is something the
    consumer should see (``evaluation.turbulence`` warns on it) rather than
    something to smooth over here.
    """
    obs_x, obs_y, obs_z = points
    solver = str(truth_access["truth_solver_name"])
    with _truth_window(truth_access, window) as window_truth:
        times = np.asarray(window_truth["time"].values, dtype=float)
        frames = [
            np.asarray(
                _sensor_component_timeseries(
                    window_truth.isel(time=index), obs_x, obs_y, obs_z, solver
                )
                .transpose("component", "sensor")
                .values,
                dtype=np.float32,
            )
            for index in range(window_truth.sizes["time"])
        ]

    cadence = float(np.median(np.diff(times))) if times.size > 1 else float("nan")
    logger.warning(
        "Truth probes fall back to the stored cadence (%.4g s): %s", cadence, reason
    )
    return _probe_dataset(
        np.stack(frames, axis=0),
        points,
        sensor_set_labels,
        times - times[0] if times.size else times,
        window_index=window,
        member_source="truth",
        output_frequency=cadence,
        spinup_time=0.0,
        cadence_fallback_reason=reason,
    )


def _truth_probes(
    cfg: DictConfig,
    truth_access: dict,
    run_dir: pathlib.Path,
    window: int,
    points: tuple[np.ndarray, np.ndarray, np.ndarray],
    sensor_set_labels: np.ndarray,
    *,
    sim_time: float,
    output_frequency: float,
    spinup_time: float,
    truth_source: str,
) -> xarray.Dataset:
    """Truth probe series for the window: a high-rate re-run when possible."""
    with _truth_window(truth_access, window) as window_truth:
        truth_start = window_truth.isel(time=0).load()

    reason: Optional[str] = "probes.truth_source=stored"
    if truth_source != "stored":
        reason = _rerun_blocker(cfg, truth_access, run_dir, truth_start)
    if reason is not None:
        return _stored_truth_probes(
            truth_access, window, points, sensor_set_labels, reason
        )

    # Local import, and deliberately below the fallback return: this script is
    # pylbm-only, and a truth that cannot be re-run must not trigger the LBM
    # submodule bootstrap.
    from pylbm.utils.warm_start_utils import clean_all_restart_files

    truth_model = instantiate(
        cfg.truth_model.forward_model,
        results_dir=None,
        # Its own experiment dir, never the assimilation run's ``runcase``.
        experiment_name=_PROBE_EXPERIMENT,
        simulation_time=sim_time,
        output_frequency=output_frequency,
        spinup_time=spinup_time,
    )
    instantiate(cfg.truth_model.prepare, forward_model=truth_model)
    # The same clean slate the members get in :func:`_member_models` (usually the
    # very same directory, but not if the two models' temp dirs are overridden
    # apart). A previous invocation's truth restart is not this run's history
    # either, and leaving it would give the truth a non-equilibrium template the
    # members deliberately do not have -- an asymmetry in the one comparison this
    # script exists to make. Its own warm start, written from the stored truth
    # frame, follows in ``_probe_window``.
    clean_all_restart_files(truth_model.dirs)
    values, achieved = _probe_window(
        truth_model,
        points,
        truth_start,
        _truth_window_params(run_dir, window, sim_time),
    )
    if abs(achieved - output_frequency) > _CADENCE_SPREAD_TOLERANCE * achieved:
        logger.info(
            "The truth re-run's achieved cadence is %.5g s, not the requested "
            "%.5g s (dt = C_l/C_u is quantised by the truth's velocity_magnitude)",
            achieved,
            output_frequency,
        )
    return _probe_dataset(
        values,
        points,
        sensor_set_labels,
        np.arange(values.shape[0], dtype=float) * achieved,
        window_index=window,
        member_source="truth",
        output_frequency=achieved,
        spinup_time=spinup_time,
    )


# ---------------------------------------------------------------------------
# Member probes
# ---------------------------------------------------------------------------


def _member_count(
    windows_dir: pathlib.Path, window: int, max_members: Optional[int]
) -> Optional[int]:
    """How many members to re-probe, from the window's posterior params file.

    Taken from the artifact rather than from ``ensemble.ensemble_size`` so a
    differently-composed invocation cannot silently probe a different number of
    members than the run has. ``None`` when the window has no posterior params,
    i.e. there is nothing to re-probe.
    """
    params_path = windows_dir / f"window_{window}_posterior_params.nc"
    if not params_path.exists():
        logger.warning(
            "No %s -- this run has no window %d to re-probe.", params_path, window
        )
        return None
    with xarray.open_dataset(params_path) as ds:
        n_members = int(ds.sizes["ensemble"])
    if max_members is not None:
        n_members = min(n_members, int(max_members))
    return n_members


def _member_models(
    cfg: DictConfig,
    n_members: int,
    *,
    sim_time: float,
    output_frequency: float,
    spinup_time: float,
) -> tuple[list[Any], pathlib.Path]:
    """``(per-member forward models, their clone root)``, from one compiled template.

    The clones are made exactly as the ensemble forward model makes them, but
    under their own ``probe_experiments/`` directory so a probe re-run never
    writes into an assimilation run's member dirs -- the template itself already
    runs under ``_PROBE_EXPERIMENT`` rather than ``runcase``. Built once and reused
    for both member sources: cloning is cheap, but ``prepare`` rebuilds the solver.
    The clone root is returned so the caller can remove it once the probes are
    written (each clone otherwise keeps a restart file forever).

    **A clone starts with an empty ``restart/``, always.** ``_prepare_warmstart``
    reads the newest restart it finds as the ghost-cell / non-equilibrium template
    and takes ``nt0`` from its iteration, so any restart present at clone time is
    a field from some *other* run that this member's re-run would then be seeded
    from and continue the clock of. There are two such fields and neither is the
    member's: the truth's final restart, which the previous invocation left in the
    shared ``_PROBE_EXPERIMENT`` dir this clone is copied from (see the note in
    :func:`run`), and -- because ``clone_root`` is only removed on success and the
    copy is ``dirs_exist_ok=True`` -- an aborted invocation's own leftover in the
    clone dir. The member's legitimate warm start is unaffected: it is the restart
    ``_probe_window`` writes from the window state file it was handed, one line
    later, and it is written into this now-empty dir.
    """
    # Local import: this script is pylbm-only, and importing pylbm runs the LBM
    # submodule bootstrap -- a run dir that no-ops early should not trigger it.
    from pylbm.utils.forward_model_utils import create_new_forward_model
    from pylbm.utils.warm_start_utils import clean_all_restart_files

    template = instantiate(
        cfg.assim_model.forward_model,
        results_dir=None,
        experiment_name=_PROBE_EXPERIMENT,
        simulation_time=sim_time,
        output_frequency=output_frequency,
        spinup_time=spinup_time,
    )
    instantiate(cfg.assim_model.prepare, forward_model=template)
    # Cleared at the source as well as in every clone: the source is the shared
    # `_PROBE_EXPERIMENT` dir the TRUTH re-run also uses, so this is what makes a
    # repeat invocation reproduce the first one on both sides of the comparison
    # rather than handing the truth a template the members do not have.
    clean_all_restart_files(template.dirs)
    clone_root = template.dirs.temp_dir / "probe_experiments"
    models: list[Any] = []
    for member in range(n_members):
        model = create_new_forward_model(template, clone_root, f"{member:03d}")
        # Cleared per clone too, not only at the source: the clone dir can itself
        # be a leftover, and this is where the invariant has to hold.
        clean_all_restart_files(model.dirs)
        models.append(model)
    return models, clone_root


def _member_probes(
    models: Sequence[Any],
    windows_dir: pathlib.Path,
    window: int,
    source: str,
    points: tuple[np.ndarray, np.ndarray, np.ndarray],
    sensor_set_labels: np.ndarray,
    *,
    output_frequency: float,
    spinup_time: float,
    num_parallel_processes: int,
    num_cpus_per_process: int,
    state_bearing_smoother: bool,
) -> Optional[xarray.Dataset]:
    """Re-probe the window's prior or posterior members; ``None`` if unavailable.

    **Which field the members start from is recorded, not assumed.** ESMDA
    re-forecasts one window from one warm-start state with successively updated
    parameters, so the prior and posterior forecasts of a window share their
    initial field and the prior re-run may legitimately be seeded from
    ``window_{w}_posterior_state.nc``. That is the *normal* path, because
    ``run.save_prior_state`` defaults false -- but only for a parameter-only
    smoother. For a state-bearing one (``state_and_parameter`` /
    ``state_and_dynamic``) the initial field is exactly what the update changed,
    so seeding the prior re-run from the posterior state would make S4's prior
    envelope partly circular; that combination is refused instead, and either way
    the choice lands in the ``initial_state_source`` attribute.
    """
    params_path = windows_dir / f"window_{window}_{source}_params.nc"
    if not params_path.exists():
        logger.warning("No %s -- skipping the %s probes.", params_path.name, source)
        return None
    state_path = windows_dir / f"window_{window}_{source}_state.nc"
    state_source = source
    if not state_path.exists():
        if source == "prior" and state_bearing_smoother:
            logger.error(
                "No %s: this run used a state-bearing smoother, whose update "
                "changes the very field the prior forecast starts from, so seeding "
                "the prior probes from the POSTERIOR state would make S4's prior "
                "envelope partly circular. Re-run the assimilation with "
                "run.save_prior_state=true, or drop probes.include_prior.",
                state_path.name,
            )
            return None
        state_path = windows_dir / f"window_{window}_posterior_state.nc"
        state_source = "posterior"
    if not state_path.exists():
        logger.warning(
            "No window_%d state file -- skipping the %s probes.", window, source
        )
        return None

    with xarray.open_dataset(params_path) as ds:
        members = list(range(min(len(models), int(ds.sizes["ensemble"]))))

    logger.info(
        "Re-probing %d %s members of window %d at ~%.4g s cadence, from %s",
        len(members),
        source,
        window,
        output_frequency,
        state_path.name,
    )
    series, cadences, kept = _run_members(
        models[: len(members)],
        members,
        state_path,
        [_window_params(params_path, member) for member in members],
        points,
        num_parallel_processes,
        num_cpus_per_process,
    )
    if not series:
        # Nothing survived: that is a systematic failure (a bad binary, a wrong
        # grid), not an unlucky member, so it must not exit 0 with no file.
        raise RuntimeError(
            f"Every {source} member's probe re-run failed; see the warnings above "
            "for each member's exception."
        )

    values, cadences, kept = _stack_members(series, cadences, kept)
    # One shared (fs, nperseg) downstream, so the ensemble gets ONE cadence: the
    # median of what the members achieved. They differ because dt = C_l/C_u is
    # quantised by each member's own velocity_magnitude.
    cadence = float(np.median(cadences))
    spread = float(np.max(cadences) - np.min(cadences))
    if spread > _CADENCE_SPREAD_TOLERANCE * cadence:
        logger.warning(
            "The %s members' achieved cadences span %.5g s about %.5g s (%.1f %%); "
            "one shared sampling frequency fits them only approximately, so the "
            "high-frequency end of the spectrum is smeared by that much",
            source,
            spread,
            cadence,
            100.0 * spread / cadence,
        )
    return _probe_dataset(
        values,
        points,
        sensor_set_labels,
        np.arange(values.shape[1], dtype=float) * cadence,
        window_index=window,
        member_source=source,
        output_frequency=cadence,
        spinup_time=spinup_time,
        member_indices=kept,
        output_frequency_spread=spread,
        initial_state_source=state_source,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    probes_cfg = cfg.get("probes", {}) or {}
    # The probe files live beside the artifacts they are compared against, i.e.
    # in the ESMDA run dir -- resolved exactly as run_esmda.py resolves its own
    # output dir (``resolve_output_dir`` would instead point at this job's
    # timestamped Hydra dir, where the metric/figure stages would never find
    # them).
    run_dir = pathlib.Path(probes_cfg.get("run_dir") or cfg.paths.results_dir)
    truth_access = read_yaml(run_dir / "truth_access.yaml")
    missing = [key for key in _REQUIRED_TRUTH_ACCESS if truth_access.get(key) is None]
    if not truth_access or missing:
        logger.warning(
            "%s in %s -- nothing to probe (expected a finished run_esmda.py output "
            "dir).",
            (
                "No truth_access.yaml"
                if not truth_access
                else f"truth_access.yaml is missing {', '.join(missing)}"
            ),
            run_dir,
        )
        return

    solver = str(cfg.assim_model.solver_name)
    if solver != _SOLVER:
        raise ValueError(
            f"Probe re-runs are implemented for the {_SOLVER!r} assimilation model "
            f"only (this run's is {solver!r})."
        )
    _check_config_matches_run(cfg, run_dir)

    num_windows = int(truth_access["num_windows"])
    sim_time = float(truth_access["sim_time"])
    requested = int(probes_cfg.get("window_index", -1))
    if not -num_windows <= requested < num_windows:
        raise ValueError(
            f"probes.window_index={requested} is outside the run's {num_windows} "
            "assimilation windows."
        )
    window = requested % num_windows

    windows_dir = run_dir / "windows"
    output_frequency = float(probes_cfg.get("output_frequency", 1.0))
    spinup_time = float(probes_cfg.get("spinup_time", cfg.time.spinup_time))
    # Checked before any solver runs: a run dir without this window's posterior
    # params has nothing to re-probe, and the truth re-run below is expensive.
    n_members = _member_count(windows_dir, window, probes_cfg.get("max_members", None))
    if n_members is None:
        return

    # Also before any solver runs: the whole point of the re-run is a spectrum,
    # and how WIDE a band gets scored is fixed by the SAMPLE COUNT alone (the
    # cadence cancels out of the in-band bin count -- see
    # evaluation.turbulence.spectral_band_bins). Two separate things can go wrong
    # and both are silent until after the solve, so both are reported here.
    #
    # The hard floor first: a 120 s window at a 1 s cadence yields 120 samples
    # against a floor of 264, so the records would be written, the snapshots
    # deleted, and `spectral_metrics` would then no-op after ~20 minutes of
    # solving. But clearing that floor is NOT the bar -- at it the band is four
    # bins wide, an RMS over less than a decade of frequency, and figure S4 draws
    # a `-5/3` guide over it regardless. So the band width is reported on every
    # run and warned about below a decade (bins 1..B span exactly a factor of B,
    # hence B >= 10); a caller who wants a slope out of S4 needs that much.
    #
    # Warn rather than refuse in both cases: the series is still valid data for
    # the window statistics, and the stored-cadence fallback cannot choose its
    # own sampling.
    # Rounded the same way `_probe_window` sizes the window, so the two cannot
    # disagree by a frame on a cadence that does not divide the window.
    n_samples = int(round(sim_time / output_frequency))
    floor = minimum_spectral_samples()
    decade = minimum_spectral_samples(SPECTRAL_BAND_DECADE_BINS)
    scored_bins = spectral_band_bins(n_samples)
    if n_samples < floor:
        logger.warning(
            "This re-run yields %d samples per sensor (%.4g s window / %.4g s "
            "cadence), under the %d a spectral comparison needs at all, so "
            "spectral_metrics and S4 will no-op. Use "
            "probes.output_frequency<=%.4g for a decade-wide band at this "
            "window length.",
            n_samples,
            sim_time,
            output_frequency,
            floor,
            sim_time / decade,
        )
    elif scored_bins < SPECTRAL_BAND_DECADE_BINS:
        logger.warning(
            "This re-run yields %d samples per sensor (%.4g s window / %.4g s "
            "cadence), which scores only %d frequency bins -- a factor-of-%d "
            "band, under the decade a -5/3 reading off figure S4 needs. The "
            "numbers will be produced but are an RMS over a very narrow band. "
            "Use probes.output_frequency<=%.4g (>=%d samples) for %d bins.",
            n_samples,
            sim_time,
            output_frequency,
            scored_bins,
            scored_bins,
            sim_time / decade,
            decade,
            SPECTRAL_BAND_DECADE_BINS,
        )
    else:
        logger.info(
            "This re-run yields %d samples per sensor and will score %d "
            "frequency bins under the cutoff.",
            n_samples,
            scored_bins,
        )

    points, sensor_set_labels = _probe_points(build_sensor_sets(cfg))

    # The member clones are made FIRST, and the ordering matters -- but it is not
    # what makes this safe, and on its own it was not enough. Both models are
    # constructed under `_PROBE_EXPERIMENT`, so they share one experiment dir;
    # `_probe_window`'s `finally` prunes restarts but deliberately keeps the
    # latest, so a truth solve leaves ITS final field in `probe_runcase/restart/`.
    # `create_new_forward_model` copies the whole experiment dir into every clone,
    # and `_prepare_warmstart` then uses whatever restart it finds as the template
    # for ghost cells and non-equilibrium content -- which would seed every member
    # with the truth's stress field, i.e. inject truth information into the member
    # side of a truth-vs-member diagnostic and bias the LSD optimistically.
    # Cloning first only keeps the FIRST invocation clean: nothing clears
    # `probe_runcase/restart/` between runs, so a re-probe of the same window at a
    # different cadence -- the normal iteration -- clones the previous
    # invocation's truth restart. `_member_models` therefore clears the restart
    # dir at the clone source and in every clone, and `_truth_probes` does the
    # same on its side; the ordering is kept because it costs nothing and the
    # in-invocation leak is one fewer thing to depend on the clearing for.
    models, clone_root = _member_models(
        cfg,
        n_members,
        sim_time=sim_time,
        output_frequency=output_frequency,
        spinup_time=spinup_time,
    )

    # A re-probe of a window that already has member records must not leave the
    # OLD members beside the NEW truth: the window_index guard downstream cannot
    # tell them apart (same window), so a failed member phase would otherwise
    # leave a 0.25 s truth to be compared against 1 s members from the previous
    # invocation.
    for suffix in ("", "_prior"):
        (windows_dir / f"window_{window}_probes{suffix}.nc").unlink(missing_ok=True)

    truth = _truth_probes(
        cfg,
        truth_access,
        run_dir,
        window,
        points,
        sensor_set_labels,
        sim_time=sim_time,
        output_frequency=output_frequency,
        spinup_time=spinup_time,
        truth_source=str(probes_cfg.get("truth_source", "auto")),
    )
    truth.to_netcdf(run_dir / "truth_probes.nc")

    sources = ["posterior"]
    if bool(probes_cfg.get("include_prior", False)):
        sources.append("prior")
    for source in sources:
        members = _member_probes(
            models,
            windows_dir,
            window,
            source,
            points,
            sensor_set_labels,
            output_frequency=output_frequency,
            spinup_time=spinup_time,
            num_parallel_processes=int(cfg.ensemble.num_parallel_processes),
            num_cpus_per_process=int(cfg.ensemble.num_cpus_per_process),
            state_bearing_smoother=_state_bearing_smoother(cfg, run_dir),
        )
        if members is None:
            continue
        suffix = "" if source == "posterior" else f"_{source}"
        members.to_netcdf(windows_dir / f"window_{window}_probes{suffix}.nc")

    # Only on success, like the job runners' RUN_TEMP_DIR trap: each clone holds a
    # restart file (and, on a failure, the evidence), so a finished run cleans them
    # up and a broken one leaves them for the post-mortem.
    shutil.rmtree(clone_root, ignore_errors=True)
    print(f"Saved probe series for window {window} in {run_dir}")


@hydra.main(  # type: ignore[misc]
    version_base=None, config_path="../../conf", config_name="run_probe_series"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
