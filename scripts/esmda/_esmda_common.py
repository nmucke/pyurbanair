"""Shared post-processing helpers for the ESMDA three-script pipeline.

``scripts/esmda/run_esmda.py`` runs the assimilation and saves the raw artifacts;
``scripts/esmda/compute_esmda_metrics.py`` turns those into ``run_summary.yaml`` and
``scripts/esmda/make_esmda_figures.py`` draws the figures. The metric and figure
stages both need the same lazy truth access, sensor-series extraction and
small scalar/YAML helpers, so they live here instead of being duplicated.

Everything is read-only with respect to the run directory except the explicit
``*.to_netcdf`` / ``_write_yaml`` writes the callers perform; the truth state is
always opened lazily (see :func:`open_truth`) so a multi-GB truth is never held
in memory in full.
"""

# mypy: ignore-errors
# Legacy untyped helper module: it predates the strict mypy config and is
# type-checked transitively whenever a script importing it is committed.
# Waived wholesale rather than annotated piecemeal; drop this when typed.

from __future__ import annotations

import logging
import pathlib
import re

import numpy as np
import xarray
import yaml
from data_assimilation.interpolation import interpolate_dataarray_at_points
from data_assimilation.observation_operator import ObservationOperator
from evaluation.turbulence import probe_spectra
from omegaconf import OmegaConf

from pyurbanair.config.hydra_helpers import (
    create_observation_points,
    create_validation_points,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Run-directory loaders (config + persisted truth-access view + sensor sets)
# ---------------------------------------------------------------------------


def load_run_config(run_dir: pathlib.Path):
    """Re-load the composed Hydra config saved by run_esmda.py."""
    return OmegaConf.load(run_dir / "config.yaml")


def read_yaml(path: pathlib.Path) -> dict:
    """Load a small YAML mapping (``{}`` if the file is missing/empty)."""
    path = pathlib.Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_sensor_sets(cfg) -> dict:
    """``{"assimilation": (x, y, z)[, "validation": (x, y, z)]}`` from the obs config.

    The validation set is the held-out sensors (scored but never assimilated);
    it is only present when the obs config defines it.
    """
    obs_x, obs_y, obs_z = create_observation_points(cfg.obs)
    sensor_sets = {"assimilation": (obs_x, obs_y, obs_z)}
    validation_points = create_validation_points(cfg.obs)
    if validation_points is not None:
        sensor_sets["validation"] = validation_points
    return sensor_sets


# ---------------------------------------------------------------------------
# Lazy truth-state access (only load the slices that are actually needed)
# ---------------------------------------------------------------------------

_X_COORDS = ("x", "xt", "xm")


def truth_x_min(ds):
    """Smallest x face/centre coordinate of a (truth) dataset, or 0.0 if none.

    Prefers the staggered face coordinate (``xm``) so the domain *edges* align
    when computing the offset; falls back to ``x``/``xt`` for non-udales grids.
    """
    for c in ("xm", "x", "xt"):
        if c in ds.coords:
            return float(np.asarray(ds[c].values).min())
    return 0.0


def open_truth(true_state_path, n_total, x_offset=0.0, start_idx=0, t_offset=0.0):
    """Lazily open the truth state, limited to ``n_total`` frames from ``start_idx``.

    Uses ``open_dataset`` (not ``load_dataset``) so the data stays on disk; the
    caller's subsequent ``.isel``/reduction then materialises only the slice it
    needs. A multi-GB truth is therefore never pulled into memory in full -- the
    window loop reads one window at a time, the plots one z-plane at a time.

    ``start_idx`` drops the leading frames before the chosen start time, so the
    assimilation horizon begins partway into a pre-simulated truth (e.g. after a
    spin-up). ``t_offset`` then rebases the kept frames' time coordinate so the
    chosen start time becomes t=0, keeping the window loop and plots on a
    [0, final_time) axis regardless of where the truth was sliced.

    ``x_offset`` shifts every x coordinate so a truth saved in its own frame
    (e.g. x in [0, 100]) lines up with the simulation domain (e.g. x in
    [-20, 80]). Applied on every open so the observation operator, the window
    loop and the plots all see the truth in domain coordinates.
    """
    ds = xarray.open_dataset(true_state_path)
    if n_total is not None:
        ds = ds.isel(time=slice(start_idx, start_idx + n_total))
    elif start_idx:
        ds = ds.isel(time=slice(start_idx, None))
    if t_offset and "time" in ds.coords:
        ds = ds.assign_coords(time=ds["time"] - t_offset)
    if x_offset:
        shifted = {c: ds[c] + x_offset for c in _X_COORDS if c in ds.coords}
        if shifted:
            ds = ds.assign_coords(shifted)
    return ds


# ---------------------------------------------------------------------------
# Sensor time-series extraction (truth vs ensemble at fixed points)
# ---------------------------------------------------------------------------


def _sensor_component_timeseries(state, obs_x, obs_y, obs_z, solver_name):
    """Per-component ``(u, v, w)`` velocity time series at each sensor point.

    Trilinearly interpolates u/v/w (each on its own staggered grid, resolved via
    an ``ObservationOperator``'s solver-specific dim mapping) at the sensor
    locations, keeping any leading dims (``ensemble``, ``time``). Returns a
    DataArray with a leading ``component`` dim: ``(component, ..., time, sensor)``.
    The velocity magnitude |U| is :func:`evaluation.sensors.sensor_magnitude` of
    this (used for the sensor figures); the full vector is used for the sensor
    error metrics.
    """
    op = ObservationOperator(
        obs_x=list(np.asarray(obs_x, dtype=float)),
        obs_y=list(np.asarray(obs_y, dtype=float)),
        obs_z=list(np.asarray(obs_z, dtype=float)),
        obs_states=["u", "v", "w"],
        solver_name=solver_name,
    )
    comps = []
    for var in ("u", "v", "w"):
        dims = op.dim_mapping[var]
        comps.append(
            interpolate_dataarray_at_points(
                state[var],
                x_dim=dims["x"],
                y_dim=dims["y"],
                z_dim=dims["z"],
                obs_x=op.obs_x,
                obs_y=op.obs_y,
                obs_z=op.obs_z,
            )
        )
    return xarray.concat(comps, dim="component").assign_coords(
        component=["u", "v", "w"]
    )


def _concat_sensor_pieces(pieces):
    """Concatenate per-window sensor series along ``time`` for each sensor set."""
    return {
        name: (
            parts[0]
            if len(parts) == 1
            else xarray.concat(parts, dim="time", join="override")
        )
        for name, parts in pieces.items()
    }


def ensemble_sensor_series(
    state_paths, sensor_sets, solver_name, sim_time, on_member=None
):
    """Ensemble per-component ``(u, v, w)`` sensor series across rollout windows.

    Interpolates u/v/w at every sensor set's points (keeping ``component`` +
    ``ensemble`` + ``time``), rebasing each window's local time onto a single
    global axis (window ``w`` starts at ``w*sim_time``) so it lines up with the
    truth. Returns ``{name: DataArray(component, ensemble, time, sensor)}``.

    **One member at a time.** A window state file is
    ``(ensemble, time, z, y, x)`` per component and runs to gigabytes, so it is
    opened lazily and sliced per member: peak memory is three components of one
    member's window, not the whole ensemble. (This function used to ``.load()``
    each file whole -- the last such site in the post-processing stack.) The
    per-member slice *is* materialised, because the interpolation reads
    ``.values``; every sensor set is interpolated from that one slice so a
    second set costs no extra read.

    ``on_member`` is called as ``on_member(window_index, member_index, member)``
    with that same materialised slice, before it is dropped. It exists so the
    WP1.4 mean fields can be accumulated inside this pass rather than in a
    second one: at Barcelona scale the window files total tens of GB and a
    second full read of them is not affordable, while a callback on a slice
    already in memory costs no I/O at all. It is only called when the state
    carries an ``ensemble`` axis -- everything hanging off it is per member.
    """
    pieces = {name: [] for name in sensor_sets}
    for w, path in enumerate(state_paths):
        with xarray.open_dataset(path) as ds:
            t = (
                np.asarray(ds["time"].values, dtype=float)
                if "time" in ds.coords
                else None
            )
            for name, vel in _window_sensor_series(
                ds, sensor_sets, solver_name, on_member, w
            ).items():
                if t is not None and "time" in vel.dims:
                    vel = vel.assign_coords(time=(t - t[0]) + w * sim_time)
                pieces[name].append(vel)
    return _concat_sensor_pieces(pieces)


def _window_sensor_series(ds, sensor_sets, solver_name, on_member, window_index):
    """One window's ``{name: DataArray(component, ensemble, time, sensor)}``.

    Members are read and interpolated one at a time from the lazily-opened
    ``ds``; every sensor set is interpolated from the member slice already in
    memory, so a second set costs no extra read, and ``on_member`` (see
    :func:`ensemble_sensor_series`) sees the same slice.
    """
    if "ensemble" not in ds.dims:
        return {
            name: _sensor_component_timeseries(ds, ox, oy, oz, solver_name)
            for name, (ox, oy, oz) in sensor_sets.items()
        }

    members = {name: [] for name in sensor_sets}
    for m in range(ds.sizes["ensemble"]):
        # A slice, not an index: it keeps the ``ensemble`` dim, so the pieces
        # concatenate back into the layout the whole-file load produced.
        member = ds[["u", "v", "w"]].isel(ensemble=slice(m, m + 1)).load()
        for name, (ox, oy, oz) in sensor_sets.items():
            members[name].append(
                _sensor_component_timeseries(member, ox, oy, oz, solver_name)
            )
        if on_member is not None:
            on_member(window_index, m, member)
        # No ``member.close()``: ``.load()``'s result owns no file handle, so
        # the call would be a no-op that reads as if it released something. The
        # slice is dropped at the end of the iteration, which is the real thing.
    return {
        name: (parts[0] if len(parts) == 1 else xarray.concat(parts, dim="ensemble"))
        for name, parts in members.items()
    }


def truth_sensor_series(
    true_state_path,
    n_total,
    x_offset,
    start_idx,
    t_offset,
    sensor_sets,
    solver_name,
    num_windows,
    n_per_window,
):
    """Truth per-component ``(u, v, w)`` sensor series, one window at a time.

    Mirrors the window loop's memory discipline: only one window's worth of the
    (potentially multi-GB) truth is held at once. The truth's ``time`` axis is
    already global, so the per-window pieces concatenate directly. Returns
    ``{name: DataArray(component, time, sensor)}``.
    """
    pieces = {name: [] for name in sensor_sets}
    for w in range(num_windows):
        ts = open_truth(true_state_path, n_total, x_offset, start_idx, t_offset).isel(
            time=slice(w * n_per_window, (w + 1) * n_per_window)
        )
        for name, (ox, oy, oz) in sensor_sets.items():
            pieces[name].append(
                _sensor_component_timeseries(ts, ox, oy, oz, solver_name)
            )
        ts.close()
    return _concat_sensor_pieces(pieces)


# ---------------------------------------------------------------------------
# High-rate probe records (phase 3: the spectral check)
#
# The probe records are written by ``scripts/esmda/run_probe_series.py``, an
# optional rerun of one window at ~1 s output -- never by the assimilation, whose
# own cadence (10 s in a 300 s window) is ~30x too coarse for a Welch spectrum.
# So a run dir *usually* has none, and every helper here treats their absence as
# the normal case: the metric and figure stages both call in, and both skip their
# spectral block when they get ``None`` (master-plan invariant 3).
# ---------------------------------------------------------------------------

_TRUTH_PROBES = "truth_probes.nc"
_PROBE_STEM = re.compile(r"window_(\d+)_probes")


def probe_record_paths(
    run_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path | None] | None:
    """``(truth, posterior, prior | None)`` probe records in ``run_dir``.

    The probe rerun covers **one** window, and ``truth_probes.nc`` is a single
    unversioned file the rerun overwrites, so which window the *truth* record
    describes is decided by whichever invocation ran last. That makes the truth's
    ``window_index`` attribute the authority here: the matching posterior record is
    the only one it may be paired with. Re-probing window 2 after window 1 leaves
    ``windows/window_1_probes.nc`` on disk beside a window-2 truth, and pairing
    those would produce a plausible figure and a wrong LSD, so a truth whose window
    has no posterior record is refused outright rather than fallen back on.

    The fallback to the highest-numbered record applies only to a truth record that
    carries no ``window_index`` at all (a hand-made or pre-WP3.2 file), and it is
    logged.

    The prior record is optional by design: prior probe reruns double the rerun
    cost and only feed figure S4's grey envelope, so ``None`` in the third slot is
    a normal outcome, not a degradation.

    Returns ``None`` when there is no truth record or no window record at all,
    which is every run dir that never had a probe rerun.
    """
    run_dir = pathlib.Path(run_dir)
    truth_path = run_dir / _TRUTH_PROBES
    windows: dict[int, pathlib.Path] = {}
    for path in sorted((run_dir / "windows").glob("window_*_probes.nc")):
        match = _PROBE_STEM.fullmatch(path.stem)
        if match:
            windows[int(match.group(1))] = path
    if not truth_path.exists() or not windows:
        return None

    wanted: int | None = None
    try:
        # Metadata only: the attribute is read without touching the series.
        with xarray.open_dataset(truth_path) as truth:
            recorded = truth.attrs.get("window_index")
        wanted = None if recorded is None else int(recorded)
    except (OSError, TypeError, ValueError) as error:
        logger.info("Cannot read %s: %s", truth_path.name, error)
        return None

    if wanted is None:
        index = max(windows)
        logger.info(
            "%s carries no window_index; pairing it with the highest-numbered "
            "posterior probe record (window %s) -- check that they are from the "
            "same rerun",
            _TRUTH_PROBES,
            index,
        )
    elif wanted not in windows:
        logger.warning(
            "%s covers window %s but the only posterior probe records are for %s. "
            "The truth record is overwritten by every rerun, so this is a stale "
            "pairing rather than a choice: the spectral check is skipped. Re-run "
            "scripts/esmda/run_probe_series.py for one window, or delete the stale "
            "window_*_probes.nc",
            _TRUTH_PROBES,
            wanted,
            sorted(windows),
        )
        return None
    else:
        index = wanted
    prior_path = windows[index].with_name(f"window_{index}_probes_prior.nc")
    return truth_path, windows[index], prior_path if prior_path.exists() else None


def _same_probe_window(
    truth: xarray.Dataset, members: xarray.Dataset, path: pathlib.Path
) -> bool:
    """Whether a member record describes the same window as the truth record.

    The filename says one thing and the file's own ``window_index`` says another
    only when something went wrong -- a record moved by hand, or a rerun that wrote
    a different window than the name it was given. Both are silent failures
    otherwise: the spectra would be of two different windows of the flow, which
    still *looks* like a comparison. Records with no ``window_index`` at all pass,
    since there is nothing to contradict.
    """
    theirs = members.attrs.get("window_index")
    ours = truth.attrs.get("window_index")
    if theirs is None or ours is None or int(theirs) == int(ours):
        return True
    logger.warning(
        "%s records window %s but %s records window %s; the spectra would compare "
        "two different windows, so this pairing is refused",
        path.name,
        theirs,
        _TRUTH_PROBES,
        ours,
    )
    return False


def probe_spectra_bundle(run_dir: pathlib.Path) -> dict | None:
    """Matched Welch spectra from the run's probe records, or ``None``.

    The one place the probe records are located, opened and reduced, called by
    both the metric stage (which turns the bundle into ``spectral_metrics``) and
    the figure stage (which draws it as S4) -- so the annotated distances in the
    figure and the numbers in ``run_summary.yaml`` come off the same arrays.

    The records are the small artifact of the rerun (a handful of sensors x a few
    thousand frames, megabytes), not window state files, so they are read whole;
    invariant 2 is about the multi-GB state files this never touches.

    ``None``, with the reason logged, when there are no probe records or they
    cannot be compared -- see :func:`evaluation.turbulence.probe_spectra` for the
    latter.
    """
    paths = probe_record_paths(run_dir)
    if paths is None:
        logger.info(
            "No probe records in %s; the spectral check is skipped (run "
            "scripts/esmda/run_probe_series.py on this run dir to write them)",
            run_dir,
        )
        return None

    truth_path, posterior_path, prior_path = paths
    try:
        with (
            xarray.open_dataset(truth_path) as truth,
            xarray.open_dataset(posterior_path) as posterior,
        ):
            if not _same_probe_window(truth, posterior, posterior_path):
                return None
            if prior_path is None:
                return probe_spectra(truth, posterior)
            with xarray.open_dataset(prior_path) as prior:
                if not _same_probe_window(truth, prior, prior_path):
                    # The prior only feeds S4's envelope, so a stale one costs
                    # itself rather than the whole check.
                    return probe_spectra(truth, posterior)
                return probe_spectra(truth, posterior, prior)
    except (OSError, ValueError, KeyError) as error:
        # A record that exists but cannot be read (a rerun killed mid-write) costs
        # the spectral block alone, like every other optional artifact here.
        logger.warning("Cannot read the probe records in %s: %s", run_dir, error)
        return None


# ---------------------------------------------------------------------------
# Observation-space diagnostics (phase 2)
# ---------------------------------------------------------------------------


def obs_diagnostics_bundle(run_dir: pathlib.Path) -> dict | None:
    """Per-iteration ``O_N`` from the run's observation-space files, or ``None``.

    The one place ``windows/window_{w}_{obs,pred_obs}.nc`` are located, opened
    and reduced, called by both the metric stage (which turns the bundle into
    ``esmda_diagnostics``) and the figure stage (which draws it as D3) — so the
    boxes in the figure and the numbers in ``run_summary.yaml`` come off the
    same arrays.

    The files are KB-scale (``N_d × M`` floats per iteration), not window state
    files, so they are read whole; invariant 2 concerns the multi-GB state files
    this never touches.

    Returns ``{"per_step": [np.ndarray, ...], "num_observations": int,
    "num_windows": int, "per_window": [np.ndarray (L, M), ...],
    "window_indices": [int, ...]}``. ``per_step`` pools members across windows,
    one entry per iteration; ``per_window`` keeps them separate, which is what
    D3 needs to avoid conflating window 0's cold-start prior with a later
    window's extrapolated one, and ``window_indices`` says which window each
    entry is so a dropped one does not silently renumber the rest.

    **Bounded by ``truth_access.yaml``'s ``num_windows``**, like every other
    consumer in ``compute_esmda_metrics``. ``paths.results_dir`` is a fixed,
    non-timestamped path and the window loop never clears ``windows/``, so a
    rerun with fewer windows (or a crash partway through one) leaves a previous
    run's files in place; globbing alone would pool them into this run's
    diagnostic while every other block in the same ``run_summary.yaml`` covered
    only the current windows.

    **And by the run's own ``save_obs_diagnostics``**, read from
    ``run_info.yaml``. The window-count bound above catches a *shrinking* rerun;
    it cannot catch one that merely turned the flag off — which
    ``conf/run_esmda.yaml`` offers as the way to reproduce the pre-phase-2
    artifact set exactly — because the leftover files then still match this run's
    window count. Deciding on file existence alone would put the *previous*
    assimilation's mismatch beside this run's parameter and state metrics, and
    draw D3 from it. An absent flag (a run dir written before WP2.1) means
    unknown, not false, and falls through to the files.

    ``None``, with the reason logged, when the run has no such files — every run
    dir written before WP2.1 or with ``esmda.save_obs_diagnostics=false``.
    """
    from evaluation.scores import data_mismatch

    run_dir = pathlib.Path(run_dir)
    windows_dir = run_dir / "windows"
    configuration = read_yaml(run_dir / "run_info.yaml").get("configuration") or {}
    if configuration.get("save_obs_diagnostics") is False:
        if any(windows_dir.glob("window_*_obs.nc")):
            logger.warning(
                "Ignoring the observation-space files in %s: this run set "
                "esmda.save_obs_diagnostics=false, so they are left over from an "
                "earlier run into the same output dir",
                windows_dir,
            )
        return None
    expected = read_yaml(run_dir / "truth_access.yaml").get("num_windows")

    pairs = []
    for obs_path in sorted(windows_dir.glob("window_*_obs.nc")):
        match = re.fullmatch(r"window_(\d+)_obs", obs_path.stem)
        if not match:
            continue
        window = int(match.group(1))
        if expected is not None and window >= int(expected):
            logger.warning(
                "Ignoring %s: window %d is beyond this run's %d window(s), so it "
                "is left over from an earlier run into the same output dir",
                obs_path.name,
                window,
                int(expected),
            )
            continue
        pred_path = windows_dir / f"window_{window}_pred_obs.nc"
        if pred_path.exists():
            pairs.append((window, obs_path, pred_path))
    if not pairs:
        logger.info(
            "No observation-space files in %s; the data-mismatch diagnostic is "
            "skipped (rerun with esmda.save_obs_diagnostics=true to write them)",
            run_dir,
        )
        return None

    per_window: list[np.ndarray] = []
    window_indices: list[int] = []
    n_obs = 0
    for window, obs_path, pred_path in sorted(pairs):
        try:
            with (
                xarray.open_dataset(obs_path) as obs_ds,
                xarray.open_dataset(pred_path) as pred_ds,
            ):
                obs = obs_ds["obs"].values
                sigma = obs_ds["obs_error_std"].values
                pred = pred_ds["pred_obs"].values  # (esmda_step, obs, ensemble)
                # N_d sets the target band, so windows scored against different
                # observation counts cannot share one. The first window kept
                # fixes it; a later mismatch is a stale or reconfigured file and
                # is dropped rather than silently widening the band.
                window_n_obs = int(np.size(obs))
                if n_obs and window_n_obs != n_obs:
                    logger.warning(
                        "Ignoring window %d in %s: %d observations against the "
                        "run's %d, so it cannot share the target band",
                        window,
                        run_dir,
                        window_n_obs,
                        n_obs,
                    )
                    continue
                # One row per iteration; ``data_mismatch`` validates the shapes.
                per_window.append(
                    np.stack(
                        [
                            data_mismatch(obs, pred[i], sigma)
                            for i in range(pred.shape[0])
                        ]
                    )
                )
                window_indices.append(window)
                n_obs = window_n_obs
        except (OSError, ValueError, KeyError) as error:
            # A window killed mid-write costs itself, not the whole diagnostic.
            logger.warning(
                "Cannot read the observation-space files for window %d in %s: %s",
                window,
                run_dir,
                error,
            )

    if not per_window:
        return None

    # Windows may in principle differ in iteration count (a config changed
    # between reruns into one dir); pool over whatever each window actually has
    # rather than assuming a rectangle.
    n_steps = max(w.shape[0] for w in per_window)
    per_step = [
        np.concatenate([w[i] for w in per_window if w.shape[0] > i])
        for i in range(n_steps)
    ]
    return {
        "per_step": per_step,
        "per_window": per_window,
        "window_indices": window_indices,
        "num_observations": n_obs,
        "num_windows": len(per_window),
    }


# ---------------------------------------------------------------------------
# Scalar / YAML helpers
# ---------------------------------------------------------------------------


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to plain Python for safe YAML."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_native(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_yaml(data, path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(_to_native(data), f, sort_keys=False, default_flow_style=False)
