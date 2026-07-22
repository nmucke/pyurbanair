"""Nudging driver for **periodic** pypalm runs — the counterpart of pyudales's
periodic nudging path.

PALM's ``nudge`` scheme (``large_scale_forcing_nudging_mod.f90``) relaxes the
*horizontal-mean* profile of each prognostic quantity toward a target read from
an ASCII ``NUDGING_DATA`` file, with a per-height / per-time relaxation
timescale ``tnudge``. This is the same relaxation physics uDALES applies, so a
cyclic PALM run driven this way and a cyclic uDALES run driven by
``timedepnudge.inp`` hold the domain to the same mean wind.

We nudge **u and v only**: the ``w``/``pt``/``q`` columns carry PALM's
``-999999`` sentinel in every row, which disables nudging for those quantities
(``nudge_u/v/w/pt/q`` flags in the reader). The near-wall exemption uDALES
expresses via ``nnudge`` is emulated by a huge ``tnudge`` below a cutoff height.

PALM's ``nudging`` switch requires ``large_scale_forcing = .T.`` (LSF0001),
which in turn wants an ``LSF_DATA`` file. We stage a physically **inert**
``LSF_DATA`` (both its surface and profile halves disable themselves via
non-fatal paths) so the nudging term is the only large-scale forcing. See
``write_inert_lsf_data`` and ``docs/plans/palm_nudging_driver_plan.md``.

The schedule builders (``_extract_schedule``, ``_prepend_spinup_plateau``,
``_build_uv_profiles``) are shared with the dynamic-driver path — the two
drivers turn params into the same (time, u_profiles, v_profiles) triple.
"""

import logging
import pathlib
from typing import Optional

import numpy as np
import xarray

from .dynamic_driver_utils import (
    _build_uv_profiles,
    _extract_schedule,
    _prepend_spinup_plateau,
    is_time_varying_params,
)
from .inflow_utils import angle_to_velocity
from .vertical_profile import build_profile_shape

logger = logging.getLogger(__name__)


# PALM's per-quantity "disable nudging" sentinel: a column that is this value in
# every row/time switches the corresponding nudge_* flag off (reader guards at
# large_scale_forcing_nudging_mod.f90:1120-1144).
NUDGE_SENTINEL = -999999.0

# Relaxation timescale written below the ``nnudge_meters`` cutoff. Large enough
# that the effective nudging there is negligible (uDALES excludes those levels
# outright); PALM floors the timescale to ``dt_3d`` but never nudges on a 1e9 s
# scale in any finite run.
_TNUDGE_DISABLED = 1.0e9

# Seconds added past ``end_time`` for the terminal bracketing snapshot the time
# interpolator needs, and for the inert LSF_DATA times.
_TERMINAL_PAD = 1.0
_LSF_PAD = 1.0e6


def _nudging_heights(nz: int, dz: float, nnudge_meters: float) -> np.ndarray:
    """Build the (0-based, native) height column PALM height-interpolates against.

    Rows: a ``z=0`` anchor, PALM's native scalar-grid cell centres
    (``arange(nz)*dz + 0.5*dz``, no ``zmin`` offset — the reader checks these
    against its own 0-based ``zu`` and errors, LSF0019, if the profile tops out
    below the model grid), and a top row two ``dz`` above the last centre so the
    profile comfortably brackets ``zu(nzt+1)``.

    When ``nnudge_meters > 0`` a pair of rows straddling the cutoff
    (``cutoff ∓ ε``) is inserted so the ``tnudge`` step from the disabled value
    to the active value is sharp rather than linearly ramped between adjacent
    cell centres.
    """
    cell_centres = np.arange(nz) * dz + 0.5 * dz
    z_top = float(cell_centres[-1]) + 2.0 * dz
    heights = [0.0, *cell_centres.tolist(), z_top]

    if nnudge_meters > 0.0 and nnudge_meters < z_top:
        eps = 0.05 * dz
        heights.extend([nnudge_meters - eps, nnudge_meters + eps])

    return np.unique(np.asarray(heights, dtype=float))


def _tnudge_column(
    heights: np.ndarray, tnudge: float, nnudge_meters: float
) -> np.ndarray:
    """Per-height ``tnudge`` column: ``tnudge`` at/above the cutoff, a huge
    (effectively disabling) value below it."""
    return np.where(heights >= nnudge_meters, float(tnudge), _TNUDGE_DISABLED)


def _append_terminal_snapshot(
    time_seconds: np.ndarray,
    inflow_angle: np.ndarray,
    velocity_magnitude: np.ndarray,
    terminal_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append a final snapshot (repeating the last values) past ``end_time``.

    PALM's time interpolation needs a snapshot bracketing ``end_time`` from
    above, so the target is well-defined for the whole run.
    """
    if terminal_time <= time_seconds[-1]:
        terminal_time = float(time_seconds[-1]) + _TERMINAL_PAD
    return (
        np.concatenate([time_seconds, [terminal_time]]),
        np.concatenate([inflow_angle, [inflow_angle[-1]]]),
        np.concatenate([velocity_magnitude, [velocity_magnitude[-1]]]),
    )


def write_nudging_data(
    path: pathlib.Path,
    times: np.ndarray,
    heights: np.ndarray,
    tnudge_column: np.ndarray,
    u_profiles: np.ndarray,
    v_profiles: np.ndarray,
) -> None:
    """Write an ASCII ``NUDGING_DATA`` file PALM's ``nudge_init`` reads.

    One block per time snapshot::

        # <time_seconds>
        <z>  <tnudge(z)>  <u(z)>  <v(z)>  -999999.0  -999999.0  -999999.0
        ...

    Columns are ``height tnudge u v w pt q``; ``w``/``pt``/``q`` are the
    sentinel so only u and v are nudged. No header lines (unlike ``LSF_DATA``).

    Shapes: ``times`` (T,); ``heights`` and ``tnudge_column`` (Nz,);
    ``u_profiles`` and ``v_profiles`` (T, Nz).
    """
    times = np.asarray(times, dtype=float)
    heights = np.asarray(heights, dtype=float)
    tnudge_column = np.asarray(tnudge_column, dtype=float)
    u_profiles = np.asarray(u_profiles, dtype=float)
    v_profiles = np.asarray(v_profiles, dtype=float)

    n_time = times.shape[0]
    n_z = heights.shape[0]
    if tnudge_column.shape != (n_z,):
        raise ValueError(
            f"tnudge_column length {tnudge_column.shape} != heights length ({n_z},)"
        )
    if u_profiles.shape != (n_time, n_z) or v_profiles.shape != (n_time, n_z):
        raise ValueError(
            f"u/v profile shape mismatch: got u={u_profiles.shape}, "
            f"v={v_profiles.shape}, expected ({n_time}, {n_z})"
        )

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t_idx in range(n_time):
            f.write(f"# {times[t_idx]:.6f}\n")
            for k in range(n_z):
                f.write(
                    f"{heights[k]:14.6f}{tnudge_column[k]:18.6f}"
                    f"{u_profiles[t_idx, k]:16.6f}{v_profiles[t_idx, k]:16.6f}"
                    f"{NUDGE_SENTINEL:14.1f}{NUDGE_SENTINEL:14.1f}"
                    f"{NUDGE_SENTINEL:14.1f}\n"
                )

    logger.info(
        "Wrote NUDGING_DATA %s with %d time snapshots and %d levels",
        path,
        n_time,
        n_z,
    )


def write_inert_lsf_data(path: pathlib.Path, end_time: float) -> None:
    """Write a physically inert ``LSF_DATA`` file.

    Exists only to satisfy the ``nudging`` → ``large_scale_forcing`` constraint
    (LSF0001). Both halves disable themselves via PALM's non-fatal paths: the
    single surface row is beyond ``end_time`` so ``lsf_surf`` is turned off
    (LSF0012, warning); the bare ``#`` is consumed by the reader's skip loop;
    and the ``# <time>`` profile marker (also beyond ``end_time``) makes the
    profile search exit before reading any rows, turning off ``lsf_vert``
    (LSF0016). The nudging term is then the only large-scale forcing. See
    ``docs/plans/palm_nudging_driver_plan.md`` facts 5-6.
    """
    t_far = float(end_time) + _LSF_PAD
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# pyurbanair inert LSF_DATA — exists only to satisfy nudging's",
        "# large_scale_forcing requirement; lsf_surf and lsf_vert both disable.",
        "# columns(surface): time shf qsws pt q p",
        f"{t_far:.6f}  0.0  0.0  0.0  0.0  0.0",
        "#",
        f"# {t_far:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n")
    logger.info(
        "Wrote inert LSF_DATA %s (all times past end_time=%.1f)", path, end_time
    )


def apply_nudging_driver(
    *,
    params: xarray.Dataset,
    nudge_path: pathlib.Path,
    lsf_path: pathlib.Path,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    nz: int,
    profile_config: Optional[dict],
    tnudge: float,
    nnudge_meters: float,
    spinup_time: float,
    simulation_time: Optional[float],
) -> xarray.Dataset:
    """Write ``NUDGING_DATA`` + inert ``LSF_DATA`` for a periodic run.

    Handles static (a 2-snapshot constant schedule bracketing the run) and
    time-varying params (the params' ``time`` array, a spinup plateau when
    ``spinup_time > 0``, plus a terminal snapshot past ``end_time``) with the
    same relaxation-target construction as the dynamic driver.

    Returns a scalar-valued ``xarray.Dataset`` holding the t=0 values so the
    caller can still populate ``ug_surface``/``vg_surface`` and the static
    ``u_profile``/``v_profile`` init entries — the run then starts consistent
    with the initial nudging target.
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    dz = (zmax - zmin) / nz

    heights = _nudging_heights(nz, dz, nnudge_meters)
    # Height axis is 0-based (native, matching PALM's zu); the profile *shape*
    # is evaluated at physical heights so a non-zero zmin is honoured — the same
    # split the dynamic driver makes (dynamic_driver_utils.apply_time_varying_inflow).
    shape = build_profile_shape(
        profile_config, heights=heights + zmin, zsize=zmax - zmin
    )
    tnudge_column = _tnudge_column(heights, tnudge, nnudge_meters)

    end_time = float(simulation_time or 0.0) + float(spinup_time)

    if is_time_varying_params(params):
        time_s, angles, speeds = _extract_schedule(params)
        time_s, angles, speeds = _prepend_spinup_plateau(
            time_s, angles, speeds, spinup_time
        )
        time_s, angles, speeds = _append_terminal_snapshot(
            time_s, angles, speeds, end_time + _TERMINAL_PAD
        )
    else:
        # Static params: a two-snapshot constant schedule bracketing the run.
        angle0 = float(params["inflow_angle"].item())
        speed0 = float(params["velocity_magnitude"].item())
        time_s = np.array([0.0, end_time + _TERMINAL_PAD], dtype=float)
        angles = np.array([angle0, angle0], dtype=float)
        speeds = np.array([speed0, speed0], dtype=float)

    u_profiles, v_profiles = _build_uv_profiles(angles, speeds, shape)

    write_nudging_data(
        nudge_path, time_s, heights, tnudge_column, u_profiles, v_profiles
    )
    write_inert_lsf_data(lsf_path, end_time)

    logger.info(
        "Nudging driver: tnudge=%.1fs, nnudge_meters=%.1fm, %d snapshots, "
        "%d levels (NUDGING_DATA + inert LSF_DATA)",
        tnudge,
        nnudge_meters,
        time_s.shape[0],
        heights.shape[0],
    )

    u0_init, v0_init = angle_to_velocity(float(angles[0]), float(speeds[0]))
    return xarray.Dataset(
        data_vars={
            "inflow_angle": float(angles[0]),
            "velocity_magnitude": float(speeds[0]),
        },
        attrs={"u0_init": float(u0_init), "v0_init": float(v0_init)},
    )


def remove_nudging_files(nudge_path: pathlib.Path, lsf_path: pathlib.Path) -> None:
    """Delete stale ``NUDGING_DATA`` / ``LSF_DATA`` files if present (no-op else).

    Keeps an inflow_outflow run (or a re-run of the same experiment dir) from
    inheriting the LSF apparatus a previous periodic run staged.
    """
    for path in (pathlib.Path(nudge_path), pathlib.Path(lsf_path)):
        if path.exists():
            path.unlink()
            logger.info("Removed stale nudging file %s", path)
