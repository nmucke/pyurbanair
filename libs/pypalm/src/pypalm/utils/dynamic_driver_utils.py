"""Time-varying inflow for pypalm via a PALM dynamic-driver NetCDF.

PALM's ``turbulent_inflow`` module with ``turbulent_inflow_method =
'read_from_file'`` reads ``inflow_plane_u/v/w(time_inflow, zu, y)`` from a
NetCDF file co-located with the ``_p3d`` namelist, then linearly interpolates
in time between snapshots at the dirichlet/radiation E/W boundary. This is
the only PALM mechanism for time-varying inflow compatible with
``bc_lr='dirichlet/radiation'`` + ``bc_ns='cyclic'``.

Mirrors the intent of pyudales's ``apply_time_varying_inflow`` /
``write_timedepnudge_file`` and pylbm's ``write_uvel_time_file``: detect the
time dim, prepend a constant plateau at the initial values during spinup,
reuse the same ``build_profile_shape`` once, and write a backend-specific
time-series file.
"""

import logging
import pathlib
from typing import Optional

import numpy as np
import xarray

from .inflow_utils import angle_to_velocity
from .p3d_utils import P3DFile
from .vertical_profile import build_profile_shape

logger = logging.getLogger(__name__)


def is_time_varying_params(params: Optional[xarray.Dataset]) -> bool:
    """Return True if any inflow param has a ``time`` dim.

    Matches pyudales/pylbm's per-variable check, which is strictly more
    permissive than pypalm's original Dataset-level check. Allows a mixed
    case where one parameter varies in time and the other is scalar.
    """
    if params is None:
        return False
    for name in ("inflow_angle", "velocity_magnitude"):
        if name in params and "time" in params[name].dims:
            return True
    return False


def _extract_schedule(
    params: xarray.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull (time_seconds, inflow_angle, velocity_magnitude) as 1-D arrays.

    Broadcasts scalar variables to the length of ``time``.
    """
    if "time" not in params.coords and "time" not in params.dims:
        raise ValueError("params has no 'time' coord/dim; not a time-varying schedule")

    time_seconds = np.asarray(params["time"].values, dtype=float)
    n = time_seconds.shape[0]

    def _broadcast(name: str) -> np.ndarray:
        var = params[name]
        arr = np.asarray(var.values, dtype=float)
        if "time" in var.dims:
            return arr
        return np.full(n, float(arr.item() if arr.ndim == 0 else arr[0]))

    return time_seconds, _broadcast("inflow_angle"), _broadcast("velocity_magnitude")


def _prepend_spinup_plateau(
    time_seconds: np.ndarray,
    inflow_angle: np.ndarray,
    velocity_magnitude: np.ndarray,
    spinup_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepend a t=0 snapshot with initial values and shift user times.

    Matches pyudales.nudging_utils (line 245) and pylbm.params_utils (line
    152). During the [0, spinup_time] interval the inflow is held constant at
    the user's t=0 values.
    """
    if spinup_time <= 0:
        return time_seconds, inflow_angle, velocity_magnitude
    return (
        np.concatenate([[0.0], time_seconds + spinup_time]),
        np.concatenate([[inflow_angle[0]], inflow_angle]),
        np.concatenate([[velocity_magnitude[0]], velocity_magnitude]),
    )


def _build_uv_profiles(
    inflow_angle: np.ndarray,
    velocity_magnitude: np.ndarray,
    shape: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_profiles, v_profiles) with shape ``(T, Nz)``.

    Profile shape is evaluated once (outside) and broadcast across time.
    """
    u0, v0 = angle_to_velocity(inflow_angle, velocity_magnitude)
    u0 = np.asarray(u0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    u_profiles = u0[:, None] * shape[None, :]
    v_profiles = v0[:, None] * shape[None, :]
    return u_profiles, v_profiles


DEFAULT_PT_SURFACE = 300.0
"""Default PALM reference potential temperature (K).

Matches PALM's internal default ``pt_surface = 300.0_wp``. The dynamic
driver requires ``inflow_plane_pt`` whenever ``neutral = .false.`` (which
is PALM's default). We feed a spatially/temporally constant 300 K so that
an unstratified run sees no pt gradient from the inflow.
"""


def write_dynamic_driver_file(
    path: pathlib.Path,
    time_seconds: np.ndarray,
    u_profiles: np.ndarray,
    v_profiles: np.ndarray,
    z: np.ndarray,
    zw: np.ndarray,
    y: np.ndarray,
    pt_surface: float = DEFAULT_PT_SURFACE,
) -> None:
    """Write ``<case>_dynamic`` NetCDF with the variables PALM requires.

    Shapes (PALM enforces exact dim lengths in turbulent_inflow_init):
      - time_seconds: (T,) — PALM requires time_inflow[0] == 0.0
      - u_profiles, v_profiles: (T, Nz) — broadcast across y to (T, Nz, Ny)
      - z: (Nz,) scalar-grid heights; len must equal PALM's ``nz``
      - zw: (Nz-1,) w-grid heights; len must equal PALM's ``nz-1``
      - y: (Ny,) spanwise cell centres; len must equal PALM ``ny+1``
        (i.e. pypalm's ``self.ny``)

    Variables written:
      - inflow_plane_u/v(time_inflow, z, y) — from u/v_profiles
      - inflow_plane_w(time_inflow, zw, y) — zeros (mean flow only)
      - inflow_plane_e(time_inflow, z, y) — zeros (no SGS-TKE inflow)
      - inflow_plane_pt(time_inflow, z, y) — constant pt_surface
        (required unless ``neutral = .T.``, which is not PALM's default)
    """
    time_seconds = np.asarray(time_seconds, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    zw = np.asarray(zw, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    u_profiles = np.asarray(u_profiles, dtype=np.float32)
    v_profiles = np.asarray(v_profiles, dtype=np.float32)

    if time_seconds.size == 0 or time_seconds[0] != 0.0:
        raise ValueError(
            f"time_inflow must start at 0.0 (PALM requirement); got {time_seconds[:1]}"
        )

    n_time = time_seconds.shape[0]
    n_z = z.shape[0]
    n_zw = zw.shape[0]
    n_y = y.shape[0]
    if u_profiles.shape != (n_time, n_z) or v_profiles.shape != (n_time, n_z):
        raise ValueError(
            f"u/v profile shape mismatch: got u={u_profiles.shape}, v={v_profiles.shape}, "
            f"expected ({n_time}, {n_z})"
        )

    u_plane = np.broadcast_to(u_profiles[:, :, None], (n_time, n_z, n_y)).astype(
        np.float32
    )
    v_plane = np.broadcast_to(v_profiles[:, :, None], (n_time, n_z, n_y)).astype(
        np.float32
    )
    w_plane = np.zeros((n_time, n_zw, n_y), dtype=np.float32)
    e_plane = np.zeros((n_time, n_z, n_y), dtype=np.float32)
    pt_plane = np.full((n_time, n_z, n_y), float(pt_surface), dtype=np.float32)

    # PALM's turbulent_inflow module expects the scalar-height dim named
    # "z" (see turbulent_inflow_mod.f90:179 where char_zu is literally set
    # to 'z' with a comment "should be 'z'") and the w-height dim named
    # "zw". Dim lengths are checked strictly: z == nz, zw == nz-1,
    # y == ny+1 (TUI0013/TUI0014/TUI0015 in turbulent_inflow_init).
    ds = xarray.Dataset(
        data_vars={
            "inflow_plane_u": (
                ("time_inflow", "z", "y"),
                u_plane,
                {"long_name": "u-component at inflow", "units": "m s-1"},
            ),
            "inflow_plane_v": (
                ("time_inflow", "z", "y"),
                v_plane,
                {"long_name": "v-component at inflow", "units": "m s-1"},
            ),
            "inflow_plane_w": (
                ("time_inflow", "zw", "y"),
                w_plane,
                {"long_name": "w-component at inflow", "units": "m s-1"},
            ),
            "inflow_plane_e": (
                ("time_inflow", "z", "y"),
                e_plane,
                {"long_name": "SGS-TKE at inflow", "units": "m2 s-2"},
            ),
            "inflow_plane_pt": (
                ("time_inflow", "z", "y"),
                pt_plane,
                {"long_name": "potential temperature at inflow", "units": "K"},
            ),
        },
        coords={
            "time_inflow": ("time_inflow", time_seconds, {"units": "s"}),
            "z": ("z", z, {"long_name": "Height at scalar grid", "units": "m"}),
            "zw": ("zw", zw, {"long_name": "Height at w grid", "units": "m"}),
            "y": ("y", y, {"long_name": "Spanwise cell centre", "units": "m"}),
        },
    )

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # PALM 25.10's turbulent_inflow_init does `ANY( time == fill_time )`
    # (turbulent_inflow_mod.f90:1396) where fill_time is read from the
    # `_FillValue` attribute, or left uninitialized when absent. Two
    # pitfalls (cf. jobs 9950168, 9950191):
    #   - xarray's default _FillValue=NaN, combined with PALM's -Ofast
    #     build (implies -ffinite-math-only), lets gfortran fold
    #     `time == NaN` to .TRUE. → false TUI0018.
    #   - omitting the attribute leaves fill_time as 0.0 (zeroed Fortran
    #     memory). PALM also REQUIRES time_inflow[0] == 0.0 (TUI0017),
    #     so the check trivially matches.
    # Pick a finite, non-NaN sentinel that can't appear in any inflow data:
    # the canonical netCDF default fill 9.96921e+36.
    fill = np.float32(9.9692099683868690e36)
    encoding = {name: {"_FillValue": fill} for name in list(ds.variables)}
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)
    logger.info(
        "Wrote dynamic driver with %d time points to %s", n_time, path
    )


def remove_dynamic_driver_file(path: pathlib.Path) -> None:
    """Delete a stale dynamic driver file if it exists (no-op otherwise)."""
    path = pathlib.Path(path)
    if path.exists():
        path.unlink()
        logger.info("Removed stale dynamic driver file %s", path)


def apply_time_varying_inflow(
    *,
    params: xarray.Dataset,
    p3d_path: pathlib.Path,
    driver_path: pathlib.Path,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    nz: int,
    ny: int,
    profile_config: Optional[dict],
    spinup_time: float,
) -> xarray.Dataset:
    """Orchestrator: write the driver file and toggle the turbulent_inflow namelist.

    Returns a scalar-valued ``xarray.Dataset`` (the t=0 snapshot) so the
    caller can still populate ``ug_surface``/``vg_surface`` and the static
    ``u_profile``/``v_profile`` entries for initialisation.
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    dz = (zmax - zmin) / nz
    dy = (ymax - ymin) / ny
    # PALM's dim checks demand: driver z dim == nz, zw dim == nz-1,
    # y dim == pypalm.ny (= PALM's namelist ny+1).
    z = np.arange(nz) * dz + 0.5 * dz + zmin
    zw = np.arange(1, nz) * dz + zmin
    y = np.arange(ny) * dy + 0.5 * dy + ymin

    shape = build_profile_shape(profile_config, heights=z, zsize=zmax - zmin)

    time_s, angles, speeds = _extract_schedule(params)
    time_s, angles, speeds = _prepend_spinup_plateau(
        time_s, angles, speeds, spinup_time
    )
    u_profiles, v_profiles = _build_uv_profiles(angles, speeds, shape)

    write_dynamic_driver_file(
        path=driver_path,
        time_seconds=time_s,
        u_profiles=u_profiles,
        v_profiles=v_profiles,
        z=z,
        zw=zw,
        y=y,
    )

    # PALM gates turbulent_inflow via the presence of a
    # &turbulent_inflow_parameters namelist block (see
    # turbulent_inflow_mod.f90:544-582). The module activates iff that block
    # is present AND switch_off_module == .false. — no boolean in
    # &initialization_parameters.
    p3d = P3DFile(p3d_path)
    p3d.set_value("turbulent_inflow_parameters", "switch_off_module", False)
    p3d.set_string(
        "turbulent_inflow_parameters", "turbulent_inflow_method", "read_from_file"
    )
    p3d.write()

    u0_init, v0_init = angle_to_velocity(float(angles[0]), float(speeds[0]))
    return xarray.Dataset(
        data_vars={
            "inflow_angle": float(angles[0]),
            "velocity_magnitude": float(speeds[0]),
        },
        attrs={"u0_init": float(u0_init), "v0_init": float(v0_init)},
    )


def disable_turbulent_inflow(p3d_path: pathlib.Path) -> None:
    """Ensure the turbulent_inflow module is off for the static path.

    Idempotent: safe to call even if the block was never written. Prevents a
    prior time-varying run from leaving a sticky
    ``&turbulent_inflow_parameters`` block that would re-enable the module
    when falling back to static params.
    """
    p3d = P3DFile(p3d_path)
    if not p3d.has_section("turbulent_inflow_parameters"):
        return
    p3d.set_value("turbulent_inflow_parameters", "switch_off_module", True)
    p3d.write()
