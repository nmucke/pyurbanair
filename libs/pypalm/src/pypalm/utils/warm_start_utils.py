"""Warm-start for pypalm via PALM's dynamic-driver ``init_atmosphere_*`` fields.

PALM 25.10 initializes the full 3D interior from ``init_atmosphere_u/v/w[/pt/qv]``
(level-of-detail 2) variables in the PIDS_DYNAMIC NetCDF when
``initializing_actions = 'read_from_file'`` (``init_3d_model.f90:478`` →
``netcdf_data_input_mod.f90:2095-2690``). This is pypalm's warm-start
mechanism: the resolved velocity field a previous window returned is injected
as the next window's initial condition.

Unlike pyudales/pylbm there is **no "carry" of subgrid state** — PALM never
reads SGS-TKE from file (there is no ``init_atmosphere_e``; grep confirms zero
matches) and re-derives ``e`` from the mean field
(``turbulence_closure_mod.f90:1130-1170``). Warm-start is therefore *stateless*:
it consumes only the handed-in xarray state and writes a driver file, so it
needs no per-member persistence and does not interact with the ensemble's
failure-resample path (a resampled member simply warm-starts from its donor's
state, which the base ensemble already supplies).

Grid contract (verified against the bundled PALM source):

* The vertical ``z``/``zw`` coordinate **values** are checked to within
  ``0.1*dz`` against PALM's 0-based ``zu``/``zw`` (``DRV0005``,
  ``netcdf_data_input_mod.f90:2267-2288``). So the file must use 0-based
  ``zu(k)=(k-0.5)*dz`` and ``zw(k)=k*dz`` — **not** the physical-bounds offset
  that ``_load_and_postprocess_state`` adds. We sample the (offset) state at the
  matching physical heights but label the file axis 0-based.
* The horizontal ``x``/``xu``/``y``/``yv`` axes are **length-checked only**
  (``DRV0004``); their values are never read, so any offset is harmless.
* Staggering (file dim order is NetCDF/C order): ``init_atmosphere_u(z,y,xu)``,
  ``init_atmosphere_v(z,yv,x)``, ``init_atmosphere_w(zw,y,x)``,
  ``init_atmosphere_pt(z,y,x)``.
* Dim lengths, for pypalm's ``nx``/``ny``/``nz`` grid-point counts (PALM
  namelist ``nx=self.nx-1``, ``ny=self.ny-1``, ``nz=self.nz``):
  ``x=nx``, ``xu=nx-1``, ``y=ny``, ``yv=ny-1``, ``z=nz``, ``zw=nz-1``.
* The fields must contain **no** ``_FillValue`` (``DRV0008``); we replace any
  NaN with 0 (PALM zeros topography-occluded cells itself anyway).
"""

import logging
import pathlib
from typing import Optional

import numpy as np
import xarray

logger = logging.getLogger(__name__)


# Canonical netCDF default fill — a finite, non-NaN sentinel that cannot appear
# in any realistic flow field. PALM reads ``_FillValue`` and fatally rejects a
# field that contains it (DRV0008); a NaN _FillValue combined with PALM's
# -ffinite-math-only build folds to spurious matches, so we mirror the inflow
# writer's choice of a large finite sentinel.
_FILL = np.float32(9.9692099683868690e36)

Bounds = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]


def _squeeze_state(state: xarray.Dataset) -> xarray.Dataset:
    """Reduce ``state`` to a single 3D frame (drop ``time``/``ensemble``)."""
    if "ensemble" in state.dims:
        state = state.isel(ensemble=-1, drop=True)
    if "time" in state.dims:
        state = state.isel(time=-1, drop=True)
    return state


def _interp_to(
    da: xarray.DataArray,
    x_phys: np.ndarray,
    y_phys: np.ndarray,
    z_phys: np.ndarray,
) -> np.ndarray:
    """Interpolate ``da`` onto the physical target grid → ``(z, y, x)`` f32.

    ``_load_and_postprocess_state`` unifies only the vertical axis (``w``
    interpolated onto ``z``); the horizontal staggers are left intact, so ``u``
    arrives on ``xu``, ``v`` on ``yv`` and ``w``/scalars on ``x``/``y``. We
    therefore interpolate on whichever horizontal dim names the array actually
    carries (``x_phys``/``y_phys`` being the physical sample points for the
    target staggering). Linear with extrapolation (staggered/edge targets fall
    just outside the cell-centred source); any residual NaN is zeroed so the
    written field carries no fill values (DRV0008).
    """
    xdim = "xu" if "xu" in da.dims else "x"
    ydim = "yv" if "yv" in da.dims else "y"
    out = da.interp(
        {xdim: x_phys, ydim: y_phys, "z": z_phys},
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    ).transpose("z", ydim, xdim)
    arr = np.asarray(out.values, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_init_atmosphere_dataset(
    state: xarray.Dataset,
    bounds: Bounds,
    nx: int,
    ny: int,
    nz: int,
    pt_surface: float,
) -> xarray.Dataset:
    """Build the ``init_atmosphere_*`` LOD=2 fields + coords from ``state``.

    ``nx``/``ny``/``nz`` are pypalm's grid-point counts (``self.nx`` etc.); the
    PALM namelist values are ``nx-1``/``ny-1``/``nz``. ``state`` must carry
    ``u``/``v``/``w`` on dims/coords ``z``/``y``/``x`` in the physical frame
    (the offset applied by ``_load_and_postprocess_state``).
    """
    state = _squeeze_state(state)
    for var in ("u", "v", "w"):
        if var not in state:
            raise ValueError(f"warm-start state is missing required variable '{var}'")

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = (
        (float(b[0]), float(b[1])) for b in bounds
    )
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz

    # File axes: 0-based PALM-native values. z/zw are value-checked (DRV0005);
    # x/xu/y/yv are length-checked only but we still use native values.
    xu_file = np.arange(nx - 1, dtype=np.float64) * dx
    x_file = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    yv_file = np.arange(ny - 1, dtype=np.float64) * dy
    y_file = (np.arange(ny, dtype=np.float64) + 0.5) * dy
    z_file = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    zw_file = (np.arange(1, nz, dtype=np.float64)) * dz

    # Physical sample points: the file axes shifted onto the state's frame.
    xu_phys = xu_file + xmin
    x_phys = x_file + xmin
    yv_phys = yv_file + ymin
    y_phys = y_file + ymin
    z_phys = z_file + zmin
    zw_phys = zw_file + zmin

    u = _interp_to(state["u"], xu_phys, y_phys, z_phys)   # (z, y, xu)
    v = _interp_to(state["v"], x_phys, yv_phys, z_phys)   # (z, yv, x)
    w = _interp_to(state["w"], x_phys, y_phys, zw_phys)   # (zw, y, x)
    pt = np.full((nz, ny, nx), float(pt_surface), dtype=np.float32)  # (z, y, x)

    lod = {"lod": np.int32(2)}
    ds = xarray.Dataset(
        data_vars={
            "init_atmosphere_u": (("z", "y", "xu"), u, {**lod, "units": "m s-1"}),
            "init_atmosphere_v": (("z", "yv", "x"), v, {**lod, "units": "m s-1"}),
            "init_atmosphere_w": (("zw", "y", "x"), w, {**lod, "units": "m s-1"}),
            "init_atmosphere_pt": (("z", "y", "x"), pt, {**lod, "units": "K"}),
        },
        coords={
            "x": ("x", x_file.astype(np.float32), {"units": "m"}),
            "xu": ("xu", xu_file.astype(np.float32), {"units": "m"}),
            "y": ("y", y_file.astype(np.float32), {"units": "m"}),
            "yv": ("yv", yv_file.astype(np.float32), {"units": "m"}),
            "z": ("z", z_file.astype(np.float32), {"units": "m"}),
            "zw": ("zw", zw_file.astype(np.float32), {"units": "m"}),
        },
    )
    return ds


def write_warmstart_driver(
    driver_path: pathlib.Path,
    state: xarray.Dataset,
    bounds: Bounds,
    nx: int,
    ny: int,
    nz: int,
    pt_surface: float,
) -> None:
    """Write/augment the PIDS_DYNAMIC driver with ``init_atmosphere_*`` fields.

    When a dynamic driver already exists (time-varying inflow wrote its
    ``inflow_plane_*`` planes), the ``init_atmosphere_*`` variables are merged
    into that same file — PALM reads the two via independent routines and they
    share the (0-based) ``z``/``zw``/``y`` axes. Otherwise a fresh driver
    holding only the init fields is created (the static-inflow warm-start case).
    """
    driver_path = pathlib.Path(driver_path)
    init_ds = build_init_atmosphere_dataset(state, bounds, nx, ny, nz, pt_surface)

    if driver_path.exists():
        with xarray.open_dataset(driver_path) as existing:
            existing = existing.load()
        # join/compat="override": the inflow file already carries 0-based
        # z/zw/y of matching length, so align positionally and keep its coords
        # (and the new xu/x/yv from init_ds).
        merged = xarray.merge(
            [existing, init_ds], join="override", compat="override"
        )
    else:
        merged = init_ds

    # Preserve the inflow writer's _FillValue convention across every variable
    # (incl. time_inflow, whose fill handling guards TUI0018), and stamp the
    # finite sentinel on the init fields so PALM's DRV0008 check passes.
    encoding = {name: {"_FillValue": _FILL} for name in merged.variables}

    driver_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(driver_path, engine="netcdf4", encoding=encoding)
    logger.info("Wrote warm-start init_atmosphere fields to %s", driver_path)
