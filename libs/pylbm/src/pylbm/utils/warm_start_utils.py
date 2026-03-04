"""Utilities for handling LBM warmstart/restart files."""

import pathlib
import re
from typing import Optional

import numpy as np
import xarray
from scipy.io import FortranFile

from .dir_utils import DirectoryPaths
from .infile_utils import Infile
from .state_utils import VELOCITY_SCALE_TO_PHYSICAL

RESTART_FILE_PATTERN = re.compile(
    r"^(?P<prefix>restart|turbulence|theta|pottemp|tracer)_(?P<tile>\d{4})_(?P<iteration>\d{6})\.uf$"
)
MAIN_RESTART_PATTERN = re.compile(r"^restart_\d{4}_(?P<iteration>\d{6})\.uf$")


def _restart_dir(dirs: DirectoryPaths) -> pathlib.Path:
    """Return the restart directory path."""
    return dirs.experiment_dir / "restart"


def identify_latest_restart_iteration(dirs: DirectoryPaths) -> Optional[int]:
    """
    Find the latest available main restart iteration.

    Returns:
        Latest iteration number if any main restart file exists, otherwise None.
    """
    restart_dir = _restart_dir(dirs)
    if not restart_dir.exists():
        return None

    iterations: list[int] = []
    for path in restart_dir.iterdir():
        if not path.is_file():
            continue
        match = MAIN_RESTART_PATTERN.match(path.name)
        if match is None:
            continue
        iterations.append(int(match.group("iteration")))

    return max(iterations) if iterations else None


def remove_old_restart_files(
    dirs: DirectoryPaths,
    keep_iteration: Optional[int] = None,
) -> None:
    """
    Remove restart files from older iterations.

    If keep_iteration is None, the newest available main restart iteration is kept.
    """
    restart_dir = _restart_dir(dirs)
    if not restart_dir.exists():
        return

    if keep_iteration is None:
        keep_iteration = identify_latest_restart_iteration(dirs)

    if keep_iteration is None:
        return

    for path in restart_dir.iterdir():
        if not path.is_file():
            continue
        match = RESTART_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        iteration = int(match.group("iteration"))
        if iteration < keep_iteration:
            path.unlink(missing_ok=True)


def clean_output_files(dirs: DirectoryPaths) -> None:
    """Remove LBM netCDF output files from the output directory."""
    for output_file in dirs.output_dir.glob("out_*.nc"):
        output_file.unlink(missing_ok=True)


def _get_mod_dimensions_int(dirs: DirectoryPaths, key: str, default: int) -> int:
    """Read integer parameter from mod_dimensions.F90."""
    pattern = re.compile(rf"integer,\s*parameter\s*::\s*{re.escape(key)}\s*=\s*(\d+)")
    with open(dirs.mod_dimensions_path, "r") as f:
        for line in f:
            stripped = line.strip()
            # Ignore commented examples in mod_dimensions.F90
            if stripped.startswith("!"):
                continue
            match = pattern.search(line)
            if match is not None:
                return int(match.group(1))
    return default


def _load_state_dataset(state: xarray.Dataset | pathlib.Path) -> xarray.Dataset:
    """Load xarray state dataset from dataset object or NetCDF path."""
    if isinstance(state, pathlib.Path):
        return xarray.open_dataset(state, engine="netcdf4").load()
    return state


def _select_last_non_spatial_dims(da: xarray.DataArray) -> xarray.DataArray:
    """Select last index for all non-spatial dimensions."""
    indexers = {dim: -1 for dim in da.dims if dim not in ("x", "y", "z")}
    if indexers:
        da = da.isel(indexers)
    return da


def _get_state_variable(state: xarray.Dataset, name: str) -> np.ndarray:
    """Extract state variable as (z, y, x) float32 array."""
    if name not in state:
        raise ValueError(f"State dataset is missing required variable '{name}'")
    da = _select_last_non_spatial_dims(state[name])
    if set(("x", "y", "z")) - set(da.dims):
        raise ValueError(
            f"Variable '{name}' must contain dims x, y, z. Found dims: {da.dims}"
        )
    da = da.transpose("z", "y", "x")
    return da.values.astype(np.float32)


def _get_state_blanking_mask(state: xarray.Dataset) -> Optional[np.ndarray]:
    """
    Return a fluid-cell mask (z,y,x) from optional blanking variable.

    Convention: blanking == 0 indicates fluid, non-zero indicates solid/blocked.
    """
    if "blanking" not in state:
        return None
    da = _select_last_non_spatial_dims(state["blanking"])
    if set(("x", "y", "z")) - set(da.dims):
        return None
    da = da.transpose("z", "y", "x")
    blanking = da.values.astype(np.float32)
    return blanking < 0.5


def _read_infile_bool(infile: Infile, keys: list[str], default: bool = False) -> bool:
    for key in keys:
        value = infile.get_value_as_bool(key)
        if value is not None:
            return value
    return default


def _read_infile_int(infile: Infile, keys: list[str], default: int = 0) -> int:
    for key in keys:
        value = infile.get_value_as_int(key)
        if value is not None:
            return value
    return default


def _fill_ghost_cells(
    field_xyz: np.ndarray, periodic_x: bool, periodic_y: bool, periodic_z: bool
) -> None:
    """Fill ghost cells in-place for array shaped (x+2, y+2, z+2)."""
    # x-direction
    if periodic_x:
        field_xyz[0, 1:-1, 1:-1] = field_xyz[-2, 1:-1, 1:-1]
        field_xyz[-1, 1:-1, 1:-1] = field_xyz[1, 1:-1, 1:-1]
    else:
        field_xyz[0, 1:-1, 1:-1] = field_xyz[1, 1:-1, 1:-1]
        field_xyz[-1, 1:-1, 1:-1] = field_xyz[-2, 1:-1, 1:-1]

    # y-direction
    if periodic_y:
        field_xyz[:, 0, 1:-1] = field_xyz[:, -2, 1:-1]
        field_xyz[:, -1, 1:-1] = field_xyz[:, 1, 1:-1]
    else:
        field_xyz[:, 0, 1:-1] = field_xyz[:, 1, 1:-1]
        field_xyz[:, -1, 1:-1] = field_xyz[:, -2, 1:-1]

    # z-direction
    if periodic_z:
        field_xyz[:, :, 0] = field_xyz[:, :, -2]
        field_xyz[:, :, -1] = field_xyz[:, :, 1]
    else:
        field_xyz[:, :, 0] = field_xyz[:, :, 1]
        field_xyz[:, :, -1] = field_xyz[:, :, -2]


def _build_equilibrium_restart_distribution(
    rho_xyz: np.ndarray,
    u_xyz: np.ndarray,
    v_xyz: np.ndarray,
    w_xyz: np.ndarray,
    ibgk: int,
) -> np.ndarray:
    """
    Build D3Q27 equilibrium distribution f(l,x,y,z), including 3rd-order option.
    """
    cs2 = 1.0 / 3.0
    cs4 = 1.0 / 9.0
    cs6 = 1.0 / 27.0
    inv1cs2 = 1.0 / cs2
    inv2cs4 = 1.0 / (2.0 * cs4)
    inv6cs6 = 1.0 / (6.0 * cs6)
    ratio = inv6cs6 / inv2cs4

    cxs = np.array(
        [
            0,
            1,
            -1,
            0,
            0,
            0,
            0,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            0,
            0,
            -1,
            1,
            0,
            0,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
        ],
        dtype=np.float32,
    )
    cys = np.array(
        [
            0,
            0,
            0,
            1,
            -1,
            0,
            0,
            1,
            -1,
            -1,
            1,
            0,
            0,
            1,
            -1,
            0,
            0,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
        ],
        dtype=np.float32,
    )
    czs = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            -1,
            1,
            0,
            0,
            0,
            0,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
        ],
        dtype=np.float32,
    )
    weights = np.array(
        [
            8.0 / 27.0,
            2.0 / 27.0,
            2.0 / 27.0,
            2.0 / 27.0,
            2.0 / 27.0,
            2.0 / 27.0,
            2.0 / 27.0,
            1.0 / 54.0,
            1.0 / 54.0,
            1.0 / 54.0,
            1.0 / 54.0,
            1.0 / 54.0,
            1.0 / 54.0,
            1.0 / 54.0,
            1.0 / 54.0,
            1.0 / 54.0,
            1.0 / 54.0,
            1.0 / 54.0,
            1.0 / 54.0,
            1.0 / 216.0,
            1.0 / 216.0,
            1.0 / 216.0,
            1.0 / 216.0,
            1.0 / 216.0,
            1.0 / 216.0,
            1.0 / 216.0,
            1.0 / 216.0,
        ],
        dtype=np.float32,
    )

    vel = [u_xyz, v_xyz, w_xyz]
    dens = rho_xyz

    # A0_2(p,q) and A0_3(p,q,r)
    a0_2 = [[dens * vel[p] * vel[q] * inv2cs4 for q in range(3)] for p in range(3)]
    a0_3 = None
    if ibgk == 3:
        a0_3 = [
            [[a0_2[p][q] * vel[r] * ratio for r in range(3)] for q in range(3)]
            for p in range(3)
        ]

    delta = np.eye(3, dtype=np.float32)
    nl = 27
    feq = np.zeros((nl, *dens.shape), dtype=np.float32)

    for l in range(nl):
        cu = cxs[l] * vel[0] + cys[l] * vel[1] + czs[l] * vel[2]
        tmp = dens * (1.0 + cu * inv1cs2)

        # 2nd-order Hermite contribution
        for p in range(3):
            for q in range(3):
                h2 = cxs[l] if p == 0 else (cys[l] if p == 1 else czs[l])
                h2 = h2 * (cxs[l] if q == 0 else (cys[l] if q == 1 else czs[l]))
                h2 = h2 - cs2 * delta[p, q]
                tmp = tmp + h2 * a0_2[p][q]

        # 3rd-order Hermite contribution for ibgk=3 and l>1
        if ibgk == 3 and l > 0 and a0_3 is not None:
            for p in range(3):
                cp = cxs[l] if p == 0 else (cys[l] if p == 1 else czs[l])
                for q in range(3):
                    cq = cxs[l] if q == 0 else (cys[l] if q == 1 else czs[l])
                    for r in range(3):
                        cr = cxs[l] if r == 0 else (cys[l] if r == 1 else czs[l])
                        h3 = cp * cq * cr - cs2 * (
                            cp * delta[q, r] + cq * delta[p, r] + cr * delta[p, q]
                        )
                        tmp = tmp + h3 * a0_3[p][q][r]

        feq[l] = weights[l] * tmp

    return feq


def _compute_macrovars_from_distribution(
    f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute rho/u/v/w from distribution f(l,x,y,z)."""
    cxs = np.array(
        [
            0,
            1,
            -1,
            0,
            0,
            0,
            0,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            0,
            0,
            -1,
            1,
            0,
            0,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
        ],
        dtype=np.float32,
    ).reshape((27, 1, 1, 1))
    cys = np.array(
        [
            0,
            0,
            0,
            1,
            -1,
            0,
            0,
            1,
            -1,
            -1,
            1,
            0,
            0,
            1,
            -1,
            0,
            0,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
        ],
        dtype=np.float32,
    ).reshape((27, 1, 1, 1))
    czs = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            -1,
            1,
            0,
            0,
            0,
            0,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
        ],
        dtype=np.float32,
    ).reshape((27, 1, 1, 1))

    rho = np.sum(f, axis=0)
    # Prevent division-by-zero if template contains degenerate cells.
    rho_safe = np.where(np.abs(rho) < 1e-12, 1e-12, rho)
    u = np.sum(f * cxs, axis=0) / rho_safe
    v = np.sum(f * cys, axis=0) / rho_safe
    w = np.sum(f * czs, axis=0) / rho_safe
    return (
        rho.astype(np.float32),
        u.astype(np.float32),
        v.astype(np.float32),
        w.astype(np.float32),
    )


def _copy_auxiliary_restart_files(
    dirs: DirectoryPaths,
    source_iteration: int,
    target_iteration: int,
) -> None:
    """Copy non-main restart files from one iteration to another."""
    restart_dir = _restart_dir(dirs)
    for path in restart_dir.iterdir():
        if not path.is_file():
            continue
        match = RESTART_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        prefix = match.group("prefix")
        iteration = int(match.group("iteration"))
        tile = match.group("tile")
        if prefix == "restart" or iteration != source_iteration:
            continue
        target_name = f"{prefix}_{tile}_{target_iteration:06d}.uf"
        target_path = restart_dir / target_name
        target_path.write_bytes(path.read_bytes())


def _try_load_restart_distribution(
    restart_file: pathlib.Path,
    nx: int,
    ny: int,
    nz: int,
) -> Optional[np.ndarray]:
    """Try loading restart distribution f from a Fortran unformatted file."""
    if not restart_file.exists():
        return None
    try:
        with FortranFile(str(restart_file), "r") as f:
            i, j, k, l, f_flat = f.read_record(
                np.int32, np.int32, np.int32, np.int32, np.float32
            )
        if int(i) != nx or int(j) != ny or int(k) != nz or int(l) != 27:
            return None
        expected_size = 27 * (nx + 2) * (ny + 2) * (nz + 2)
        if f_flat.size != expected_size:
            return None
        f_data = np.reshape(
            f_flat.astype(np.float32),
            (27, nx + 2, ny + 2, nz + 2),
            order="F",
        )
        return np.asfortranarray(f_data)
    except Exception:
        return None


def write_restart_file_from_xarray(
    state: xarray.Dataset | pathlib.Path,
    dirs: DirectoryPaths,
    restart_iteration: Optional[int] = None,
) -> int:
    """
    Write a LBM restart file from xarray state and return used iteration.

    This constructs an equilibrium distribution from rho/u/v/w and writes:
    `restart/restart_0000_<iter>.uf`.

    Limitations:
    - Only single-tile restarts are supported.
    - If special restart companions are required (turbulence/theta/pottemp/tracer),
      this function requires an existing restart template iteration to copy from.
    """
    state_ds = _load_state_dataset(state)
    rho_zyx = _get_state_variable(state_ds, "rho")
    u_zyx = _get_state_variable(state_ds, "u")
    v_zyx = _get_state_variable(state_ds, "v")
    w_zyx = _get_state_variable(state_ds, "w")
    # pylbm forward model outputs velocity in m/s; LBM restart expects lattice units
    scale_to_lattice = 1.0 / VELOCITY_SCALE_TO_PHYSICAL
    u_zyx = (u_zyx * scale_to_lattice).astype(np.float32)
    v_zyx = (v_zyx * scale_to_lattice).astype(np.float32)
    w_zyx = (w_zyx * scale_to_lattice).astype(np.float32)
    fluid_mask_zyx = _get_state_blanking_mask(state_ds)

    nz, ny, nx = rho_zyx.shape
    if (
        u_zyx.shape != (nz, ny, nx)
        or v_zyx.shape != (nz, ny, nx)
        or w_zyx.shape != (nz, ny, nx)
    ):
        raise ValueError(
            "State variables rho/u/v/w must all have matching (z,y,x) shape."
        )

    nx_cfg = _get_mod_dimensions_int(dirs, "nx", nx)
    ny_cfg = _get_mod_dimensions_int(dirs, "ny", ny)
    nz_cfg = _get_mod_dimensions_int(dirs, "nz", nz)
    ntiles_cfg = _get_mod_dimensions_int(dirs, "ntiles", 1)
    ntracer_cfg = _get_mod_dimensions_int(dirs, "ntracer", 0)
    if (nx, ny, nz) != (nx_cfg, ny_cfg, nz_cfg):
        raise ValueError(
            f"State shape (x={nx}, y={ny}, z={nz}) does not match compiled LBM grid "
            f"(x={nx_cfg}, y={ny_cfg}, z={nz_cfg})."
        )
    if ntiles_cfg != 1:
        raise NotImplementedError(
            "xarray-to-restart initialization currently supports ntiles=1 only."
        )

    latest_iteration = identify_latest_restart_iteration(dirs)
    if restart_iteration is None:
        restart_iteration = 1 if latest_iteration is None else latest_iteration

    infile = Infile(dirs.infile_path)
    ibnd = _read_infile_int(infile, ["ibnd"], default=1)
    jbnd = _read_infile_int(infile, ["jbnd"], default=0)
    kbnd = _read_infile_int(infile, ["kbnd"], default=22)
    ibgk = _read_infile_int(infile, ["ibgk"], default=3)
    inflow_turbulence = _read_infile_bool(
        infile, ["inflowturbulence", "lturb"], default=False
    )
    nturbines = _read_infile_int(infile, ["nturbines"], default=0)
    iablvisc = _read_infile_int(infile, ["iablvisc"], default=0)
    needs_aux_files = (
        inflow_turbulence or nturbines > 0 or iablvisc == 2 or ntracer_cfg > 0
    )

    restart_dir = _restart_dir(dirs)
    restart_dir.mkdir(parents=True, exist_ok=True)

    if (
        needs_aux_files
        and latest_iteration is not None
        and restart_iteration != latest_iteration
    ):
        _copy_auxiliary_restart_files(
            dirs=dirs,
            source_iteration=latest_iteration,
            target_iteration=restart_iteration,
        )
    elif needs_aux_files and latest_iteration is None:
        raise NotImplementedError(
            "This configuration requires auxiliary restart files "
            "(turbulence/theta/pottemp/tracer), but no existing restart template "
            "was found. Run one cold simulation first to generate template files."
        )

    # Prefer template-based update to preserve consistent ghost/boundary data.
    template_f = None
    if latest_iteration is not None:
        template_restart = restart_dir / f"restart_0000_{latest_iteration:06d}.uf"
        template_f = _try_load_restart_distribution(
            restart_file=template_restart,
            nx=nx,
            ny=ny,
            nz=nz,
        )

    # Convert from (z,y,x) to (x,y,z)
    rho_xyz_interior = np.transpose(rho_zyx, (2, 1, 0)).astype(np.float32)
    u_xyz_interior = np.transpose(u_zyx, (2, 1, 0)).astype(np.float32)
    v_xyz_interior = np.transpose(v_zyx, (2, 1, 0)).astype(np.float32)
    w_xyz_interior = np.transpose(w_zyx, (2, 1, 0)).astype(np.float32)

    shape_xyz = (nx + 2, ny + 2, nz + 2)
    rho_xyz = np.zeros(shape_xyz, dtype=np.float32)
    u_xyz = np.zeros(shape_xyz, dtype=np.float32)
    v_xyz = np.zeros(shape_xyz, dtype=np.float32)
    w_xyz = np.zeros(shape_xyz, dtype=np.float32)
    rho_xyz[1:-1, 1:-1, 1:-1] = rho_xyz_interior
    u_xyz[1:-1, 1:-1, 1:-1] = u_xyz_interior
    v_xyz[1:-1, 1:-1, 1:-1] = v_xyz_interior
    w_xyz[1:-1, 1:-1, 1:-1] = w_xyz_interior

    periodic_x = ibnd == 0
    periodic_y = jbnd == 0
    periodic_z = kbnd == 0
    _fill_ghost_cells(rho_xyz, periodic_x, periodic_y, periodic_z)
    _fill_ghost_cells(u_xyz, periodic_x, periodic_y, periodic_z)
    _fill_ghost_cells(v_xyz, periodic_x, periodic_y, periodic_z)
    _fill_ghost_cells(w_xyz, periodic_x, periodic_y, periodic_z)

    feq = _build_equilibrium_restart_distribution(
        rho_xyz=rho_xyz,
        u_xyz=u_xyz,
        v_xyz=v_xyz,
        w_xyz=w_xyz,
        ibgk=ibgk,
    )
    feq = np.asfortranarray(feq.astype(np.float32))

    if template_f is not None:
        # Preserve non-equilibrium content from template restart for better stability:
        # f_new = f_template - feq(template_macro) + feq(target_macro)
        rho_t, u_t, v_t, w_t = _compute_macrovars_from_distribution(template_f)
        # Build target macro fields in interior, defaulting to template values where
        # xarray values are invalid or correspond to blanked/solid cells.
        rho_target = rho_t[1:-1, 1:-1, 1:-1].copy()
        u_target = u_t[1:-1, 1:-1, 1:-1].copy()
        v_target = v_t[1:-1, 1:-1, 1:-1].copy()
        w_target = w_t[1:-1, 1:-1, 1:-1].copy()

        valid_mask = (
            np.isfinite(rho_xyz_interior)
            & np.isfinite(u_xyz_interior)
            & np.isfinite(v_xyz_interior)
            & np.isfinite(w_xyz_interior)
            & (rho_xyz_interior > 1e-6)
        )
        if fluid_mask_zyx is not None:
            fluid_mask_xyz = np.transpose(fluid_mask_zyx, (2, 1, 0))
            valid_mask = valid_mask & fluid_mask_xyz

        # Apply bounded increments toward xarray target to avoid restart shocks.
        rho_delta = rho_xyz_interior - rho_target
        u_delta = u_xyz_interior - u_target
        v_delta = v_xyz_interior - v_target
        w_delta = w_xyz_interior - w_target

        rho_delta = np.clip(rho_delta, -0.02, 0.02)
        u_delta = np.clip(u_delta, -0.01, 0.01)
        v_delta = np.clip(v_delta, -0.01, 0.01)
        w_delta = np.clip(w_delta, -0.01, 0.01)

        blend = 0.5
        rho_target[valid_mask] = rho_target[valid_mask] + blend * rho_delta[valid_mask]
        u_target[valid_mask] = u_target[valid_mask] + blend * u_delta[valid_mask]
        v_target[valid_mask] = v_target[valid_mask] + blend * v_delta[valid_mask]
        w_target[valid_mask] = w_target[valid_mask] + blend * w_delta[valid_mask]

        # Keep values in a numerically sane range for restart consistency.
        rho_target = np.clip(rho_target, 1e-6, np.inf).astype(np.float32)
        u_target = np.clip(u_target, -0.25, 0.25).astype(np.float32)
        v_target = np.clip(v_target, -0.25, 0.25).astype(np.float32)
        w_target = np.clip(w_target, -0.25, 0.25).astype(np.float32)

        rho_xyz[1:-1, 1:-1, 1:-1] = rho_target
        u_xyz[1:-1, 1:-1, 1:-1] = u_target
        v_xyz[1:-1, 1:-1, 1:-1] = v_target
        w_xyz[1:-1, 1:-1, 1:-1] = w_target
        _fill_ghost_cells(rho_xyz, periodic_x, periodic_y, periodic_z)
        _fill_ghost_cells(u_xyz, periodic_x, periodic_y, periodic_z)
        _fill_ghost_cells(v_xyz, periodic_x, periodic_y, periodic_z)
        _fill_ghost_cells(w_xyz, periodic_x, periodic_y, periodic_z)

        feq = _build_equilibrium_restart_distribution(
            rho_xyz=rho_xyz,
            u_xyz=u_xyz,
            v_xyz=v_xyz,
            w_xyz=w_xyz,
            ibgk=ibgk,
        )

        feq_template = _build_equilibrium_restart_distribution(
            rho_xyz=rho_t,
            u_xyz=u_t,
            v_xyz=v_t,
            w_xyz=w_t,
            ibgk=ibgk,
        )
        f_new = np.asfortranarray(template_f.astype(np.float32))
        f_new[:, 1:-1, 1:-1, 1:-1] = (
            template_f[:, 1:-1, 1:-1, 1:-1]
            - feq_template[:, 1:-1, 1:-1, 1:-1]
            + feq[:, 1:-1, 1:-1, 1:-1]
        )
        feq = np.asfortranarray(f_new.astype(np.float32))

    restart_file = restart_dir / f"restart_0000_{restart_iteration:06d}.uf"
    with FortranFile(str(restart_file), "w") as f:
        f.write_record(
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.int32(27),
            np.ravel(feq, order="F").astype(np.float32),
        )

    return restart_iteration
