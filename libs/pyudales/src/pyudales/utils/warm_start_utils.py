"""Utilities for handling warm start settings for uDALES."""

import logging
import os
import pathlib
import pdb
import re
import shutil

import numpy as np
from xarray import Dataset

from .dir_utils import DirectoryPaths
from .namoptions_utils import NamoptionsFile

logger = logging.getLogger(__name__)


def _fill_halo_cells(
    ws_3d: np.ndarray,
    offset_x: int,
    offset_y: int,
    offset_z: int,
    itot: int,
    jtot: int,
    ktot: int,
    bc_periodic_x: bool,
    bc_periodic_y: bool,
) -> None:
    """
    Fill halo cells in a 3D warmstart array based on boundary conditions.

    This modifies ws_3d in-place to ensure halo cells are consistent with
    the interior after any modifications (e.g., perturbations).

    For periodic boundaries: halos are filled from the opposite side of the interior.
    For non-periodic boundaries: halos are copied from the nearest interior value.

    Args:
        ws_3d: 3D array in Fortran order (x, y, z) to modify in-place.
        offset_x: Starting x-index of interior region.
        offset_y: Starting y-index of interior region.
        offset_z: Starting z-index of interior region.
        itot: Size of interior in x-direction.
        jtot: Size of interior in y-direction.
        ktot: Size of interior in z-direction.
        bc_periodic_x: True if x-direction uses periodic boundary conditions.
        bc_periodic_y: True if y-direction uses periodic boundary conditions.
    """
    # Interior index bounds
    ix_start, ix_end = offset_x, offset_x + itot
    iy_start, iy_end = offset_y, offset_y + jtot
    iz_start, iz_end = offset_z, offset_z + ktot

    # Get array shape
    nx, ny, nz = ws_3d.shape

    # Fill x-direction halos (left and right)
    if bc_periodic_x:
        # Periodic: copy from opposite side of interior
        # Left halo gets data from right side of interior
        for i in range(offset_x):
            src_i = ix_end - offset_x + i  # Map to right side of interior
            ws_3d[i, iy_start:iy_end, iz_start:iz_end] = ws_3d[
                src_i, iy_start:iy_end, iz_start:iz_end
            ]
        # Right halo gets data from left side of interior
        for i in range(ix_end, nx):
            src_i = ix_start + (i - ix_end)  # Map to left side of interior
            ws_3d[i, iy_start:iy_end, iz_start:iz_end] = ws_3d[
                src_i, iy_start:iy_end, iz_start:iz_end
            ]
    else:
        # Non-periodic: copy from nearest interior value (Neumann-like)
        # Left halo: copy from leftmost interior
        for i in range(offset_x):
            ws_3d[i, iy_start:iy_end, iz_start:iz_end] = ws_3d[
                ix_start, iy_start:iy_end, iz_start:iz_end
            ]
        # Right halo: copy from rightmost interior
        for i in range(ix_end, nx):
            ws_3d[i, iy_start:iy_end, iz_start:iz_end] = ws_3d[
                ix_end - 1, iy_start:iy_end, iz_start:iz_end
            ]

    # Fill y-direction halos (front and back)
    # Note: Now include x-halos since they're already filled
    if bc_periodic_y:
        # Periodic: copy from opposite side of interior
        for j in range(offset_y):
            src_j = iy_end - offset_y + j
            ws_3d[:, j, iz_start:iz_end] = ws_3d[:, src_j, iz_start:iz_end]
        for j in range(iy_end, ny):
            src_j = iy_start + (j - iy_end)
            ws_3d[:, j, iz_start:iz_end] = ws_3d[:, src_j, iz_start:iz_end]
    else:
        # Non-periodic: copy from nearest interior value
        for j in range(offset_y):
            ws_3d[:, j, iz_start:iz_end] = ws_3d[:, iy_start, iz_start:iz_end]
        for j in range(iy_end, ny):
            ws_3d[:, j, iz_start:iz_end] = ws_3d[:, iy_end - 1, iz_start:iz_end]

    # Fill z-direction halos (bottom and top)
    # Z typically has different handling - usually copy from nearest (Neumann at ground)
    # Note: Include both x and y halos since they're now filled
    for k in range(offset_z):
        ws_3d[:, :, k] = ws_3d[:, :, iz_start]
    for k in range(iz_end, nz):
        ws_3d[:, :, k] = ws_3d[:, :, iz_end - 1]


def set_trestart(dirs: DirectoryPaths) -> None:
    """
    Set trestart to runtime in the namoptions file.

    Sets trestart = runtime in the &RUN section of the namoptions file.
    This is used for warm restart functionality where the restart time
    should match the runtime of the previous simulation.

    Args:
        dirs: DirectoryPaths instance containing experiment_dir and experiment_name.
    """
    namoptions_path = dirs.experiment_dir / f"namoptions.{dirs.experiment_name}"

    if not namoptions_path.exists():
        logger.warning(
            f"namoptions.{dirs.experiment_name} not found, skipping trestart update"
        )
        return

    namoptions = NamoptionsFile(namoptions_path)

    # Get runtime value from &RUN section
    runtime_value = namoptions.get_value("RUN", "runtime")

    if runtime_value is not None:
        # Clean up the runtime value (remove trailing period if present, but preserve decimals)
        runtime_clean = runtime_value.rstrip(".")
        if "." in runtime_value:
            runtime_clean = runtime_value

        # Set trestart to runtime in &RUN section
        namoptions.set_value("RUN", "trestart", runtime_clean)
        namoptions.write()

        logger.info(
            f"Updated trestart to {runtime_clean} (runtime) in namoptions.{dirs.experiment_name}"
        )
    else:
        logger.warning(
            f"runtime not found in &RUN section of namoptions.{dirs.experiment_name}, "
            "cannot set trestart"
        )


def identify_warmstart_file(
    dirs: DirectoryPaths,
) -> str:
    """
    Identify the warmstart file and return it in x-format.

    Returns the warmstart filename in the format 'initd{timestamp}_xxx_xxx.{experiment_name}'
    where xxx_xxx is a wildcard pattern that matches processor numbers.

    Args:
        dirs: DirectoryPaths instance containing output_dir and experiment_name.

    Returns:
        Warmstart filename in x-format (e.g., 'initd00000440_xxx_xxx.300').

    Raises:
        ValueError: If no warmstart file is found in output_dir/{experiment_name}.
    """
    # Pattern to match and extract timestamp: initd{timestamp}_{proc1}_{proc2}.{experiment_name}
    pattern = re.compile(rf"^initd(\d+)_\d+_\d+\.{re.escape(dirs.experiment_name)}$")
    output_experiment_dir = dirs.output_dir.joinpath(dirs.experiment_name)

    if not output_experiment_dir.exists():
        os.makedirs(output_experiment_dir, exist_ok=True)

    for item in output_experiment_dir.iterdir():
        if item.is_file():
            match = pattern.match(item.name)
            if match:
                timestamp = match.group(1)
                # Return in x-format: initd{timestamp}_xxx_xxx.{experiment_name}
                # return f"initd{timestamp}_xxx_xxx.{dirs.experiment_name}"
                return f"initd{timestamp}_000_000.{dirs.experiment_name}"

    # Try copying a warmstart file from the default directory if none found
    warmstart_dir = dirs.cwd / "libs" / "pyudales" / "warmstart_files"
    if os.path.exists(warmstart_dir):
        for fname in os.listdir(warmstart_dir):
            # Match files with .nc or .300 (suffix is not known; can be parametric)
            if fname.startswith("initd") and ("." in fname):
                src_file = os.path.join(warmstart_dir, fname)
                # New file with same basename but replace suffix after last '.' with dirs.experiment_name
                parts = fname.rsplit(".", 1)
                if len(parts) == 2:
                    new_fname = f"{parts[0]}.{dirs.experiment_name}"
                else:
                    new_fname = fname  # fallback
                dst_file = output_experiment_dir / new_fname
                shutil.copy2(src_file, dst_file)
                # Return in x-format: e.g., initd{timestamp}_000_000.{dirs.experiment_name}
                match = re.match(r"^initd(\d+)_\d+_\d+\.", new_fname)
                if match:
                    timestamp = match.group(1)
                    return f"initd{timestamp}_000_000.{dirs.experiment_name}"
                else:
                    # fallback: return the filename just copied
                    return new_fname

    raise ValueError(f"No warmstart file found in {output_experiment_dir}")


def set_warm_start(
    dirs: DirectoryPaths,
) -> None:
    """
    Set warm start settings in the namoptions file.

    Looks for warmstart files in output_dir/{experiment_name} (actual files with processor numbers),
    extracts the timestamp, and writes the x-format to namoptions.

    Sets lwarmstart to .true. and startfile to the pattern matching warmstart files.
    The startfile format is 'initd{timestamp}_xxx_xxx.<experiment_name>' where
    xxx_xxx is a wildcard pattern that matches processor numbers.

    Args:
        dirs: DirectoryPaths instance containing experiment_dir, output_dir, and experiment_name.
    """
    namoptions_path = dirs.experiment_dir / f"namoptions.{dirs.experiment_name}"

    if not namoptions_path.exists():
        logger.warning(
            f"namoptions.{dirs.experiment_name} not found, skipping warm start setup"
        )
        return

    namoptions = NamoptionsFile(namoptions_path)

    # Look for actual warmstart files in output_dir/{experiment_name}
    # Pattern to match actual files: initd{timestamp}_{proc1}_{proc2}.{experiment_name}
    pattern = re.compile(rf"^initd(\d+)_\d+_\d+\.{re.escape(dirs.experiment_name)}$")

    output_experiment_dir = dirs.output_dir / dirs.experiment_name

    if not output_experiment_dir.exists():
        logger.warning(
            f"Output experiment directory {output_experiment_dir} does not exist, "
            "cannot find warmstart files"
        )
        return

    # Find a warmstart file to extract the timestamp
    warmstart_file = None
    for item in output_experiment_dir.iterdir():
        if item.is_file():
            match = pattern.match(item.name)
            if match:
                warmstart_file = item.name
                timestamp = match.group(1)
                break

    if warmstart_file is None:
        logger.warning(
            f"No warmstart files found in {output_experiment_dir}, cannot set startfile"
        )
        return

    # Create x-format startfile value
    startfile_value = f"initd{timestamp}_xxx_xxx.{dirs.experiment_name}"

    # Set lwarmstart to .true.
    namoptions.set_value("RUN", "lwarmstart", ".true.")

    # Set startfile (with quotes as it's a string value)
    namoptions.set_value("RUN", "startfile", f"'{startfile_value}'")

    namoptions.write()

    logger.info(
        f"Set lwarmstart = .true. and startfile = '{startfile_value}' "
        f"in namoptions.{dirs.experiment_name}"
    )


def move_warmstart_files(
    dirs: DirectoryPaths,
    warmstart_dir: pathlib.Path,
) -> None:
    """
    Move warmstart files to the warmstart directory.

    When trestart is enabled, the model generates files in the format:
    'initd00000440_000_000.<experiment_name>'

    This function finds all files matching this pattern in the output_dir
    and moves them to the warmstart_dir.

    Args:
        dirs: DirectoryPaths instance containing output_dir and experiment_name.
        warmstart_dir: Directory where warmstart files should be moved.
    """
    if not dirs.output_dir.exists():
        logger.warning(
            f"Output directory {dirs.output_dir} does not exist, "
            "cannot move warmstart files"
        )
        return

    # Create warmstart directory if it doesn't exist
    warmstart_dir.mkdir(parents=True, exist_ok=True)

    # Pattern: initd followed by digits, then _digits_digits., then experiment_name
    # Example: initd00000440_000_000.300
    pattern = re.compile(rf"^initd\d+_\d+_\d+\.{re.escape(dirs.experiment_name)}$")

    output_experiment_dir = dirs.output_dir.joinpath(dirs.experiment_name)

    moved_files = []
    for item in output_experiment_dir.iterdir():
        if item.is_file() and pattern.match(item.name):
            target_path = warmstart_dir / item.name
            # Remove target if it exists
            if target_path.exists():
                target_path.unlink()
            shutil.move(str(item), str(target_path))
            moved_files.append(item.name)

    if moved_files:
        logger.info(
            f"Moved {len(moved_files)} warmstart file(s) to {warmstart_dir}: "
            f"{', '.join(moved_files)}"
        )
    else:
        logger.debug(
            f"No warmstart files found matching pattern 'initd*_*_*.{dirs.experiment_name}' "
            f"in {dirs.output_dir}"
        )


def clean_output_except_warmstart_files(
    dirs: DirectoryPaths,
) -> None:
    """
    Remove all files in output_dir/{experiment_name} except for warmstart files.

    This function keeps only files matching the warmstart pattern:
    'initd{timestamp}_{proc1}_{proc2}.{experiment_name}'

    All other files and directories in the output experiment directory are removed.

    Args:
        dirs: DirectoryPaths instance containing output_dir and experiment_name.
    """
    output_experiment_dir = dirs.output_dir / dirs.experiment_name

    if not output_experiment_dir.exists():
        logger.debug(
            f"Output experiment directory {output_experiment_dir} does not exist, "
            "nothing to clean"
        )
        return

    # Pattern to match warmstart files: initd{timestamp}_{proc1}_{proc2}.{experiment_name}
    warmstart_pattern = re.compile(
        rf"^initd\d+_\d+_\d+\.{re.escape(dirs.experiment_name)}$"
    )

    removed_files = []
    kept_files = []

    for item in output_experiment_dir.iterdir():
        if item.is_file():
            if warmstart_pattern.match(item.name):
                kept_files.append(item.name)
            else:
                item.unlink(missing_ok=True)
                removed_files.append(item.name)
        elif item.is_dir():
            shutil.rmtree(item)
            removed_files.append(f"{item.name}/ (directory)")

    if removed_files:
        logger.info(
            f"Removed {len(removed_files)} file(s) from {output_experiment_dir}, "
            f"kept {len(kept_files)} warmstart file(s): {', '.join(kept_files) if kept_files else 'none'}"
        )
    else:
        logger.debug(
            f"No files to remove in {output_experiment_dir}, "
            f"kept {len(kept_files)} warmstart file(s)"
        )


def remove_old_warmstart_files(
    dirs: DirectoryPaths,
) -> None:
    """
    Remove old warmstart files, keeping only the newest ones.

    The timestamp following 'initd' in warmstart filenames is always increasing.
    This function finds all warmstart files, identifies the maximum timestamp,
    and removes all files with timestamps less than the maximum.

    Args:
        dirs: DirectoryPaths instance containing output_dir and experiment_name.
    """
    output_experiment_dir = dirs.output_dir / dirs.experiment_name

    if not output_experiment_dir.exists():
        logger.debug(
            f"Output experiment directory {output_experiment_dir} does not exist, "
            "nothing to clean"
        )
        return

    # Pattern to match warmstart files and extract timestamp: initd{timestamp}_{proc1}_{proc2}.{experiment_name}
    warmstart_pattern = re.compile(
        rf"^initd(\d+)_\d+_\d+\.{re.escape(dirs.experiment_name)}$"
    )

    # Find all warmstart files and extract their timestamps
    warmstart_files = []
    for item in output_experiment_dir.iterdir():
        if item.is_file():
            match = warmstart_pattern.match(item.name)
            if match:
                timestamp = int(match.group(1))
                warmstart_files.append((timestamp, item))

    if not warmstart_files:
        logger.debug(f"No warmstart files found in {output_experiment_dir}")
        return

    # Find the maximum timestamp
    max_timestamp = max(timestamp for timestamp, _ in warmstart_files)

    # Remove files with timestamps less than the maximum
    removed_files = []
    kept_files = []

    for timestamp, item in warmstart_files:
        if timestamp < max_timestamp:
            item.unlink(missing_ok=True)
            removed_files.append(item.name)
        else:
            kept_files.append(item.name)

    if removed_files:
        logger.info(
            f"Removed {len(removed_files)} old warmstart file(s) from {output_experiment_dir}, "
            f"kept {len(kept_files)} newest file(s) with timestamp {max_timestamp}"
        )
    else:
        logger.debug(
            f"All warmstart files in {output_experiment_dir} are already the newest "
            f"(timestamp {max_timestamp}), nothing to remove"
        )


def update_warmstart_file_from_xarray(
    state: Dataset,
    dirs: DirectoryPaths,
    warmstart_file: pathlib.Path,
) -> None:
    """Update an existing warmstart file with flow variables from xarray.

    This function takes an existing warmstart file (from a previous uDALES run),
    reads its structure, updates only the flow variables (u0, v0, w0, pres0)
    from the xarray state, and writes it with a new timestamp.

    This approach is more robust than creating warmstart files from scratch
    because uDALES uses 2DECOMP&FFT library which has complex memory layouts
    that are difficult to replicate in Python.

    IMPORTANT: Flow variable injection is currently limited due to grid mismatch:
    - Fielddump (xarray source): Contains interior cells only (itot × jtot × ktot)
    - Warmstart (target): Contains interior + halos + 2DECOMP&FFT padding

    For example, a 128×128×8 grid may have:
    - Fielddump: 128×128×8 = 131,072 elements
    - Warmstart: 195×195×8 = 304,200 elements (includes halos and FFT padding)

    When shapes don't match, the function keeps template values for flow variables
    but still updates the time record (timee, dt).

    Args:
        state: An xarray.Dataset containing the flow variables to update.
        dirs: A DirectoryPaths object containing the paths for the simulation.
        warmstart_file: The name of the warmstart file to update (used to extract timestamp).
    """
    import glob

    import numpy as np
    from scipy.io import FortranFile

    if "time" in state.dims and len(state.time) > 1:
        logger.info(
            "Multiple time steps found in state, selecting the last one for warmstart."
        )
        state = state.isel(time=-1)

    output_experiment_dir = dirs.output_dir / dirs.experiment_name

    # Find an existing warmstart file to use as template
    # template_pattern = str(
    #     output_experiment_dir / f"initd*_000_000.{dirs.experiment_name}"
    # )
    # template_files = sorted(glob.glob(template_pattern))

    # if not template_files:
    #     raise FileNotFoundError(
    #         f"No existing warmstart file found matching {template_pattern}. "
    #         "Run a cold start first to generate the warmstart file structure."
    #     )

    # # Use the most recent warmstart file as template
    # template_file = template_files[-1]
    # logger.info(f"Using {template_file} as template for warmstart file")

    # Read all records from template
    # Note: uDALES uses double precision (REAL*8 / float64), not single precision
    records = []
    with FortranFile(str(warmstart_file), "r") as f:
        try:
            while True:
                rec = f.read_record(dtype=np.float64)
                records.append(rec)
        except Exception:
            pass  # End of file

    if len(records) < 13:
        raise ValueError(
            f"Template file has only {len(records)} records, expected at least 13"
        )

    # Get the shapes from the template
    # Records: 0=mindist, 1=wall, 2=u0, 3=v0, 4=w0, 5=pres0, 6=thl0, 7=e120, 8=ekm, 9=qt0, 10=ql0, 11=ql0h, 12=timee/dt
    flow_var_shape = records[2].shape[0]  # Shape of u0

    # Variable name mapping
    name_map = {"u0": "u", "v0": "v", "w0": "w", "pres0": "pres"}

    # Try to determine grid dimensions from namoptions
    namoptions_path = dirs.experiment_dir / f"namoptions.{dirs.experiment_name}"
    namoptions = NamoptionsFile(namoptions_path)
    itot = int(namoptions.get_value("DOMAIN", "itot") or 128)
    jtot = int(namoptions.get_value("DOMAIN", "jtot") or 128)
    ktot = int(namoptions.get_value("DOMAIN", "ktot") or 8)

    # Read boundary conditions (1 = periodic, 2 = profile/inflow, 3 = driver)
    # Default to periodic (1) if not specified
    bcxm = int(namoptions.get_value("INPS", "BCxm") or 1)
    bcym = int(namoptions.get_value("INPS", "BCym") or 1)
    bc_periodic_x = bcxm == 1
    bc_periodic_y = bcym == 1

    # Calculate warmstart dimensions by factorizing the array size
    # The warmstart is flattened in column-major (Fortran) order: (nx, ny, nz)
    ws_size = flow_var_shape
    fd_size = itot * jtot * ktot

    # Try to find warmstart dimensions that make sense
    # Warmstart z-dimension should match ktot (or ktot+1 for kh=1)
    ws_nz = ktot  # Most likely
    if ws_size % ws_nz != 0:
        ws_nz = ktot + 1  # Try with kh=1

    if ws_size % ws_nz == 0:
        ws_xy = ws_size // ws_nz
        ws_nx = int(np.sqrt(ws_xy))
        # Try to find exact square root or nearby factors
        while ws_nx > 0 and ws_nx * ws_nx != ws_xy:
            if ws_xy % ws_nx == 0:
                ws_ny = ws_xy // ws_nx
                if abs(ws_nx - ws_ny) < 10:  # Accept near-square shapes
                    break
            ws_nx -= 1
        else:
            ws_nx = int(np.sqrt(ws_xy))
            ws_ny = ws_nx
    else:
        ws_nx, ws_ny, ws_nz = 0, 0, 0

    logger.info(
        f"Grid dimensions - fielddump: {itot}x{jtot}x{ktot}={fd_size}, "
        f"warmstart: {ws_nx}x{ws_ny}x{ws_nz}={ws_size}"
    )

    # Calculate offsets for placing interior data within warmstart array
    # The interior is centered within the warmstart array (halo on both sides)
    if ws_nx > 0 and ws_nx >= itot and ws_ny >= jtot:
        offset_x = (ws_nx - itot) // 2
        offset_y = (ws_ny - jtot) // 2
        offset_z = 0  # z typically starts at kb=1 in both
        can_map = True
        logger.info(
            f"Grid mapping: interior at offset ({offset_x}, {offset_y}, {offset_z})"
        )
    else:
        can_map = False
        logger.warning(
            f"Cannot determine grid mapping, will use template values for flow variables"
        )

    # Update flow variables from xarray state
    for idx, var_name in enumerate(["u0", "v0", "w0", "pres0"], start=2):
        xr_name = name_map.get(var_name, var_name)
        if xr_name in state:
            # Get the data
            fd_data = state[xr_name].values.astype(np.float64)

            # Handle any extra dimensions (squeeze out singleton dimensions)
            while fd_data.ndim > 3:
                fd_data = (
                    fd_data.squeeze(axis=0) if fd_data.shape[0] == 1 else fd_data[-1]
                )

            # The xarray data may have different shape, need to map to warmstart grid
            if fd_data.size == flow_var_shape:
                # Direct match - just use it
                records[idx] = fd_data.flatten()
                logger.info(f"Updated {var_name} from xarray state (direct match)")
            elif can_map and fd_data.size == fd_size:
                # Grid mapping: place fielddump interior into warmstart array
                #
                # Fortran writes warmstart with: (((u0(i,j,k), i=...), j=...), k=...)
                # This means i (x) varies fastest, then j (y), then k (z)
                # So the flattened array is in Fortran order (column-major)
                #
                # xarray/NetCDF stores as (time, z, y, x), so after squeezing time:
                # fd_data shape is (ktot, jtot, itot) with C ordering
                #
                # To convert: reshape warmstart with Fortran order to get (nx, ny, nz)

                # Reshape warmstart template to 3D using Fortran order
                # Shape is (nx, ny, nz) = (ws_nx, ws_ny, ws_nz) with Fortran memory layout
                ws_3d = records[idx].reshape((ws_nx, ws_ny, ws_nz), order="F")

                # Reshape fielddump data - it's in C order (z, y, x) from xarray
                # Need to convert to Fortran order (x, y, z)
                fd_3d = fd_data.reshape((ktot, jtot, itot))  # (z, y, x) in C order
                fd_3d_fortran = np.asfortranarray(
                    fd_3d.transpose(2, 1, 0)
                )  # -> (x, y, z) Fortran order

                # Copy interior data to the warmstart array
                # The interior region: [offset_x:offset_x+itot, offset_y:offset_y+jtot, offset_z:offset_z+ktot]
                ws_3d[
                    offset_x : offset_x + itot,
                    offset_y : offset_y + jtot,
                    offset_z : offset_z + ktot,
                ] = fd_3d_fortran

                # Fill halo cells based on boundary conditions
                # This ensures consistency after perturbations to the interior
                _fill_halo_cells(
                    ws_3d,
                    offset_x,
                    offset_y,
                    offset_z,
                    itot,
                    jtot,
                    ktot,
                    bc_periodic_x,
                    bc_periodic_y,
                )

                # Flatten back to 1D (Fortran order)
                records[idx] = ws_3d.flatten(order="F")
                logger.info(f"Updated {var_name} from xarray state (with grid mapping)")
            else:
                # Shape mismatch and can't map
                logger.warning(
                    f"Shape mismatch for {var_name}: xarray has {fd_data.size} elements "
                    f"(expected {fd_size} for interior), template has {flow_var_shape} elements. "
                    f"Using template values."
                )
        else:
            logger.debug(
                f"Variable {xr_name} not found in state, keeping template values"
            )

    # Update timee from state if available
    if "time" in state.coords:
        timee = float(state.time.values)
    elif "time" in state:
        timee = float(state.time.values)
    else:
        timee = 0.0

    # Get dt from namoptions (reuse the namoptions object from earlier)
    dtmax_str = namoptions.get_value("RUN", "dtmax")
    dt = float(dtmax_str) if dtmax_str is not None else 1.0

    # Update timee and dt in record 12 (preserve any additional values in the record)
    # Record 12 may have more than 2 elements - only update the first two
    if records[12].size >= 2:
        records[12][0] = np.float64(timee)
        records[12][1] = np.float64(dt)
        logger.info(f"Updated timee={timee}, dt={dt} in time record")
    else:
        logger.warning(
            f"Record 12 has unexpected size {records[12].size}, keeping original values"
        )

    # Write new warmstart file
    # new_file = output_experiment_dir / warmstart_file.name
    new_file = warmstart_file
    with FortranFile(str(new_file), "w") as f:
        for rec in records:
            f.write_record(rec)

    logger.info(f"Wrote updated warmstart file: {new_file}")


def write_warmstart_file_from_xarray(
    state: Dataset,
    dirs: DirectoryPaths,
    ntrun: int,
) -> None:
    """Write warmstart files from an xarray Dataset for all processors.

    NOTE: This function attempts to create warmstart files from scratch,
    which may not work correctly due to 2DECOMP&FFT library memory layouts.
    For more reliable warm starts, use update_warmstart_file_from_xarray()
    which modifies existing warmstart files created by uDALES.

    This function mimics the behavior of the `writerestartfiles` subroutine
    in `modsave.f90` of the u-dales source code. It creates Fortran
    unformatted binary files that can be used as warmstart files for a
    u-dales simulation.

    The function reads nprocx and nprocy from namoptions and creates one
    warmstart file per processor, decomposing the full domain state into
    sub-domains with appropriate halo regions.

    Args:
        state: An xarray.Dataset containing the variables to write.
            The variable names should match the Fortran variables in u-dales,
            or common aliases (e.g., 'u' for 'u0'). The data arrays are
            expected to have dimensions in (z, y, x) order and represent
            the FULL domain (not decomposed).
        dirs: A DirectoryPaths object containing the paths for the simulation.
        ntrun: The run number (timestamp) for the warmstart file.
    """
    import numpy as np
    from scipy.io import FortranFile

    if "time" in state.dims and len(state.time) > 1:
        logger.info(
            "Multiple time steps found in state, selecting the last one for warmstart."
        )
        state = state.isel(time=-1)

    namoptions_path = dirs.experiment_dir / f"namoptions.{dirs.experiment_name}"
    if not namoptions_path.exists():
        raise FileNotFoundError(f"Namoptions file not found at {namoptions_path}")
    namoptions = NamoptionsFile(namoptions_path)

    # Get processor grid configuration
    nprocx_str = namoptions.get_value("RUN", "nprocx")
    nprocy_str = namoptions.get_value("RUN", "nprocy")
    nprocx = int(nprocx_str) if nprocx_str is not None else 1
    nprocy = int(nprocy_str) if nprocy_str is not None else 1

    nsv_str = namoptions.get_value("SCALARS", "nsv")
    nsv = int(nsv_str) if nsv_str is not None else 0
    dtmax_str = namoptions.get_value("RUN", "dtmax")
    dtmax = float(dtmax_str) if dtmax_str is not None else 1.0

    # Get full domain dimensions
    itot = int(namoptions.get_value("DOMAIN", "itot"))  # type: ignore[arg-type]
    jtot = int(namoptions.get_value("DOMAIN", "jtot"))  # type: ignore[arg-type]
    ktot = int(namoptions.get_value("DOMAIN", "ktot"))  # type: ignore[arg-type]

    # Calculate per-processor domain size (without halos)
    imax = itot // nprocx
    jmax = jtot // nprocy

    # Halo sizes (standard uDALES values)
    ih = 3  # halo in x-direction
    jh = 3  # halo in y-direction
    kh = 1  # halo in z-direction

    # Per-processor shapes
    # Note: In uDALES, the k-dimension only has a top halo (ke+kh), not a bottom halo
    # This is because kb=1 (ground level) and there are no cells below ground
    shape_3d_no_halos_proc = (ktot, jmax, imax)
    shape_3d_with_halos_proc = (ktot + kh, jmax + 2 * jh, imax + 2 * ih)

    # Variable name mapping (Fortran name -> xarray name)
    name_map = {"u0": "u", "v0": "v", "w0": "w", "pres0": "pres", "timee": "time"}

    def get_full_var_array(name: str) -> np.ndarray | None:
        """Get the full domain variable array from state."""
        if name in state:
            return state[name].values
        if name in name_map and name_map[name] in state:
            return state[name_map[name]].values
        return None

    def decompose_and_add_halos(
        full_array: np.ndarray | None,
        myidx: int,
        myidy: int,
        shape_no_halos: tuple,
        shape_with_halos: tuple,
        default_value: float = 0.0,
        n_dim: int = 0,
    ) -> np.ndarray:
        """Extract sub-domain for a processor and add halo regions.

        In uDALES, the array dimensions are:
        - k (z): from kb to ke+kh (NO bottom halo, only top halo of size kh)
        - j (y): from jb-jh to je+jh (both halos of size jh)
        - i (x): from ib-ih to ie+ih (both halos of size ih)

        Args:
            full_array: Full domain array (z, y, x) or None for default.
            myidx: Processor index in x-direction.
            myidy: Processor index in y-direction.
            shape_no_halos: Shape without halos for this processor (ktot, jmax, imax).
            shape_with_halos: Shape with halos (ktot+kh, jmax+2*jh, imax+2*ih).
            default_value: Default value if full_array is None.
            n_dim: If > 0, prepend this dimension (for multi-component fields).

        Returns:
            Sub-domain array with halos added.
        """
        kmax, jmax_local, imax_local = shape_no_halos
        kmax_h, jmax_h, imax_h = shape_with_halos

        if n_dim > 0:
            out_shape = (n_dim,) + shape_with_halos
        else:
            out_shape = shape_with_halos

        if full_array is None:
            return np.full(out_shape, default_value, dtype=np.float32)

        # Handle multi-component fields
        if n_dim > 0 and full_array.ndim == 4:
            result = np.zeros(out_shape, dtype=np.float32)
            for d in range(n_dim):
                result[d] = decompose_and_add_halos(
                    full_array[d],
                    myidx,
                    myidy,
                    shape_no_halos,
                    shape_with_halos,
                    default_value,
                    n_dim=0,
                )
            return result

        # Calculate start indices in full domain for this processor
        i_start = myidx * imax_local
        j_start = myidy * jmax_local

        # Get full domain dimensions
        kmax_full = full_array.shape[0]
        jmax_full = full_array.shape[1]
        imax_full = full_array.shape[2]

        # Create output array with halos
        result = np.zeros(shape_with_halos, dtype=np.float32)

        # Fill interior (the actual sub-domain data)
        # Note: k starts at 0 (no bottom halo), j and i start at jh and ih respectively
        # Interior indices in result: [0:kmax, jh:jh+jmax, ih:ih+imax]
        result[0:kmax, jh : jh + jmax_local, ih : ih + imax_local] = full_array[
            :kmax, j_start : j_start + jmax_local, i_start : i_start + imax_local
        ]

        # Fill halos with periodic boundary conditions
        # Note: NO bottom z-halo in uDALES (kb=1, not kb-kh)

        # Top z-halo (k >= kmax): extrapolate or use zeros
        # In uDALES, the top halo is typically zero or extrapolated
        # We'll use zeros as it's what the model expects for initialization
        # (The actual halo exchange happens during the simulation)
        for k in range(kmax, kmax_h):
            result[k, jh : jh + jmax_local, ih : ih + imax_local] = 0.0

        # Left x-halo (i < ih): get from left neighbor (periodic)
        for i in range(ih):
            i_src = (i_start - ih + i) % imax_full
            result[0:kmax, jh : jh + jmax_local, i] = full_array[
                :kmax, j_start : j_start + jmax_local, i_src
            ]

        # Right x-halo (i >= ih + imax): get from right neighbor (periodic)
        for i in range(ih + imax_local, imax_h):
            i_src = (i_start + i - ih) % imax_full
            result[0:kmax, jh : jh + jmax_local, i] = full_array[
                :kmax, j_start : j_start + jmax_local, i_src
            ]

        # Bottom y-halo (j < jh): get from bottom neighbor (periodic)
        for j in range(jh):
            j_src = (j_start - jh + j) % jmax_full
            result[0:kmax, j, ih : ih + imax_local] = full_array[
                :kmax, j_src, i_start : i_start + imax_local
            ]

        # Top y-halo (j >= jh + jmax): get from top neighbor (periodic)
        for j in range(jh + jmax_local, jmax_h):
            j_src = (j_start + j - jh) % jmax_full
            result[0:kmax, j, ih : ih + imax_local] = full_array[
                :kmax, j_src, i_start : i_start + imax_local
            ]

        # Corner halos (combinations of x and y halos)
        for i in range(ih):
            i_src = (i_start - ih + i) % imax_full
            for j in range(jh):
                j_src = (j_start - jh + j) % jmax_full
                result[0:kmax, j, i] = full_array[:kmax, j_src, i_src]
            for j in range(jh + jmax_local, jmax_h):
                j_src = (j_start + j - jh) % jmax_full
                result[0:kmax, j, i] = full_array[:kmax, j_src, i_src]

        for i in range(ih + imax_local, imax_h):
            i_src = (i_start + i - ih) % imax_full
            for j in range(jh):
                j_src = (j_start - jh + j) % jmax_full
                result[0:kmax, j, i] = full_array[:kmax, j_src, i_src]
            for j in range(jh + jmax_local, jmax_h):
                j_src = (j_start + j - jh) % jmax_full
                result[0:kmax, j, i] = full_array[:kmax, j_src, i_src]

        return result

    def get_var_scalar(name: str) -> float:
        """Helper to get a scalar value from state or create a default."""
        if name in state:
            val = state[name].values
            return float(val.item()) if hasattr(val, "item") else float(val)
        if name in name_map and name_map[name] in state:
            val = state[name_map[name]].values
            return float(val.item()) if hasattr(val, "item") else float(val)

        logger.warning(f"Variable '{name}' not found in state, using default.")
        if name == "timee":
            return 0.0
        if name == "dt":
            return dtmax
        return 0.0

    # Get full domain arrays once
    full_arrays = {
        "mindist": get_full_var_array("mindist"),
        "wall": get_full_var_array("wall"),
        "u0": get_full_var_array("u0"),
        "v0": get_full_var_array("v0"),
        "w0": get_full_var_array("w0"),
        "pres0": get_full_var_array("pres0"),
        "thl0": get_full_var_array("thl0"),
        "e120": get_full_var_array("e120"),
        "ekm": get_full_var_array("ekm"),
        "qt0": get_full_var_array("qt0"),
        "ql0": get_full_var_array("ql0"),
        "ql0h": get_full_var_array("ql0h"),
    }
    if nsv > 0:
        full_arrays["sv0"] = get_full_var_array("sv0")

    # Get scalar values
    timee = get_var_scalar("timee")
    dt = get_var_scalar("dt")

    output_experiment_dir = dirs.output_dir / dirs.experiment_name
    output_experiment_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Writing warmstart files for {nprocx}x{nprocy} = {nprocx * nprocy} processors"
    )

    # Write warmstart file for each processor
    for myidy in range(nprocy):
        for myidx in range(nprocx):
            cmyidx = f"{myidx:03d}"
            cmyidy = f"{myidy:03d}"

            name = f"initd{ntrun:08d}_{cmyidx}_{cmyidy}.{dirs.experiment_name}"
            filepath = output_experiment_dir / name

            with FortranFile(filepath, "w") as f:
                # mindist and wall use no-halo shape
                f.write_record(
                    decompose_and_add_halos(
                        full_arrays["mindist"],
                        myidx,
                        myidy,
                        shape_3d_no_halos_proc,
                        shape_3d_no_halos_proc,  # no halos for mindist
                        default_value=0.0,
                    ).astype(np.float32)
                )
                f.write_record(
                    decompose_and_add_halos(
                        full_arrays["wall"],
                        myidx,
                        myidy,
                        shape_3d_no_halos_proc,
                        shape_3d_no_halos_proc,  # no halos for wall
                        default_value=0.0,
                        n_dim=5,
                    ).astype(np.float32)
                )

                # Flow variables use with-halo shape
                f.write_record(
                    decompose_and_add_halos(
                        full_arrays["u0"],
                        myidx,
                        myidy,
                        shape_3d_no_halos_proc,
                        shape_3d_with_halos_proc,
                        default_value=0.0,
                    ).astype(np.float32)
                )
                f.write_record(
                    decompose_and_add_halos(
                        full_arrays["v0"],
                        myidx,
                        myidy,
                        shape_3d_no_halos_proc,
                        shape_3d_with_halos_proc,
                        default_value=0.0,
                    ).astype(np.float32)
                )
                f.write_record(
                    decompose_and_add_halos(
                        full_arrays["w0"],
                        myidx,
                        myidy,
                        shape_3d_no_halos_proc,
                        shape_3d_with_halos_proc,
                        default_value=0.0,
                    ).astype(np.float32)
                )
                f.write_record(
                    decompose_and_add_halos(
                        full_arrays["pres0"],
                        myidx,
                        myidy,
                        shape_3d_no_halos_proc,
                        shape_3d_with_halos_proc,
                        default_value=0.0,
                    ).astype(np.float32)
                )
                f.write_record(
                    decompose_and_add_halos(
                        full_arrays["thl0"],
                        myidx,
                        myidy,
                        shape_3d_no_halos_proc,
                        shape_3d_with_halos_proc,
                        default_value=0.0,
                    ).astype(np.float32)
                )
                f.write_record(
                    decompose_and_add_halos(
                        full_arrays["e120"],
                        myidx,
                        myidy,
                        shape_3d_no_halos_proc,
                        shape_3d_with_halos_proc,
                        default_value=5.0e-5,
                    ).astype(np.float32)
                )
                f.write_record(
                    decompose_and_add_halos(
                        full_arrays["ekm"],
                        myidx,
                        myidy,
                        shape_3d_no_halos_proc,
                        shape_3d_with_halos_proc,
                        default_value=1.5e-5,
                    ).astype(np.float32)
                )
                f.write_record(
                    decompose_and_add_halos(
                        full_arrays["qt0"],
                        myidx,
                        myidy,
                        shape_3d_no_halos_proc,
                        shape_3d_with_halos_proc,
                        default_value=0.0,
                    ).astype(np.float32)
                )
                f.write_record(
                    decompose_and_add_halos(
                        full_arrays["ql0"],
                        myidx,
                        myidy,
                        shape_3d_no_halos_proc,
                        shape_3d_with_halos_proc,
                        default_value=0.0,
                    ).astype(np.float32)
                )
                f.write_record(
                    decompose_and_add_halos(
                        full_arrays["ql0h"],
                        myidx,
                        myidy,
                        shape_3d_no_halos_proc,
                        shape_3d_with_halos_proc,
                        default_value=0.0,
                    ).astype(np.float32)
                )
                f.write_record(np.float32(timee), np.float32(dt))

            if nsv > 0:
                name_s = f"inits{ntrun:08d}_{cmyidx}_{cmyidy}.{dirs.experiment_name}"
                filepath_s = output_experiment_dir / name_s
                with FortranFile(filepath_s, "w") as f:
                    f.write_record(
                        decompose_and_add_halos(
                            full_arrays["sv0"],
                            myidx,
                            myidy,
                            shape_3d_no_halos_proc,
                            shape_3d_with_halos_proc,
                            default_value=0.0,
                            n_dim=nsv,
                        ).astype(np.float32)
                    )
                    f.write_record(np.float32(timee))

            logger.debug(f"Wrote warmstart file: {name}")
