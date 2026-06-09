import logging
import pathlib

logger = logging.getLogger(__name__)

import numpy as np
import trimesh
from pylbm.utils import DirectoryPaths


def _load_single_mesh(stl_path: str | pathlib.Path) -> trimesh.Trimesh:
    """
    Load an STL file as a single concatenated trimesh.Trimesh.

    A ``trimesh.Scene`` (multi-geometry STL) is concatenated into one mesh so
    that the occupancy ray-casting sees a single triangle soup. The building
    geometry is intentionally kept whole here: unlike the old bounding-box
    approach we do NOT split into connected components, because the voxel
    occupancy is evaluated per grid column and naturally preserves arbitrary
    footprints, courtyards and concave shapes.

    Args:
        stl_path: Path to the .stl file.

    Returns:
        A single trimesh.Trimesh containing all geometry.
    """
    loaded = trimesh.load(stl_path)

    if isinstance(loaded, trimesh.Scene):
        scene_meshes = [
            m for m in loaded.geometry.values() if isinstance(m, trimesh.Trimesh)
        ]
        logger.info("Loaded scene with %s separate meshes", len(scene_meshes))
        if len(scene_meshes) == 0:
            raise ValueError(f"No valid meshes found in {stl_path}")
        mesh = trimesh.util.concatenate(scene_meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Could not load a valid mesh from {stl_path}")

    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh loaded from {stl_path} has no faces")

    logger.info(
        "Loaded mesh with %s vertices / %s faces, bounds %s",
        len(mesh.vertices),
        len(mesh.faces),
        mesh.bounds.tolist(),
    )
    return mesh


def compute_solid_occupancy(
    stl_path: str | pathlib.Path,
    nx: int,
    ny: int,
    nz: int,
    domain_bounds: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Build a 3D boolean solid-occupancy mask over the LBM grid from an STL mesh.

    The buildings in these STLs are open-bottomed extrusions resting on z=0, so
    they are not watertight and ``mesh.contains`` is unreliable. Instead we use
    a vertical ray-cast height field: for each (i, j) grid column we cast one
    ray straight down through the column centre and take the highest mesh hit as
    that column's roof height. Cells whose centre lies at or below the roof are
    marked solid. Columns with no roof hit (open courtyards, gaps between
    buildings, the area outside the footprint) stay empty, so holes and
    arbitrary/concave footprints are preserved exactly.

    Args:
        stl_path: Path to the .stl file.
        nx, ny, nz: Number of interior grid cells in x, y, z.
        domain_bounds: Physical bounds {'xmin', 'xmax', ...}. If None, the mesh
            bounding box is used.

    Returns:
        Boolean array ``solid`` of shape (nx, ny, nz). ``solid[i, j, k]`` is True
        when interior cell (i+1, j+1, k+1) in 1-based Fortran indexing is solid.
    """
    mesh = _load_single_mesh(stl_path)

    # Determine the physical domain used for the grid mapping.
    if domain_bounds is None:
        b = mesh.bounds
        xmin, ymin, zmin = float(b[0, 0]), float(b[0, 1]), float(b[0, 2])
        xmax, ymax, zmax = float(b[1, 0]), float(b[1, 1]), float(b[1, 2])
    else:
        xmin, ymin, zmin = (
            domain_bounds["xmin"],
            domain_bounds["ymin"],
            domain_bounds["zmin"],
        )
        xmax, ymax, zmax = (
            domain_bounds["xmax"],
            domain_bounds["ymax"],
            domain_bounds["zmax"],
        )

    # Cell sizes and cell-centre coordinate convention (matches forward_model.py:
    # x_grid = (arange(nx) + 0.5) * dx + xmin).
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz

    x_centers = (np.arange(nx) + 0.5) * dx + xmin
    y_centers = (np.arange(ny) + 0.5) * dy + ymin
    z_centers = (np.arange(nz) + 0.5) * dz + zmin

    # One downward ray per (i, j) column, originating just above the domain top.
    # Columns whose centre falls outside the mesh's xy extent cannot produce a
    # hit; we still cast them (cheap, and keeps indexing trivial).
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    n_cols = nx * ny
    origins = np.column_stack(
        [
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            np.full(n_cols, max(zmax, float(mesh.bounds[1, 2])) + max(dz, 1.0)),
        ]
    )
    directions = np.tile(np.array([0.0, 0.0, -1.0]), (n_cols, 1))

    logger.info(
        "Casting %s vertical rays (%sx%s grid) for solid occupancy...",
        n_cols,
        nx,
        ny,
    )

    # multiple_hits=True returns every intersection; we reduce to the per-column
    # maximum z (the roof). Using intersects_location keeps the hit -> ray
    # mapping explicit via index_ray.
    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=True,
    )

    # roof_height[col] = highest hit z for that column; columns with no hit keep
    # -inf and are excluded below (so they stay fluid).
    roof_height = np.full(n_cols, -np.inf)
    if len(index_ray) > 0:
        np.maximum.at(roof_height, index_ray, locations[:, 2])

    roof_height = roof_height.reshape(nx, ny)

    # A cell is solid when its centre z is at/below the column roof (and the
    # column actually had a roof hit). A flat ground plane at z=0 yields
    # roof ~ 0 < z_centers[0], so no cells are filled -> ground stays fluid.
    solid = z_centers[None, None, :] <= roof_height[:, :, None]
    solid &= np.isfinite(roof_height)[:, :, None]

    logger.info(
        "Solid occupancy: %s / %s cells (%.2f%%), %s occupied columns",
        int(solid.sum()),
        solid.size,
        100.0 * solid.sum() / solid.size,
        int((solid.any(axis=2)).sum()),
    )

    return solid


def occupancy_to_column_runs(solid: np.ndarray) -> list[dict[str, int]]:
    """
    Run-length encode a solid-occupancy mask along z, per (i, j) column.

    Args:
        solid: Boolean array of shape (nx, ny, nz) in 0-based grid order.

    Returns:
        A list of run dicts with 1-based Fortran indices:
        ``{"i": I, "j": J, "ks": Z0, "ke": Z1}`` for each contiguous solid run
        [Z0, Z1] in column (I, J). A pure height field yields one run per
        occupied column; overhangs/gaps yield multiple runs.
    """
    nx, ny, nz = solid.shape
    runs: list[dict[str, int]] = []

    # Find occupied columns first to avoid scanning empty ones.
    occupied_i, occupied_j = np.nonzero(solid.any(axis=2))
    for i, j in zip(occupied_i.tolist(), occupied_j.tolist()):
        column = solid[i, j]
        # Detect run boundaries via padded diff: rising edge starts a run,
        # falling edge ends it.
        padded = np.concatenate(([False], column, [False]))
        edges = np.diff(padded.astype(np.int8))
        starts = np.nonzero(edges == 1)[0]
        ends = np.nonzero(edges == -1)[0]  # exclusive end (0-based)
        for z0, z1 in zip(starts.tolist(), ends.tolist()):
            runs.append(
                {
                    "i": i + 1,  # 1-based Fortran
                    "j": j + 1,
                    "ks": z0 + 1,
                    "ke": z1,  # inclusive 1-based end (exclusive 0-based == inclusive 1-based)
                }
            )

    return runs


def get_building_grid_indices(
    stl_path: str | pathlib.Path,
    nx: int,
    ny: int,
    nz: int,
    domain_bounds: dict[str, float] | None = None,
    verbose: bool = True,
) -> list[dict[str, int]]:
    """
    Compute solid-occupancy column runs for the STL over the LBM grid.

    This replaces the old bounding-box implementation: it returns one entry per
    contiguous solid z-run per grid column (see :func:`occupancy_to_column_runs`)
    rather than one rectangular block per building, which preserves courtyards
    and arbitrary footprints.

    Args:
        stl_path: Path to the .stl file.
        nx, ny, nz: Number of discrete cells in x, y, z directions.
        domain_bounds: Physical bounds of the simulation domain
            {'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'}. If None, the STL
            bounds are used.
        verbose: Unused; kept for backwards-compatible signature.

    Returns:
        A list of dictionaries with 1-based Fortran column-run indices:
        ``{"i", "j", "ks", "ke"}``.
    """
    solid = compute_solid_occupancy(
        stl_path=stl_path,
        nx=nx,
        ny=ny,
        nz=nz,
        domain_bounds=domain_bounds,
    )
    runs = occupancy_to_column_runs(solid)

    if len(runs) == 0:
        raise ValueError(
            f"No solid cells found for {stl_path}; check domain bounds/resolution"
        )

    logger.info("Generated %s solid column runs from occupancy mask.", len(runs))
    return runs


def generate_fortran_code(
    buildings_indices: list[dict[str, int]],
    nx: int,
    ny: int,
    nz: int,
    module_name: str = "m_runcase",
    subroutine_name: str = "runcase",
    filename: str = "runcase.f90",
) -> str:
    """
    Function 2: Emit the solid column runs as a Fortran geometry module.

    Each run dict (from :func:`occupancy_to_column_runs`) becomes one blanking
    line for a single (i, j) column over its solid z-range, e.g.
    ``blanking(ioff+12, joff+34, 1:5)=.true.``. This run-length encoding keeps
    the generated file compact and self-contained.

    Args:
        buildings_indices: List of column-run dicts with keys
            ``{"i", "j", "ks", "ke"}`` (1-based Fortran indices).
        nx, ny, nz: Grid dimensions (kept for signature compatibility).
        module_name: Name of the Fortran module.
        subroutine_name: Name of the Fortran subroutine.
        filename: Output file path.
    """

    # Template strings for the Fortran boilerplate
    header = (
        f"module {module_name}\n"
        f"contains\n"
        f"subroutine {subroutine_name}(blanking)\n"
        f"   use mod_dimensions, only : nx, nyg, nz\n"
        f"   implicit none\n"
        f"   logical, intent(inout) :: blanking(0:nx+1,0:nyg+1,0:nz+1)\n"
        f"   integer ioff\n"
        f"   integer joff\n\n"
        f"   ioff=0\n"
        f"   joff=0\n"
    )

    footer = "end subroutine\nend module\n"

    body = ""

    for b in buildings_indices:
        body += (
            f"   blanking(ioff+{b['i']}, joff+{b['j']}, "
            f"{b['ks']}:{b['ke']})=.true.\n"
        )

    full_code = header + body + footer

    # Write to file
    output_path = pathlib.Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(full_code)

    return full_code


def update_solid_objects_init(
    solid_objects_init_path: pathlib.Path,
    experiment_name: str,
) -> None:
    """
    Update m_solid_objects_init.F90 to include the generated geometry module.

    This function:
    1. Adds `use m_{experiment_name}` to the use statements if not present
    2. Adds a `case('{experiment_name}')` block in the select case statement
       that calls the geometry subroutine

    Args:
        solid_objects_init_path: Path to m_solid_objects_init.F90
        experiment_name: Name of the experiment (e.g. "runcase", "city2")
    """
    if not solid_objects_init_path.exists():
        logger.warning(
            "m_solid_objects_init.F90 not found at %s", solid_objects_init_path
        )
        return

    module_name = f"m_{experiment_name}"
    use_statement = f"   use {module_name}"

    # Read the file
    with open(solid_objects_init_path, "r") as f:
        lines = f.readlines()

    # Check if use statement already exists (check for exact module name)
    has_use = any(f"use {module_name}" in line for line in lines)

    # Check if case statement exists and is correctly implemented
    case_pattern = f"case('{experiment_name}')"
    case_line_idx = None
    case_end_idx = None
    case_correct = False

    for i, line in enumerate(lines):
        if case_pattern in line:
            case_line_idx = i
            # Find where this case block ends (next case or end select)
            case_end_idx = i + 1
            for j in range(i + 1, len(lines)):
                if lines[j].strip().startswith("case(") or lines[j].strip().startswith(
                    "end select"
                ):
                    case_end_idx = j
                    break

            # Check if case is empty (immediately followed by another case)
            is_empty = case_end_idx > i + 1 and lines[i + 1].strip().startswith("case(")

            if is_empty:
                # Empty case is always incorrect
                case_correct = False
                break

            # Check if the case block has the correct call and no wrong calls
            has_correct_call = False
            has_wrong_calls = False

            # Look through all lines in the case block
            for j in range(i + 1, case_end_idx):
                line_content = lines[j]
                # Check for correct call: call {experiment_name}(blanking_global)
                if f"call {experiment_name}(blanking_global)" in line_content:
                    has_correct_call = True
                # Check for any call statements
                elif "call " in line_content:
                    # If it's calling something other than our experiment, it's wrong
                    if f"call {experiment_name}" not in line_content:
                        has_wrong_calls = True
                    # If it's calling our experiment but with wrong signature
                    elif f"call {experiment_name}" in line_content:
                        if "(blanking_global)" not in line_content:
                            has_wrong_calls = True

            # Case is correct only if it has exactly the correct call and no wrong calls
            case_correct = has_correct_call and not has_wrong_calls
            break

    modified = False

    # Step 1: Add use statement if missing
    if not has_use:
        # Find the insertion point (after other use m_* statements, before MPI section)
        insert_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith("use m_") and not line.strip().startswith(
                "use m_mpi"
            ):
                # Keep track of the last use m_* line
                insert_idx = i + 1
            elif line.strip().startswith("#ifdef MPI") or line.strip().startswith(
                "implicit none"
            ):
                # Stop before MPI section or implicit none
                if insert_idx is not None:
                    break
                insert_idx = i
                break

        if insert_idx is None:
            # Fallback: insert after use m_dump_elevation
            for i, line in enumerate(lines):
                if "use m_dump_elevation" in line:
                    insert_idx = i + 1
                    break

        if insert_idx is not None:
            lines.insert(insert_idx, use_statement + "\n")
            modified = True
            logger.info("Added use statement: %s", use_statement)

    # Step 2: Add or fix case statement
    if not case_correct:
        # Also check for and fix empty cases that might be before our target case
        # (e.g., empty cylinder case before runcase)
        lines_deleted_before = 0
        if case_line_idx is not None:
            # Check if there's an empty case immediately before our target case
            if (
                case_line_idx > 0
                and lines[case_line_idx - 1].strip().startswith("case(")
                and lines[case_line_idx - 1].strip() != f"case('{experiment_name}')"
            ):
                # Check if the previous case is empty (falls through to our case)
                prev_case_line = case_line_idx - 1
                # If the line immediately after the previous case is our case, it's empty
                if prev_case_line + 1 == case_line_idx:
                    # Remove the empty case line
                    del lines[prev_case_line]
                    lines_deleted_before = 1
                    modified = True
                    logger.info(
                        "Removed empty case statement before '%s'",
                        experiment_name,
                    )
                    # Adjust case_line_idx and case_end_idx since we deleted a line
                    case_line_idx -= 1
                    if case_end_idx is not None:
                        case_end_idx -= 1

        if case_line_idx is not None and case_end_idx is not None:
            # Case exists but is broken - need to fix it
            # Remove the broken case block
            del lines[case_line_idx:case_end_idx]
            modified = True
            logger.info("Removed broken case statement for '%s'", experiment_name)

        # Find the select case block and add/fix the case
        # Need to recalculate indices after deletion
        in_select_case = False
        case_insert_idx = None

        for i, line in enumerate(lines):
            if "select case(trim(experiment))" in line:
                in_select_case = True
            elif in_select_case and line.strip().startswith("case("):
                # Track the last case statement (but skip 'airfoil' which stops execution)
                if "'airfoil'" not in line:
                    # Find the end of this case block (next case or end select)
                    case_block_end = i + 1
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip().startswith("case(") or lines[
                            j
                        ].strip().startswith("end select"):
                            case_block_end = j
                            break
                    # Insert after the entire case block, not just the case line
                    case_insert_idx = case_block_end
            elif in_select_case and line.strip().startswith("end select"):
                # Insert before end select (or before airfoil if it exists)
                if case_insert_idx is None:
                    case_insert_idx = i
                break

        if case_insert_idx is not None:
            # Generate the case block
            case_block = f"""      case('{experiment_name}')
         call {experiment_name}(blanking_global)
         lsolids=.true.
"""
            lines.insert(case_insert_idx, case_block)
            modified = True
            if case_line_idx is not None:
                logger.info("Fixed case statement for '%s'", experiment_name)
            else:
                logger.info("Added case statement for '%s'", experiment_name)

    # Write back if modified
    if modified:
        with open(solid_objects_init_path, "w") as f:
            f.writelines(lines)
        logger.info(
            "Updated m_solid_objects_init.F90 to include %s geometry",
            experiment_name,
        )
    else:
        logger.info(
            "m_solid_objects_init.F90 already includes %s geometry",
            experiment_name,
        )


# --- Helper Wrapper for Testing ---
def process_stl_to_fortran(
    stl_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    nx: int,
    ny: int,
    nz: int,
    bounds: dict[str, float] | None = None,
) -> str:
    """
    Orchestrator function for better readability.
    """
    logger.info("Processing %s...", stl_path)

    # Step 1: Get indices
    building_data = get_building_grid_indices(
        stl_path=stl_path,
        nx=nx,
        ny=ny,
        nz=nz,
        domain_bounds=bounds,
    )

    # Step 2: Generate Fortran
    code_str = generate_fortran_code(
        buildings_indices=building_data,
        nx=nx,
        ny=ny,
        nz=nz,
        filename=output_path,  # type: ignore[arg-type]
    )

    logger.info("Fortran code written to %s", output_path)
    return code_str


# --- Main API function matching the old interface ---
def stl_to_lbm_geometry(
    stl_path: str | pathlib.Path,
    dirs: DirectoryPaths,
    nx: int,
    ny: int,
    nz: int,
    bounds: (
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None
    ) = None,
) -> None:
    """
    Convert an STL file to a Fortran geometry module for LBM simulation.

    This function wraps the new alternative implementation using get_building_grid_indices
    and generate_fortran_code.

    Args:
        stl_path: Path to the input STL file
        dirs: DirectoryPaths object containing all relevant paths (including experiment_dir
              and executable_path).
        nx: Grid resolution in x-direction
        ny: Grid resolution in y-direction
        nz: Grid resolution in z-direction
        bounds: Optional bounding box as ((xmin, xmax), (ymin, ymax), (zmin, zmax))
                in physical coordinates. If None, uses the mesh bounding box.
        scale: Optional scaling factor (not yet implemented in new version)
        translate: Optional translation (not yet implemented in new version)

    Returns:
        None. Writes the Fortran file to output_path.
    """
    stl_path = pathlib.Path(stl_path)

    # Convert bounds from tuple format to dict format if provided
    domain_bounds = None
    if bounds is not None:
        domain_bounds = {
            "xmin": float(bounds[0][0]),
            "xmax": float(bounds[0][1]),
            "ymin": float(bounds[1][0]),
            "ymax": float(bounds[1][1]),
            "zmin": float(bounds[2][0]),
            "zmax": float(bounds[2][1]),
        }

    # Step 1: Get building grid indices
    building_data = get_building_grid_indices(
        stl_path=stl_path,
        nx=nx,
        ny=ny,
        nz=nz,
        domain_bounds=domain_bounds,
    )

    # Step 2: Generate Fortran code
    generate_fortran_code(
        buildings_indices=building_data,
        nx=nx,
        ny=ny,
        nz=nz,
        module_name=f"m_{dirs.experiment_name}",
        subroutine_name=dirs.experiment_name,
        filename=dirs.lbm_src_path / f"m_{dirs.experiment_name}.F90",  # type: ignore[arg-type]
    )

    # Step 3: Update m_solid_objects_init.F90 to use the generated geometry
    update_solid_objects_init(
        solid_objects_init_path=dirs.lbm_src_path / "m_solid_objects_init.F90",
        experiment_name=dirs.experiment_name,
    )
