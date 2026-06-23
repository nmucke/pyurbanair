import logging
import pathlib

logger = logging.getLogger(__name__)

import numpy as np
import trimesh
from pylbm.utils import DirectoryPaths
from scipy.io import FortranFile


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


def write_bathymetry_file(
    solid: np.ndarray,
    dirs: DirectoryPaths,
    nx: int,
    ny: int,
    nz: int,
) -> pathlib.Path:
    """
    Write the building mask as a compact Fortran-unformatted bathymetry file.

    This replaces the old per-gridpoint ``m_<experiment>.F90`` module (one
    ``blanking(...)=.true.`` statement per solid column, ~10⁴ lines for a city
    block) which was too large to compile. The LBM binary instead loads this
    file at run time via ``m_read_bathymetry``; the layout mirrors that routine::

        record 1:  nx, nyg(=ny), nz                       (3 × int32)
        record 2:  blanking(0:nx+1, 0:nyg+1, 0:nz+1)      (logical, column-major)

    The interior cells carry the solid occupancy; the one-cell ghost shell stays
    fluid (the Fortran side zeroes its halos too). Logicals are written as int32
    1/0, the canonical representation for the default 4-byte LOGICAL used by the
    LBM build (matching how the codebase already round-trips restart ``.uf``
    files through :class:`scipy.io.FortranFile`).

    Args:
        solid: Boolean occupancy of shape ``(nx, ny, nz)`` (0-based grid order).
        dirs: DirectoryPaths; the file is written into ``experiment_dir`` (the
            run cwd, and the dir cloned per ensemble member) and named after the
            experiment so ``read_bathymetry(blanking, '<experiment>')`` finds it.
        nx, ny, nz: Interior grid dimensions.

    Returns:
        Path to the written ``bathymetry_<experiment>.uf`` file.
    """
    # Full padded mask including the ghost shell, in Fortran (i,j,k) order.
    blanking = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.int32)
    blanking[1 : nx + 1, 1 : ny + 1, 1 : nz + 1] = solid.astype(np.int32)

    out_path = dirs.experiment_dir / f"bathymetry_{dirs.experiment_name}.uf"
    with FortranFile(str(out_path), "w") as f:
        # nyg == ny in the serial build set_experiment configures.
        f.write_record(np.array([nx, ny, nz], dtype=np.int32))
        f.write_record(np.ravel(blanking, order="F").astype(np.int32))

    logger.info(
        "Wrote bathymetry geometry (%s solid cells, %.2f%%) to %s",
        int(solid.sum()),
        100.0 * solid.sum() / solid.size,
        out_path,
    )
    return out_path


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
    Wire m_solid_objects_init.F90 to load this experiment's geometry at run time.

    Ensures (idempotently):

    1. ``use m_read_bathymetry`` is present among the module use statements.
    2. a ``case('<experiment_name>')`` branch in the
       ``select case(trim(experiment))`` calls
       ``read_bathymetry(blanking_global, '<experiment_name>')`` and sets
       ``lsolids=.true.``.

    This replaces the previous approach, which generated a per-experiment
    geometry module (``m_<experiment_name>.F90``) and called it directly. Any
    leftover ``use m_<experiment_name>`` from that approach is removed here so the
    file does not reference a module that no longer exists.

    Args:
        solid_objects_init_path: Path to m_solid_objects_init.F90.
        experiment_name: Name of the experiment (e.g. "runcase").
    """
    if not solid_objects_init_path.exists():
        logger.warning(
            "m_solid_objects_init.F90 not found at %s", solid_objects_init_path
        )
        return

    lines = solid_objects_init_path.read_text().splitlines()

    # The case body the LBM binary executes for this experiment.
    desired_case = [
        f"      case('{experiment_name}')",
        f"         call read_bathymetry(blanking_global,'{experiment_name}')",
        "         lsolids=.true.",
    ]

    # --- 0. Drop the obsolete generated-module use statement, if present. ---
    lines = [
        line for line in lines if line.strip() != f"use m_{experiment_name}"
    ]

    # --- 1. Ensure `use m_read_bathymetry`. ---
    if not any(line.strip() == "use m_read_bathymetry" for line in lines):
        insert_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("use m_") and not stripped.startswith("use m_mpi"):
                insert_idx = i + 1
            elif stripped.startswith("implicit") or stripped.startswith("#ifdef MPI"):
                break
        if insert_idx is None:
            insert_idx = next(
                (i for i, l in enumerate(lines) if l.strip().startswith("implicit")),
                None,
            )
        if insert_idx is not None:
            lines.insert(insert_idx, "   use m_read_bathymetry")
            logger.info("Added 'use m_read_bathymetry' to m_solid_objects_init.F90")

    # --- 2. Replace or insert the case('<experiment>') block. ---
    select_idx = next(
        (i for i, l in enumerate(lines) if "select case(trim(experiment))" in l),
        None,
    )
    if select_idx is None:
        logger.warning(
            "No 'select case(trim(experiment))' found in %s; cannot wire geometry.",
            solid_objects_init_path,
        )
        solid_objects_init_path.write_text("\n".join(lines) + "\n")
        return

    end_idx = next(
        (
            i
            for i in range(select_idx + 1, len(lines))
            if lines[i].strip().startswith("end select")
        ),
        len(lines),
    )

    case_start = next(
        (
            i
            for i in range(select_idx + 1, end_idx)
            if lines[i].strip() == f"case('{experiment_name}')"
        ),
        None,
    )
    if case_start is not None:
        # Replace the existing block (up to the next case / end select).
        case_end = next(
            (
                i
                for i in range(case_start + 1, end_idx)
                if lines[i].strip().startswith("case(")
            ),
            end_idx,
        )
        if lines[case_start:case_end] != desired_case:
            lines[case_start:case_end] = desired_case
            logger.info("Updated case('%s') to use read_bathymetry.", experiment_name)
    else:
        # Insert before the airfoil stop-case if present, else before end select.
        insert_at = next(
            (
                i
                for i in range(select_idx + 1, end_idx)
                if lines[i].strip().startswith("case('airfoil')")
            ),
            end_idx,
        )
        lines[insert_at:insert_at] = desired_case
        logger.info("Added case('%s') calling read_bathymetry.", experiment_name)

    solid_objects_init_path.write_text("\n".join(lines) + "\n")


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
    Convert an STL file to LBM geometry: a compact bathymetry file the binary
    loads at run time via ``m_read_bathymetry``.

    Earlier versions emitted a per-gridpoint ``m_<experiment>.F90`` module (one
    ``blanking(...)=.true.`` per solid column), which grew to tens of thousands
    of lines for a city block and was too large to compile. Now the solid
    occupancy is written to ``bathymetry_<experiment>.uf`` and ``read_bathymetry``
    is wired into ``m_solid_objects_init.F90``.

    Args:
        stl_path: Path to the input STL file.
        dirs: DirectoryPaths object (provides experiment_dir, lbm_src_path,
              experiment_name).
        nx: Grid resolution in x-direction.
        ny: Grid resolution in y-direction.
        nz: Grid resolution in z-direction.
        bounds: Optional bounding box as ((xmin, xmax), (ymin, ymax), (zmin, zmax))
                in physical coordinates. If None, uses the mesh bounding box.

    Returns:
        None.
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

    # Step 1: Voxelize the STL into a solid-occupancy mask over the LBM grid.
    solid = compute_solid_occupancy(
        stl_path=stl_path,
        nx=nx,
        ny=ny,
        nz=nz,
        domain_bounds=domain_bounds,
    )
    if int(solid.sum()) == 0:
        raise ValueError(
            f"No solid cells found for {stl_path}; check domain bounds/resolution"
        )

    # Step 2: Write the compact bathymetry file (read at run time, not compiled).
    write_bathymetry_file(solid=solid, dirs=dirs, nx=nx, ny=ny, nz=nz)

    # Step 3: Wire m_solid_objects_init.F90 to call read_bathymetry for this case.
    update_solid_objects_init(
        solid_objects_init_path=dirs.lbm_src_path / "m_solid_objects_init.F90",
        experiment_name=dirs.experiment_name,
    )

    # Step 4: Drop any stale generated geometry module from the old approach so
    # the makefile's `ls *.F90` does not recompile the obsolete (huge) file.
    stale_module = dirs.lbm_src_path / f"m_{dirs.experiment_name}.F90"
    stale_module.unlink(missing_ok=True)
