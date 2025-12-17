"""
Convert STL files to LBM geometry Fortran modules.

This module provides functionality to convert 3D STL mesh files into
Fortran geometry modules compatible with the LBM simulation.
"""

import pathlib
import sys
from typing import Optional

try:
    import numpy as np
    import trimesh
except ImportError:
    raise ImportError(
        "trimesh and numpy are required for STL conversion. "
        "Install with: pip install trimesh numpy"
    )


def stl_to_lbm_geometry(
    stl_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    module_name: str = "m_obstacle",
    subroutine_name: str = "obstacle",
    nx: int = 200,
    ny: int = 120,
    nz: int = 96,
    bounds: Optional[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    ] = None,
    scale: Optional[float] = None,
    translate: Optional[tuple[float, float, float]] = None,
) -> None:
    """
    Convert an STL file to a Fortran geometry module for LBM simulation.

    Args:
        stl_path: Path to the input STL file
        output_path: Path where the output Fortran file will be written
        module_name: Name of the Fortran module (default: "m_obstacle")
        subroutine_name: Name of the Fortran subroutine (default: "obstacle")
        nx: Grid resolution in x-direction
        ny: Grid resolution in y-direction
        nz: Grid resolution in z-direction
        bounds: Optional bounding box as ((xmin, xmax), (ymin, ymax), (zmin, zmax))
                in physical coordinates. If None, uses the mesh bounding box.
        scale: Optional scaling factor to apply to the mesh before voxelization
        translate: Optional translation (dx, dy, dz) to apply before voxelization

    Returns:
        None. Writes the Fortran file to output_path.
    """
    stl_path = pathlib.Path(stl_path)
    output_path = pathlib.Path(output_path)

    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    # Load the STL mesh
    try:
        mesh = trimesh.load(str(stl_path))
        if isinstance(mesh, trimesh.Scene):
            # If it's a scene, combine all meshes
            mesh = trimesh.util.concatenate(
                [m for m in mesh.geometry.values() if isinstance(m, trimesh.Trimesh)]
            )

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Could not load a valid mesh from {stl_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load STL file {stl_path}: {e}")

    # Apply transformations if specified
    if scale is not None:
        mesh.apply_scale(scale)

    if translate is not None:
        mesh.apply_translation(translate)

    # Get or set bounds
    if bounds is None:
        bbox = mesh.bounds
        bounds = (
            (float(bbox[0, 0]), float(bbox[1, 0])),  # x bounds
            (float(bbox[0, 1]), float(bbox[1, 1])),  # y bounds
            (float(bbox[0, 2]), float(bbox[1, 2])),  # z bounds
        )
    else:
        bounds = (
            (float(bounds[0][0]), float(bounds[0][1])),
            (float(bounds[1][0]), float(bounds[1][1])),
            (float(bounds[2][0]), float(bounds[2][1])),
        )

    xmin, xmax = bounds[0]
    ymin, ymax = bounds[1]
    zmin, zmax = bounds[2]

    # Validate bounds
    if xmax <= xmin or ymax <= ymin or zmax <= zmin:
        raise ValueError(
            f"Invalid bounds: x=({xmin}, {xmax}), y=({ymin}, {ymax}), z=({zmin}, {zmax}). "
            "Max must be greater than min for all dimensions."
        )

    # Create grid cell centers in physical coordinates
    # The grid covers the full bounds, so we have nx cells from xmin to xmax
    dx = (xmax - xmin) / nx if nx > 0 else 1.0
    dy = (ymax - ymin) / ny if ny > 0 else 1.0
    dz = (zmax - zmin) / nz if nz > 0 else 1.0

    # Cell centers: first cell center is at xmin + dx/2, last at xmax - dx/2
    x_coords = (
        np.linspace(xmin + dx / 2, xmax - dx / 2, nx)
        if nx > 1
        else np.array([(xmin + xmax) / 2])
    )
    y_coords = (
        np.linspace(ymin + dy / 2, ymax - dy / 2, ny)
        if ny > 1
        else np.array([(ymin + ymax) / 2])
    )
    z_coords = (
        np.linspace(zmin + dz / 2, zmax - dz / 2, nz)
        if nz > 1
        else np.array([(zmin + zmax) / 2])
    )

    # Create voxel grid by checking if cell centers are inside the mesh
    # Create all grid points
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Check which points are inside the mesh
    # This can be slow for large meshes, but is accurate
    print(
        f"Checking {len(grid_points)} grid points against mesh...",
        file=__import__("sys").stderr,
    )
    contains = mesh.contains(grid_points)

    # Reshape back to grid shape (nx, ny, nz)
    obstacle_grid = contains.reshape(nx, ny, nz)

    # Check if any obstacles were found
    num_obstacles = int(np.sum(obstacle_grid))
    if num_obstacles == 0:
        print(
            f"Warning: No obstacle cells found in STL mesh. "
            f"This might indicate:\n"
            f"  - The mesh bounds don't match the grid bounds\n"
            f"  - The mesh is outside the grid domain\n"
            f"  - The mesh needs scaling or translation\n"
            f"  Mesh bounds: x=[{xmin:.3f}, {xmax:.3f}], "
            f"y=[{ymin:.3f}, {ymax:.3f}], z=[{zmin:.3f}, {zmax:.3f}]\n"
            f"  Grid: nx={nx}, ny={ny}, nz={nz}",
            file=sys.stderr,
        )

    # Generate Fortran code
    fortran_code = _generate_fortran_code(
        module_name=module_name,
        subroutine_name=subroutine_name,
        obstacle_grid=obstacle_grid,
        nx=nx,
        ny=ny,
        nz=nz,
    )

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(fortran_code)

    print(f"Generated Fortran geometry file: {output_path}", file=sys.stderr)
    print(
        f"Total obstacle cells: {num_obstacles} out of {nx * ny * nz} total cells",
        file=sys.stderr,
    )


def _generate_fortran_code(
    module_name: str,
    subroutine_name: str,
    obstacle_grid: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> str:
    """
    Generate Fortran code from obstacle grid.

    Args:
        module_name: Name of the Fortran module
        subroutine_name: Name of the Fortran subroutine
        obstacle_grid: 3D boolean array (nx, ny, nz) where True indicates obstacle
        nx, ny, nz: Grid dimensions

    Returns:
        Fortran source code as string
    """
    lines = [
        f"module {module_name}",
        "contains",
        f"subroutine {subroutine_name}(lsolids,blanking)",
        "   use mod_dimensions",
        "   implicit none",
        "   logical, intent(out)   :: lsolids",
        "   logical, intent(inout) :: blanking(0:nx+1,0:ny+1,0:nz+1)",
        "   integer :: i, j, k",
        "#ifdef _CUDA",
        "   attributes(device) :: blanking",
        "#endif",
        "",
        "   lsolids=.true.",
        "",
        "! Set obstacle cells based on STL geometry",
    ]

    # Generate code to set obstacle cells
    # We'll iterate through all cells and set blanking where obstacles exist
    obstacle_count = 0
    max_line_length = 100

    # Collect all obstacle cell coordinates
    obstacle_cells = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if obstacle_grid[i, j, k]:
                    obstacle_cells.append(
                        (i + 1, j + 1, k + 1)
                    )  # Convert to 1-based indexing
                    obstacle_count += 1

    # If too many obstacles, write them in batches
    if obstacle_count > 0:
        # Write obstacles in a more compact format
        lines.append("   ! Obstacle cells from STL geometry")

        # Group by k-level for better readability
        k_levels: dict[int, list[tuple[int, int]]] = {}
        for i, j, k in obstacle_cells:
            if k not in k_levels:
                k_levels[k] = []
            k_levels[k].append((i, j))

        # Write code for each k level
        for k in sorted(k_levels.keys()):
            cells = k_levels[k]
            if len(cells) == 1:
                i, j = cells[0]
                lines.append(f"   blanking({i},{j},{k})=.true.")
            else:
                # Group by i for compactness
                i_groups: dict[int, list[int]] = {}
                for i, j in cells:
                    if i not in i_groups:
                        i_groups[i] = []
                    i_groups[i].append(j)

                for i in sorted(i_groups.keys()):
                    js = sorted(i_groups[i])
                    if len(js) == 1:
                        lines.append(f"   blanking({i},{js[0]},{k})=.true.")
                    elif len(js) == 2:
                        lines.append(f"   blanking({i},{js[0]}:{js[1]},{k})=.true.")
                    else:
                        # Write individual cells or ranges
                        # Try to find consecutive ranges
                        ranges = _find_consecutive_ranges(js)
                        for start, end in ranges:
                            if start == end:
                                lines.append(f"   blanking({i},{start},{k})=.true.")
                            else:
                                lines.append(
                                    f"   blanking({i},{start}:{end},{k})=.true."
                                )

    lines.append("")
    lines.append("end subroutine")
    lines.append("end module")

    return "\n".join(lines)


def _find_consecutive_ranges(numbers: list[int]) -> list[tuple[int, int]]:
    """Find consecutive ranges in a sorted list of integers."""
    if not numbers:
        return []

    ranges = []
    start = numbers[0]
    end = numbers[0]

    for num in numbers[1:]:
        if num == end + 1:
            end = num
        else:
            ranges.append((start, end))
            start = num
            end = num

    ranges.append((start, end))
    return ranges


# Example usage:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert STL file to LBM geometry Fortran module"
    )
    parser.add_argument("stl_file", help="Path to input STL file")
    parser.add_argument("-o", "--output", help="Output Fortran file path", default=None)
    parser.add_argument(
        "--module-name", default="m_obstacle", help="Fortran module name"
    )
    parser.add_argument(
        "--subroutine-name", default="obstacle", help="Fortran subroutine name"
    )
    parser.add_argument(
        "--nx", type=int, default=200, help="Grid resolution in x-direction"
    )
    parser.add_argument(
        "--ny", type=int, default=120, help="Grid resolution in y-direction"
    )
    parser.add_argument(
        "--nz", type=int, default=96, help="Grid resolution in z-direction"
    )
    parser.add_argument(
        "--scale", type=float, default=None, help="Scaling factor for mesh"
    )
    parser.add_argument(
        "--translate",
        nargs=3,
        type=float,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Translation offset (dx dy dz)",
    )

    args = parser.parse_args()

    stl_path = pathlib.Path(args.stl_file)
    if args.output:
        output_path = pathlib.Path(args.output)
    else:
        output_path = stl_path.parent / f"{stl_path.stem}.F90"

    translate_tuple = None
    if args.translate:
        translate_tuple = tuple(args.translate)

    stl_to_lbm_geometry(
        stl_path=stl_path,
        output_path=output_path,
        module_name=args.module_name,
        subroutine_name=args.subroutine_name,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        scale=args.scale,
        translate=translate_tuple,
    )
