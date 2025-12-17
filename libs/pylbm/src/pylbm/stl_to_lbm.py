import os
import pathlib
import sys
from collections import defaultdict

import numpy as np
import trimesh


def _split_buildings_edge_based(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """
    Split mesh into building components using edge-based connectivity analysis.

    Inspired by u-dales splitBuildings.m, this uses face-to-face connectivity
    via shared edges rather than just vertex connectivity. This can better
    separate buildings that are topologically connected.

    Args:
        mesh: trimesh.Trimesh object

    Returns:
        List of trimesh.Trimesh objects, one per building component
    """
    if len(mesh.faces) == 0:
        return []

    # Remove ground faces first (faces where all vertices have z=0)
    # This is similar to deleteGround.m in u-dales
    z_coords = mesh.vertices[:, 2]
    ground_faces_mask = np.all(z_coords[mesh.faces] == 0, axis=1)
    building_faces_mask = ~ground_faces_mask

    if np.sum(building_faces_mask) == 0:
        # All faces are ground, return empty
        return []

    # Get only building faces
    building_faces = mesh.faces[building_faces_mask]

    # Build edge-to-face mapping
    # Each edge is represented as a sorted tuple (v1, v2) where v1 < v2
    edge_to_faces = defaultdict(list)

    for face_idx, face in enumerate(building_faces):
        # Add all three edges of this triangle
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]])),
        ]
        for edge in edges:
            edge_to_faces[edge].append(face_idx)

    # Build face adjacency graph
    # Two faces are adjacent if they share an edge
    face_adjacency = defaultdict(set)
    for edge, face_indices in edge_to_faces.items():
        # If an edge is shared by multiple faces, those faces are adjacent
        if len(face_indices) > 1:
            for i in range(len(face_indices)):
                for j in range(i + 1, len(face_indices)):
                    face_adjacency[face_indices[i]].add(face_indices[j])
                    face_adjacency[face_indices[j]].add(face_indices[i])

    # Find connected components using BFS/DFS
    visited = set()
    components = []

    def dfs(face_idx: int, component: list[int]) -> None:
        """Depth-first search to find all connected faces."""
        visited.add(face_idx)
        component.append(face_idx)
        for neighbor in face_adjacency[face_idx]:
            if neighbor not in visited:
                dfs(neighbor, component)

    # Find all connected components
    for face_idx in range(len(building_faces)):
        if face_idx not in visited:
            component: list[int] = []
            dfs(face_idx, component)
            if len(component) > 0:
                components.append(component)

    # Create separate meshes for each component
    building_meshes: list[trimesh.Trimesh] = []
    for component_face_indices in components:
        # Get faces for this component
        component_faces = building_faces[component_face_indices]

        # Find unique vertices used by this component
        used_vertex_indices = np.unique(component_faces.flatten())
        component_vertices = mesh.vertices[used_vertex_indices]

        # Remap face indices to new vertex indices
        vertex_map = {
            old_idx: new_idx for new_idx, old_idx in enumerate(used_vertex_indices)
        }
        remapped_faces = np.array(
            [[vertex_map[v] for v in face] for face in component_faces]
        )

        # Create new mesh for this building component
        try:
            building_mesh = trimesh.Trimesh(
                vertices=component_vertices, faces=remapped_faces
            )
            if len(building_mesh.faces) > 0:
                building_meshes.append(building_mesh)
        except Exception as e:
            print(f"Warning: Failed to create mesh for component: {e}", file=sys.stderr)
            continue

    return building_meshes


def get_building_grid_indices(
    stl_path: str | pathlib.Path,
    nx: int,
    ny: int,
    nz: int,
    domain_bounds: dict[str, float] | None = None,
    verbose: bool = True,
) -> list[dict[str, int]]:
    """
    Function 1: Generates a list of building dimensions in grid points.

    Args:
        stl_path (str): Path to the .stl file.
        nx, ny, nz (int): Number of discrete points in x, y, z directions.
        domain_bounds (dict, optional): Physical bounds of the simulation domain
                                        {'xmin': float, 'xmax': float, ...}.
                                        If None, the STL bounds are used.

    Returns:
        list: A list of dictionaries, where each dict contains the
              start and end indices for x, y, and z (1-based for Fortran).
    """

    # 1. Load the mesh
    loaded = trimesh.load(stl_path)

    # 2. Handle Scene (multiple meshes) or single mesh
    # Split into connected components (separate buildings)
    buildings_meshes: list[trimesh.Trimesh] = []

    if isinstance(loaded, trimesh.Scene):
        # Extract all meshes from scene
        scene_meshes = [
            m for m in loaded.geometry.values() if isinstance(m, trimesh.Trimesh)
        ]
        print(f"Loaded scene with {len(scene_meshes)} separate meshes", file=sys.stderr)

        # Split each mesh in the scene into connected components
        # (in case some meshes contain multiple buildings)
        for idx, mesh in enumerate(scene_meshes):
            if len(mesh.vertices) == 0:
                print(
                    f"Warning: Mesh {idx+1} in scene has no vertices, skipping",
                    file=sys.stderr,
                )
                continue

            # Use edge-based splitting for better building detection
            split_meshes = _split_buildings_edge_based(mesh)
            if len(split_meshes) > 0:
                buildings_meshes.extend(split_meshes)
                if len(split_meshes) > 1:
                    print(
                        f"  Mesh {idx+1} split into {len(split_meshes)} components",
                        file=sys.stderr,
                    )
            else:
                buildings_meshes.append(mesh)
                print(f"  Mesh {idx+1} kept as single component", file=sys.stderr)

    elif isinstance(loaded, trimesh.Trimesh):
        # Single mesh - use edge-based splitting for better building detection
        print(
            "Splitting mesh into connected components using edge-based analysis...",
            file=sys.stderr,
        )
        split_meshes = _split_buildings_edge_based(loaded)
        if len(split_meshes) > 0:
            buildings_meshes = split_meshes
            print(
                f"Found {len(buildings_meshes)} connected components", file=sys.stderr
            )
        else:
            buildings_meshes = [loaded]
            print(
                "Split returned empty, using original mesh as single building",
                file=sys.stderr,
            )
    else:
        raise ValueError(f"Could not load a valid mesh from {stl_path}")

    if len(buildings_meshes) == 0:
        raise ValueError(f"No valid meshes found in {stl_path}")

    # Filter out very small meshes and ground planes (likely artifacts)
    filtered_meshes: list[trimesh.Trimesh] = []

    # First, calculate overall domain bounds for filtering
    temp_all_bounds: dict[str, float] | None = None
    for mesh in buildings_meshes:
        bbox = mesh.bounds
        if temp_all_bounds is None:
            temp_all_bounds = {
                "xmin": float(bbox[0, 0]),
                "xmax": float(bbox[1, 0]),
                "ymin": float(bbox[0, 1]),
                "ymax": float(bbox[1, 1]),
                "zmin": float(bbox[0, 2]),
                "zmax": float(bbox[1, 2]),
            }
        else:
            temp_all_bounds["xmin"] = min(temp_all_bounds["xmin"], float(bbox[0, 0]))
            temp_all_bounds["xmax"] = max(temp_all_bounds["xmax"], float(bbox[1, 0]))
            temp_all_bounds["ymin"] = min(temp_all_bounds["ymin"], float(bbox[0, 1]))
            temp_all_bounds["ymax"] = max(temp_all_bounds["ymax"], float(bbox[1, 1]))
            temp_all_bounds["zmin"] = min(temp_all_bounds["zmin"], float(bbox[0, 2]))
            temp_all_bounds["zmax"] = max(temp_all_bounds["zmax"], float(bbox[1, 2]))

    domain_x = temp_all_bounds["xmax"] - temp_all_bounds["xmin"]  # type: ignore[index]
    domain_y = temp_all_bounds["ymax"] - temp_all_bounds["ymin"]  # type: ignore[index]
    domain_z = temp_all_bounds["zmax"] - temp_all_bounds["zmin"]  # type: ignore[index]

    for idx, mesh in enumerate(buildings_meshes):
        bbox = mesh.bounds
        x_size = bbox[1, 0] - bbox[0, 0]
        y_size = bbox[1, 1] - bbox[0, 1]
        z_size = bbox[1, 2] - bbox[0, 2]

        # Filter out very flat structures (likely ground planes or walls)
        if z_size < 0.1:  # Less than 0.1 units tall
            print(
                f"Filtering out building {idx+1}: too flat (height={z_size:.3f})",
                file=sys.stderr,
            )
            continue

        # Filter out structures that cover too much of the domain (likely ground planes/bases)
        # A ground plane typically covers >90% of the domain in x and y
        if domain_x > 0 and domain_y > 0:
            coverage_x = x_size / domain_x
            coverage_y = y_size / domain_y
            height_ratio = z_size / domain_z if domain_z > 0 else 0

            # Filter if it covers >90% in both x and y
            # This catches ground planes/bases that span the entire domain
            # Even if they're tall, if they cover the entire x-y plane, they're likely a base/platform
            if coverage_x > 0.90 and coverage_y > 0.90:
                # Additional check: if height is also very high (>80% of domain), it might be a large building
                # But if there are other buildings, this is likely still a base
                # For now, filter anything covering >90% of both dimensions
                print(
                    f"Filtering out building {idx+1}: appears to be ground plane/base "
                    f"(coverage={100*coverage_x:.1f}% x {100*coverage_y:.1f}%, "
                    f"height={z_size:.2f}, height_ratio={100*height_ratio:.1f}%)",
                    file=sys.stderr,
                )
                continue

        filtered_meshes.append(mesh)

    buildings_meshes = filtered_meshes

    if len(buildings_meshes) == 0:
        raise ValueError(
            f"No valid buildings found after filtering (all were too flat)"
        )

    print(
        f"Detected {len(buildings_meshes)} unique buildings in the STL (after filtering).",
        file=sys.stderr,
    )

    # 3. Determine Physical Domain for Grid Mapping
    # Calculate overall bounds from all meshes
    all_bounds: dict[str, float] | None = None
    for mesh in buildings_meshes:
        bbox = mesh.bounds
        if all_bounds is None:
            all_bounds = {
                "xmin": float(bbox[0, 0]),
                "xmax": float(bbox[1, 0]),
                "ymin": float(bbox[0, 1]),
                "ymax": float(bbox[1, 1]),
                "zmin": float(bbox[0, 2]),
                "zmax": float(bbox[1, 2]),
            }
        else:
            all_bounds["xmin"] = min(all_bounds["xmin"], float(bbox[0, 0]))
            all_bounds["xmax"] = max(all_bounds["xmax"], float(bbox[1, 0]))
            all_bounds["ymin"] = min(all_bounds["ymin"], float(bbox[0, 1]))
            all_bounds["ymax"] = max(all_bounds["ymax"], float(bbox[1, 1]))
            all_bounds["zmin"] = min(all_bounds["zmin"], float(bbox[0, 2]))
            all_bounds["zmax"] = max(all_bounds["zmax"], float(bbox[1, 2]))

    # Use provided bounds or calculated bounds
    if domain_bounds is None:
        xmin = all_bounds["xmin"]  # type: ignore[index]
        xmax = all_bounds["xmax"]  # type: ignore[index]
        ymin = all_bounds["ymin"]  # type: ignore[index]
        ymax = all_bounds["ymax"]  # type: ignore[index]
        zmin = all_bounds["zmin"]  # type: ignore[index]
        zmax = all_bounds["zmax"]  # type: ignore[index]
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

    # Calculate cell sizes (physical units per grid index)
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz

    buildings_indices: list[dict[str, int]] = []

    # 4. Iterate through each building and calculate grid indices
    for idx, mesh in enumerate(buildings_meshes):
        b_min = mesh.bounds[0]
        b_max = mesh.bounds[1]

        # Debug: print building physical bounds
        print(
            f"Building {idx+1}: physical bounds "
            f"x=[{b_min[0]:.3f}, {b_max[0]:.3f}], "
            f"y=[{b_min[1]:.3f}, {b_max[1]:.3f}], "
            f"z=[{b_min[2]:.3f}, {b_max[2]:.3f}]",
            file=sys.stderr,
        )

        # Convert Physical coordinates to Grid Indices
        # Fortran arrays are 1-based: valid domain is 1:nx, 1:ny, 1:nz
        # Ghost cells are at 0, nx+1, ny+1, nz+1
        # CRITICAL: Buildings cannot be at boundaries (index 1 or nx/ny)
        # Valid building indices: 2 to (nx-1) for x and y, 1 to nz for z

        # Start Indices: use floor to get the cell containing the minimum coordinate
        is_raw = int(np.floor((b_min[0] - xmin) / dx)) + 1
        js_raw = int(np.floor((b_min[1] - ymin) / dy)) + 1
        ks = int(np.floor((b_min[2] - zmin) / dz)) + 1

        # End Indices: calculate raw indices first
        ie_raw = (b_max[0] - xmin) / dx
        je_raw = (b_max[1] - ymin) / dy
        ke_raw = (b_max[2] - zmin) / dz

        # For x and y: use floor for end indices
        ie_raw_int = int(np.floor(ie_raw)) + 1
        je_raw_int = int(np.floor(je_raw)) + 1

        # For z: use ceil to include the top cell
        ke = int(np.ceil(ke_raw))

        # CRITICAL: Ensure buildings don't touch boundaries
        # Buildings must be in range 2 to (nx-1) for x and y
        # If building would start at index 1, move it to index 2
        # If building would end at index nx, move it to index nx-1
        is_ = max(2, min(nx - 1, is_raw))  # Clamp to [2, nx-1]
        ie = max(2, min(nx - 1, ie_raw_int))  # Clamp to [2, nx-1]
        js = max(2, min(ny - 1, js_raw))  # Clamp to [2, ny-1]
        je = max(2, min(ny - 1, je_raw_int))  # Clamp to [2, ny-1]

        # For z: boundaries are allowed (1 to nz)
        ks = max(1, min(nz, ks))
        ke = max(1, min(nz, ke))

        building_data = {"is": is_, "ie": ie, "js": js, "je": je, "ks": ks, "ke": ke}

        # Ensure start <= end for all dimensions
        if building_data["is"] > building_data["ie"]:
            building_data["ie"] = building_data["is"]
        if building_data["js"] > building_data["je"]:
            building_data["je"] = building_data["js"]
        if building_data["ks"] > building_data["ke"]:
            building_data["ke"] = building_data["ks"]

        # Final validation: ensure buildings are not at boundaries
        if building_data["is"] == 1 or building_data["ie"] == nx:
            print(
                f"Warning: Building {idx+1} would be at x-boundary, "
                f"adjusted from i=[{is_raw}, {ie_raw_int}] to i=[{building_data['is']}, {building_data['ie']}]",
                file=sys.stderr,
            )
        if building_data["js"] == 1 or building_data["je"] == ny:
            print(
                f"Warning: Building {idx+1} would be at y-boundary, "
                f"adjusted from j=[{js_raw}, {je_raw_int}] to j=[{building_data['js']}, {building_data['je']}]",
                file=sys.stderr,
            )

        # Debug: print grid indices
        print(
            f"Building {idx+1}: grid indices "
            f"i=[{building_data['is']}, {building_data['ie']}], "
            f"j=[{building_data['js']}, {building_data['je']}], "
            f"k=[{building_data['ks']}, {building_data['ke']}]",
            file=sys.stderr,
        )

        buildings_indices.append(building_data)

    return buildings_indices


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
    Function 2: Takes the output of function one and generates the Fortran code.

    Args:
        buildings_indices: List of building index dictionaries
        nx, ny, nz: Grid dimensions
        module_name: Name of the Fortran module
        subroutine_name: Name of the Fortran subroutine
        filename: Output file path
    """

    # Template strings for the Fortran boilerplate
    header = f"""module {module_name}
contains
subroutine {subroutine_name}(lsolids,blanking)
   use mod_dimensions
   implicit none
   logical, intent(out)   :: lsolids
   logical, intent(inout) :: blanking(0:{nx}+1,0:{ny}+1,0:{nz}+1)
   integer :: i, j, k
#ifdef _CUDA
   attributes(device) :: blanking
#endif

   lsolids=.true.

! Set obstacle cells based on STL geometry (rectangular buildings)
   ! {len(buildings_indices)} buildings
"""

    footer = """
end subroutine
end module
"""

    body = ""

    for idx, b in enumerate(buildings_indices):
        # Format: blanking(5:47,113:128,1:4)=.true.
        line = f"   ! Building {idx + 1}\n"
        line += f"   blanking({b['is']}:{b['ie']},{b['js']}:{b['je']},{b['ks']}:{b['ke']})=.true.\n"
        body += line

    full_code = header + body + footer

    # Write to file
    output_path = pathlib.Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(full_code)

    return full_code


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
    print(f"Processing {stl_path}...")

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
        filename=output_path,
    )

    print(f"Fortran code written to {output_path}")
    return code_str


# --- Main API function matching the old interface ---
def stl_to_lbm_geometry(
    stl_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    module_name: str = "m_runcase",
    subroutine_name: str = "runcase",
    nx: int = 200,
    ny: int = 120,
    nz: int = 96,
    bounds: (
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None
    ) = None,
    scale: float | None = None,
    translate: tuple[float, float, float] | None = None,
) -> None:
    """
    Convert an STL file to a Fortran geometry module for LBM simulation.

    This function wraps the new alternative implementation using get_building_grid_indices
    and generate_fortran_code.

    Args:
        stl_path: Path to the input STL file
        output_path: Path where the output Fortran file will be written
        module_name: Name of the Fortran module (default: "m_runcase")
        subroutine_name: Name of the Fortran subroutine (default: "runcase")
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
    output_path = pathlib.Path(output_path)

    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")

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

    # Note: scale and translate are not yet implemented in the new version
    # They would need to be applied to the mesh before calling get_building_grid_indices
    if scale is not None or translate is not None:
        print(
            "Warning: scale and translate parameters are not yet implemented "
            "in the alternative STL conversion. They will be ignored.",
            file=sys.stderr,
        )

    # Step 1: Get building grid indices
    building_data = get_building_grid_indices(
        stl_path, nx, ny, nz, domain_bounds=domain_bounds
    )

    # Step 2: Generate Fortran code
    generate_fortran_code(  # type: ignore[arg-type]
        building_data,
        nx,
        ny,
        nz,
        module_name=module_name,
        subroutine_name=subroutine_name,
        filename=output_path,
    )

    print(f"Generated Fortran geometry file: {output_path}", file=sys.stderr)
