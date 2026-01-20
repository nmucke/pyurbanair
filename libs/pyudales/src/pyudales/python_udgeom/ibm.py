"""Immersed Boundary Method (IBM) module for uDALES preprocessing."""

import os
import pathlib
import subprocess
from typing import Optional

import numpy as np
import trimesh


def write_ibm_files_using_fortran(
    TR: trimesh.Trimesh,
    xgrid_u: np.ndarray,
    ygrid_u: np.ndarray,
    zgrid_u: np.ndarray,
    xgrid_v: np.ndarray,
    ygrid_v: np.ndarray,
    zgrid_v: np.ndarray,
    xgrid_w: np.ndarray,
    ygrid_w: np.ndarray,
    zgrid_w: np.ndarray,
    xgrid_c: np.ndarray,
    ygrid_c: np.ndarray,
    zgrid_c: np.ndarray,
    fpath: str,
    dx: float,
    dy: float,
    itot: int,
    jtot: int,
    ktot: int,
    stl_ground: bool,
    diag_neighbs: bool,
    periodic_x: bool,
    periodic_y: bool,
    toolsdir: Optional[str] = None,
    tol_mypoly: float = 5e-4,
    n_threads: int = 8,
) -> tuple:
    """
    Write IBM files using Fortran routines.

    This function interfaces with the Fortran preprocessing code.
    Returns counts array similar to MATLAB version.

    Args:
        TR: Triangulation mesh
        xgrid_u, ygrid_u, zgrid_u: Grid coordinates for u-grid
        xgrid_v, ygrid_v, zgrid_v: Grid coordinates for v-grid
        xgrid_w, ygrid_w, zgrid_w: Grid coordinates for w-grid
        xgrid_c, ygrid_c, zgrid_c: Grid coordinates for c-grid
        fpath: Path to write files
        dx, dy: Grid spacing
        itot, jtot, ktot: Grid dimensions
        stl_ground: Whether STL includes ground facets
        diag_neighbs: Whether to use diagonal neighbors
        periodic_x, periodic_y: Periodic boundary conditions
        tol_mypoly: Tolerance for mypoly
        n_threads: Number of threads

    Returns:
        ncounts array with various counts
    """
    # This is a placeholder - the actual implementation would call Fortran code
    # Similar to how MATLAB version does it
    current_path = os.getcwd()

    # Find Fortran preprocessing directory
    # Use toolsdir if provided, otherwise try to find it from environment or relative path
    if toolsdir:
        fortran_path = pathlib.Path(toolsdir) / "IBM" / "IBM_preproc_fortran"
    else:
        # Try environment variable
        da_toolsdir = os.getenv("DA_TOOLSDIR")
        if da_toolsdir:
            fortran_path = pathlib.Path(da_toolsdir) / "IBM" / "IBM_preproc_fortran"
        else:
            # Fallback: try relative to script location
            script_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent
            fortran_path = script_dir / "libs" / "pyudales" / "u-dales" / "tools" / "IBM" / "IBM_preproc_fortran"

    if not fortran_path.exists():
        raise FileNotFoundError(
            f"Fortran preprocessing directory not found: {fortran_path}\n"
            f"Please ensure DA_TOOLSDIR is set correctly or toolsdir parameter is provided."
        )

    os.chdir(fpath)

    # Write input files for Fortran code
    # Match MATLAB fprintf format exactly: %15.10f 
    # Note: Python's f"{val:15.10f}" right-aligns, creating leading spaces, but Fortran's f15.10 format
    # can handle this. The 'x' in Fortran format means "skip whitespace", so multiple spaces are OK.
    with open(os.path.join(fpath, "inmypoly_inp_info.txt"), "w") as f:
        # Line 1: dx dy (f15.10 format) - MATLAB: fprintf(fileID,'%15.10f %15.10f\n',[dx dy]');
        f.write(f"{dx:15.10f} {dy:15.10f}\n")
        # Line 2: itot jtot ktot (i5 format) - MATLAB: fprintf(fileID,'%5d %5d %5d\n',[itot jtot ktot]');
        f.write(f"{itot:5d} {jtot:5d} {ktot:5d}\n")
        # Line 3: tol (f15.10 format) - MATLAB: fprintf(fileID,'%15.10f\n',tol_mypoly);
        f.write(f"{tol_mypoly:15.10f}\n")
        # Lines 4-7: Ray directions (f15.10 format) - MATLAB uses same format
        # MATLAB: fprintf(fileID,'%15.10f %15.10f %15.10f\n',Dir_ray_u);
        # Use numpy array formatting to match MATLAB exactly
        dir_ray_u = np.array([0.0, 0.0, 1.0])
        dir_ray_v = np.array([0.0, 0.0, 1.0])
        dir_ray_w = np.array([0.0, 0.0, 1.0])
        dir_ray_c = np.array([0.0, 0.0, 1.0])
        f.write(f"{dir_ray_u[0]:15.10f} {dir_ray_u[1]:15.10f} {dir_ray_u[2]:15.10f}\n")
        f.write(f"{dir_ray_v[0]:15.10f} {dir_ray_v[1]:15.10f} {dir_ray_v[2]:15.10f}\n")
        f.write(f"{dir_ray_w[0]:15.10f} {dir_ray_w[1]:15.10f} {dir_ray_w[2]:15.10f}\n")
        f.write(f"{dir_ray_c[0]:15.10f} {dir_ray_c[1]:15.10f} {dir_ray_c[2]:15.10f}\n")
        # Line 8: n_vert n_fcts (i8 format) - MATLAB: fprintf(fileID,'%8d %8d\n',[size(TR.Points, 1) size(TR.ConnectivityList, 1)]);
        f.write(f"{TR.vertices.shape[0]:8d} {TR.faces.shape[0]:8d}\n")
        # Line 9: n_threads (i4 format) - MATLAB: fprintf(fileID,'%4d\n',n_threads);
        f.write(f"{n_threads:4d}\n")
        # Line 10: stl_ground diag_neighbs periodic_x periodic_y (i1 format)
        # MATLAB: fprintf(fileID,'%d %d %d %d\n',[stl_ground diag_neighbs periodic_x periodic_y]);
        # Fortran reads as: (i1,x,i1,x,i1,x,i1) - single digit integers with spaces
        f.write(f"{int(stl_ground)} {int(diag_neighbs)} {int(periodic_x)} {int(periodic_y)}\n")
        f.flush()  # Ensure file is written before Fortran reads it
        os.fsync(f.fileno())  # Force write to disk

    # Write grid files - ensure they're flushed
    zhgrid_file = os.path.join(fpath, "zhgrid.txt")
    zfgrid_file = os.path.join(fpath, "zfgrid.txt")
    vertices_file = os.path.join(fpath, "vertices.txt")
    np.savetxt(zhgrid_file, zgrid_w, fmt="%15.10f")
    np.savetxt(zfgrid_file, zgrid_c, fmt="%15.10f")
    np.savetxt(vertices_file, TR.vertices, fmt="%15.10f %15.10f %15.10f")

    # Calculate face centers and normals
    face_centers = []
    face_normals = []
    for face in TR.faces:
        v0, v1, v2 = TR.vertices[face]
        center = (v0 + v1 + v2) / 3.0
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / np.linalg.norm(normal)
        face_centers.append(center)
        face_normals.append(normal)

    face_centers = np.array(face_centers)
    face_normals = np.array(face_normals)

    # Write faces file
    with open(os.path.join(fpath, "faces.txt"), "w") as f:
        for i, face in enumerate(TR.faces):
            f.write(
                f"{face[0]+1:8d} {face[1]+1:8d} {face[2]+1:8d} "
                f"{face_centers[i, 0]:15.10f} {face_centers[i, 1]:15.10f} {face_centers[i, 2]:15.10f} "
                f"{face_normals[i, 0]:15.10f} {face_normals[i, 1]:15.10f} {face_normals[i, 2]:15.10f}\n"
            )

    # Compile and run Fortran code
    os.chdir(fortran_path)
    compile_cmd = [
        "gfortran",
        "-O3",
        "-fopenmp",
        "in_mypoly_functions.f90",
        "boundaryMasking.f90",
        "matchFacetsCells.f90",
        "IBM_preproc_io.f90",
        "IBM_preproc_main.f90",
        "-o",
        "IBM_preproc.exe",
    ]
    subprocess.run(compile_cmd, check=True)

    # Copy executable to fpath
    import shutil

    shutil.copy("IBM_preproc.exe", fpath)
    # Clean up mod files
    for mod_file in ["in_mypoly_functions.mod", "boundaryMasking.mod", "matchFacets2Cells.mod", "IBM_preproc_io.mod"]:
        if os.path.exists(mod_file):
            os.remove(mod_file)
    os.remove("IBM_preproc.exe")

    # Run Fortran executable
    os.chdir(fpath)
    run_cmd = ["./IBM_preproc.exe"]
    subprocess.run(run_cmd, check=True)

    # Clean up
    os.remove("IBM_preproc.exe")
    os.remove("inmypoly_inp_info.txt")
    os.remove("faces.txt")
    os.remove("vertices.txt")
    os.remove("zfgrid.txt")
    os.remove("zhgrid.txt")

    # Read output files and count points
    # The Fortran code writes info_fort.txt with the counts
    info_fort_file = os.path.join(fpath, "info_fort.txt")
    if os.path.exists(info_fort_file):
        # Read ncounts from info_fort.txt (skip first row header, read all values from second row)
        # MATLAB: readmatrix('info_fort.txt', 'Range', [2,1]) reads row 2, all columns
        ncounts = np.loadtxt(info_fort_file, skiprows=1)
        # Ensure it's a 1D array (in case it's read as 2D)
        if ncounts.ndim > 1:
            ncounts = ncounts.flatten()
        # Ensure we have exactly 13 values
        if len(ncounts) != 13:
            raise ValueError(f"Expected 13 values in info_fort.txt, got {len(ncounts)}")
    else:
        # Fallback: count from individual files
        nfcts = TR.faces.shape[0]

        # Read solid points
        solid_u = np.loadtxt(os.path.join(fpath, "solid_u.txt"), skiprows=1, dtype=int)
        solid_v = np.loadtxt(os.path.join(fpath, "solid_v.txt"), skiprows=1, dtype=int)
        solid_w = np.loadtxt(os.path.join(fpath, "solid_w.txt"), skiprows=1, dtype=int)
        solid_c = np.loadtxt(os.path.join(fpath, "solid_c.txt"), skiprows=1, dtype=int)

        # Read fluid boundary points
        fluid_boundary_u = np.loadtxt(os.path.join(fpath, "fluid_boundary_u.txt"), skiprows=1, dtype=int)
        fluid_boundary_v = np.loadtxt(os.path.join(fpath, "fluid_boundary_v.txt"), skiprows=1, dtype=int)
        fluid_boundary_w = np.loadtxt(os.path.join(fpath, "fluid_boundary_w.txt"), skiprows=1, dtype=int)
        fluid_boundary_c = np.loadtxt(os.path.join(fpath, "fluid_boundary_c.txt"), skiprows=1, dtype=int)

        # Read facet sections
        facet_sections_u = np.loadtxt(os.path.join(fpath, "facet_sections_u.txt"), skiprows=1)
        facet_sections_v = np.loadtxt(os.path.join(fpath, "facet_sections_v.txt"), skiprows=1)
        facet_sections_w = np.loadtxt(os.path.join(fpath, "facet_sections_w.txt"), skiprows=1)
        facet_sections_c = np.loadtxt(os.path.join(fpath, "facet_sections_c.txt"), skiprows=1)

        # Create ncounts array (similar to MATLAB version)
        ncounts = np.array(
            [
                nfcts,  # 1: nfcts
                solid_u.shape[0],  # 2: nsolpts_u
                solid_v.shape[0],  # 3: nsolpts_v
                solid_w.shape[0],  # 4: nsolpts_w
                solid_c.shape[0],  # 5: nsolpts_c
                fluid_boundary_u.shape[0],  # 6: nbndpts_u
                fluid_boundary_v.shape[0],  # 7: nbndpts_v
                fluid_boundary_w.shape[0],  # 8: nbndpts_w
                fluid_boundary_c.shape[0],  # 9: nbndpts_c
                facet_sections_u.shape[0],  # 10: nfctsecs_u
                facet_sections_v.shape[0],  # 11: nfctsecs_v
                facet_sections_w.shape[0],  # 12: nfctsecs_w
                facet_sections_c.shape[0],  # 13: nfctsecs_c
            ]
        )

    os.chdir(current_path)

    return ncounts


def write_ibm_files(*args, **kwargs):
    """Write IBM files using MATLAB routines (deprecated)."""
    raise NotImplementedError("MATLAB-based IBM file writing not implemented in Python")


