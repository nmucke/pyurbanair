"""Python implementation of write_inputs.m for uDALES preprocessing."""

import os
import pathlib
import sys
from typing import Optional

import numpy as np
import trimesh

from . import ibm, preprocessing, seb


def write_inputs(expnr: int, exppath: Optional[str] = None, toolsdir: Optional[str] = None) -> None:
    """
    Main function to generate input files for uDALES.

    Args:
        expnr: Experiment number
        exppath: Path to experiments directory
        toolsdir: Path to tools directory (for Fortran executables)
    """
    if expnr is None:
        raise ValueError("Error: No input argument provided. The script will terminate.")

    expnr_str = f"{expnr:03d}"

    # Get environment variables
    da_expdir = os.getenv("DA_EXPDIR")
    da_toolsdir = os.getenv("DA_TOOLSDIR")

    if exppath is None:
        exppath = da_expdir if da_expdir else "."

    if toolsdir is None:
        toolsdir = da_toolsdir if da_toolsdir else "."

    fpath = os.path.join(exppath, expnr_str)
    namoptionsfile = os.path.join(fpath, f"namoptions.{expnr_str}")

    if not os.path.exists(namoptionsfile):
        raise FileNotFoundError(f"namoptions.{expnr_str} not found in {fpath}")

    # Change to experiment directory
    original_dir = os.getcwd()
    os.chdir(fpath)

    try:
        # Create preprocessing object
        r = preprocessing.Preprocessing(expnr, exppath)

        if r.iexpnr != expnr:
            raise ValueError(
                f"Error: appropriate iexpnr must be set under &RUN in namoptions. "
                f"iexpnr should be the same as the experiment case name. "
                f"Got {r.iexpnr}, expected {expnr}"
            )

        # Set defaults
        preprocessing.Preprocessing.set_defaults(r)

        # Generate grids
        preprocessing.Preprocessing.generate_xygrid(r)
        preprocessing.Preprocessing.generate_zgrid(r)

        # Generate and write lscale
        preprocessing.Preprocessing.generate_lscale(r)
        preprocessing.Preprocessing.write_lscale(r)
        print(f"Written lscal.inp.{r.expnr}")

        # Generate and write prof
        preprocessing.Preprocessing.generate_prof(r)
        preprocessing.Preprocessing.write_prof(r)
        print(f"Written prof.inp.{r.expnr}")

        # Handle scalars
        if r.nsv > 0:
            preprocessing.Preprocessing.generate_scalar(r)
            preprocessing.Preprocessing.write_scalar(r)
            print(f"Written scalar.inp.{r.expnr}")
            if r.lscasrc or r.lscasrcl:
                preprocessing.Preprocessing.generate_scalarsources(r)
                preprocessing.Preprocessing.write_scalarsources(r)
                print(f"Written scalarsources.inp.{r.expnr}")

        # Handle trees
        if r.ltrees or r.ltreesfile:
            print("Generating trees")
            # Note: Tree generation would need to be implemented
            # preprocessing.generate_trees_from_namoptions(r)
            # preprocessing.write_trees(r)
            # print(f"Written trees.inp.{r.expnr}")
            raise NotImplementedError("Tree generation not yet implemented in Python")

        # Handle factypes
        factypes_file = f"factypes.inp.{expnr_str}"
        if os.path.exists(factypes_file):
            r.factypes = np.loadtxt(factypes_file, skiprows=3)
        else:
            preprocessing.Preprocessing.write_factypes(r)
            print(f"Written factypes.inp.{r.expnr}")

        # Handle IBM (geometry)
        if r.libm:
            # Read STL file
            stl_path = os.path.join(fpath, r.stl_file)
            if not os.path.exists(stl_path):
                raise FileNotFoundError(f"STL file not found: {stl_path}")

            TR = trimesh.load(stl_path)
            nfcts = TR.faces.shape[0]
            preprocessing.Preprocessing.set_nfcts(r, nfcts)

            calculate_facet_sections_uvw = r.iwallmom > 1
            calculate_facet_sections_c = r.ltempeq or r.lmoist or r.lwritefac

            if r.gen_geom:
                # Set up grids
                xgrid_c = r.xf
                ygrid_c = r.yf
                zgrid_c = r.zf

                xgrid_u = r.xh[:-1]
                ygrid_u = r.yf
                zgrid_u = r.zf

                xgrid_v = r.xf
                ygrid_v = r.yh[:-1]
                zgrid_v = r.zf

                xgrid_w = r.xf
                ygrid_w = r.yf
                zgrid_w = r.zh[:-1]

                diag_neighbs = r.diag_neighbs
                stl_ground = r.stl_ground
                periodic_x = r.BCxm == 1
                periodic_y = r.BCym == 1

                # Determine which routines to use
                if r.isolid_bound == 1:
                    lmypolyfortran = True
                    lmypoly = False
                elif r.isolid_bound == 2:
                    lmypolyfortran = False
                    lmypoly = True
                elif r.isolid_bound == 3:
                    lmypolyfortran = False
                    lmypoly = False
                else:
                    raise ValueError("Unrecognised option for fluid/solid point classification")

                if r.ifacsec == 1:
                    lmatchFacetsToCellsFortran = True
                elif r.ifacsec == 2:
                    lmatchFacetsToCellsFortran = False
                else:
                    raise ValueError("Unrecognised option for facet section calculation")

                # Write IBM files
                if lmypolyfortran and lmatchFacetsToCellsFortran:
                    ncounts = ibm.write_ibm_files_using_fortran(
                        TR,
                        xgrid_u,
                        ygrid_u,
                        zgrid_u,
                        xgrid_v,
                        ygrid_v,
                        zgrid_v,
                        xgrid_w,
                        ygrid_w,
                        zgrid_w,
                        xgrid_c,
                        ygrid_c,
                        zgrid_c,
                        fpath,
                        r.dx,
                        r.dy,
                        r.itot,
                        r.jtot,
                        r.ktot,
                        stl_ground,
                        diag_neighbs,
                        periodic_x,
                        periodic_y,
                        toolsdir=toolsdir,
                    )
                else:
                    raise NotImplementedError(
                        "MATLAB-based IBM file writing not implemented in Python. "
                        "Use isolid_bound=1 and ifacsec=1 to use Fortran routines."
                    )

                # Update namoptions with counts
                # ncounts indices: [nfcts, nsolpts_u, nsolpts_v, nsolpts_w, nsolpts_c,
                #                  nbndpts_u, nbndpts_v, nbndpts_w, nbndpts_c,
                #                  nfctsecs_u, nfctsecs_v, nfctsecs_w, nfctsecs_c]
                preprocessing.Preprocessing.update_namoptions(
                    namoptionsfile, "&WALLS", "nfctsecs_c", int(ncounts[12])
                )
                preprocessing.Preprocessing.update_namoptions(
                    namoptionsfile, "&WALLS", "nfctsecs_w", int(ncounts[11])
                )
                preprocessing.Preprocessing.update_namoptions(
                    namoptionsfile, "&WALLS", "nfctsecs_v", int(ncounts[10])
                )
                preprocessing.Preprocessing.update_namoptions(
                    namoptionsfile, "&WALLS", "nfctsecs_u", int(ncounts[9])
                )

                preprocessing.Preprocessing.update_namoptions(
                    namoptionsfile, "&WALLS", "nbndpts_c", int(ncounts[8])
                )
                preprocessing.Preprocessing.update_namoptions(
                    namoptionsfile, "&WALLS", "nbndpts_w", int(ncounts[7])
                )
                preprocessing.Preprocessing.update_namoptions(
                    namoptionsfile, "&WALLS", "nbndpts_v", int(ncounts[6])
                )
                preprocessing.Preprocessing.update_namoptions(
                    namoptionsfile, "&WALLS", "nbndpts_u", int(ncounts[5])
                )

                preprocessing.Preprocessing.update_namoptions(
                    namoptionsfile, "&WALLS", "nsolpts_c", int(ncounts[4])
                )
                preprocessing.Preprocessing.update_namoptions(
                    namoptionsfile, "&WALLS", "nsolpts_w", int(ncounts[3])
                )
                preprocessing.Preprocessing.update_namoptions(
                    namoptionsfile, "&WALLS", "nsolpts_v", int(ncounts[2])
                )
                preprocessing.Preprocessing.update_namoptions(
                    namoptionsfile, "&WALLS", "nsolpts_u", int(ncounts[1])
                )

                preprocessing.Preprocessing.update_namoptions(
                    namoptionsfile, "&WALLS", "nfcts", int(ncounts[0])
                )

            else:
                # Copy geometry files from geom_path
                if not r.geom_path:
                    raise ValueError("Need to specify the path to geometry files")
                import shutil

                geom_path = pathlib.Path(r.geom_path)
                for pattern in ["solid_*", "fluid_boundary_*"]:
                    for src_file in geom_path.glob(pattern):
                        shutil.copy(src_file, fpath)

                if calculate_facet_sections_uvw:
                    for pattern in ["facet_sections_u*", "facet_sections_v*", "facet_sections_w*"]:
                        for src_file in geom_path.glob(pattern):
                            shutil.copy(src_file, fpath)

                if calculate_facet_sections_c:
                    for src_file in geom_path.glob("facet_sections_c*"):
                        shutil.copy(src_file, fpath)

            # Set facet types
            if r.read_types:
                facet_types = np.loadtxt(r.types_path, skiprows=1)
            else:
                facet_types = np.ones(r.nfcts, dtype=int)  # defaults to type 1

            # Set type of facets that are not linked with (heat) fluid points to 0
            if calculate_facet_sections_c:
                facet_sections_c_fromfile = np.loadtxt(
                    os.path.join(fpath, "facet_sections_c.txt"), skiprows=1
                )
                facets_used = np.unique(facet_sections_c_fromfile[:, 0].astype(int))
                facets_unused = np.setdiff1d(np.arange(1, r.nfcts + 1), facets_used)
                np.savetxt(
                    f"facets_unused.{r.expnr}",
                    facets_unused,
                    fmt="%d",
                )

            # Calculate face normals
            face_normals = []
            for face in TR.faces:
                v0, v1, v2 = TR.vertices[face]
                normal = np.cross(v1 - v0, v2 - v0)
                normal = normal / np.linalg.norm(normal)
                face_normals.append(normal)
            face_normals = np.array(face_normals)

            preprocessing.Preprocessing.write_facets(r, facet_types, face_normals)
            print(f"Written facets.inp.{r.expnr}")

            # Calculate and write facet areas
            area_facets = seb.facet_areas(TR.faces, TR.vertices)
            preprocessing.Preprocessing.write_facetarea(r, area_facets)
            print(f"Written facetarea.inp.{r.expnr}")

            # Handle energy balance
            if r.lEB:
                # View factors calculation would go here
                # This is a placeholder - full implementation would require View3D integration
                print("Energy balance calculations not yet fully implemented in Python")
                # The following would need View3D and shortwave calculations:
                # - Calculate view factors
                # - Calculate shortwave radiation
                # - Write netsw, svf, vf files
                # - Write initial facet temperatures

    finally:
        os.chdir(original_dir)


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: write_inputs.py <expnr> [exppath] [toolsdir]")
        sys.exit(1)

    expnr = int(sys.argv[1])
    exppath = sys.argv[2] if len(sys.argv) > 2 else None
    toolsdir = sys.argv[3] if len(sys.argv) > 3 else None

    write_inputs(expnr, exppath, toolsdir)


if __name__ == "__main__":
    main()

