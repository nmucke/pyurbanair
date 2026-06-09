import pathlib
import re

import numpy as np
from scipy import ndimage

from pylbm.stl_to_lbm import process_stl_to_fortran

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "stl_to_lbm"

# New voxel-occupancy output: one blanking line per (i, j) column over a solid
# z-run, e.g. ``blanking(ioff+12, joff+34, 1:5)=.true.``.
_COLUMN_PATTERN = re.compile(
    r"blanking\(\s*ioff\+\s*(\d+)\s*,\s*joff\+\s*(\d+)\s*,\s*"
    r"(\d+)\s*:\s*(\d+)\s*\)\s*=\s*\.true\.",
    flags=re.IGNORECASE,
)


def _extract_column_runs(fortran_text: str) -> list[tuple[int, int, int, int]]:
    return [tuple(map(int, m)) for m in _COLUMN_PATTERN.findall(fortran_text)]


def test_stl_to_lbm_voxel_occupancy_on_xie_castro(tmp_path: pathlib.Path) -> None:
    """The Xie & Castro array of uniform cubes should map to a set of disjoint,
    roughly cube-shaped solid blocks, with no column clamped to the x-boundary.
    """
    stl_path = FIXTURE_DIR / "xie_castro_2008_STL.stl"
    generated_fortran_path = tmp_path / "generated_city3.F90"

    nx, ny, nz = 64, 64, 8

    generated_text = process_stl_to_fortran(
        stl_path=stl_path,
        output_path=generated_fortran_path,
        nx=nx,
        ny=ny,
        nz=nz,
        bounds={
            "xmin": 0.0,
            "xmax": 80.0,
            "ymin": 0.0,
            "ymax": 80.0,
            "zmin": 0.0,
            "zmax": 40.0,
        },
    )

    runs = _extract_column_runs(generated_text)
    assert len(runs) > 0

    # Fortran header/footer sanity.
    assert generated_text.startswith("module m_")
    assert "subroutine" in generated_text
    assert "use mod_dimensions, only : nx, nyg, nz" in generated_text
    assert generated_text.rstrip().endswith("end module")

    # All indices within interior bounds.
    for i, j, z0, z1 in runs:
        assert 1 <= i <= nx
        assert 1 <= j <= ny
        assert 1 <= z0 <= z1 <= nz

    # Reconstruct the footprint and verify several disjoint cube blocks.
    footprint = np.zeros((nx, ny), dtype=bool)
    for i, j, _, _ in runs:
        footprint[i - 1, j - 1] = True

    labels, n_components = ndimage.label(footprint)
    # The benchmark is a regular array of separated cubes.
    assert n_components >= 8

    # Each block should be compact (roughly cube-shaped, not a long smear):
    # bounding-box fill ratio close to 1 for these axis-aligned cubes.
    for lab in range(1, n_components + 1):
        ii, jj = np.nonzero(labels == lab)
        area = len(ii)
        bbox_area = (ii.max() - ii.min() + 1) * (jj.max() - jj.min() + 1)
        assert area / bbox_area > 0.8

    # Regression check for the right-wall artifact: nothing clamped to the
    # outer x boundary.
    assert max(r[0] for r in runs) < nx


def _hollow_square_ring(outer: float, inner: float, height: float):
    """Build a hollow square ring (a courtyard wall) as an explicit triangle
    soup, without any CSG/shapely backend. Outer footprint is [0, outer]^2, the
    inner courtyard hole is centered with side ``inner``. The ring is extruded
    to ``height`` with a roof but no floor (open bottom, like the real STLs).

    Only a roof is strictly needed for the downward ray-cast occupancy: the roof
    is solid over the ring but absent over the courtyard, so the courtyard
    columns get no roof hit and stay fluid.
    """
    import trimesh

    lo = (outer - inner) / 2.0
    hi = lo + inner

    # Four roof quads (annulus) at z=height, each split into two triangles.
    # Layout (top view): bottom strip, top strip, left strip, right strip.
    quads = [
        [(0, 0), (outer, 0), (outer, lo), (0, lo)],  # bottom strip
        [(0, hi), (outer, hi), (outer, outer), (0, outer)],  # top strip
        [(0, lo), (lo, lo), (lo, hi), (0, hi)],  # left strip
        [(hi, lo), (outer, lo), (outer, hi), (hi, hi)],  # right strip
    ]
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for q in quads:
        base = len(verts)
        for (x, y) in q:
            verts.append([x, y, height])
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])

    return trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces))


def test_voxel_occupancy_preserves_interior_holes(tmp_path: pathlib.Path) -> None:
    """An open-bottom box with a courtyard must keep the courtyard fluid."""
    ring = _hollow_square_ring(outer=10.0, inner=4.0, height=5.0)
    stl_path = tmp_path / "ring.stl"
    ring.export(stl_path)

    from pylbm.stl_to_lbm import compute_solid_occupancy

    nx, ny, nz = 20, 20, 10
    solid = compute_solid_occupancy(
        stl_path,
        nx,
        ny,
        nz,
        domain_bounds={
            "xmin": 0.0, "xmax": 10.0,
            "ymin": 0.0, "ymax": 10.0,
            "zmin": 0.0, "zmax": 5.0,
        },
    )

    footprint = solid.any(axis=2)
    # Center columns (the courtyard) must be empty.
    assert not footprint[nx // 2, ny // 2]
    # The walls around it must be solid.
    assert footprint.sum() > 0
    # The empty center is an enclosed hole, not connected to the border.
    empty = ~footprint
    lab, _ = ndimage.label(empty)
    center_label = lab[nx // 2, ny // 2]
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    assert center_label not in border
