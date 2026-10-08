"""The building mask is the solvers' own; `on_grid` moves a field between grids."""

import pathlib

import numpy as np
import xarray as xr
from evaluation.turbulence import on_grid, stl_solid_mask

REPO = pathlib.Path(__file__).resolve().parents[2]
XIE_CASTRO = REPO / "geometries/xie_and_castro/xie_castro_2008_STL.stl"

# Corner indices (4 iz + 2 iy + ix) of each box face, as two triangles. The
# roof (4, 5, 7, 6) splits along its (x0, y0) -> (x1, y1) diagonal.
_FACES = [
    (0, 1, 3, 2),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 3, 7, 5),
]


def _write_boxes(path, boxes):
    """Binary STL of closed boxes ``((x0, x1), (y0, y1), (z0, z1))``."""
    tris = []
    for (x0, x1), (y0, y1), (z0, z1) in boxes:
        v = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)])
        for a, b, c, d in _FACES:
            tris += [v[[a, b, c]], v[[a, c, d]]]
    records = np.zeros(
        len(tris),
        [("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")],
    )
    records["vertices"] = tris
    path.write_bytes(bytes(80) + np.uint32(len(tris)).tobytes() + records.tobytes())
    return path


def _below_roofs(boxes, z, y, x):
    """The 2.5-D expectation: solid at or below the highest box top over a column."""
    roof = np.full((y.size, x.size), -np.inf)
    for (x0, x1), (y0, y1), (_, z1) in boxes:
        over = (y[:, None] >= y0) & (y[:, None] <= y1) & (x >= x0) & (x <= x1)
        roof = np.where(over, np.maximum(roof, z1), roof)
    return z[:, None, None] <= roof


def test_centres_on_walls_roofs_and_the_roof_diagonal_are_solid(tmp_path):
    box = ((2.0, 6.0), (2.0, 6.0), (0.0, 3.0))
    stl = _write_boxes(tmp_path / "box.stl", [box])
    # Centres on both walls in x and y, on the roof diagonal (3, 3)...(5, 5)
    # and on the roof itself (z = 3).
    z, y, x = np.array([0.5, 1.5, 2.5, 3.0, 3.5]), np.arange(9.0), np.arange(9.0)

    mask = stl_solid_mask(stl, z, y, x)

    expected = np.zeros_like(mask)
    expected[:4, 2:7, 2:7] = True
    assert (mask == expected).all()
    assert mask[0, [3, 4, 5], [3, 4, 5]].all()


def test_a_centre_just_off_a_wall_is_fluid(tmp_path):
    stl = _write_boxes(tmp_path / "box.stl", [((2.0, 6.0), (2.0, 6.0), (0.0, 3.0))])
    x = np.array([1.9, 2.0, 2.1, 5.9, 6.0, 6.1])

    mask = stl_solid_mask(stl, np.array([1.0, 2.0]), np.array([4.0, 5.0]), x)

    assert mask[0, 0].tolist() == [False, True, True, True, True, False]


def test_overlapping_and_stacked_boxes_are_solid_below_the_highest_roof(tmp_path):
    # Faces inside other boxes give even crossing counts there: a parity rule
    # would open fluid holes in the overlap and under the stacked box.
    boxes = [
        ((2.0, 6.0), (2.0, 6.0), (0.0, 3.0)),
        ((4.0, 8.0), (2.0, 6.0), (0.0, 5.0)),
        ((2.0, 4.0), (2.0, 4.0), (3.0, 4.0)),
    ]
    stl = _write_boxes(tmp_path / "boxes.stl", boxes)
    z, y, x = np.arange(6.0) + 0.5, np.arange(10.0) + 0.5, np.arange(10.0) + 0.5

    mask = stl_solid_mask(stl, z, y, x)

    assert (mask == _below_roofs(boxes, z, y, x)).all()
    assert mask[:5, 3, 5].all()  # the overlap, up to B's roof


def test_xie_castro_mask_is_pylbms_voxelisation():
    """On the canonical grid every cube wall runs through a row of centres."""
    from pylbm.stl_to_lbm import compute_solid_occupancy

    n, bounds = (30, 40, 16), ((-20.0, 40.0), (0.0, 80.0), (0.0, 32.0))
    x, y, z = (
        lo + (np.arange(k) + 0.5) * (hi - lo) / k for k, (lo, hi) in zip(n, bounds)
    )
    domain = {
        f"{a}{m}": bounds[i][j]
        for i, a in enumerate("xyz")
        for j, m in enumerate(("min", "max"))
    }

    mask = stl_solid_mask(XIE_CASTRO, z, y, x)
    lbm = compute_solid_occupancy(XIE_CASTRO, *n, domain_bounds=domain)

    assert mask.sum() == 1368
    assert (mask == lbm.transpose(2, 1, 0)).all()


def test_on_grid_interpolates_only_the_axes_that_differ():
    z_fine, z, y = np.arange(7.0) / 2, np.arange(4.0), np.arange(3.0)
    dims = ("z", "y", "x")
    field = xr.DataArray(
        np.broadcast_to(z_fine[:, None, None], (7, 3, 2)),
        dims=dims,
        coords={"z": z_fine, "y": y, "x": [0.0, 1.0]},
    )
    grid = xr.DataArray(
        np.zeros((4, 3, 2)), dims=dims, coords={"z": z, "y": y, "x": [0.0, 1.0]}
    )

    moved = on_grid(field, grid)

    np.testing.assert_allclose(
        moved.values, np.broadcast_to(z[:, None, None], (4, 3, 2))
    )
