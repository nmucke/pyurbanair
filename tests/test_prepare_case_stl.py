"""Tests for the standalone STL preparation tool (``tools/prepare_case_stl.py``).

The tool merges a raw (buildings + ground) STL pair into a single domain-frame
STL for the forward models. "Matches the old STL files" here means the output
conforms to the **same contract that makes the known-good Xie & Castro STL
usable by uDALES** — not the same geometry. That contract, verified against the
real xie file in ``test_reference_xie_file_defines_the_contract``, is:

* a single, valid, non-empty triangle mesh;
* **binary** STL on disk, so its size is exactly ``84 + 50 * nfaces``
  (uDALES reads ``nfcts = TR.faces.shape[0]`` straight from it);
* in the **domain frame**: ``bounds.min == (0, 0, 0)`` and all vertices ``>= 0``
  (the grid spans ``0 -> xlen/ylen/zsize`` with the floor at ``z = 0``).

Most tests run on small synthetic inputs that reproduce the awkward features of
the real exports (off-origin real-world coordinates, an over-sized ground patch,
a buildings mesh with inconsistent winding and several disjoint bodies). A
slow, opt-in test runs the real ``buildings.stl`` / ``groundpatched_extrude.stl``
when present and ``PYURBANAIR_RUN_STL_INTEGRATION`` is set.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib

import numpy as np
import pytest
import trimesh

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
XIE_STL = REPO_ROOT / "examples" / "xie_and_castro" / "xie_castro_2008_STL.stl"


def _load_tool():
    path = REPO_ROOT / "tools" / "prepare_case_stl.py"
    spec = importlib.util.spec_from_file_location("prepare_case_stl", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


tool = _load_tool()


# --------------------------------------------------------------------------- #
# Contract helpers
# --------------------------------------------------------------------------- #
def assert_domain_frame(mesh: trimesh.Trimesh, tol: float = 1e-3) -> None:
    """Same frame as the xie STL: origin at 0 and no negative coordinates."""
    assert len(mesh.faces) > 0
    lo = mesh.bounds[0]
    assert abs(lo[0]) < tol, f"min x should be 0, got {lo[0]}"
    assert abs(lo[1]) < tol, f"min y should be 0, got {lo[1]}"
    assert abs(lo[2]) < tol, f"min z (floor) should be 0, got {lo[2]}"
    assert (mesh.vertices >= -tol).all(), "all vertices must be >= 0 (inside the domain)"


def assert_binary_stl(path: pathlib.Path, nfaces: int) -> None:
    """A binary STL is exactly 80-byte header + uint32 count + 50 bytes/face."""
    assert path.stat().st_size == 84 + 50 * nfaces, "output is not a binary STL"


# --------------------------------------------------------------------------- #
# Synthetic inputs mimicking the real export format
# --------------------------------------------------------------------------- #
# Real-world-ish origin so the tool must translate it back to (0, 0, 0).
OFFSET = np.array([1000.0, -2000.0])
GROUND_TOP = -3.0
GROUND_BOTTOM = -5.0
BUILDING_HEIGHT = 20.0


def _make_buildings() -> trimesh.Trimesh:
    """Two separate boxes (multi-body); the first has a subset of its faces
    flipped so its winding is genuinely *inconsistent* across shared edges
    (mimicking the broken real ``buildings.stl``). Flipping whole disjoint
    bodies would not do it -- each body would stay internally consistent."""
    b1 = trimesh.creation.box(extents=(10.0, 10.0, BUILDING_HEIGHT))
    b1.apply_translation([OFFSET[0], OFFSET[1], GROUND_TOP + BUILDING_HEIGHT / 2])
    faces = b1.faces.copy()
    faces[::2] = faces[::2][:, ::-1]  # reverse winding on every other face
    b1 = trimesh.Trimesh(vertices=b1.vertices.copy(), faces=faces, process=False)

    b2 = trimesh.creation.box(extents=(8.0, 8.0, BUILDING_HEIGHT))
    b2.apply_translation([OFFSET[0] + 25.0, OFFSET[1] + 5.0, GROUND_TOP + BUILDING_HEIGHT / 2])
    return trimesh.util.concatenate([b1, b2])


def _make_ground() -> trimesh.Trimesh:
    """A flat slab whose footprint is much larger than the buildings.

    Finely subdivided so its triangles are small, like a real terrain mesh —
    the tool crops by face *centroid*, which only works on fine tessellations
    (a single coarse box would have all its triangle centroids near the corners
    and get dropped entirely).
    """
    g = trimesh.creation.box(extents=(120.0, 120.0, GROUND_TOP - GROUND_BOTTOM))
    g.apply_translation([OFFSET[0], OFFSET[1], (GROUND_TOP + GROUND_BOTTOM) / 2])
    return g.subdivide_to_size(8.0)


@pytest.fixture
def synthetic_inputs(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    b_path = tmp_path / "buildings.stl"
    g_path = tmp_path / "ground.stl"
    _make_buildings().export(b_path)
    _make_ground().export(g_path)
    return b_path, g_path


def _prepare(b_path, g_path, **kw):
    defaults = dict(
        center=None,
        size=None,
        margin=0.0,
        z_datum="min",
        max_faces=None,
        repair_buildings=True,
        rotate_buildings=0.0,
        rotate_ground=0.0,
    )
    defaults.update(kw)
    return tool.prepare(b_path, g_path, **defaults)


# --------------------------------------------------------------------------- #
# The contract, anchored on the real known-good file
# --------------------------------------------------------------------------- #
def test_reference_xie_file_defines_the_contract():
    """The old, working xie STL satisfies exactly the invariants we enforce."""
    assert XIE_STL.exists()
    m = trimesh.load(XIE_STL, force="mesh")
    assert_domain_frame(m)
    assert_binary_stl(XIE_STL, len(m.faces))


def test_output_matches_reference_contract(synthetic_inputs, tmp_path):
    b_path, g_path = synthetic_inputs
    mesh = _prepare(b_path, g_path)
    assert_domain_frame(mesh)

    written = tool.write_outputs(
        mesh,
        examples_root=tmp_path / "examples",
        case="barcelona",
        output_name="buildings.stl",
        dry_run=False,
    )
    # Every written file obeys the same binary-STL contract as the xie file.
    for dest in written:
        reloaded = trimesh.load(dest, force="mesh")
        assert_domain_frame(reloaded)
        assert_binary_stl(dest, len(reloaded.faces))
        # Round-trips without changing geometry.
        assert len(reloaded.faces) == len(mesh.faces)
        np.testing.assert_allclose(reloaded.bounds, mesh.bounds, atol=1e-4)


# --------------------------------------------------------------------------- #
# Geometry-preserving behaviour
# --------------------------------------------------------------------------- #
def test_writes_one_shared_stl(synthetic_inputs, tmp_path):
    b_path, g_path = synthetic_inputs
    examples_root = tmp_path / "examples"
    (shared,) = tool.write_outputs(
        _prepare(b_path, g_path),
        examples_root=examples_root,
        case="barcelona",
        output_name="buildings.stl",
        dry_run=False,
    )
    # Exactly one real file lives at examples/<case>/...; all backends reference
    # it by path (no per-backend copies or symlinks).
    assert shared == examples_root / "barcelona" / "buildings.stl"
    assert shared.is_file() and not shared.is_symlink()
    assert not (examples_root / "udales" / "barcelona" / "buildings.stl").exists()
    assert not (examples_root / "lbm" / "barcelona" / "buildings.stl").exists()
    assert not (examples_root / "palm" / "barcelona" / "buildings.stl").exists()


def test_dry_run_writes_nothing(synthetic_inputs, tmp_path):
    b_path, g_path = synthetic_inputs
    mesh = _prepare(b_path, g_path)
    written = tool.write_outputs(
        mesh,
        examples_root=tmp_path / "examples",
        case="barcelona",
        output_name="buildings.stl",
        dry_run=True,
    )
    assert not any(p.exists() for p in written)


def test_merged_mesh_keeps_both_buildings_and_ground(synthetic_inputs):
    b_path, g_path = synthetic_inputs
    mesh = _prepare(b_path, g_path)
    zmax = mesh.bounds[1, 2]
    # Ground sits near z=0; buildings rise to ~building height above it.
    assert zmax > BUILDING_HEIGHT * 0.8, "buildings should be present (tall geometry)"
    face_z = mesh.triangles.mean(axis=1)[:, 2]
    assert (face_z < (GROUND_TOP - GROUND_BOTTOM) + 1.0).any(), "ground layer should be present"
    assert (face_z > BUILDING_HEIGHT * 0.8).any(), "building tops should be present"


def test_translation_puts_window_origin_at_zero(synthetic_inputs):
    b_path, g_path = synthetic_inputs
    # Force a known square window so we can predict the resulting extents.
    mesh = _prepare(b_path, g_path, center=tuple(OFFSET), size=(60.0, 60.0))
    np.testing.assert_allclose(mesh.bounds[0], [0.0, 0.0, 0.0], atol=1e-4)
    # ~60 m window. Centroid-cropping leaves triangles overhanging the window
    # edge, so allow slack of roughly one ground-triangle (~8 m) per side.
    assert 40.0 <= mesh.bounds[1, 0] <= 60.0 + 20.0
    assert 40.0 <= mesh.bounds[1, 1] <= 60.0 + 20.0


@pytest.mark.parametrize("z_datum", ["min", "ground"])
def test_z_datum_never_leaves_geometry_below_floor(synthetic_inputs, z_datum):
    b_path, g_path = synthetic_inputs
    mesh = _prepare(b_path, g_path, z_datum=z_datum)
    assert mesh.bounds[0, 2] >= -1e-4, "no solid may sit below the z=0 floor"


# --------------------------------------------------------------------------- #
# Independent building vs. ground rotation
# --------------------------------------------------------------------------- #
def _rect_building_inputs(tmp_path) -> tuple[pathlib.Path, pathlib.Path]:
    """A single rectangular building (long in x) on a wide flat ground."""
    b = trimesh.creation.box(extents=(40.0, 10.0, BUILDING_HEIGHT))
    b.apply_translation([OFFSET[0], OFFSET[1], GROUND_TOP + BUILDING_HEIGHT / 2])
    g = trimesh.creation.box(extents=(140.0, 140.0, GROUND_TOP - GROUND_BOTTOM))
    g.apply_translation([OFFSET[0], OFFSET[1], (GROUND_TOP + GROUND_BOTTOM) / 2])
    g = g.subdivide_to_size(8.0)
    b_path, g_path = tmp_path / "b.stl", tmp_path / "g.stl"
    b.export(b_path)
    g.export(g_path)
    return b_path, g_path


def _tall_body_xy_extents(mesh) -> tuple[float, float]:
    bodies = [
        x for x in mesh.split(only_watertight=False)
        if x.bounds[1, 2] > BUILDING_HEIGHT * 0.8
    ]
    assert bodies, "no tall building body found"
    tall = max(bodies, key=lambda m: m.bounds[1, 2])
    return float(tall.extents[0]), float(tall.extents[1])


def test_rotate_buildings_is_independent_of_ground(tmp_path):
    """--rotate-buildings rotates only the buildings; --rotate-ground does not."""
    b_path, g_path = _rect_building_inputs(tmp_path)
    win = dict(center=tuple(OFFSET), size=(90.0, 90.0))

    # baseline: the building footprint is long in x.
    x0, y0 = _tall_body_xy_extents(_prepare(b_path, g_path, **win))
    assert x0 > y0 * 2, "baseline building should be long in x"

    # rotating ONLY the buildings 90 deg swaps the footprint to long-in-y.
    xb, yb = _tall_body_xy_extents(
        _prepare(b_path, g_path, rotate_buildings=90.0, **win)
    )
    assert yb > xb * 2, "rotating the buildings should make them long in y"

    # rotating ONLY the ground must NOT change the building's orientation.
    xg, yg = _tall_body_xy_extents(
        _prepare(b_path, g_path, rotate_ground=90.0, **win)
    )
    assert xg > yg * 2, "ground rotation must not rotate the buildings"


def test_shared_rotate_flag_rotates_both(tmp_path):
    """The CLI --rotate is the shared default for both per-mesh angles."""
    b_path, g_path = _rect_building_inputs(tmp_path)
    examples_root = tmp_path / "examples"
    rc = tool.main(
        [str(b_path), str(g_path), "--rotate", "90",
         "--center", str(OFFSET[0]), str(OFFSET[1]), "--size", "90",
         "--examples-root", str(examples_root)]
    )
    assert rc == 0
    mesh = trimesh.load(examples_root / "barcelona" / "buildings.stl", force="mesh")
    xb, yb = _tall_body_xy_extents(mesh)
    assert yb > xb * 2, "--rotate should yaw the buildings too"


# --------------------------------------------------------------------------- #
# PALM regression: a deep basement must not lift the ground into a plateau
# --------------------------------------------------------------------------- #
def _built_fraction(mesh, n: int = 60, thresh: float = 2.0) -> float:
    """Top-down built fraction, mimicking PALM's height-map rasterization."""
    b = mesh.bounds
    cx, cy = (b[0, 0] + b[1, 0]) / 2, (b[0, 1] + b[1, 1]) / 2
    w = min(b[1, 0] - b[0, 0], b[1, 1] - b[0, 1]) * 0.8
    xs = np.linspace(cx - w / 2, cx + w / 2, n)
    ys = np.linspace(cy - w / 2, cy + w / 2, n)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    o = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, b[1, 2] + 1.0)])
    d = np.tile([0.0, 0.0, -1.0], (o.shape[0], 1))
    loc, idr, _ = trimesh.ray.ray_triangle.RayMeshIntersector(mesh).intersects_location(
        o, d, multiple_hits=False
    )
    h = np.zeros(o.shape[0])
    for p, r in zip(loc, idr):
        h[r] = max(h[r], p[2])
    return float((h > thresh).mean())


def test_deep_basement_does_not_lift_ground_into_plateau(tmp_path):
    """Regression for the PALM 'one big building' bug.

    A building basement dipping far below the ground used to drag the whole
    near-zero ground sheet up (via z_datum='min' + the lift guard), so PALM's
    top-down map read a solid plateau. With z_datum='ground' + floor clipping the
    ground stays low and streets stay open.
    """
    g = trimesh.creation.box(extents=(120.0, 120.0, GROUND_TOP - GROUND_BOTTOM))
    g.apply_translation([OFFSET[0], OFFSET[1], (GROUND_TOP + GROUND_BOTTOM) / 2])
    g = g.subdivide_to_size(8.0)
    # one small building with a DEEP basement (z from -30 to +20).
    bld = trimesh.creation.box(extents=(20.0, 20.0, 50.0))
    bld.apply_translation([OFFSET[0], OFFSET[1], -5.0])
    b_path, g_path = tmp_path / "b.stl", tmp_path / "g.stl"
    bld.export(b_path)
    g.export(g_path)

    mesh = _prepare(
        b_path, g_path, z_datum="ground", center=tuple(OFFSET), size=(100.0, 100.0)
    )
    assert mesh.bounds[0, 2] >= -1e-4, "no solid may sit below the z=0 floor"
    frac = _built_fraction(mesh)
    assert frac < 0.4, f"streets should be open after the fix, got built fraction {frac:.2f}"


# --------------------------------------------------------------------------- #
# pylbm regression: flattening makes the ground a strippable z==0 plane
# --------------------------------------------------------------------------- #
def test_flatten_ground_makes_strippable_z0_plane(tmp_path):
    """Option A: the ground must collapse to an EXACT z=0 plane so pylbm's
    'all vertices z==0' rule removes it; otherwise the (sloped) terrain sheet is
    mis-read as one giant building (see docs/palm_ground_topography_issue.md)."""
    # Sloped terrain: a flat slab would trivially clamp to z=0, so tilt the
    # ground in x to represent genuine relief (the case pylbm can't strip).
    b = trimesh.creation.box(extents=(40.0, 10.0, BUILDING_HEIGHT))
    b.apply_translation([OFFSET[0], OFFSET[1], GROUND_TOP + BUILDING_HEIGHT / 2])
    g = trimesh.creation.box(extents=(140.0, 140.0, GROUND_TOP - GROUND_BOTTOM))
    g.apply_translation([OFFSET[0], OFFSET[1], (GROUND_TOP + GROUND_BOTTOM) / 2])
    g = g.subdivide_to_size(8.0)
    gv = g.vertices.copy()
    gv[:, 2] += 0.1 * (gv[:, 0] - OFFSET[0])  # ~10 m relief across the window
    g.vertices = gv
    b_path, g_path = tmp_path / "b.stl", tmp_path / "g.stl"
    b.export(b_path)
    g.export(g_path)
    win = dict(center=tuple(OFFSET), size=(90.0, 90.0))

    flat = _prepare(b_path, g_path, z_datum="ground", flatten_ground=True, **win)
    z = flat.vertices[:, 2]
    gmask = np.all(z[flat.faces] == 0.0, axis=1)  # pylbm's exact ground test
    assert gmask.sum() > 0, "flattened ground must produce exact z==0 faces"
    # Those z==0 faces ARE the ground: they span essentially the whole footprint.
    gverts = flat.vertices[np.unique(flat.faces[gmask])]
    cov_x = np.ptp(gverts[:, 0]) / np.ptp(flat.vertices[:, 0])
    cov_y = np.ptp(gverts[:, 1]) / np.ptp(flat.vertices[:, 1])
    assert cov_x > 0.9 and cov_y > 0.9, "the z==0 plane should span the domain"

    # With terrain kept, the (tilted/elevated) ground is NOT an exact z==0 plane,
    # which is exactly the case pylbm fails to strip.
    kept = _prepare(b_path, g_path, z_datum="ground", flatten_ground=False, **win)
    zk = kept.vertices[:, 2]
    kept_ground = np.all(zk[kept.faces] == 0.0, axis=1)
    kgv = kept.vertices[np.unique(kept.faces[kept_ground])] if kept_ground.any() else None
    spans = kgv is not None and (
        np.ptp(kgv[:, 0]) / np.ptp(kept.vertices[:, 0]) > 0.9
        and np.ptp(kgv[:, 1]) / np.ptp(kept.vertices[:, 1]) > 0.9
    )
    assert not spans, "without flattening there should be no full-domain z==0 plane"


# --------------------------------------------------------------------------- #
# Unit behaviour of the building blocks
# --------------------------------------------------------------------------- #
def test_crop_xy_drops_faces_outside_window():
    ground = _make_ground()
    lo = np.array([OFFSET[0] - 30.0, OFFSET[1] - 30.0])
    hi = np.array([OFFSET[0] + 30.0, OFFSET[1] + 30.0])
    cropped = tool._crop_xy(ground, lo, hi)
    assert 0 < len(cropped.faces) < len(ground.faces)
    centroids = cropped.triangles.mean(axis=1)
    assert (centroids[:, 0] >= lo[0] - 1e-6).all() and (centroids[:, 0] <= hi[0] + 1e-6).all()
    assert (centroids[:, 1] >= lo[1] - 1e-6).all() and (centroids[:, 1] <= hi[1] + 1e-6).all()


def test_repair_makes_building_winding_consistent():
    raw = _make_buildings()
    assert not raw.is_winding_consistent, "synthetic input is intentionally inconsistent"
    fixed = tool._repair_buildings(raw)
    assert fixed.is_winding_consistent, "repair should make the winding consistent"


def test_default_crop_keeps_all_buildings_when_centroid_is_off_center(tmp_path):
    """The default (size=None) window must cover the full buildings footprint.

    Regression: the window was once centred on ``mesh.centroid`` (area/volume
    weighted centre of mass) while sized to the full ``extents``. On a mesh whose
    mass is lopsided -- one big body plus a far-off small one -- that off-centre
    window slides past the far edge and silently drops the small building. The
    centre must be the bounding-box centre so every building is kept.
    """
    big = trimesh.creation.box(extents=(40.0, 40.0, BUILDING_HEIGHT))
    big.apply_translation([OFFSET[0], OFFSET[1], GROUND_TOP + BUILDING_HEIGHT / 2])
    far = trimesh.creation.box(extents=(6.0, 6.0, BUILDING_HEIGHT))
    far.apply_translation([OFFSET[0] + 120.0, OFFSET[1], GROUND_TOP + BUILDING_HEIGHT / 2])
    buildings = trimesh.util.concatenate([big, far])
    # Mass is dominated by the big box, so the centroid sits far from the
    # geometric (bounding-box) centre -- exactly the failure condition.
    assert abs(buildings.centroid[0] - buildings.bounds.mean(axis=0)[0]) > 10.0

    b_path = tmp_path / "buildings.stl"
    g_path = tmp_path / "ground.stl"
    buildings.export(b_path)
    g = trimesh.creation.box(extents=(300.0, 300.0, GROUND_TOP - GROUND_BOTTOM))
    g.apply_translation([OFFSET[0] + 60.0, OFFSET[1], (GROUND_TOP + GROUND_BOTTOM) / 2])
    g.subdivide_to_size(8.0).export(g_path)

    mesh = _prepare(b_path, g_path)
    # Both buildings must survive: the footprint spans ~143 m (big box near edge
    # to far box far edge). The buggy centroid-centred window only reached the
    # big box (~40 m), so anything well past that proves the far box was kept.
    assert mesh.bounds[1, 0] - mesh.bounds[0, 0] > 100.0, "far building was clipped"
    # And both should split out as separate solid bodies (plus the ground).
    tall = [b for b in mesh.split(only_watertight=False)
            if b.bounds[1, 2] > BUILDING_HEIGHT * 0.8]
    assert len(tall) >= 2, "expected both buildings to be present"


def test_empty_crop_raises(synthetic_inputs):
    b_path, g_path = synthetic_inputs
    # A window far from the geometry leaves nothing -> a clear error.
    with pytest.raises(ValueError):
        _prepare(b_path, g_path, center=(50_000.0, 50_000.0), size=(10.0, 10.0))


# --------------------------------------------------------------------------- #
# End-to-end CLI on synthetic inputs
# --------------------------------------------------------------------------- #
def test_cli_main_writes_outputs(synthetic_inputs, tmp_path):
    b_path, g_path = synthetic_inputs
    examples_root = tmp_path / "examples"
    rc = tool.main(
        [
            str(b_path),
            str(g_path),
            "--case",
            "barcelona",
            "--output-name",
            "buildings.stl",
            "--examples-root",
            str(examples_root),
        ]
    )
    assert rc == 0
    shared = examples_root / "barcelona" / "buildings.stl"
    assert shared.is_file()
    # No per-backend copies/symlinks; all backends reference the shared file.
    assert not (examples_root / "udales" / "barcelona" / "buildings.stl").exists()
    m = trimesh.load(shared, force="mesh")
    assert_domain_frame(m)
    assert_binary_stl(shared, len(m.faces))


# --------------------------------------------------------------------------- #
# Opt-in: run against the real raw exports if they are present
# --------------------------------------------------------------------------- #
REAL_BUILDINGS = REPO_ROOT / "buildings.stl"
REAL_GROUND = REPO_ROOT / "groundpatched_extrude.stl"


@pytest.mark.skipif(
    not (
        REAL_BUILDINGS.exists()
        and REAL_GROUND.exists()
        and os.environ.get("PYURBANAIR_RUN_STL_INTEGRATION")
    ),
    reason="set PYURBANAIR_RUN_STL_INTEGRATION=1 and provide the real STLs to run "
    "(loads the ~450 MB ground mesh; slow)",
)
def test_real_files_produce_usable_stl(tmp_path):
    examples_root = tmp_path / "examples"
    rc = tool.main(
        [
            str(REAL_BUILDINGS),
            str(REAL_GROUND),
            "--case",
            "barcelona",
            "--size",
            "200",  # keep facet count bounded for the test
            "--examples-root",
            str(examples_root),
        ]
    )
    assert rc == 0
    dest = examples_root / "barcelona" / "buildings.stl"
    m = trimesh.load(dest, force="mesh")
    assert_domain_frame(m)
    assert_binary_stl(dest, len(m.faces))
    assert len(m.faces) > 1000  # real geometry, not empty
