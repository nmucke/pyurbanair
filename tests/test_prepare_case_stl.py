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
XIE_STL = REPO_ROOT / "examples" / "lbm" / "xie_and_castro" / "xie_castro_2008_STL.stl"


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
def test_writes_one_shared_stl_with_udales_symlink(synthetic_inputs, tmp_path):
    b_path, g_path = synthetic_inputs
    examples_root = tmp_path / "examples"
    shared, udales_link = tool.write_outputs(
        _prepare(b_path, g_path),
        examples_root=examples_root,
        case="barcelona",
        output_name="buildings.stl",
        dry_run=False,
    )
    # Exactly one real file lives at examples/<case>/...
    assert shared == examples_root / "barcelona" / "buildings.stl"
    assert shared.is_file() and not shared.is_symlink()
    # uDALES gets a relative symlink to that one file (not a second copy).
    assert udales_link == examples_root / "udales" / "barcelona" / "buildings.stl"
    assert udales_link.is_symlink()
    assert not os.path.isabs(os.readlink(udales_link)), "symlink should be relative"
    assert udales_link.resolve() == shared.resolve()
    # No per-backend copies for lbm/palm — they reference the shared file by path.
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
    udales_link = examples_root / "udales" / "barcelona" / "buildings.stl"
    assert shared.is_file()
    assert udales_link.is_symlink() and udales_link.resolve() == shared.resolve()
    for dest in (shared, udales_link):
        m = trimesh.load(dest, force="mesh")
        assert_domain_frame(m)
        assert_binary_stl(dest, len(m.faces))


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
    dest = examples_root / "udales" / "barcelona" / "buildings.stl"
    m = trimesh.load(dest, force="mesh")
    assert_domain_frame(m)
    assert_binary_stl(dest, len(m.faces))
    assert len(m.faces) > 1000  # real geometry, not empty
